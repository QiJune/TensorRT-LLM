# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OpenAI-server facade over the engine-client pipeline.

``OpenAIServingPipeline`` serves eligible chat/completions requests through
the frontend-owned pipeline (processor -> engine client -> assembler ->
formatter). Ineligible requests are handled per deployment mode:

- Co-located: ``try_chat``/``try_completion`` return ``None`` after a debug
  log; the caller falls back to the in-process path. Never an error.
- Detached: there is no in-process path, so ineligible requests receive a
  typed capability-error response and nothing crosses the engine boundary.
"""

from __future__ import annotations

import json
import traceback
import uuid
from http import HTTPStatus
from typing import Any, AsyncGenerator, Optional, Union

from fastapi.responses import JSONResponse, StreamingResponse

from tensorrt_llm.engine_api.contracts import (
    EngineClient,
    EngineErrorCode,
    EngineRequest,
    FrontendOutputConfig,
    RuntimeSamplingConfig,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.frontend.eligibility import (
    EligibilityResult,
    PipelineDeploymentMode,
    check_request,
)
from tensorrt_llm.serve.frontend.openai_formatters import (
    ChatFormatterParams,
    CompletionFormatterParams,
    format_chat_response,
    format_chat_stream_chunks,
    format_completion_response,
    format_completion_stream_chunks,
)
from tensorrt_llm.serve.frontend.request_processor import FrontendProcessor, ProcessedInput
from tensorrt_llm.serve.frontend.response_assembler import FrontendResponseAssembler

__all__ = ["OpenAIServingPipeline"]


def _capability_error_response(reason: str) -> JSONResponse:
    """Typed capability rejection used in detached mode."""
    return JSONResponse(
        status_code=HTTPStatus.BAD_REQUEST,
        content={
            "error": {
                "message": reason,
                "type": "unsupported_capability",
                "code": EngineErrorCode.UNSUPPORTED_CAPABILITY.value,
                "param": None,
            }
        },
    )


def _messages_have_multimodal(messages: list) -> bool:
    """Detect non-text content parts in raw chat messages."""
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") not in (None, "text"):
                    return True
    return False


def _normalize_single_prompt(prompt: Any) -> Optional[Union[str, list[int]]]:
    """Return the single prompt of a completions request, or None if batched."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        if len(prompt) == 0:
            return None
        if isinstance(prompt[0], int):
            return prompt
        if len(prompt) == 1 and isinstance(prompt[0], (str, list)):
            return prompt[0]
    return None


class OpenAIServingPipeline:
    """Serves eligible OpenAI requests through the engine-client pipeline.

    Args:
        engine_client: The engine boundary client (legacy adapter co-located,
            or a socket-backed client).
        processor: The shared frontend input pipeline.
        model_label: Model name echoed in responses.
        mode: Deployment mode deciding fallback-vs-reject for ineligible
            requests.
        reasoning_parser / tool_parser / tool_call_id_type: Formatter
            configuration mirroring the server's historical postprocessing
            arguments.
    """

    def __init__(
        self,
        engine_client: EngineClient,
        processor: FrontendProcessor,
        *,
        model_label: str,
        mode: PipelineDeploymentMode = PipelineDeploymentMode.COLOCATED,
        reasoning_parser: Optional[str] = None,
        tool_parser: Optional[str] = None,
        tool_call_id_type: str = "random",
    ) -> None:
        self._client = engine_client
        self._processor = processor
        self._model_label = model_label
        self._mode = mode
        self._reasoning_parser = reasoning_parser
        self._tool_parser = tool_parser
        self._tool_call_id_type = tool_call_id_type

    @property
    def mode(self) -> PipelineDeploymentMode:
        return self._mode

    @property
    def engine_client(self) -> EngineClient:
        return self._client

    def _handle_ineligible(self, decision: EligibilityResult, endpoint: str):
        if self._mode is PipelineDeploymentMode.COLOCATED:
            logger.debug(
                f"engine-client pipeline fallback for {endpoint} request: {decision.reason}"
            )
            return None
        return _capability_error_response(decision.reason)

    # --- chat -----------------------------------------------------------

    async def try_chat(self, request: Any, raw_request: Any = None):
        """Serve a chat request, or return None to fall back (co-located)."""
        if getattr(request, "logit_bias", None) is not None:
            return self._handle_ineligible(
                EligibilityResult(False, "logit_bias tensors cannot cross the engine boundary"),
                "chat",
            )
        sampling_params = request.to_sampling_params(
            vocab_size=None, reasoning_parser=self._reasoning_parser, backend="pytorch"
        )
        decision = check_request(
            sampling_params,
            endpoint="chat",
            has_multimodal=_messages_have_multimodal(request.messages),
            lora_request=getattr(request, "lora_request", None),
            disaggregated_params=request.disaggregated_params,
        )
        if not decision:
            return self._handle_ineligible(decision, "chat")

        from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines

        conversation, mm_coroutines, mm_placeholder_counts = parse_chat_messages_coroutines(
            request.messages, self._processor._model_config, None
        )
        mm_data, mm_embeddings = await mm_coroutines
        if mm_data or mm_embeddings:
            return self._handle_ineligible(
                EligibilityResult(False, "multimodal requests are served by the in-process path"),
                "chat",
            )

        tool_dicts = (
            None if request.tools is None else [tool.model_dump() for tool in request.tools]
        )
        if request.prompt_token_ids is not None:
            processed = self._processor.process_text(request.prompt_token_ids, sampling_params)
        else:
            processed = self._processor.process_chat(
                conversation,
                sampling_params,
                add_generation_prompt=request.add_generation_prompt,
                mm_placeholder_counts=mm_placeholder_counts,
                tools=tool_dicts,
                documents=request.documents,
                chat_template=request.chat_template,
                chat_template_kwargs=request.chat_template_kwargs,
            )

        params = ChatFormatterParams.from_request(request, model=self._model_label)
        params.reasoning_parser = self._reasoning_parser
        params.tool_parser = self._tool_parser
        params.tool_call_id_type = self._tool_call_id_type
        params.tokenizer = self._processor.tokenizer
        params.num_prompt_tokens = len(processed.prompt_token_ids)
        role = "assistant" if request.add_generation_prompt else request.messages[-1]["role"]
        if (
            conversation
            and conversation[-1].get("content")
            and conversation[-1].get("role") == role
        ):
            params.last_message_content = conversation[-1]["content"]

        if request.stream:
            return StreamingResponse(
                self._stream(processed, params, format_chat_stream_chunks),
                media_type="text/event-stream",
            )
        view = await self._collect(processed, streaming=False)
        return format_chat_response(view, params)

    # --- completions ------------------------------------------------------

    async def try_completion(self, request: Any, raw_request: Any = None):
        """Serve a completions request, or return None to fall back (co-located)."""
        if getattr(request, "logit_bias", None) is not None:
            return self._handle_ineligible(
                EligibilityResult(False, "logit_bias tensors cannot cross the engine boundary"),
                "completions",
            )
        prompt = _normalize_single_prompt(request.prompt)
        sampling_params = request.to_sampling_params(vocab_size=None, backend="pytorch")
        decision = check_request(
            sampling_params,
            endpoint="completions",
            lora_request=getattr(request, "lora_request", None),
            disaggregated_params=request.disaggregated_params,
            num_prompts=1 if prompt is not None else 2,
        )
        if decision and request.echo and not isinstance(prompt, str):
            decision = EligibilityResult(
                False, "echo with pre-tokenized prompts is served by the in-process path"
            )
        if not decision:
            return self._handle_ineligible(decision, "completions")

        processed = self._processor.process_text(prompt, sampling_params)
        params = CompletionFormatterParams.from_request(request, model=self._model_label)
        params.tokenizer = self._processor.tokenizer
        params.num_prompt_tokens = len(processed.prompt_token_ids)
        params.prompt = processed.prompt

        if request.stream:
            return StreamingResponse(
                self._stream(processed, params, format_completion_stream_chunks),
                media_type="text/event-stream",
            )
        view = await self._collect(processed, streaming=False)
        return format_completion_response(view, params)

    # --- shared pipeline drive ----------------------------------------------

    def _submit(self, processed: ProcessedInput, streaming: bool):
        request_id = uuid.uuid4().hex
        engine_request = EngineRequest(
            request_id=request_id,
            prompt_token_ids=list(processed.prompt_token_ids),
            sampling=processed.sampling,
            streaming=streaming,
        )
        handle = self._client.submit(engine_request)
        assembler = _make_assembler(
            request_id,
            processed.sampling,
            processed.output_config,
            streaming,
            self._processor.tokenizer,
        )
        return handle, assembler

    async def _collect(self, processed: ProcessedInput, streaming: bool):
        handle, assembler = self._submit(processed, streaming)
        async for event in handle.aevents():
            assembler.consume(event)
            if assembler.done:
                break
        _raise_on_assembly_error(assembler)
        return assembler.view

    async def _stream(
        self, processed: ProcessedInput, params: Any, formatter
    ) -> AsyncGenerator[str, None]:
        try:
            handle, assembler = self._submit(processed, streaming=True)
            async for event in handle.aevents():
                assembler.consume(event)
                if assembler.error is not None:
                    raise RuntimeError(assembler.error.message)
                for chunk in formatter(assembler.view, params):
                    yield chunk
                if assembler.done:
                    break
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(traceback.format_exc())
            # StreamingResponse commits HTTP 200 before the first chunk;
            # terminate the stream with an SSE error event like the
            # historical handlers do.
            error_data = json.dumps(
                {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": None,
                        "param": None,
                    }
                }
            )
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"


def _make_assembler(
    request_id: str,
    sampling: RuntimeSamplingConfig,
    output_config: FrontendOutputConfig,
    streaming: bool,
    tokenizer: Any,
) -> FrontendResponseAssembler:
    return FrontendResponseAssembler(
        request_id,
        output_config,
        num_sequences=sampling.best_of or sampling.n,
        num_returns=sampling.n,
        use_beam_search=sampling.use_beam_search,
        streaming=streaming,
        tokenizer=tokenizer if output_config.detokenize else None,
    )


def _raise_on_assembly_error(assembler: FrontendResponseAssembler) -> None:
    if assembler.error is not None:
        raise RuntimeError(f"Generation failed: {assembler.error.message}")
