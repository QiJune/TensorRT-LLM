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
    EngineClientError,
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

# Mirrors executor.request.DEFAULT_REQUEST_PRIORITY (kept local so the
# detached frontend stays import-light — executor.request pulls torch).
DEFAULT_REQUEST_PRIORITY: float = 0.5

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


def _parse_text_conversation(messages: list) -> list[dict[str, Any]]:
    """Flatten text-only chat messages into role/content dicts."""
    conversation = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
        conversation.append({"role": message.get("role"), "content": content or ""})
    return conversation


# W3C distributed-tracing headers (mirrors tracing.TRACE_HEADERS; inlined so
# the detached frontend stays import-light — llmapi.tracing pulls torch).
_TRACE_HEADER_NAMES = ("traceparent", "tracestate")


def _extract_trace_context(raw_request: Any) -> Optional[dict]:
    """Extract distributed-tracing headers from the HTTP request, if any.

    The engine forwards these opaquely; a request carrying trace headers keeps
    its distributed-tracing propagation on the engine-client path.
    """
    headers = getattr(raw_request, "headers", None)
    if not headers:
        return None
    lower_map = {k.lower(): v for k, v in headers.items()}
    extracted = {h: lower_map[h] for h in _TRACE_HEADER_NAMES if h in lower_map}
    return extracted or None


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
        allow_request_chat_template: bool = False,
        gather_generation_logits: bool = False,
    ) -> None:
        self._client = engine_client
        self._processor = processor
        self._model_label = model_label
        self._mode = mode
        self._reasoning_parser = reasoning_parser
        self._tool_parser = tool_parser
        self._tool_call_id_type = tool_call_id_type
        self._allow_request_chat_template = allow_request_chat_template
        self._gather_generation_logits = gather_generation_logits

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
        # The server's chat-template policy applies before any routing
        # decision: co-located, the in-process path raises its historical
        # error; detached, the request is rejected typed.
        if getattr(request, "chat_template", None) is not None and not (
            self._allow_request_chat_template
        ):
            return self._handle_ineligible(
                EligibilityResult(
                    False,
                    "request-level chat templates are disabled by server policy",
                ),
                "chat",
            )
        sampling_params = request.to_sampling_params(
            vocab_size=None,
            gather_generation_logits=self._gather_generation_logits,
            reasoning_parser=self._reasoning_parser,
            backend="pytorch",
        )
        # Apply the historical chat path's sampling mutations before the
        # eligibility check so requests it would turn into guided decoding
        # (strict tools) or raw-special-token output are routed correctly.
        if self._tool_parser and request.tools:
            from tensorrt_llm.serve.tool_parser.strict_mode import (
                build_tool_strict_guided_decoding_params,
            )
            from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory

            tool_parser_cls = ToolParserFactory.parsers.get(self._tool_parser.lower())
            if tool_parser_cls and getattr(tool_parser_cls, "needs_raw_special_tokens", False):
                sampling_params.skip_special_tokens = False
            if sampling_params.guided_decoding is None:
                strict_guided = build_tool_strict_guided_decoding_params(
                    request.tools, self._tool_parser
                )
                if strict_guided is not None:
                    sampling_params.guided_decoding = strict_guided
        # Routing metadata the pipeline does not map to the engine request:
        # served by the in-process path so multi-turn routing and
        # hierarchy-aware scheduling are not silently dropped.
        if getattr(request, "conversation_params", None) is not None:
            return self._handle_ineligible(
                EligibilityResult(False, "conversation_params are served by the in-process path"),
                "chat",
            )
        if getattr(request, "agent_hierarchy", None) is not None:
            return self._handle_ineligible(
                EligibilityResult(
                    False, "agent_hierarchy scheduling is served by the in-process path"
                ),
                "chat",
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

        if self._mode is PipelineDeploymentMode.DETACHED:
            # Import-light text-only parsing: multimodal content was already
            # rejected above, so plain role/content extraction is the whole
            # surface.
            conversation = _parse_text_conversation(request.messages)
            mm_placeholder_counts: list = []
        else:
            from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines

            conversation, mm_coroutines, mm_placeholder_counts = parse_chat_messages_coroutines(
                request.messages, self._processor._model_config, None
            )
            mm_data, mm_embeddings = await mm_coroutines
            if mm_data or mm_embeddings:
                return self._handle_ineligible(
                    EligibilityResult(
                        False, "multimodal requests are served by the in-process path"
                    ),
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

        cache_salt = getattr(request, "cache_salt", None)
        priority = getattr(request, "priority", DEFAULT_REQUEST_PRIORITY)
        trace_context = _extract_trace_context(raw_request)
        if request.stream:
            return StreamingResponse(
                self._stream(
                    processed,
                    params,
                    format_chat_stream_chunks,
                    cache_salt=cache_salt,
                    priority=priority,
                    trace_context=trace_context,
                ),
                media_type="text/event-stream",
            )
        view = await self._collect(
            processed,
            streaming=False,
            cache_salt=cache_salt,
            priority=priority,
            trace_context=trace_context,
        )
        return format_chat_response(view, params)

    # --- completions ------------------------------------------------------

    async def try_completion(self, request: Any, raw_request: Any = None):
        """Serve a completions request, or return None to fall back (co-located)."""
        if getattr(request, "logit_bias", None) is not None:
            return self._handle_ineligible(
                EligibilityResult(False, "logit_bias tensors cannot cross the engine boundary"),
                "completions",
            )
        # conversation_params (sticky multi-turn routing) is not mapped to the
        # engine request; served in-process so the conversation id is not
        # silently dropped (mirrors the chat path).
        if getattr(request, "conversation_params", None) is not None:
            return self._handle_ineligible(
                EligibilityResult(False, "conversation_params are served by the in-process path"),
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

        cache_salt = getattr(request, "cache_salt", None)
        priority = getattr(request, "priority", DEFAULT_REQUEST_PRIORITY)
        trace_context = _extract_trace_context(raw_request)
        if request.stream:
            return StreamingResponse(
                self._stream(
                    processed,
                    params,
                    format_completion_stream_chunks,
                    cache_salt=cache_salt,
                    priority=priority,
                    trace_context=trace_context,
                ),
                media_type="text/event-stream",
            )
        view = await self._collect(
            processed,
            streaming=False,
            cache_salt=cache_salt,
            priority=priority,
            trace_context=trace_context,
        )
        return format_completion_response(view, params)

    # --- shared pipeline drive ----------------------------------------------

    def _submit(
        self,
        processed: ProcessedInput,
        streaming: bool,
        *,
        cache_salt: Optional[str] = None,
        priority: float = DEFAULT_REQUEST_PRIORITY,
        trace_context: Optional[dict] = None,
    ):
        request_id = uuid.uuid4().hex
        engine_request = EngineRequest(
            request_id=request_id,
            prompt_token_ids=list(processed.prompt_token_ids),
            sampling=processed.sampling,
            streaming=streaming,
            cache_salt=cache_salt,
            priority=priority,
            trace_context=dict(trace_context) if trace_context else None,
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

    async def _collect(
        self,
        processed: ProcessedInput,
        streaming: bool,
        *,
        cache_salt: Optional[str] = None,
        priority: float = DEFAULT_REQUEST_PRIORITY,
        trace_context: Optional[dict] = None,
    ):
        handle, assembler = self._submit(
            processed,
            streaming,
            cache_salt=cache_salt,
            priority=priority,
            trace_context=trace_context,
        )
        # The handle aborts the engine request and forgets its bookkeeping in
        # its own generator ``finally`` when this loop stops early (terminal
        # event, or the awaiting coroutine cancelled by a client disconnect).
        async for event in handle.aevents():
            assembler.consume(event)
            if assembler.done:
                break
        _raise_on_assembly_error(assembler)
        return assembler.view

    async def _stream(
        self,
        processed: ProcessedInput,
        params: Any,
        formatter,
        *,
        cache_salt: Optional[str] = None,
        priority: float = DEFAULT_REQUEST_PRIORITY,
        trace_context: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        handle, assembler = self._submit(
            processed,
            streaming=True,
            cache_salt=cache_salt,
            priority=priority,
            trace_context=trace_context,
        )
        try:
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
        # On client disconnect the surrounding StreamingResponse cancels this
        # generator; closing it cascades to ``handle.aevents()``, whose own
        # ``finally`` aborts the still-running engine request and forgets it.


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
        # Raise the typed engine error so the endpoint (co-located) and the
        # detached app convert it into a structured OpenAI error response
        # rather than an unhandled 500.
        raise EngineClientError(assembler.error)
