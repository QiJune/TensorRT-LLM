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
"""Python LLM API facade over the engine-client pipeline.

Routes eligible text requests from ``LLM.generate_async`` through the shared
frontend pipeline while preserving the request-output interface: the wrapper
returned here iterates one engine event per step (the same cadence as one
runtime response on the historical path) and exposes the same output fields
(``text``, ``token_ids``, ``logprobs``, ``finish_reason``, ``stop_reason``,
diffs, cumulative streaming behavior).

The LLM API stays an in-process Python facade; it never runs against a
detached engine.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional, Union

from tensorrt_llm.engine_api.contracts import EngineClient, EngineRequest
from tensorrt_llm.serve.frontend.eligibility import check_request
from tensorrt_llm.serve.frontend.request_processor import FrontendProcessor
from tensorrt_llm.serve.frontend.response_assembler import FrontendResponseAssembler

__all__ = ["EnginePipelineRequestOutput", "LlmApiEnginePipeline"]

# The only prompt-dict keys the pipeline maps to a plain decoder prompt. Any
# other key (multimodal, encoder/decoder ids, star-attention query, multi-item
# scoring, ...) makes the request ineligible so it is not silently dropped.
_SUPPORTED_PROMPT_KEYS = frozenset({"prompt", "prompt_token_ids"})


class EnginePipelineRequestOutput:
    """Request-output view over the engine-client pipeline.

    Interface-compatible with the historical request output for the fields
    and iteration protocols the text path uses. Each iteration step consumes
    one engine event, mirroring one runtime response of the historical path.
    """

    def __init__(
        self,
        handle: Any,
        assembler: FrontendResponseAssembler,
        prompt: Optional[str],
        prompt_token_ids: list[int],
    ) -> None:
        self._handle = handle
        self._assembler = assembler
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self._sync_events = None
        self._async_events = None

    @property
    def request_id(self) -> str:
        return self._handle.request_id

    @property
    def outputs(self) -> list:
        return self._assembler.view.outputs

    @property
    def finished(self) -> bool:
        return self._assembler.done

    @property
    def error(self) -> Optional[str]:
        engine_error = self._assembler.error
        return None if engine_error is None else engine_error.message

    def abort(self) -> None:
        self._handle.abort()

    def aborted(self) -> bool:
        view_outputs = self._assembler.view._outputs
        return any(output.finish_reason == "cancelled" for output in view_outputs)

    def _step_sync(self) -> None:
        if self._sync_events is None:
            self._sync_events = self._handle.events()
        event = next(self._sync_events)
        self._assembler.consume(event)

    async def _step_async(self) -> None:
        if self._async_events is None:
            self._async_events = self._handle.aevents()
        event = await self._async_events.__anext__()
        self._assembler.consume(event)

    def result(self, timeout: Optional[float] = None) -> "EnginePipelineRequestOutput":
        while not self.finished:
            self._step_sync()
        return self

    async def aresult(self) -> "EnginePipelineRequestOutput":
        while not self.finished:
            await self._step_async()
        return self

    def __await__(self):
        return self.aresult().__await__()

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration
        self._step_sync()
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.finished:
            raise StopAsyncIteration
        await self._step_async()
        return self

    def __repr__(self) -> str:
        return (
            f"EnginePipelineRequestOutput(request_id={self.request_id!r}, "
            f"prompt_token_ids={self.prompt_token_ids}, outputs={self.outputs}, "
            f"finished={self.finished})"
        )


class LlmApiEnginePipeline:
    """Engine-client pipeline entry for the Python LLM API text path."""

    def __init__(self, engine_client: EngineClient, processor: FrontendProcessor) -> None:
        self._client = engine_client
        self._processor = processor

    @property
    def engine_client(self) -> EngineClient:
        return self._client

    def try_generate_async(
        self,
        inputs: Any,
        sampling_params: Any,
        *,
        streaming: bool = False,
        lora_request: Any = None,
        prompt_adapter_request: Any = None,
        disaggregated_params: Any = None,
        cache_salt: Optional[str] = None,
        priority: float = 0.5,
        **unsupported_kwargs: Any,
    ) -> Optional[EnginePipelineRequestOutput]:
        """Serve an eligible text request, or return None to fall back.

        Any keyword the pipeline does not map (multimodal params, KV
        retention config, scheduling/conversation params, ...) makes the
        request ineligible when set. ``cache_salt`` (KV-cache isolation) and
        ``priority`` (scheduling) are mapped onto the engine request rather
        than forcing a fallback.
        """
        prompt = _normalize_text_inputs(inputs)
        if prompt is None:
            return None
        blocking_kwargs = [name for name, value in unsupported_kwargs.items() if value is not None]
        if blocking_kwargs:
            return None
        decision = check_request(
            sampling_params,
            endpoint="llm_api_text",
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            disaggregated_params=disaggregated_params,
        )
        if not decision:
            return None

        processed = self._processor.process_text(prompt, sampling_params)
        request_id = uuid.uuid4().hex
        engine_request = EngineRequest(
            request_id=request_id,
            prompt_token_ids=list(processed.prompt_token_ids),
            sampling=processed.sampling,
            streaming=streaming,
            cache_salt=cache_salt,
            priority=priority,
        )
        handle = self._client.submit(engine_request)
        assembler = FrontendResponseAssembler(
            request_id,
            processed.output_config,
            num_sequences=processed.sampling.best_of or processed.sampling.n,
            num_returns=processed.sampling.n,
            use_beam_search=processed.sampling.use_beam_search,
            streaming=streaming,
            tokenizer=self._processor.tokenizer if processed.output_config.detokenize else None,
        )
        return EnginePipelineRequestOutput(
            handle, assembler, processed.prompt, list(processed.prompt_token_ids)
        )


def _normalize_text_inputs(inputs: Any) -> Optional[Union[str, list[int]]]:
    """Extract a text-path prompt from LLM API inputs; None means ineligible."""
    if isinstance(inputs, str):
        return inputs
    # PreprocessedInputs fast path: token ids pass through untouched as long
    # as no multimodal/query/encoder payload rides along.
    preprocessed_ids = getattr(inputs, "prompt_token_ids", None)
    if preprocessed_ids is not None and not isinstance(inputs, dict):
        if (
            getattr(inputs, "multimodal_params", None) is None
            and getattr(inputs, "query_token_ids", None) is None
            and getattr(inputs, "encoder_input_token_ids", None) is None
        ):
            return list(preprocessed_ids)
        return None
    if isinstance(inputs, list) and inputs and isinstance(inputs[0], int):
        return inputs
    if isinstance(inputs, dict):
        # Reject any dict carrying a field the pipeline does not map —
        # multimodal data/uuids, mm processor kwargs, star-attention
        # query_token_ids, multi-item scoring, encoder/decoder input ids, etc.
        # The historical _preprocess handles or raises for these, so silently
        # dropping the extra payload and submitting a plain decoder prompt
        # would change behavior. Only a bare text/token prompt is eligible.
        if set(inputs.keys()) - _SUPPORTED_PROMPT_KEYS:
            return None
        if "prompt" in inputs and isinstance(inputs["prompt"], str):
            return inputs["prompt"]
        if "prompt_token_ids" in inputs:
            token_ids = inputs["prompt_token_ids"]
            if isinstance(token_ids, list) and (not token_ids or isinstance(token_ids[0], int)):
                return token_ids
    return None
