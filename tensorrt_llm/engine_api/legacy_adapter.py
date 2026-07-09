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
"""Adapter exposing today's executor machinery through the engine boundary contracts.

``LegacyEngineClientAdapter`` wraps an existing (unmodified)
:class:`~tensorrt_llm.executor.executor.GenerationExecutor` — proxy, worker,
or RPC variant — and normalizes its raw runtime responses into typed
:class:`~tensorrt_llm.engine_api.contracts.EngineEvent` streams.

Key properties of the adapter:

- Submissions disable runtime-side detokenization and carry no callable of
  any kind (no postprocessing formatter, no hooks). Stop sequences reach the
  runtime pre-tokenized so the engine side never needs a tokenizer.
- Events are token-level and raw: no detokenized text, no stop trimming.
  All output shaping happens in the frontend that consumes the events.
- Raw runtime response objects never escape to callers.

This module imports the heavy executor stack; it must only be imported on
the engine side (never by the detached frontend).
"""

from __future__ import annotations

import threading
from typing import Any, AsyncIterator, Iterator, Optional

from tensorrt_llm.engine_api.contracts import (
    EngineClient,
    EngineClientError,
    EngineError,
    EngineErrorCode,
    EngineEvent,
    EngineRequest,
    RequestHandle,
    RuntimeSamplingConfig,
    TerminalKind,
    TokenLogprob,
)
from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.result import GenerationResult, LogProbsResult, ResponseWrapper
from tensorrt_llm.executor.utils import ErrorResponse, is_llm_response
from tensorrt_llm.sampling_params import SamplingParams

__all__ = ["LegacyEngineClientAdapter"]

# tensorrt_llm.bindings FinishReason values, matched by name to avoid a
# hard import of the C++ bindings in unit-test paths that fake responses.
_FINISH_REASON_BY_NAME = {
    "NOT_FINISHED": None,
    "END_ID": "stop",
    "STOP_WORDS": "stop",
    "LENGTH": "length",
    "TIMED_OUT": "timeout",
    "CANCELLED": "cancelled",
}


def _finish_reason_name(finish_reason: Any) -> str:
    """Return the enum-style name of a runtime finish reason."""
    name = getattr(finish_reason, "name", None)
    if name is None:
        name = str(finish_reason).rsplit(".", maxsplit=1)[-1]
    return name


def _to_sampling_params(request: EngineRequest) -> SamplingParams:
    """Build engine-facing SamplingParams from the data-only request.

    Runtime detokenization is always disabled: the engine emits token ids
    only. Stop sequences are injected pre-tokenized
    (``stop_sequence_token_ids``) so no tokenizer is needed engine-side;
    authoritative stop-string detection stays with the frontend.
    """
    cfg: RuntimeSamplingConfig = request.sampling
    params = SamplingParams(
        max_tokens=cfg.max_tokens,
        n=cfg.n,
        best_of=cfg.best_of,
        use_beam_search=cfg.use_beam_search,
        end_id=cfg.end_id,
        pad_id=cfg.pad_id,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        top_p_min=cfg.top_p_min,
        top_p_reset_ids=cfg.top_p_reset_ids,
        top_p_decay=cfg.top_p_decay,
        min_p=cfg.min_p,
        seed=cfg.seed,
        min_tokens=cfg.min_tokens,
        presence_penalty=cfg.presence_penalty,
        frequency_penalty=cfg.frequency_penalty,
        repetition_penalty=cfg.repetition_penalty,
        length_penalty=cfg.length_penalty if cfg.length_penalty is not None else 1.0,
        early_stopping=cfg.early_stopping if cfg.early_stopping is not None else 1,
        no_repeat_ngram_size=cfg.no_repeat_ngram_size,
        beam_search_diversity_rate=cfg.beam_search_diversity_rate,
        beam_width_array=cfg.beam_width_array,
        stop_token_ids=list(cfg.stop_token_ids) if cfg.stop_token_ids else None,
        bad_token_ids=list(cfg.bad_token_ids) if cfg.bad_token_ids else None,
        ignore_eos=cfg.ignore_eos,
        logprobs=cfg.logprobs,
        prompt_logprobs=cfg.prompt_logprobs,
        logprobs_simple_format=cfg.logprobs_simple_format,
        prompt_logprobs_simple_format=cfg.prompt_logprobs_simple_format,
        return_perf_metrics=cfg.return_perf_metrics,
        detokenize=False,
    )
    if cfg.stop_sequence_token_ids:
        # Placeholder markers keep `_get_stop_words()` emitting the
        # pre-tokenized sequences; the marker strings themselves are never
        # used for text matching on this path (detokenization is off).
        params.stop = [""] * len(cfg.stop_sequence_token_ids)
        params._stop_word_ids = [list(seq) for seq in cfg.stop_sequence_token_ids]
    if cfg.bad_sequence_token_ids:
        params.bad = [""] * len(cfg.bad_sequence_token_ids)
        params._bad_word_ids = [list(seq) for seq in cfg.bad_sequence_token_ids]
    return params


class _ResponseNormalizer:
    """Translates one request's raw runtime responses into EngineEvents."""

    def __init__(self, request: EngineRequest) -> None:
        self._request = request
        self._event_index: dict[int, int] = {}
        self._terminated: set[int] = set()
        self._prompt_echo_sent = False
        self._finished = False

    @property
    def finished(self) -> bool:
        return self._finished

    def _next_index(self, sequence_index: int) -> int:
        index = self._event_index.get(sequence_index, 0)
        self._event_index[sequence_index] = index + 1
        return index

    def _first_event_metadata(self) -> dict[str, Any]:
        if self._prompt_echo_sent:
            return {}
        self._prompt_echo_sent = True
        return {"prompt_token_ids": list(self._request.prompt_token_ids)}

    def _terminal_error_event(self, message: str, code: EngineErrorCode) -> list[EngineEvent]:
        self._finished = True
        events = []
        error = EngineError(code=code, message=message, request_id=self._request.request_id)
        # Terminate every known sequence exactly once (at least sequence 0).
        open_sequences = (set(self._event_index) or {0}) - self._terminated
        for sequence_index in sorted(open_sequences):
            self._terminated.add(sequence_index)
            events.append(
                EngineEvent(
                    request_id=self._request.request_id,
                    sequence_index=sequence_index,
                    event_index=self._next_index(sequence_index),
                    terminal_kind=TerminalKind.ERROR,
                    error=error,
                    **self._first_event_metadata(),
                )
            )
        return events

    def normalize(self, response: Any) -> list[EngineEvent]:
        """Translate one raw runtime response into zero or more events."""
        logprobs_result: Optional[LogProbsResult] = None
        perf_metrics: Optional[dict[str, float]] = None
        if isinstance(response, ResponseWrapper):
            logprobs_result = response.logprobs
            perf_metrics = response.request_perf_metrics
            response = response._response

        if isinstance(response, ErrorResponse):
            return self._terminal_error_event(
                str(response.error_msg), EngineErrorCode.REQUEST_FAILED
            )
        if is_llm_response(response):
            if response.has_error():
                return self._terminal_error_event(
                    str(response.error_msg), EngineErrorCode.REQUEST_FAILED
                )
            return self._normalize_result(response.result, logprobs_result, perf_metrics)
        raise EngineClientError(
            EngineError(
                code=EngineErrorCode.INTERNAL_ERROR,
                message=f"unexpected runtime response type {type(response).__name__}",
                request_id=self._request.request_id,
            )
        )

    def _normalize_result(
        self,
        result: Any,
        logprobs_result: Optional[LogProbsResult],
        perf_metrics: Optional[dict[str, float]],
    ) -> list[EngineEvent]:
        if hasattr(result, "_result") and isinstance(getattr(result, "_result"), bytes):
            result.deserialize()

        is_final = bool(result.is_final)
        if is_final:
            self._finished = True

        use_beam_search = self._request.sampling.use_beam_search
        finish_reasons = result.finish_reasons
        events = []
        if use_beam_search:
            for beam_index in range(len(result.output_token_ids)):
                events.append(
                    self._sequence_event(
                        result,
                        sequence_index=beam_index,
                        source_index=beam_index,
                        cumulative=True,
                        finish_reasons=finish_reasons,
                        logprobs_result=logprobs_result,
                        perf_metrics=perf_metrics,
                        request_final=is_final,
                    )
                )
        else:
            events.append(
                self._sequence_event(
                    result,
                    sequence_index=getattr(result, "sequence_index", 0),
                    source_index=0,
                    cumulative=False,
                    finish_reasons=finish_reasons,
                    logprobs_result=logprobs_result,
                    perf_metrics=perf_metrics,
                    request_final=is_final,
                )
            )
        return [event for event in events if event is not None]

    def _sequence_event(
        self,
        result: Any,
        sequence_index: int,
        source_index: int,
        cumulative: bool,
        finish_reasons: Any,
        logprobs_result: Optional[LogProbsResult],
        perf_metrics: Optional[dict[str, float]],
        request_final: bool,
    ) -> Optional[EngineEvent]:
        if sequence_index in self._terminated:
            return None

        reason_name = "NOT_FINISHED"
        if finish_reasons:
            reason_name = _finish_reason_name(finish_reasons[source_index])
        finish_reason = _FINISH_REASON_BY_NAME.get(reason_name)
        cancelled = reason_name == "CANCELLED"
        sequence_finished = (
            (finish_reason is not None and not cancelled) or cancelled or request_final
        )

        terminal_kind = None
        if sequence_finished:
            self._terminated.add(sequence_index)
            terminal_kind = TerminalKind.ABORTED if cancelled else TerminalKind.FINISHED
            if finish_reason is None and not cancelled:
                # Final response without an explicit reason (e.g. the
                # context-only half of a disaggregated request).
                finish_reason = "not_finished"

        token_ids = list(result.output_token_ids[source_index])

        stop_reason = None
        if reason_name in ("END_ID", "STOP_WORDS"):
            stop_reason = self._token_level_stop_reason(reason_name, token_ids)

        logprobs = None
        if logprobs_result is not None and logprobs_result.generation is not None:
            logprobs = _plain_logprobs(logprobs_result.generation)
        elif getattr(result, "log_probs", None):
            logprobs = _plain_logprobs(result.log_probs[source_index])

        cumulative_logprob = None
        if getattr(result, "cum_log_probs", None):
            cumulative_logprob = result.cum_log_probs[source_index]

        metadata = self._first_event_metadata()
        if logprobs_result is not None and logprobs_result.prompt is not None:
            metadata.setdefault("prompt_token_ids", list(self._request.prompt_token_ids))
            metadata["prompt_logprobs"] = _plain_logprobs(logprobs_result.prompt)

        metrics: dict[str, float] = {}
        if perf_metrics:
            metrics.update(perf_metrics)
        for name in ("decoding_iter", "cached_tokens", "avg_decoded_tokens_per_iter"):
            value = getattr(result, name, None)
            if value is not None:
                metrics[name] = float(value)

        return EngineEvent(
            request_id=self._request.request_id,
            sequence_index=sequence_index,
            event_index=self._next_index(sequence_index),
            token_ids=token_ids,
            cumulative=cumulative,
            logprobs=logprobs,
            cumulative_logprob=cumulative_logprob,
            finish_reason=finish_reason
            if terminal_kind is TerminalKind.FINISHED
            else ("cancelled" if cancelled else None),
            stop_reason=stop_reason,
            terminal_kind=terminal_kind,
            disaggregated_metadata=_disaggregated_metadata(result),
            metrics=metrics or None,
            **metadata,
        )

    def _token_level_stop_reason(self, reason_name: str, token_ids: list[int]) -> Optional[int]:
        """Best-effort token-level stop attribution; string attribution is frontend work."""
        cfg = self._request.sampling
        if reason_name == "END_ID":
            return cfg.end_id
        if cfg.stop_token_ids and token_ids and token_ids[-1] in cfg.stop_token_ids:
            return token_ids[-1]
        return None


def _plain_logprobs(payload: Any) -> Any:
    """Convert runtime logprob payloads into contract-plain data."""
    if payload is None:
        return None
    plain = []
    for entry in payload:
        if isinstance(entry, dict):
            plain.append(
                {
                    int(token_id): TokenLogprob(
                        logprob=float(getattr(item, "logprob", item)),
                        rank=getattr(item, "rank", None),
                    )
                    for token_id, item in entry.items()
                }
            )
        else:
            plain.append(float(entry))
    return plain


def _disaggregated_metadata(result: Any) -> Optional[dict[str, Any]]:
    """Opaque passthrough of disaggregated-serving handoff state."""
    context_phase_params = getattr(result, "context_phase_params", None)
    if context_phase_params is None:
        return None
    metadata: dict[str, Any] = {"request_type": "context_only"}
    for source_name, name in (
        ("first_gen_tokens", "first_gen_tokens"),
        ("req_id", "ctx_request_id"),
        ("opaque_state", "opaque_state"),
        ("draft_tokens", "draft_tokens"),
        ("ctx_dp_rank", "ctx_dp_rank"),
        ("disagg_info_endpoint", "ctx_info_endpoint"),
    ):
        value = getattr(context_phase_params, source_name, None)
        if value is not None:
            metadata[name] = value
    return metadata


class _LegacyRequestHandle(RequestHandle):
    """Handle backed by the legacy result queue; yields normalized events only."""

    def __init__(
        self,
        adapter: "LegacyEngineClientAdapter",
        request: EngineRequest,
        result: GenerationResult,
    ) -> None:
        self._adapter = adapter
        self._request = request
        self._result = result
        self._normalizer = _ResponseNormalizer(request)

    @property
    def request_id(self) -> str:
        return self._request.request_id

    def events(self) -> Iterator[EngineEvent]:
        while not self._normalizer.finished:
            response = self._result.queue.get()
            yield from self._normalizer.normalize(response)
        self._adapter._forget(self._request.request_id)

    async def aevents(self) -> AsyncIterator[EngineEvent]:
        if self._result.aqueue is None:
            raise EngineClientError(
                EngineError(
                    code=EngineErrorCode.INVALID_REQUEST,
                    message="async iteration requires an event loop at submission time",
                    request_id=self._request.request_id,
                )
            )
        while not self._normalizer.finished:
            response = await self._result.aqueue.get()
            for event in self._normalizer.normalize(response):
                yield event
        self._adapter._forget(self._request.request_id)

    def abort(self) -> None:
        self._adapter.abort(self._request.request_id)


class LegacyEngineClientAdapter(EngineClient):
    """EngineClient over an unmodified legacy ``GenerationExecutor``.

    Args:
        executor: Any concrete ``GenerationExecutor`` (IPC proxy, in-process
            worker, or RPC proxy). The adapter never mutates it and never
            renames its API; it only submits through ``generate_async`` and
            consumes the raw per-request response queues.
        capabilities: Optional override of the advertised capability set.
    """

    def __init__(
        self,
        executor: GenerationExecutor,
        capabilities: Optional[dict[str, Any]] = None,
    ) -> None:
        if executor.postproc_config.num_postprocess_workers > 0:
            raise EngineClientError(
                EngineError(
                    code=EngineErrorCode.UNSUPPORTED_CAPABILITY,
                    message="postprocess worker processes are not supported on the "
                    "engine-client path; run with num_postprocess_workers=0",
                )
            )
        self._executor = executor
        self._capabilities = capabilities or _default_capabilities()
        self._handles: dict[str, _LegacyRequestHandle] = {}
        self._request_ids: dict[str, int] = {}
        self._lock = threading.Lock()

    # --- data plane -------------------------------------------------------

    def submit(self, request: EngineRequest) -> RequestHandle:
        if not isinstance(request, EngineRequest):
            raise EngineClientError(
                EngineError(
                    code=EngineErrorCode.INVALID_REQUEST,
                    message=f"expected EngineRequest, got {type(request).__name__}",
                )
            )
        with self._lock:
            if request.request_id in self._request_ids:
                raise EngineClientError(
                    EngineError(
                        code=EngineErrorCode.INVALID_REQUEST,
                        message=f"duplicate request_id {request.request_id!r}",
                        request_id=request.request_id,
                    )
                )
        sampling_params = _to_sampling_params(request)
        logits_processor = (
            request.python_extension.logits_processor if request.python_extension else None
        )
        if logits_processor is not None:
            sampling_params.logits_processor = logits_processor
        result = self._executor.generate_async(
            list(request.prompt_token_ids),
            sampling_params=sampling_params,
            streaming=request.streaming,
            trace_headers=dict(request.trace_context) if request.trace_context else None,
            cache_salt=request.cache_salt,
            arrival_time=request.arrival_time,
            priority=request.priority,
        )
        handle = _LegacyRequestHandle(self, request, result)
        with self._lock:
            self._handles[request.request_id] = handle
            self._request_ids[request.request_id] = result.request_id
        return handle

    def abort(self, request_id: str) -> None:
        with self._lock:
            legacy_id = self._request_ids.get(request_id)
        if legacy_id is None:
            raise EngineClientError(
                EngineError(
                    code=EngineErrorCode.UNKNOWN_REQUEST,
                    message=f"unknown request_id {request_id!r}",
                    request_id=request_id,
                )
            )
        self._executor.abort_request(legacy_id)

    def _forget(self, request_id: str) -> None:
        with self._lock:
            self._handles.pop(request_id, None)
            self._request_ids.pop(request_id, None)

    # --- control plane ------------------------------------------------------

    def get_capabilities(self) -> dict[str, Any]:
        return dict(self._capabilities)

    def check_health(self) -> bool:
        return self._executor.check_health()

    def get_stats(self, timeout: float) -> list[dict[str, Any]]:
        return self._executor.get_stats(timeout)

    def get_kv_events(self, timeout: float) -> list[dict[str, Any]]:
        return self._executor.get_kv_events(timeout)

    def shutdown(self) -> None:
        self._executor.shutdown()


def _default_capabilities() -> dict[str, Any]:
    return {
        "generation": {
            "streaming": True,
            "beam_search": True,
            "num_sequences": True,
            "logprobs": True,
        },
        "control": {
            "health": True,
            "stats": True,
            "kv_events": True,
        },
    }
