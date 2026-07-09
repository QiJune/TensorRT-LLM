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
"""Frontend-owned response assembly over raw token-level engine events.

Relocates the output shaping that historically ran inside the executor
result machinery — incremental detokenization, stop-string detection and
trimming (including ``include_stop_str_in_output``), finish-reason mapping,
logprobs accumulation, and streaming diff tracking — into the frontend tier.

The assembler consumes :class:`~tensorrt_llm.engine_api.contracts.EngineEvent`
streams produced by an engine client. Its input contract is strict: events
must be raw and token-level (no pre-detokenized text — the event type cannot
even represent it — and no pre-trimmed stop sequences) and must satisfy the
per-sequence ordering invariants. Violations raise typed errors.

Per-request formatting state (reasoning/tool parser instances, streaming
metadata) lives in the formatter-parameter objects keyed to the request and
persists across streaming chunks; it never crosses to the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from tensorrt_llm.engine_api.contracts import (
    ContractViolationError,
    EngineError,
    EngineEvent,
    EventOrderingChecker,
    FrontendOutputConfig,
    ProtocolViolationError,
    TerminalKind,
)

__all__ = [
    "AssembledRequestView",
    "AssembledSequenceOutput",
    "FrontendResponseAssembler",
]


@dataclass(slots=True)
class AssembledSequenceOutput:
    """Frontend-side accumulation state of one output sequence.

    Mirrors the sequence-output view the endpoint formatters consume: full
    accumulated values plus diff markers for streaming chunks.
    """

    index: int
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    cumulative_logprob: Optional[float] = None
    logprobs: list = field(default_factory=list)
    prompt_logprobs: list = field(default_factory=list)
    finish_reason: Optional[str] = None
    stop_reason: Optional[int | str] = None
    disaggregated_params: Any = None

    _last_text_len: int = field(default=0, repr=False)
    _last_token_ids_len: int = field(default=0, repr=False)
    _last_logprobs_len: int = field(default=0, repr=False)
    _incremental_states: Optional[dict] = field(default=None, repr=False)

    @property
    def length(self) -> int:
        return len(self.token_ids)

    @property
    def text_diff(self) -> str:
        return self.text[self._last_text_len :]

    @property
    def token_ids_diff(self) -> list[int]:
        return self.token_ids[self._last_token_ids_len :]

    @property
    def logprobs_diff(self) -> list:
        return self.logprobs[self._last_logprobs_len :]


class AssembledRequestView:
    """Formatter-facing view of one request's assembled outputs.

    Exposes the attributes the endpoint formatters read: ``outputs`` (with
    the same top-``n`` selection semantics the executor result machinery
    applies when ``best_of > n``), ``_done``, ``cached_tokens``,
    ``avg_decoded_tokens_per_iter``, ``context_logits``, and ``id``.
    """

    def __init__(
        self,
        request_id: str,
        num_sequences: int,
        n: int,
        use_beam_search: bool,
    ) -> None:
        self.id = request_id
        self._n = n
        self._use_beam_search = use_beam_search
        self._outputs = [AssembledSequenceOutput(i) for i in range(num_sequences)]
        self._done = False
        self.cached_tokens = 0
        self.avg_decoded_tokens_per_iter: Optional[float] = None
        self.context_logits = None

    @property
    def outputs(self) -> list[AssembledSequenceOutput]:
        if self._use_beam_search or self._n == len(self._outputs):
            return self._outputs[: self._n]
        sorted_outputs = sorted(
            self._outputs,
            key=lambda x: (
                x.cumulative_logprob if x.cumulative_logprob is not None else float("-inf")
            ),
            reverse=True,
        )
        for i, sorted_out in enumerate(sorted_outputs):
            sorted_out.index = i
        return sorted_outputs[: self._n]


class FrontendResponseAssembler:
    """Assembles one request's raw engine events into formatted output state.

    Args:
        request_id: The request the assembler serves; events for other
            request ids are rejected.
        output_config: Frontend-only output shaping configuration (stop
            strings and their tokenized forms, detokenize flag, special-token
            handling, ``include_stop_str_in_output``, stream interval).
        num_sequences: Total sequences produced by the engine (``best_of``).
        num_returns: Sequences returned to the caller (``n``).
        use_beam_search: Whether events carry cumulative beam prefixes.
        streaming: Whether the caller consumes incremental chunks.
        tokenizer: Frontend tokenizer used for detokenization; may be None
            when ``output_config.detokenize`` is False.
    """

    def __init__(
        self,
        request_id: str,
        output_config: FrontendOutputConfig,
        *,
        num_sequences: int = 1,
        num_returns: int = 1,
        use_beam_search: bool = False,
        streaming: bool = False,
        tokenizer: Any = None,
    ) -> None:
        if output_config.detokenize and tokenizer is None:
            raise ContractViolationError(
                "detokenization requested but no tokenizer provided to the assembler"
            )
        self._request_id = request_id
        self._config = output_config
        self._tokenizer = tokenizer
        self._streaming = streaming
        self._use_beam_search = use_beam_search
        self._checker = EventOrderingChecker()
        self._view = AssembledRequestView(request_id, num_sequences, num_returns, use_beam_search)
        self._terminated: set[int] = set()
        self._num_sequences = num_sequences
        self._error: Optional[EngineError] = None
        self._stop_reasons_and_words = _stop_reasons_and_words(output_config)

    @property
    def view(self) -> AssembledRequestView:
        return self._view

    @property
    def done(self) -> bool:
        return self._view._done

    @property
    def error(self) -> Optional[EngineError]:
        return self._error

    def consume(self, event: EngineEvent) -> None:
        """Fold one raw engine event into the assembled state.

        Raises:
            ContractViolationError: If the event is not an ``EngineEvent``
                or belongs to a different request.
            ProtocolViolationError: If ordering/terminal invariants are
                violated or a runtime-matched stop sequence arrives already
                trimmed (the event stream is not raw token-level data).
        """
        if not isinstance(event, EngineEvent):
            raise ContractViolationError(
                f"assembler input must be EngineEvent, got {type(event).__name__}"
            )
        if event.request_id != self._request_id:
            raise ContractViolationError(
                f"event for request {event.request_id!r} fed to assembler of {self._request_id!r}"
            )

        # Error terminals are position-independent: a socket-side engine
        # failure arrives with event_index 0 even after partial output has
        # streamed. Preserve the typed engine error rather than letting the
        # ordering checker reject it as out-of-order.
        if event.terminal_kind is TerminalKind.ERROR:
            self._error = event.error
            self._terminated.add(event.sequence_index)
            self._view._done = True
            return

        self._checker.observe(event)

        self._fold_sequence(event)
        if self._config.detokenize and self._tokenizer is not None:
            self._detokenize_and_scan_stop_strings()

    # --- token-level folding (relocated from the executor result machinery) --

    def _fold_sequence(self, event: EngineEvent) -> None:
        output = self._view._outputs[event.sequence_index]
        output._last_token_ids_len = len(output.token_ids)
        if event.cumulative:
            output.token_ids = list(event.token_ids)
        else:
            output.token_ids.extend(event.token_ids)

        if event.cumulative_logprob is not None:
            output.cumulative_logprob = event.cumulative_logprob
        if event.prompt_logprobs is not None:
            output.prompt_logprobs = event.prompt_logprobs
        if event.logprobs is not None:
            output._last_logprobs_len = len(output.logprobs)
            if event.cumulative:
                # Beam search: the token list replaced the prefix, so the
                # logprobs must replace it too — appending would leave
                # output.logprobs longer than output.token_ids.
                output.logprobs = list(event.logprobs)
            else:
                output.logprobs = output.logprobs + list(event.logprobs)

        if event.metrics:
            cached = event.metrics.get("cached_tokens")
            if cached is not None:
                self._view.cached_tokens = int(cached)
            avg_decoded = event.metrics.get("avg_decoded_tokens_per_iter")
            if avg_decoded is not None:
                self._view.avg_decoded_tokens_per_iter = avg_decoded

        if event.terminal_kind is TerminalKind.ABORTED:
            output.finish_reason = "cancelled"
            self._terminated.add(event.sequence_index)
        elif event.terminal_kind is TerminalKind.FINISHED:
            self._terminated.add(event.sequence_index)
            self._apply_finish(output, event)

        if len(self._terminated) >= self._num_sequences:
            self._view._done = True

    def _apply_finish(self, output: AssembledSequenceOutput, event: EngineEvent) -> None:
        if event.finish_reason == "stop":
            output.finish_reason = "stop"
            if event.stop_kind == "stop_sequence":
                for stop_reason, stop_ids in self._stop_reasons_and_words:
                    if output.token_ids[-len(stop_ids) :] == stop_ids:
                        output.stop_reason = stop_reason
                        if not self._config.include_stop_str_in_output:
                            output.token_ids = output.token_ids[: -len(stop_ids)]
                        break
                else:
                    raise ProtocolViolationError(
                        f"request {self._request_id!r} sequence {event.sequence_index}: "
                        "runtime reported a stop-sequence match but no configured stop "
                        "sequence matches the token tail — the event stream appears "
                        "pre-trimmed and violates the raw token-level input contract"
                    )
        elif event.finish_reason in ("length", "timeout", "not_finished"):
            output.finish_reason = event.finish_reason
        elif event.finish_reason is not None:
            output.finish_reason = event.finish_reason

    # --- text-level shaping (relocated from the detokenizing result class) --

    def _detokenize_and_scan_stop_strings(self) -> None:
        kwargs = {
            "skip_special_tokens": self._config.skip_special_tokens,
            "spaces_between_special_tokens": self._config.spaces_between_special_tokens,
        }
        for output in self._view.outputs:
            output._last_text_len = len(output.text)
            if (
                hasattr(self._tokenizer, "decode_incrementally")
                and self._streaming
                and not self._use_beam_search
            ):
                output.text, output._incremental_states = self._tokenizer.decode_incrementally(
                    output.token_ids_diff,
                    prev_text=output.text,
                    states=output._incremental_states,
                    flush=self._view._done,
                    stream_interval=self._config.stream_interval or 1,
                    **kwargs,
                )
            else:
                output.text = self._tokenizer.decode(output.token_ids, **kwargs)

            is_generating = not self._view._done
            is_finished_with_stop_or_length = (
                output.finish_reason == "stop" or output.finish_reason == "length"
            )

            if is_generating or is_finished_with_stop_or_length:
                for stop_reason, _ in self._stop_reasons_and_words:
                    if isinstance(stop_reason, str) and stop_reason in output.text:
                        stop_pos = output.text.find(stop_reason)
                        if not self._config.include_stop_str_in_output:
                            output.text = output.text[:stop_pos]
                        else:
                            output.text = output.text[: stop_pos + len(stop_reason)]

                        output.finish_reason = "stop"
                        output.stop_reason = stop_reason
                        self._view._done = True
                        break


def _stop_reasons_and_words(
    config: FrontendOutputConfig,
) -> list[tuple[int | str, list[int]]]:
    """(stop reason, token sequence) pairs, in the runtime's matching order."""
    reasons: list[int | str] = []
    words: list[list[int]] = []
    if config.stop_token_ids:
        reasons.extend(config.stop_token_ids)
        words.extend([token_id] for token_id in config.stop_token_ids)
    if config.stop_strings:
        sequences = config.stop_sequence_token_ids or []
        if len(sequences) != len(config.stop_strings):
            raise ContractViolationError(
                "FrontendOutputConfig.stop_sequence_token_ids must align with stop_strings"
            )
        reasons.extend(config.stop_strings)
        words.extend(list(sequence) for sequence in sequences)
    return list(zip(reasons, words))
