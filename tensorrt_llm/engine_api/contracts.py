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
"""Language-neutral data contracts for the frontend <-> engine boundary.

This module defines the narrow interface between the serving frontend tier
(HTTP server, tokenization, detokenization, output formatting) and the
generation engine (scheduler, model execution). The contract is designed so
that every field that crosses the boundary is plain data — representable in
msgpack without any Python-object serialization — with two explicitly
declared side channels for Python-only and tensor payloads that never cross
the language-neutral wire.

Token-id semantics
------------------
``EngineEvent.token_ids`` carries a **delta** — only the token ids generated
since the previous event for the same ``(request_id, sequence_index)`` — when
``EngineEvent.cumulative`` is False (the default; sampling paths). When
``cumulative`` is True (beam search, where earlier tokens of a beam can be
rewritten), ``token_ids`` carries the **full** output prefix for the sequence
and replaces all previously received tokens.

Sequences
---------
A request produces ``best_of`` sequences (``n`` of which are returned to the
caller); each event names its sequence via ``sequence_index``. ``event_index``
is monotonically increasing per ``(request_id, sequence_index)`` starting at 0.

Terminal invariant
------------------
Every sequence ends with **exactly one** terminal event
(``terminal_kind`` set). No events may follow a terminal event for that
sequence. ``terminal_kind == ERROR`` events must carry a typed
``EngineError``; engine-side exceptions never cross the boundary as pickled
exception objects or stack traces.

Prompt metadata
---------------
Prompt-derived metadata (``prompt_token_ids`` echo, ``prompt_logprobs``) may
only appear on the **first** event (``event_index == 0``) of a sequence.
"""

from __future__ import annotations

import dataclasses
import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, ClassVar, Iterator, Literal, Optional, Union

__all__ = [
    "ContractViolationError",
    "EngineClient",
    "EngineClientError",
    "EngineError",
    "EngineErrorCode",
    "EngineEvent",
    "EngineRequest",
    "EventOrderingChecker",
    "FinishReason",
    "FrontendOutputConfig",
    "GenerationClient",
    "ProtocolViolationError",
    "PythonExtension",
    "RequestHandle",
    "RuntimeControl",
    "RuntimeSamplingConfig",
    "StopKind",
    "TensorAuxiliaryPayload",
    "TerminalKind",
    "TokenLogprob",
    "validate_plain_data",
]

FinishReason = Literal["stop", "length", "timeout", "cancelled", "not_finished"]

# How a 'stop' finish was triggered: the end token, or a stop token sequence
# the runtime matched. Stop *strings* are a frontend concern and never appear
# here.
StopKind = Literal["end_token", "stop_sequence"]


class TerminalKind(str, enum.Enum):
    """How a sequence's event stream terminated."""

    FINISHED = "finished"
    ABORTED = "aborted"
    ERROR = "error"


class EngineErrorCode(str, enum.Enum):
    """Typed error codes carried across the boundary instead of exceptions."""

    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    UNKNOWN_REQUEST = "unknown_request"
    REQUEST_FAILED = "request_failed"
    ENGINE_UNAVAILABLE = "engine_unavailable"
    ENGINE_SHUTDOWN = "engine_shutdown"
    SLOW_CONSUMER = "slow_consumer"
    PROTOCOL_VERSION_MISMATCH = "protocol_version_mismatch"
    PROTOCOL_VIOLATION = "protocol_violation"
    INTERNAL_ERROR = "internal_error"


class ContractViolationError(ValueError):
    """A contract type was constructed with non-plain data in a plain-data field."""


class ProtocolViolationError(RuntimeError):
    """An event stream violated ordering or terminal invariants."""


@dataclass(slots=True, frozen=True)
class EngineError:
    """Typed, wire-safe error payload.

    Args:
        code: Machine-readable error category.
        message: Human-readable description. Must not contain pickled objects
            or full stack traces; a single-line summary is expected.
        request_id: The request this error pertains to, if any.
    """

    code: EngineErrorCode
    message: str
    request_id: Optional[str] = None


class EngineClientError(RuntimeError):
    """Client-facing exception wrapping a typed :class:`EngineError`."""

    def __init__(self, error: EngineError) -> None:
        super().__init__(f"[{error.code.value}] {error.message}")
        self.error = error


@dataclass(slots=True)
class TokenLogprob:
    """Log probability (and optional vocab rank) of one candidate token."""

    logprob: float
    rank: Optional[int] = None


# Per-position logprobs: either the compact form (one float per generated
# token) or the top-k form (candidate-token-id -> TokenLogprob per position).
LogprobsPayload = Union[list[float], list[dict[int, TokenLogprob]]]

_PLAIN_SCALAR_TYPES = (bool, int, float, str, bytes, type(None))
# Composite dataclass types whose fields are themselves validated recursively.
_PLAIN_COMPOSITE_TYPES = (TokenLogprob, EngineError)
_PLAIN_ENUM_TYPES = (TerminalKind, EngineErrorCode)


def validate_plain_data(value: Any, path: str = "value") -> None:
    """Recursively verify that ``value`` contains only wire-safe plain data.

    Allowed: ``None``, ``bool``, ``int``, ``float``, ``str``, ``bytes``,
    lists/tuples and dicts of allowed values (dict keys must be scalars), the
    declared enum types, and the declared plain composite dataclasses.
    Everything else — callables, tensors, arbitrary objects — is rejected.

    Raises:
        ContractViolationError: If a disallowed value is found. The error
            message names the offending path and type.
    """
    if isinstance(value, _PLAIN_ENUM_TYPES):
        return
    if isinstance(value, _PLAIN_SCALAR_TYPES):
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            validate_plain_data(item, f"{path}[{i}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, _PLAIN_SCALAR_TYPES):
                raise ContractViolationError(
                    f"{path} has a non-scalar dict key of type {type(key).__name__}"
                )
            validate_plain_data(item, f"{path}[{key!r}]")
        return
    if isinstance(value, _PLAIN_COMPOSITE_TYPES):
        for f in dataclasses.fields(value):
            validate_plain_data(getattr(value, f.name), f"{path}.{f.name}")
        return
    kind = "callable" if callable(value) else type(value).__name__
    raise ContractViolationError(
        f"{path} contains non-plain data ({kind}); only data-only fields may cross the "
        f"engine boundary — use the declared side channels for Python or tensor payloads"
    )


def _validate_dataclass_plain_fields(obj: Any, side_channels: frozenset[str]) -> None:
    """Validate every non-side-channel field of a contract dataclass."""
    cls_name = type(obj).__name__
    for f in dataclasses.fields(obj):
        if f.name in side_channels:
            continue
        validate_plain_data(getattr(obj, f.name), f"{cls_name}.{f.name}")


@dataclass(slots=True)
class PythonExtension:
    """Declared Python-only side channel of :class:`EngineRequest`.

    Carries values that are legitimately Python objects — a custom logits
    processor and dynamically attached Python-only request fields. This
    channel is capability-gated: it is only usable when frontend and engine
    share a Python process tree, and it never crosses the language-neutral
    wire.
    """

    logits_processor: Optional[Union[Callable, list[Callable]]] = None
    extra_fields: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TensorAuxiliaryPayload:
    """Declared tensor side channel (multimodal/embedding payloads).

    Reserved for future use; transporting tensors over the neutral wire
    requires a dedicated frame format and is deliberately not part of the
    plain-data contract.
    """

    tensors: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeSamplingConfig:
    """Sampling/runtime parameters that cross the boundary. Data-only.

    This is the engine-facing half of today's sampling-params bag: everything
    the scheduler and sampler need, and nothing the frontend keeps for output
    shaping (see :class:`FrontendOutputConfig`).

    ``stop_sequence_token_ids`` carries tokenized stop sequences (derived by
    the frontend from stop strings and stop token ids) so the engine can halt
    generation early; authoritative stop-*string* detection and trimming stay
    frontend-side.
    """

    max_tokens: int = 32
    n: int = 1
    best_of: Optional[int] = None
    use_beam_search: bool = False
    end_id: Optional[int] = None
    pad_id: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    top_p_min: Optional[float] = None
    top_p_reset_ids: Optional[int] = None
    top_p_decay: Optional[float] = None
    min_p: Optional[float] = None
    seed: Optional[int] = None
    min_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    early_stopping: Optional[int] = None
    no_repeat_ngram_size: Optional[int] = None
    prompt_ignore_length: Optional[int] = None
    beam_search_diversity_rate: Optional[float] = None
    beam_width_array: Optional[list[int]] = None
    stop_token_ids: Optional[list[int]] = None
    stop_sequence_token_ids: Optional[list[list[int]]] = None
    bad_token_ids: Optional[list[int]] = None
    bad_sequence_token_ids: Optional[list[list[int]]] = None
    ignore_eos: bool = False
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    logprobs_simple_format: bool = False
    prompt_logprobs_simple_format: bool = False
    return_perf_metrics: bool = False
    # Logprob computation mode ("raw" / "processed"), carried as the enum's
    # string value so it stays plain data on the wire.
    logprobs_mode: Optional[str] = None
    exclude_input_from_output: bool = True

    SIDE_CHANNEL_FIELDS: ClassVar[frozenset[str]] = frozenset()

    def __post_init__(self) -> None:
        _validate_dataclass_plain_fields(self, self.SIDE_CHANNEL_FIELDS)


@dataclass(slots=True)
class FrontendOutputConfig:
    """Output-shaping parameters owned by the frontend. Never crosses.

    Keyed by request id in the frontend's response assembly; the engine never
    sees these. Stop *strings* live here (their detection requires
    detokenized text); stop *token ids* additionally cross runtime-side in
    :class:`RuntimeSamplingConfig` so the engine can stop early.
    """

    detokenize: bool = True
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    stop_strings: Optional[list[str]] = None
    # Tokenized form of each entry in ``stop_strings`` (same order), produced
    # by the frontend's tokenizer. Used for token-level stop attribution and
    # trimming without re-tokenizing.
    stop_sequence_token_ids: Optional[list[list[int]]] = None
    stop_token_ids: Optional[list[int]] = None
    include_stop_str_in_output: bool = False
    num_return_sequences: int = 1
    stream_interval: Optional[int] = None

    SIDE_CHANNEL_FIELDS: ClassVar[frozenset[str]] = frozenset()

    def __post_init__(self) -> None:
        _validate_dataclass_plain_fields(self, self.SIDE_CHANNEL_FIELDS)


@dataclass(slots=True)
class EngineRequest:
    """A generation request as seen by the engine. Data-only plus declared side channels.

    Args:
        request_id: Frontend-assigned correlation id, unique per client.
        prompt_token_ids: Pre-tokenized prompt (the engine never tokenizes).
        sampling: Engine-facing sampling/runtime configuration.
        streaming: Whether the caller consumes incremental events.
        priority: Scheduling priority in [0.0, 1.0].
        cache_salt: Optional KV-cache salt.
        arrival_time: Frontend-observed arrival timestamp (monotonic domain).
        trace_context: Opaque W3C-style trace propagation headers.
        disaggregated_metadata: Opaque disaggregated-serving metadata; passed
            through without interpretation.
        python_extension: Declared Python-only side channel (capability-gated;
            never crosses the language-neutral wire).
        tensor_payload: Declared tensor side channel (reserved; never crosses
            the language-neutral wire).
    """

    request_id: str
    prompt_token_ids: list[int]
    sampling: RuntimeSamplingConfig = field(default_factory=RuntimeSamplingConfig)
    streaming: bool = False
    priority: float = 0.5
    cache_salt: Optional[str] = None
    arrival_time: Optional[float] = None
    trace_context: Optional[dict[str, str]] = None
    disaggregated_metadata: Optional[dict[str, Any]] = None
    python_extension: Optional[PythonExtension] = None
    tensor_payload: Optional[TensorAuxiliaryPayload] = None

    SIDE_CHANNEL_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"python_extension", "tensor_payload"}
    )

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ContractViolationError("EngineRequest.request_id must be a non-empty string")
        if not isinstance(self.sampling, RuntimeSamplingConfig):
            raise ContractViolationError(
                "EngineRequest.sampling must be a RuntimeSamplingConfig, got "
                f"{type(self.sampling).__name__}"
            )
        if not (0.0 <= self.priority <= 1.0):
            raise ContractViolationError(
                f"EngineRequest.priority must be within [0.0, 1.0], got {self.priority}"
            )
        if self.python_extension is not None and not isinstance(
            self.python_extension, PythonExtension
        ):
            raise ContractViolationError(
                "EngineRequest.python_extension must be a PythonExtension side channel"
            )
        if self.tensor_payload is not None and not isinstance(
            self.tensor_payload, TensorAuxiliaryPayload
        ):
            raise ContractViolationError(
                "EngineRequest.tensor_payload must be a TensorAuxiliaryPayload side channel"
            )
        for f in dataclasses.fields(self):
            if f.name in self.SIDE_CHANNEL_FIELDS or f.name == "sampling":
                continue
            validate_plain_data(getattr(self, f.name), f"EngineRequest.{f.name}")

    @property
    def crosses_neutral_wire(self) -> bool:
        """Whether this request is representable on the language-neutral wire."""
        return self.python_extension is None and self.tensor_payload is None


@dataclass(slots=True)
class EngineEvent:
    """One token-level event of a generation stream. Data-only plus declared side channels.

    See the module docstring for delta-vs-cumulative token-id semantics, the
    terminal invariant, and the first-event-only prompt-metadata rule.

    Args:
        request_id: Correlates the event to its :class:`EngineRequest`.
        sequence_index: Which of the request's sequences this event belongs to.
        event_index: Monotonic per-sequence counter starting at 0.
        token_ids: Newly generated token ids (delta), or the full output
            prefix when ``cumulative`` is True.
        cumulative: True when ``token_ids`` replaces previous tokens (beam
            search); False when it appends.
        logprobs: Per-position logprobs for the tokens in ``token_ids``.
        cumulative_logprob: Cumulative log probability of the whole sequence.
        prompt_token_ids: Prompt echo; first event of a sequence only.
        prompt_logprobs: Prompt logprobs; first event of a sequence only.
        finish_reason: Why generation finished; terminal events only.
        stop_kind: For ``finish_reason == "stop"``, whether the end token or a
            runtime-matched stop token sequence triggered it.
        stop_reason: The stop token id that ended generation, if any. Stop
            strings are detected frontend-side and never appear here.
        terminal_kind: Set on the sequence's single terminal event.
        error: Typed error payload; required when ``terminal_kind`` is ERROR.
        disaggregated_metadata: Opaque disaggregated-serving metadata
            passthrough (e.g. context-phase handoff state).
        metrics: Numeric per-request metrics (perf timings, cache hits).
        tensor_payload: Declared tensor side channel (reserved; never crosses
            the language-neutral wire).
    """

    request_id: str
    sequence_index: int = 0
    event_index: int = 0
    token_ids: list[int] = field(default_factory=list)
    cumulative: bool = False
    logprobs: Optional[LogprobsPayload] = None
    cumulative_logprob: Optional[float] = None
    prompt_token_ids: Optional[list[int]] = None
    prompt_logprobs: Optional[LogprobsPayload] = None
    finish_reason: Optional[FinishReason] = None
    stop_kind: Optional[StopKind] = None
    stop_reason: Optional[int] = None
    terminal_kind: Optional[TerminalKind] = None
    error: Optional[EngineError] = None
    disaggregated_metadata: Optional[dict[str, Any]] = None
    metrics: Optional[dict[str, float]] = None
    tensor_payload: Optional[TensorAuxiliaryPayload] = None

    SIDE_CHANNEL_FIELDS: ClassVar[frozenset[str]] = frozenset({"tensor_payload"})

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ContractViolationError("EngineEvent.request_id must be a non-empty string")
        if self.sequence_index < 0 or self.event_index < 0:
            raise ContractViolationError(
                "EngineEvent.sequence_index and event_index must be non-negative"
            )
        if self.terminal_kind == TerminalKind.ERROR and self.error is None:
            raise ContractViolationError(
                "EngineEvent with terminal_kind=ERROR must carry a typed EngineError"
            )
        if self.error is not None and self.terminal_kind != TerminalKind.ERROR:
            raise ContractViolationError(
                "EngineEvent.error is only valid on terminal_kind=ERROR events"
            )
        if self.finish_reason is not None and self.terminal_kind is None:
            raise ContractViolationError(
                "EngineEvent.finish_reason is only valid on terminal events"
            )
        if self.stop_kind is not None and self.finish_reason != "stop":
            raise ContractViolationError(
                "EngineEvent.stop_kind is only valid when finish_reason is 'stop'"
            )
        if self.tensor_payload is not None and not isinstance(
            self.tensor_payload, TensorAuxiliaryPayload
        ):
            raise ContractViolationError(
                "EngineEvent.tensor_payload must be a TensorAuxiliaryPayload side channel"
            )
        for f in dataclasses.fields(self):
            if f.name in self.SIDE_CHANNEL_FIELDS:
                continue
            validate_plain_data(getattr(self, f.name), f"EngineEvent.{f.name}")

    @property
    def is_terminal(self) -> bool:
        return self.terminal_kind is not None


class EventOrderingChecker:
    """Validates the per-sequence ordering and terminal invariants of an event stream.

    Feed every received event to :meth:`observe`; violations raise
    :class:`ProtocolViolationError`. One checker instance validates the
    interleaved streams of any number of requests.
    """

    def __init__(self) -> None:
        # (request_id, sequence_index) -> next expected event_index
        self._next_index: dict[tuple[str, int], int] = {}
        # (request_id, sequence_index) that have seen their terminal event
        self._terminated: set[tuple[str, int]] = set()

    def observe(self, event: EngineEvent) -> None:
        key = (event.request_id, event.sequence_index)
        if key in self._terminated:
            if event.is_terminal:
                raise ProtocolViolationError(
                    f"duplicate terminal event for request {event.request_id!r} "
                    f"sequence {event.sequence_index}"
                )
            raise ProtocolViolationError(
                f"event received after terminal event for request {event.request_id!r} "
                f"sequence {event.sequence_index}"
            )
        expected = self._next_index.get(key, 0)
        if event.event_index != expected:
            raise ProtocolViolationError(
                f"out-of-order event for request {event.request_id!r} sequence "
                f"{event.sequence_index}: expected event_index {expected}, "
                f"got {event.event_index}"
            )
        if event.event_index > 0 and (
            event.prompt_token_ids is not None or event.prompt_logprobs is not None
        ):
            raise ProtocolViolationError(
                f"prompt metadata on non-first event (event_index {event.event_index}) "
                f"for request {event.request_id!r} sequence {event.sequence_index}"
            )
        self._next_index[key] = expected + 1
        if event.is_terminal:
            self._terminated.add(key)

    def sequence_finished(self, request_id: str, sequence_index: int = 0) -> bool:
        return (request_id, sequence_index) in self._terminated

    def forget(self, request_id: str) -> None:
        """Drop tracking state for a completed/aborted request."""
        self._next_index = {k: v for k, v in self._next_index.items() if k[0] != request_id}
        self._terminated = {k for k in self._terminated if k[0] != request_id}


class RequestHandle(ABC):
    """Client-side handle to one submitted request's event stream."""

    @property
    @abstractmethod
    def request_id(self) -> str:
        """The id the request was submitted under."""

    @abstractmethod
    def events(self) -> Iterator[EngineEvent]:
        """Iterate events synchronously, blocking until the terminal event."""

    @abstractmethod
    def aevents(self) -> AsyncIterator[EngineEvent]:
        """Iterate events asynchronously until the terminal event."""

    @abstractmethod
    def abort(self) -> None:
        """Request cancellation. Idempotent; the stream still terminates."""


class GenerationClient(ABC):
    """Data-plane half of the engine boundary: submit, stream, abort."""

    @abstractmethod
    def submit(self, request: EngineRequest) -> RequestHandle:
        """Submit a request and return a handle to its event stream.

        Raises:
            EngineClientError: With a typed :class:`EngineError` when the
                request cannot be accepted (invalid, unsupported capability,
                engine unavailable).
        """

    @abstractmethod
    def abort(self, request_id: str) -> None:
        """Abort a request by id.

        Aborting an already-terminated request is an idempotent no-op.

        Raises:
            EngineClientError: With code ``UNKNOWN_REQUEST`` when the id was
                never submitted through this client.
        """


class RuntimeControl(ABC):
    """Control-plane half of the engine boundary: health, stats, events."""

    @abstractmethod
    def get_capabilities(self) -> dict[str, Any]:
        """Return the engine's advertised capability set (plain data)."""

    @abstractmethod
    def check_health(self) -> bool:
        """Return True when the engine can accept and process requests."""

    @abstractmethod
    def get_stats(self, timeout: float) -> list[dict[str, Any]]:
        """Return queued per-iteration runtime statistics."""

    @abstractmethod
    def get_kv_events(self, timeout: float) -> list[dict[str, Any]]:
        """Return queued KV-cache events."""


class EngineClient(GenerationClient, RuntimeControl, ABC):
    """The complete engine boundary: data plane plus control plane."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release client-side resources. Idempotent."""
