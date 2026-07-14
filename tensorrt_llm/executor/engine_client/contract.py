# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Typed wire contract between a serving frontend and the generation engine.

Every type in this module is a frozen, primitives-only, callable-free
dataclass carrying ``protocol_version``. Construction validates types
strictly: ``bool`` is never accepted where ``int`` is required, identifiers
must be non-empty strings, token ids are 64-bit int tuples, and
mapping-valued fields are converted to immutable views. Nothing here may
hold a live Python object graph — these types define the future
inter-process wire and must survive ``codec`` round-trips unchanged.

See ``ENGINE_CONTRACT.md`` in this package for the scope matrix, the
rejection-to-test table, and divergence notes against the design draft.
"""

import dataclasses
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Optional, Union

__all__ = [
    "ENGINE_CONTRACT_VERSION",
    "FINISH_REASONS",
    "REQUEST_STATUSES",
    "GUIDED_DECODING_MODES",
    "MAX_PROMPT_TOKENS",
    "MAX_STOP_SEQUENCES",
    "MAX_STOP_SEQUENCE_TOKENS",
    "MAX_STRING_CHARS",
    "MAX_METRICS_ENTRIES",
    "INT64_MIN",
    "INT64_MAX",
    "ContractError",
    "ContractConstructionError",
    "EngineCapabilities",
    "EngineSamplingConfig",
    "GuidedDecodingSpec",
    "EngineRequest",
    "TokenDelta",
    "Terminal",
    "RequestComplete",
    "ErrorFrame",
    "OutputFrame",
    "OUTPUT_FRAME_TYPES",
    "FrontendOutputConfig",
    "TokenizerSpec",
    "FrontendModelContext",
    "EngineHealth",
    "IterationStatsBatch",
    "KvCacheEventsBatch",
    "CACHED_TOKENS_METRIC_KEY",
    "validate_no_callables",
]

ENGINE_CONTRACT_VERSION = 1
"""int: Current engine contract protocol version."""

FINISH_REASONS = ("stop", "length", "abort", "error")
"""Valid ``Terminal.finish_reason`` values (sole carrier of finish state)."""

REQUEST_STATUSES = ("ok", "aborted", "failed")
"""Valid ``RequestComplete.status`` values."""

GUIDED_DECODING_MODES = ("json_schema", "json_object", "regex", "grammar", "structural_tag")
"""Valid ``GuidedDecodingSpec.mode`` values (schema-as-data; execution is engine-side)."""

CACHED_TOKENS_METRIC_KEY = "cached_tokens"
"""Reserved ``TokenDelta.metrics`` key carrying pre-completion cached-token accounting."""

# Resource limits enforced at construction time (the codec enforces its own
# byte-level limits on top of these; see codec.py).
MAX_PROMPT_TOKENS = 1_048_576
MAX_STOP_SEQUENCES = 1_024
MAX_STOP_SEQUENCE_TOKENS = 256
MAX_STRING_CHARS = 4 * 1024 * 1024
MAX_METRICS_ENTRIES = 256
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


class ContractError(Exception):
    """Base class for all engine-contract errors."""


class ContractConstructionError(ContractError):
    """A wire type was constructed with invalid content."""


def _fail(type_name: str, field: str, message: str) -> "ContractConstructionError":
    return ContractConstructionError(f"{type_name}.{field}: {message}")


def _check_int(type_name: str, field: str, value, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(type_name, field, f"expected int, got {type(value).__name__}")
    if not INT64_MIN <= value <= INT64_MAX:
        raise _fail(type_name, field, "outside signed 64-bit range")
    if minimum is not None and value < minimum:
        raise _fail(type_name, field, f"must be >= {minimum}, got {value}")
    return value


def _check_opt_int(type_name: str, field: str, value, *, minimum: Optional[int] = None) -> Optional[int]:
    if value is None:
        return None
    return _check_int(type_name, field, value, minimum=minimum)


def _check_float(type_name: str, field: str, value) -> float:
    if isinstance(value, bool):
        raise _fail(type_name, field, "expected float, got bool")
    if isinstance(value, int):
        value = float(value)
    if not isinstance(value, float):
        raise _fail(type_name, field, f"expected float, got {type(value).__name__}")
    if not math.isfinite(value):
        raise _fail(type_name, field, "must be finite (NaN/Inf rejected)")
    return value


def _check_opt_float(type_name: str, field: str, value) -> Optional[float]:
    if value is None:
        return None
    return _check_float(type_name, field, value)


def _check_str(type_name: str, field: str, value, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise _fail(type_name, field, f"expected str, got {type(value).__name__}")
    if not allow_empty and not value:
        raise _fail(type_name, field, "must be a non-empty string")
    if len(value) > MAX_STRING_CHARS:
        raise _fail(type_name, field, f"exceeds {MAX_STRING_CHARS} characters")
    return value


def _check_opt_str(type_name: str, field: str, value, *, allow_empty: bool = True) -> Optional[str]:
    if value is None:
        return None
    return _check_str(type_name, field, value, allow_empty=allow_empty)


def _check_bool(type_name: str, field: str, value) -> bool:
    if not isinstance(value, bool):
        raise _fail(type_name, field, f"expected bool, got {type(value).__name__}")
    return value


def _int_tuple(type_name: str, field: str, value, *, allow_empty: bool = True,
               max_items: Optional[int] = None) -> tuple:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _fail(type_name, field, f"expected a sequence of ints, got {type(value).__name__}")
    items = tuple(value)
    if not allow_empty and not items:
        raise _fail(type_name, field, "must not be empty")
    if max_items is not None and len(items) > max_items:
        raise _fail(type_name, field, f"exceeds {max_items} items")
    for i, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, int):
            raise _fail(type_name, field, f"item {i}: expected int, got {type(item).__name__}")
        if not INT64_MIN <= item <= INT64_MAX:
            raise _fail(type_name, field, f"item {i}: outside signed 64-bit range")
    return items


def _float_tuple(type_name: str, field: str, value) -> Optional[tuple]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _fail(type_name, field, f"expected a sequence of floats, got {type(value).__name__}")
    return tuple(_check_float(type_name, f"{field}[{i}]", item) for i, item in enumerate(value))


def _str_tuple(type_name: str, field: str, value, *, allow_empty_items: bool = False) -> tuple:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _fail(type_name, field, f"expected a sequence of strings, got {type(value).__name__}")
    return tuple(
        _check_str(type_name, f"{field}[{i}]", item, allow_empty=allow_empty_items)
        for i, item in enumerate(value))


def _freeze_metrics(type_name: str, field: str, value) -> Optional[Mapping]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise _fail(type_name, field, f"expected a mapping, got {type(value).__name__}")
    if len(value) > MAX_METRICS_ENTRIES:
        raise _fail(type_name, field, f"exceeds {MAX_METRICS_ENTRIES} entries")
    frozen = {}
    for key in value:
        if not isinstance(key, str) or not key:
            raise _fail(type_name, field, "metric keys must be non-empty strings")
        frozen[key] = _check_float(type_name, f"{field}[{key!r}]", value[key])
    return MappingProxyType(frozen)


def _check_protocol_version(type_name: str, value) -> int:
    version = _check_int(type_name, "protocol_version", value, minimum=1)
    return version


def validate_no_callables(obj, _path: str = "value") -> None:
    """Reject any callable anywhere in a wire value graph.

    Walks dataclasses, mappings, sequences, and primitives. Raises
    ``ContractConstructionError`` when a callable (or an unrecognized
    object that could smuggle behavior) is found.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return
    if callable(obj):
        raise ContractConstructionError(f"{_path}: callables are forbidden on the wire")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            validate_no_callables(getattr(obj, field.name), f"{_path}.{field.name}")
        return
    if isinstance(obj, Mapping):
        for key, item in obj.items():
            validate_no_callables(key, f"{_path} key")
            validate_no_callables(item, f"{_path}[{key!r}]")
        return
    if isinstance(obj, (bytes, bytearray)):
        raise ContractConstructionError(f"{_path}: raw bytes are not part of the contract")
    if isinstance(obj, Sequence):
        for i, item in enumerate(obj):
            validate_no_callables(item, f"{_path}[{i}]")
        return
    raise ContractConstructionError(
        f"{_path}: {type(obj).__name__} is not a primitive contract value")


@dataclasses.dataclass(frozen=True)
class EngineCapabilities:
    """Handshake payload naming what the engine supports.

    A request whose ``required_features`` are not a subset of ``features``
    fails before submit.
    """

    features: tuple = ()
    protocol_version: int = ENGINE_CONTRACT_VERSION

    def __post_init__(self):
        name = type(self).__name__
        object.__setattr__(self, "features", _str_tuple(name, "features", self.features))
        object.__setattr__(self, "protocol_version", _check_protocol_version(name, self.protocol_version))


@dataclasses.dataclass(frozen=True)
class EngineSamplingConfig:
    """Canonical generation data in wire form.

    Generation semantics live here; result-assembly concerns (stop strings,
    formatting, candidate selection) have no fields here — they belong to
    ``FrontendOutputConfig``. ``pad_id`` is carried so the runtime request is
    reproducible from the encoded form alone (contract divergence note 4).
    """

    max_new_tokens: int
    end_id: Optional[int] = None
    pad_id: Optional[int] = None
    stop_token_ids: tuple = ()
    stop_token_sequences: tuple = ()
    min_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    num_logprobs: Optional[int] = None
    num_prompt_logprobs: Optional[int] = None
    protocol_version: int = ENGINE_CONTRACT_VERSION

    def __post_init__(self):
        name = type(self).__name__
        object.__setattr__(self, "max_new_tokens",
                           _check_int(name, "max_new_tokens", self.max_new_tokens, minimum=1))
        object.__setattr__(self, "end_id", _check_opt_int(name, "end_id", self.end_id))
        object.__setattr__(self, "pad_id", _check_opt_int(name, "pad_id", self.pad_id))
        object.__setattr__(self, "stop_token_ids",
                           _int_tuple(name, "stop_token_ids", self.stop_token_ids))
        if isinstance(self.stop_token_sequences, (str, bytes)) or \
                not isinstance(self.stop_token_sequences, Sequence):
            raise _fail(name, "stop_token_sequences", "expected a sequence of int sequences")
        if len(self.stop_token_sequences) > MAX_STOP_SEQUENCES:
            raise _fail(name, "stop_token_sequences", f"exceeds {MAX_STOP_SEQUENCES} sequences")
        object.__setattr__(
            self, "stop_token_sequences",
            tuple(
                _int_tuple(name, f"stop_token_sequences[{i}]", seq, allow_empty=False,
                           max_items=MAX_STOP_SEQUENCE_TOKENS)
                for i, seq in enumerate(self.stop_token_sequences)))
        object.__setattr__(self, "min_tokens",
                           _check_opt_int(name, "min_tokens", self.min_tokens, minimum=0))
        object.__setattr__(self, "temperature", _check_opt_float(name, "temperature", self.temperature))
        object.__setattr__(self, "top_p", _check_opt_float(name, "top_p", self.top_p))
        object.__setattr__(self, "top_k", _check_opt_int(name, "top_k", self.top_k))
        object.__setattr__(self, "seed", _check_opt_int(name, "seed", self.seed))
        object.__setattr__(self, "repetition_penalty",
                           _check_opt_float(name, "repetition_penalty", self.repetition_penalty))
        object.__setattr__(self, "presence_penalty",
                           _check_opt_float(name, "presence_penalty", self.presence_penalty))
        object.__setattr__(self, "frequency_penalty",
                           _check_opt_float(name, "frequency_penalty", self.frequency_penalty))
        object.__setattr__(self, "num_logprobs",
                           _check_opt_int(name, "num_logprobs", self.num_logprobs, minimum=0))
        object.__setattr__(self, "num_prompt_logprobs",
                           _check_opt_int(name, "num_prompt_logprobs", self.num_prompt_logprobs, minimum=0))
        object.__setattr__(self, "protocol_version", _check_protocol_version(name, self.protocol_version))


@dataclasses.dataclass(frozen=True)
class GuidedDecodingSpec:
    """Structured-output request as data; grammar compile + masking are engine-side.

    Present in the schema from V0 but capability-gated (rejected pre-submit)
    until the engine-path validation lands.
    """

    mode: str
    payload: Optional[str] = None
    protocol_version: int = ENGINE_CONTRACT_VERSION

    def __post_init__(self):
        name = type(self).__name__
        mode = _check_str(name, "mode", self.mode)
        if mode not in GUIDED_DECODING_MODES:
            raise _fail(name, "mode", f"must be one of {GUIDED_DECODING_MODES}, got {mode!r}")
        payload = _check_opt_str(name, "payload", self.payload, allow_empty=False)
        if payload is None and mode != "json_object":
            raise _fail(name, "payload", f"required for mode {mode!r}")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "protocol_version", _check_protocol_version(name, self.protocol_version))


@dataclasses.dataclass(frozen=True)
class EngineRequest:
    """Southbound request: token ids in, sampling data, no Python object graph."""

    request_id: str
    prompt_token_ids: tuple
    sampling: EngineSamplingConfig
    guided_decoding: Optional[GuidedDecodingSpec] = None
    required_features: tuple = ()
    protocol_version: int = ENGINE_CONTRACT_VERSION

    def __post_init__(self):
        name = type(self).__name__
        object.__setattr__(self, "request_id", _check_str(name, "request_id", self.request_id))
        object.__setattr__(
            self, "prompt_token_ids",
            _int_tuple(name, "prompt_token_ids", self.prompt_token_ids, allow_empty=False,
                       max_items=MAX_PROMPT_TOKENS))
        if not isinstance(self.sampling, EngineSamplingConfig):
            raise _fail(name, "sampling", "expected EngineSamplingConfig")
        if self.guided_decoding is not None and not isinstance(self.guided_decoding, GuidedDecodingSpec):
            raise _fail(name, "guided_decoding", "expected GuidedDecodingSpec or None")
        object.__setattr__(self, "required_features",
                           _str_tuple(name, "required_features", self.required_features))
        object.__setattr__(self, "protocol_version", _check_protocol_version(name, self.protocol_version))
        validate_no_callables(self, name)


def _check_frame_common(name: str, request_id, event_seq) -> tuple:
    return (_check_str(name, "request_id", request_id),
            _check_int(name, "event_seq", event_seq, minimum=0))


@dataclasses.dataclass(frozen=True)
class TokenDelta:
    """Per-iteration token emission. Never carries completion state.

    ``new_token_ids`` is non-empty by construction: a runtime response with
    no new tokens produces no ``TokenDelta`` frame. ``logprobs``, when
    present, aligns 1:1 with ``new_token_ids``. ``prompt_logprobs`` arrives
    at most once per sequence. The reserved metrics key
    ``CACHED_TOKENS_METRIC_KEY`` carries pre-completion cached-token
    accounting.
    """

    request_id: str
    sequence_id: int
    new_token_ids: tuple
    logprobs: Optional[tuple] = None
    prompt_logprobs: Optional[tuple] = None
    metrics: Optional[Mapping] = None
    event_seq: int = 0
    protocol_version: int = ENGINE_CONTRACT_VERSION

    def __post_init__(self):
        name = type(self).__name__
        request_id, event_seq = _check_frame_common(name, self.request_id, self.event_seq)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "event_seq", event_seq)
        object.__setattr__(self, "sequence_id", _check_int(name, "sequence_id", self.sequence_id, minimum=0))
        object.__setattr__(self, "new_token_ids",
                           _int_tuple(name, "new_token_ids", self.new_token_ids, allow_empty=False))
        logprobs = _float_tuple(name, "logprobs", self.logprobs)
        if logprobs is not None and len(logprobs) != len(self.new_token_ids):
            raise _fail(name, "logprobs",
                        f"length {len(logprobs)} != new_token_ids length {len(self.new_token_ids)}")
        object.__setattr__(self, "logprobs", logprobs)
        object.__setattr__(self, "prompt_logprobs",
                           _float_tuple(name, "prompt_logprobs", self.prompt_logprobs))
        object.__setattr__(self, "metrics", _freeze_metrics(name, "metrics", self.metrics))
        object.__setattr__(self, "protocol_version", _check_protocol_version(name, self.protocol_version))


@dataclasses.dataclass(frozen=True)
class Terminal:
    """Sole carrier of finish/stop reasons; exactly once per started sequence."""

    request_id: str
    sequence_id: int
    finish_reason: str
    stop_reason: Union[int, str, None] = None
    event_seq: int = 0
    protocol_version: int = ENGINE_CONTRACT_VERSION

    def __post_init__(self):
        name = type(self).__name__
        request_id, event_seq = _check_frame_common(name, self.request_id, self.event_seq)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "event_seq", event_seq)
        object.__setattr__(self, "sequence_id", _check_int(name, "sequence_id", self.sequence_id, minimum=0))
        finish_reason = _check_str(name, "finish_reason", self.finish_reason)
        if finish_reason not in FINISH_REASONS:
            raise _fail(name, "finish_reason", f"must be one of {FINISH_REASONS}, got {finish_reason!r}")
        object.__setattr__(self, "finish_reason", finish_reason)
        if self.stop_reason is not None:
            if isinstance(self.stop_reason, bool) or not isinstance(self.stop_reason, (int, str)):
                raise _fail(name, "stop_reason", "expected int, str, or None")
            if isinstance(self.stop_reason, int):
                _check_int(name, "stop_reason", self.stop_reason)
            else:
                _check_str(name, "stop_reason", self.stop_reason, allow_empty=False)
        object.__setattr__(self, "protocol_version", _check_protocol_version(name, self.protocol_version))


@dataclasses.dataclass(frozen=True)
class RequestComplete:
    """Exactly-once request ending frame (after every started sequence's Terminal).

    ``prompt_tokens``/``completion_tokens`` are adapter-observed (draft
    DEC-2), not engine-native usage. ``cached_tokens`` is the final
    cached-token accounting: ``None`` means the source never reported it;
    when at least one ``TokenDelta`` carried ``CACHED_TOKENS_METRIC_KEY``,
    this field must equal the last per-delta value.
    """

    request_id: str
    status: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: Optional[int] = None
    metrics: Optional[Mapping] = None
    event_seq: int = 0
    protocol_version: int = ENGINE_CONTRACT_VERSION

    def __post_init__(self):
        name = type(self).__name__
        request_id, event_seq = _check_frame_common(name, self.request_id, self.event_seq)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "event_seq", event_seq)
        status = _check_str(name, "status", self.status)
        if status not in REQUEST_STATUSES:
            raise _fail(name, "status", f"must be one of {REQUEST_STATUSES}, got {status!r}")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "prompt_tokens",
                           _check_int(name, "prompt_tokens", self.prompt_tokens, minimum=0))
        object.__setattr__(self, "completion_tokens",
                           _check_int(name, "completion_tokens", self.completion_tokens, minimum=0))
        object.__setattr__(self, "cached_tokens",
                           _check_opt_int(name, "cached_tokens", self.cached_tokens, minimum=0))
        object.__setattr__(self, "metrics", _freeze_metrics(name, "metrics", self.metrics))
        object.__setattr__(self, "protocol_version", _check_protocol_version(name, self.protocol_version))


@dataclasses.dataclass(frozen=True)
class ErrorFrame:
    """Standalone request ending, only for failures before any sequence started."""

    request_id: str
    error_code: str
    message: str = ""
    event_seq: int = 0
    protocol_version: int = ENGINE_CONTRACT_VERSION

    def __post_init__(self):
        name = type(self).__name__
        request_id, event_seq = _check_frame_common(name, self.request_id, self.event_seq)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "event_seq", event_seq)
        object.__setattr__(self, "error_code", _check_str(name, "error_code", self.error_code))
        object.__setattr__(self, "message", _check_str(name, "message", self.message, allow_empty=True))
        object.__setattr__(self, "protocol_version", _check_protocol_version(name, self.protocol_version))


OUTPUT_FRAME_TYPES = (TokenDelta, Terminal, RequestComplete, ErrorFrame)
OutputFrame = Union[TokenDelta, Terminal, RequestComplete, ErrorFrame]
"""One typed output frame; a stream is a multiplex of these per request."""


@dataclasses.dataclass(frozen=True)
class FrontendOutputConfig:
    """Frontend-only result-assembly configuration, keyed by ``request_id``.

    Never crosses the frontend↔engine contract; typed and codec-tested so
    frontend state stays serializable. ``stop_sequence_reasons`` is an
    ordered association list of ``(stop_token_sequence, user_visible_reason)``
    pairs: configuration order, first match wins, collisions resolved by
    order (contract divergence note 11).
    """

    detokenize: bool = True
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    stop_strings: tuple = ()
    include_stop_str_in_output: bool = False
    stop_sequence_reasons: tuple = ()
    end_id: Optional[int] = None
    num_logprobs: Optional[int] = None
    protocol_version: int = ENGINE_CONTRACT_VERSION

    def __post_init__(self):
        name = type(self).__name__
        object.__setattr__(self, "detokenize", _check_bool(name, "detokenize", self.detokenize))
        object.__setattr__(self, "skip_special_tokens",
                           _check_bool(name, "skip_special_tokens", self.skip_special_tokens))
        object.__setattr__(
            self, "spaces_between_special_tokens",
            _check_bool(name, "spaces_between_special_tokens", self.spaces_between_special_tokens))
        stop_strings = _str_tuple(name, "stop_strings", self.stop_strings, allow_empty_items=False)
        object.__setattr__(self, "stop_strings", stop_strings)
        object.__setattr__(self, "include_stop_str_in_output",
                           _check_bool(name, "include_stop_str_in_output", self.include_stop_str_in_output))
        if isinstance(self.stop_sequence_reasons, (str, bytes)) or \
                not isinstance(self.stop_sequence_reasons, Sequence):
            raise _fail(name, "stop_sequence_reasons", "expected an ordered sequence of pairs")
        pairs = []
        for i, pair in enumerate(self.stop_sequence_reasons):
            if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
                raise _fail(name, f"stop_sequence_reasons[{i}]", "expected a (sequence, reason) pair")
            sequence = _int_tuple(name, f"stop_sequence_reasons[{i}][0]", pair[0], allow_empty=False,
                                  max_items=MAX_STOP_SEQUENCE_TOKENS)
            reason = pair[1]
            if isinstance(reason, bool) or not isinstance(reason, (int, str)):
                raise _fail(name, f"stop_sequence_reasons[{i}][1]", "expected int or str reason")
            pairs.append((sequence, reason))
        object.__setattr__(self, "stop_sequence_reasons", tuple(pairs))
        object.__setattr__(self, "end_id", _check_opt_int(name, "end_id", self.end_id))
        object.__setattr__(self, "num_logprobs",
                           _check_opt_int(name, "num_logprobs", self.num_logprobs, minimum=0))
        object.__setattr__(self, "protocol_version", _check_protocol_version(name, self.protocol_version))


@dataclasses.dataclass(frozen=True)
class TokenizerSpec:
    """Data-only tokenizer provenance sufficient to reload without guessing.

    ``files_manifest`` is an ordered tuple of ``(relative_path, sha256_hex)``
    pairs. JSON-valued configuration (added tokens, special tokens,
    normalizer, pre-tokenizer) crosses as source strings, never live
    objects: a live tokenizer handle could not survive a process boundary.
    """

    uri: str
    files_manifest: tuple = ()
    revision: Optional[str] = None
    fast: bool = True
    trust_remote_code: bool = False
    added_tokens_json: Optional[str] = None
    special_tokens_json: Optional[str] = None
    normalizer_json: Optional[str] = None
    pre_tokenizer_json: Optional[str] = None
    protocol_version: int = ENGINE_CONTRACT_VERSION

    def __post_init__(self):
        name = type(self).__name__
        object.__setattr__(self, "uri", _check_str(name, "uri", self.uri))
        if isinstance(self.files_manifest, (str, bytes)) or not isinstance(self.files_manifest, Sequence):
            raise _fail(name, "files_manifest", "expected a sequence of (path, sha256) pairs")
        manifest = []
        for i, pair in enumerate(self.files_manifest):
            if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
                raise _fail(name, f"files_manifest[{i}]", "expected a (path, sha256) pair")
            manifest.append((_check_str(name, f"files_manifest[{i}][0]", pair[0]),
                             _check_str(name, f"files_manifest[{i}][1]", pair[1])))
        object.__setattr__(self, "files_manifest", tuple(manifest))
        object.__setattr__(self, "revision", _check_opt_str(name, "revision", self.revision,
                                                            allow_empty=False))
        object.__setattr__(self, "fast", _check_bool(name, "fast", self.fast))
        object.__setattr__(self, "trust_remote_code",
                           _check_bool(name, "trust_remote_code", self.trust_remote_code))
        for field in ("added_tokens_json", "special_tokens_json", "normalizer_json",
                      "pre_tokenizer_json"):
            object.__setattr__(self, field, _check_opt_str(name, field, getattr(self, field)))
        object.__setattr__(self, "protocol_version", _check_protocol_version(name, self.protocol_version))


@dataclasses.dataclass(frozen=True)
class FrontendModelContext:
    """Data-only model context the frontend reads instead of engine internals.

    Built locally from the in-process engine in V0; a later remote detach
    only swaps the delivery (ready-handshake) without changing consumers.
    """

    tokenizer: TokenizerSpec
    capabilities: EngineCapabilities
    chat_template_source: Optional[str] = None
    chat_template_version: Optional[str] = None
    eos_id: Optional[int] = None
    pad_id: Optional[int] = None
    max_context_length: Optional[int] = None
    # Normalization inputs so the frontend never reaches into live model
    # config objects: extra stop ids from the model's generation config and
    # the architecture family (used by the callable-producing-normalization
    # eligibility gates).
    generation_stop_token_ids: tuple = ()
    model_type: Optional[str] = None
    protocol_version: int = ENGINE_CONTRACT_VERSION

    def __post_init__(self):
        name = type(self).__name__
        if not isinstance(self.tokenizer, TokenizerSpec):
            raise _fail(name, "tokenizer", "expected TokenizerSpec")
        if not isinstance(self.capabilities, EngineCapabilities):
            raise _fail(name, "capabilities", "expected EngineCapabilities")
        object.__setattr__(self, "chat_template_source",
                           _check_opt_str(name, "chat_template_source", self.chat_template_source))
        object.__setattr__(self, "chat_template_version",
                           _check_opt_str(name, "chat_template_version", self.chat_template_version,
                                          allow_empty=False))
        object.__setattr__(self, "eos_id", _check_opt_int(name, "eos_id", self.eos_id))
        object.__setattr__(self, "pad_id", _check_opt_int(name, "pad_id", self.pad_id))
        object.__setattr__(self, "max_context_length",
                           _check_opt_int(name, "max_context_length", self.max_context_length, minimum=1))
        object.__setattr__(
            self, "generation_stop_token_ids",
            _int_tuple(name, "generation_stop_token_ids",
                       self.generation_stop_token_ids))
        object.__setattr__(self, "model_type",
                           _check_opt_str(name, "model_type", self.model_type,
                                          allow_empty=False))
        object.__setattr__(self, "protocol_version", _check_protocol_version(name, self.protocol_version))
        validate_no_callables(self, name)


@dataclasses.dataclass(frozen=True)
class EngineHealth:
    """Typed health probe result."""

    healthy: bool
    detail: str = ""
    protocol_version: int = ENGINE_CONTRACT_VERSION

    def __post_init__(self):
        name = type(self).__name__
        object.__setattr__(self, "healthy", _check_bool(name, "healthy", self.healthy))
        object.__setattr__(self, "detail", _check_str(name, "detail", self.detail, allow_empty=True))
        object.__setattr__(self, "protocol_version", _check_protocol_version(name, self.protocol_version))


@dataclasses.dataclass(frozen=True)
class IterationStatsBatch:
    """Iteration stats entries; each entry is a self-contained JSON document string."""

    entries: tuple = ()
    protocol_version: int = ENGINE_CONTRACT_VERSION

    def __post_init__(self):
        name = type(self).__name__
        object.__setattr__(self, "entries", _str_tuple(name, "entries", self.entries))
        object.__setattr__(self, "protocol_version", _check_protocol_version(name, self.protocol_version))


@dataclasses.dataclass(frozen=True)
class KvCacheEventsBatch:
    """KV-cache event entries; each entry is a self-contained JSON document string."""

    entries: tuple = ()
    protocol_version: int = ENGINE_CONTRACT_VERSION

    def __post_init__(self):
        name = type(self).__name__
        object.__setattr__(self, "entries", _str_tuple(name, "entries", self.entries))
        object.__setattr__(self, "protocol_version", _check_protocol_version(name, self.protocol_version))
