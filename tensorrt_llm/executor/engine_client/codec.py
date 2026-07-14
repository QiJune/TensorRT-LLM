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
"""Strict msgpack codec for the engine contract wire types.

Every wire type in ``contract`` encodes to a msgpack envelope
``{"k": <kind>, "d": {<field>: <value>, ...}}`` and decodes back through
the contract constructors, so construction-time validation applies to
every decoded payload. The codec is strictly primitive: any value outside
``None | bool | int | float | str | list | dict`` is a typed encode error,
and there is **no pickle fallback under any flag**. Decode rejects, with
typed errors, unknown kinds, unknown fields, missing required fields,
wrong item types, newer protocol versions, msgpack extension/bin types,
non-finite floats, and payloads exceeding the resource limits below.

Encoding is canonical: dataclass fields are written in declaration order
and mapping-valued fields are written in sorted key order, so equal values
produce identical bytes (the basis of the checked-in golden fixtures).
"""

import dataclasses
import math
from collections.abc import Mapping, Sequence

import msgpack

from .contract import (ENGINE_CONTRACT_VERSION, INT64_MAX, INT64_MIN,
                       MAX_STRING_CHARS, ContractConstructionError,
                       ContractError, EngineCapabilities, EngineHealth,
                       EngineRequest, EngineSamplingConfig, ErrorFrame,
                       FrontendModelContext, FrontendOutputConfig,
                       GuidedDecodingSpec, IterationStatsBatch,
                       KvCacheEventsBatch, RequestComplete, Terminal,
                       TokenDelta, TokenizerSpec)

__all__ = [
    "MAX_MESSAGE_BYTES",
    "MAX_NESTING_DEPTH",
    "MAX_ARRAY_ITEMS",
    "MAX_MAP_ITEMS",
    "CodecError",
    "EncodeError",
    "DecodeError",
    "KIND_BY_TYPE",
    "TYPE_BY_KIND",
    "encode",
    "decode",
]

MAX_MESSAGE_BYTES = 16 * 1024 * 1024
MAX_NESTING_DEPTH = 16
MAX_ARRAY_ITEMS = 2 * 1024 * 1024
MAX_MAP_ITEMS = 4096


class CodecError(ContractError):
    """Base class for codec errors. Carries a stable ``reason`` code."""

    def __init__(self, reason: str, message: str):
        super().__init__(f"{reason}: {message}")
        self.reason = reason


class EncodeError(CodecError):
    """A value cannot be represented on the strict primitive wire."""


class DecodeError(CodecError):
    """A payload violates the wire schema, limits, or version rules."""


KIND_BY_TYPE = {
    EngineCapabilities: "engine_capabilities",
    EngineSamplingConfig: "engine_sampling_config",
    GuidedDecodingSpec: "guided_decoding_spec",
    EngineRequest: "engine_request",
    TokenDelta: "token_delta",
    Terminal: "terminal",
    RequestComplete: "request_complete",
    ErrorFrame: "error_frame",
    FrontendOutputConfig: "frontend_output_config",
    TokenizerSpec: "tokenizer_spec",
    FrontendModelContext: "frontend_model_context",
    EngineHealth: "engine_health",
    IterationStatsBatch: "iteration_stats_batch",
    KvCacheEventsBatch: "kv_cache_events_batch",
}
TYPE_BY_KIND = {kind: cls for cls, kind in KIND_BY_TYPE.items()}

# Fields whose values are nested wire types (decoded recursively through
# their own constructors). ``None`` stays ``None`` for optional fields.
_NESTED_FIELDS = {
    EngineRequest: {
        "sampling": EngineSamplingConfig,
        "guided_decoding": GuidedDecodingSpec,
    },
    FrontendModelContext: {
        "tokenizer": TokenizerSpec,
        "capabilities": EngineCapabilities,
    },
}


def _value_to_wire(value, depth: int):
    if depth > MAX_NESTING_DEPTH:
        raise EncodeError("limit_exceeded", f"nesting depth exceeds {MAX_NESTING_DEPTH}")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if not INT64_MIN <= value <= INT64_MAX:
            raise EncodeError("limit_exceeded", "integer outside signed 64-bit range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EncodeError("not_finite", "NaN/Inf floats are not encodable")
        return value
    if isinstance(value, str):
        if len(value) > MAX_STRING_CHARS:
            raise EncodeError("limit_exceeded", f"string exceeds {MAX_STRING_CHARS} characters")
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if type(value) not in KIND_BY_TYPE:
            raise EncodeError("non_primitive", f"{type(value).__name__} is not a registered wire type")
        return {
            field.name: _value_to_wire(getattr(value, field.name), depth + 1)
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        if len(value) > MAX_MAP_ITEMS:
            raise EncodeError("limit_exceeded", f"mapping exceeds {MAX_MAP_ITEMS} entries")
        keys = list(value.keys())
        for key in keys:
            if not isinstance(key, str):
                raise EncodeError("non_primitive", "mapping keys must be strings")
        wire = {}
        for key in sorted(keys):
            wire[key] = _value_to_wire(value[key], depth + 1)
        return wire
    if isinstance(value, (bytes, bytearray)):
        raise EncodeError("non_primitive", "raw bytes are not part of the contract")
    if isinstance(value, Sequence):
        if len(value) > MAX_ARRAY_ITEMS:
            raise EncodeError("limit_exceeded", f"sequence exceeds {MAX_ARRAY_ITEMS} items")
        return [_value_to_wire(item, depth + 1) for item in value]
    raise EncodeError("non_primitive", f"{type(value).__name__} is not encodable on the wire")


def encode(obj) -> bytes:
    """Encode a registered wire type to canonical msgpack bytes.

    Raises:
        EncodeError: for unregistered types, non-primitive values,
            non-finite floats, or payloads exceeding the resource limits.
    """
    kind = KIND_BY_TYPE.get(type(obj))
    if kind is None:
        raise EncodeError("unknown_type", f"{type(obj).__name__} is not a registered wire type")
    payload = {"k": kind, "d": _value_to_wire(obj, 1)}
    data = msgpack.packb(payload, use_bin_type=True)
    if len(data) > MAX_MESSAGE_BYTES:
        raise EncodeError("limit_exceeded", f"message exceeds {MAX_MESSAGE_BYTES} bytes")
    return data


def _check_raw_depth(value, depth: int) -> None:
    if depth > MAX_NESTING_DEPTH:
        raise DecodeError("limit_exceeded", f"nesting depth exceeds {MAX_NESTING_DEPTH}")
    if isinstance(value, dict):
        for key, item in value.items():
            _check_raw_depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _check_raw_depth(item, depth + 1)


def _decode_dataclass(cls, raw, path: str):
    if raw is None and cls in (GuidedDecodingSpec,):
        return None
    if not isinstance(raw, dict):
        raise DecodeError("invalid_content", f"{path}: expected a field map")
    field_map = {field.name: field for field in dataclasses.fields(cls)}
    unknown = set(raw.keys()) - set(field_map.keys())
    if unknown:
        raise DecodeError("unknown_field", f"{path}: unknown field(s) {sorted(unknown)!r}")
    version = raw.get("protocol_version", ENGINE_CONTRACT_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise DecodeError("invalid_content", f"{path}.protocol_version: expected int")
    if version > ENGINE_CONTRACT_VERSION:
        raise DecodeError(
            "version_unsupported",
            f"{path}.protocol_version {version} > supported {ENGINE_CONTRACT_VERSION}")
    nested = _NESTED_FIELDS.get(cls, {})
    kwargs = {}
    for name, field in field_map.items():
        if name not in raw:
            has_default = (field.default is not dataclasses.MISSING
                           or field.default_factory is not dataclasses.MISSING)
            if not has_default:
                raise DecodeError("missing_field", f"{path}: missing required field {name!r}")
            continue
        value = raw[name]
        if name in nested:
            value = _decode_dataclass(nested[name], value, f"{path}.{name}")
        kwargs[name] = value
    try:
        return cls(**kwargs)
    except ContractConstructionError as e:
        raise DecodeError("invalid_content", f"{path}: {e}") from e


def decode(data: bytes):
    """Decode msgpack bytes back into a registered wire type.

    Raises:
        DecodeError: for malformed msgpack, extension/bin types, unknown
            kinds/fields, missing required fields, wrong types, newer
            protocol versions, or payloads exceeding the resource limits.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise DecodeError("invalid_content", f"expected bytes, got {type(data).__name__}")
    if len(data) > MAX_MESSAGE_BYTES:
        raise DecodeError("limit_exceeded", f"message exceeds {MAX_MESSAGE_BYTES} bytes")
    try:
        payload = msgpack.unpackb(
            data,
            raw=False,
            strict_map_key=True,
            use_list=True,
            max_str_len=4 * MAX_STRING_CHARS,
            max_bin_len=0,
            max_array_len=MAX_ARRAY_ITEMS,
            max_map_len=MAX_MAP_ITEMS,
            max_ext_len=0,
        )
    except Exception as e:  # msgpack raises several unpack error types.
        raise DecodeError("malformed", f"msgpack unpack failed: {e}") from e
    if not isinstance(payload, dict) or set(payload.keys()) != {"k", "d"}:
        raise DecodeError("malformed", "expected an envelope with exactly the keys 'k' and 'd'")
    kind = payload["k"]
    if not isinstance(kind, str) or kind not in TYPE_BY_KIND:
        raise DecodeError("unknown_kind", f"unknown wire kind {kind!r}")
    _check_raw_depth(payload["d"], 1)
    return _decode_dataclass(TYPE_BY_KIND[kind], payload["d"], kind)
