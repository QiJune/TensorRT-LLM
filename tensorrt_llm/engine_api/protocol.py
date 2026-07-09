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
"""Language-neutral wire protocol for the frontend <-> engine boundary.

The wire codec is msgpack with **no pickle fallback and no Python-object
serialization of any kind**: values that are not plain data fail encoding
with a typed error. See ``PROTOCOL.md`` in this package for the versioned
protocol specification (message flow, state machine, error semantics,
slow-consumer policy, and threat notes).

Every message is one msgpack map with the envelope fields:

- ``protocol_version`` (int): protocol revision; mismatches fail the
  handshake with a typed error.
- ``message_type`` (str): one of :class:`MessageType`; undeclared types are
  rejected on decode.
- ``request_id`` (str | None): request correlation id.
- ``payload`` (map): message-type-specific body. For events this carries
  ``sequence_index``, ``event_index``, ``terminal_kind``, token ids,
  logprobs, and typed error codes; for handshakes it carries
  ``capabilities``, ``readiness_state``, and the model context.
"""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass, field
from typing import Any, Optional

import msgpack

from tensorrt_llm.engine_api.contracts import (
    ContractViolationError,
    EngineError,
    EngineErrorCode,
    EngineEvent,
    EngineRequest,
    ProtocolViolationError,
    RuntimeSamplingConfig,
    TerminalKind,
    TokenLogprob,
)

__all__ = [
    "PROTOCOL_VERSION",
    "MessageType",
    "ReadinessState",
    "WireMessage",
    "check_handshake_version",
    "decode_message",
    "encode_message",
    "engine_event_from_payload",
    "engine_event_to_payload",
    "engine_request_from_payload",
    "engine_request_to_payload",
    "error_message",
    "handshake_reply",
]

PROTOCOL_VERSION = 1


class MessageType(str, enum.Enum):
    """Declared message types of the boundary protocol."""

    SUBMIT = "submit"
    EVENT = "event"
    ABORT = "abort"
    ABORT_ACK = "abort_ack"
    CONTROL_REQUEST = "control_request"
    CONTROL_RESPONSE = "control_response"
    HANDSHAKE = "handshake"
    ERROR = "error"


class ReadinessState(str, enum.Enum):
    """Engine readiness advertised in handshakes and health responses."""

    INITIALIZING = "initializing"
    READY = "ready"
    UNHEALTHY = "unhealthy"
    SHUTTING_DOWN = "shutting_down"


@dataclass(slots=True)
class WireMessage:
    """One protocol message: envelope plus type-specific payload."""

    message_type: MessageType
    request_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)
    protocol_version: int = PROTOCOL_VERSION


def _reject_non_plain(obj: Any) -> Any:
    """Msgpack ``default`` hook: never serialize arbitrary Python objects."""
    kind = "callable" if callable(obj) else type(obj).__name__
    raise ContractViolationError(
        f"cannot serialize {kind} on the engine boundary wire; only plain data crosses"
    )


def encode_message(message: WireMessage) -> bytes:
    """Encode one message to msgpack bytes. Non-plain data fails typed."""
    body = {
        "protocol_version": message.protocol_version,
        "message_type": message.message_type.value,
        "request_id": message.request_id,
        "payload": message.payload,
    }
    try:
        return msgpack.packb(body, use_bin_type=True, default=_reject_non_plain)
    except (TypeError, ValueError) as e:
        raise ContractViolationError(
            f"wire encoding failed: {e}; only plain data crosses the engine boundary"
        ) from e


def decode_message(data: bytes) -> WireMessage:
    """Decode msgpack bytes into a message; violations raise typed errors."""
    try:
        body = msgpack.unpackb(data, raw=False, strict_map_key=False)
    except Exception as e:
        raise ProtocolViolationError(f"undecodable wire message: {e}") from e
    if not isinstance(body, dict):
        raise ProtocolViolationError(f"wire message must be a map, got {type(body).__name__}")
    raw_type = body.get("message_type")
    try:
        message_type = MessageType(raw_type)
    except ValueError:
        raise ProtocolViolationError(f"undeclared message_type {raw_type!r}") from None
    version = body.get("protocol_version")
    if not isinstance(version, int):
        raise ProtocolViolationError(f"missing or invalid protocol_version: {version!r}")
    return WireMessage(
        message_type=message_type,
        request_id=body.get("request_id"),
        payload=body.get("payload") or {},
        protocol_version=version,
    )


# --- contract type <-> payload conversion -------------------------------------


def engine_request_to_payload(request: EngineRequest) -> dict[str, Any]:
    """Flatten an EngineRequest for the wire. Side channels must be absent."""
    if not request.crosses_neutral_wire:
        raise ContractViolationError(
            "EngineRequest carries declared side channels (PythonExtension / "
            "TensorAuxiliaryPayload) which never cross the language-neutral wire"
        )
    payload = {
        f.name: getattr(request, f.name)
        for f in dataclasses.fields(request)
        if f.name not in EngineRequest.SIDE_CHANNEL_FIELDS and f.name != "sampling"
    }
    payload["sampling"] = dataclasses.asdict(request.sampling)
    return payload


def engine_request_from_payload(payload: dict[str, Any]) -> EngineRequest:
    data = dict(payload)
    sampling = data.pop("sampling", None) or {}
    return EngineRequest(sampling=RuntimeSamplingConfig(**sampling), **data)


def _logprobs_to_wire(entries: Any) -> Any:
    if entries is None:
        return None
    wire_entries = []
    for entry in entries:
        if isinstance(entry, dict):
            wire_entries.append(
                {int(token_id): [item.logprob, item.rank] for token_id, item in entry.items()}
            )
        else:
            wire_entries.append(entry)
    return wire_entries


def _logprobs_from_wire(entries: Any) -> Any:
    if entries is None:
        return None
    decoded = []
    for entry in entries:
        if isinstance(entry, dict):
            decoded.append(
                {
                    int(token_id): TokenLogprob(logprob=pair[0], rank=pair[1])
                    for token_id, pair in entry.items()
                }
            )
        else:
            decoded.append(entry)
    return decoded


def engine_event_to_payload(event: EngineEvent) -> dict[str, Any]:
    """Flatten an EngineEvent for the wire. Tensor side channel must be absent."""
    if event.tensor_payload is not None:
        raise ContractViolationError(
            "EngineEvent carries a tensor side channel which never crosses the "
            "language-neutral wire"
        )
    payload: dict[str, Any] = {
        "sequence_index": event.sequence_index,
        "event_index": event.event_index,
        "token_ids": event.token_ids,
        "cumulative": event.cumulative,
        "logprobs": _logprobs_to_wire(event.logprobs),
        "cumulative_logprob": event.cumulative_logprob,
        "prompt_token_ids": event.prompt_token_ids,
        "prompt_logprobs": _logprobs_to_wire(event.prompt_logprobs),
        "finish_reason": event.finish_reason,
        "stop_kind": event.stop_kind,
        "stop_reason": event.stop_reason,
        "terminal_kind": event.terminal_kind.value if event.terminal_kind else None,
        "disaggregated_metadata": event.disaggregated_metadata,
        "metrics": event.metrics,
    }
    if event.error is not None:
        payload["error_code"] = event.error.code.value
        payload["error_message"] = event.error.message
    return payload


def engine_event_from_payload(request_id: str, payload: dict[str, Any]) -> EngineEvent:
    error = None
    if payload.get("error_code") is not None:
        error = EngineError(
            code=EngineErrorCode(payload["error_code"]),
            message=payload.get("error_message", ""),
            request_id=request_id,
        )
    terminal_kind = payload.get("terminal_kind")
    return EngineEvent(
        request_id=request_id,
        sequence_index=payload.get("sequence_index", 0),
        event_index=payload.get("event_index", 0),
        token_ids=payload.get("token_ids") or [],
        cumulative=payload.get("cumulative", False),
        logprobs=_logprobs_from_wire(payload.get("logprobs")),
        cumulative_logprob=payload.get("cumulative_logprob"),
        prompt_token_ids=payload.get("prompt_token_ids"),
        prompt_logprobs=_logprobs_from_wire(payload.get("prompt_logprobs")),
        finish_reason=payload.get("finish_reason"),
        stop_kind=payload.get("stop_kind"),
        stop_reason=payload.get("stop_reason"),
        terminal_kind=TerminalKind(terminal_kind) if terminal_kind else None,
        error=error,
        disaggregated_metadata=payload.get("disaggregated_metadata"),
        metrics=payload.get("metrics"),
    )


# --- common message builders ---------------------------------------------------


def error_message(
    code: EngineErrorCode, message: str, request_id: Optional[str] = None
) -> WireMessage:
    return WireMessage(
        message_type=MessageType.ERROR,
        request_id=request_id,
        payload={"error_code": code.value, "error_message": message},
    )


def handshake_reply(
    capabilities: dict[str, Any],
    readiness_state: ReadinessState,
    model_context: Optional[dict[str, Any]] = None,
) -> WireMessage:
    return WireMessage(
        message_type=MessageType.HANDSHAKE,
        payload={
            "capabilities": capabilities,
            "readiness_state": readiness_state.value,
            "model_context": model_context or {},
        },
    )


def check_handshake_version(message: WireMessage) -> None:
    """Raise a typed error when the peer speaks a different protocol revision."""
    if message.protocol_version != PROTOCOL_VERSION:
        raise ProtocolViolationError(
            f"protocol_version mismatch: peer speaks {message.protocol_version}, "
            f"this side speaks {PROTOCOL_VERSION}"
        )
