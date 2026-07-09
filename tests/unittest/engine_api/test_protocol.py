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
"""Codec and envelope tests for the engine boundary wire protocol."""

import dataclasses

import pytest
import torch

from tensorrt_llm.engine_api import (
    ContractViolationError,
    EngineError,
    EngineErrorCode,
    EngineEvent,
    EngineRequest,
    ProtocolViolationError,
    PythonExtension,
    RuntimeSamplingConfig,
    TensorAuxiliaryPayload,
    TerminalKind,
    TokenLogprob,
)
from tensorrt_llm.engine_api.protocol import (
    PROTOCOL_VERSION,
    MessageType,
    ReadinessState,
    WireMessage,
    check_handshake_version,
    decode_message,
    encode_message,
    engine_event_from_payload,
    engine_event_to_payload,
    engine_request_from_payload,
    engine_request_to_payload,
    error_message,
    handshake_reply,
)


def roundtrip(message: WireMessage) -> WireMessage:
    return decode_message(encode_message(message))


def make_request(**overrides) -> EngineRequest:
    defaults = dict(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling=RuntimeSamplingConfig(
            max_tokens=16,
            temperature=0.7,
            stop_token_ids=[42],
            stop_sequence_token_ids=[[7, 8]],
            logprobs=1,
        ),
        streaming=True,
        priority=0.7,
        cache_salt="salt",
        trace_context={"traceparent": "00-abc-01"},
        disaggregated_metadata={"opaque_state": b"\x00\x01", "first_gen_tokens": [5]},
    )
    defaults.update(overrides)
    return EngineRequest(**defaults)


def make_event(**overrides) -> EngineEvent:
    defaults = dict(
        request_id="req-1",
        sequence_index=1,
        event_index=3,
        token_ids=[5, 6],
        logprobs=[{5: TokenLogprob(logprob=-0.25, rank=1)}, -0.5],
        cumulative_logprob=-0.75,
        finish_reason="stop",
        stop_kind="stop_sequence",
        stop_reason=42,
        terminal_kind=TerminalKind.FINISHED,
        metrics={"decoding_iter": 3.0},
    )
    defaults.update(overrides)
    return EngineEvent(**defaults)


class TestRoundTripEveryMessageType:
    def test_submit_round_trip(self):
        request = make_request()
        message = WireMessage(
            message_type=MessageType.SUBMIT,
            request_id=request.request_id,
            payload=engine_request_to_payload(request),
        )
        decoded = roundtrip(message)
        assert decoded.message_type is MessageType.SUBMIT
        rebuilt = engine_request_from_payload(decoded.payload)
        assert rebuilt == request

    def test_event_round_trip(self):
        event = make_event()
        message = WireMessage(
            message_type=MessageType.EVENT,
            request_id=event.request_id,
            payload=engine_event_to_payload(event),
        )
        decoded = roundtrip(message)
        rebuilt = engine_event_from_payload(decoded.request_id, decoded.payload)
        assert rebuilt == event

    def test_error_terminal_event_round_trip(self):
        event = make_event(
            token_ids=[],
            logprobs=None,
            cumulative_logprob=None,
            finish_reason=None,
            stop_kind=None,
            stop_reason=None,
            terminal_kind=TerminalKind.ERROR,
            error=EngineError(
                code=EngineErrorCode.REQUEST_FAILED, message="boom", request_id="req-1"
            ),
            metrics=None,
        )
        message = WireMessage(
            message_type=MessageType.EVENT,
            request_id="req-1",
            payload=engine_event_to_payload(event),
        )
        rebuilt = engine_event_from_payload("req-1", roundtrip(message).payload)
        assert rebuilt.error == event.error
        assert rebuilt.terminal_kind is TerminalKind.ERROR

    def test_prompt_metadata_event_round_trip(self):
        event = make_event(
            event_index=0,
            prompt_token_ids=[1, 2, 3],
            prompt_logprobs=[-0.1, -0.2, -0.3],
            finish_reason=None,
            stop_kind=None,
            stop_reason=None,
            terminal_kind=None,
        )
        message = WireMessage(
            message_type=MessageType.EVENT,
            request_id="req-1",
            payload=engine_event_to_payload(event),
        )
        rebuilt = engine_event_from_payload("req-1", roundtrip(message).payload)
        assert rebuilt == event

    def test_abort_and_ack_round_trip(self):
        abort = WireMessage(message_type=MessageType.ABORT, request_id="req-1")
        assert roundtrip(abort) == abort
        ack = WireMessage(
            message_type=MessageType.ABORT_ACK, request_id="req-1", payload={"known": True}
        )
        assert roundtrip(ack) == ack

    def test_control_round_trip(self):
        request = WireMessage(
            message_type=MessageType.CONTROL_REQUEST,
            payload={"control_id": 7, "method": "get_stats", "kwargs": {"timeout": 0.5}},
        )
        assert roundtrip(request) == request
        response = WireMessage(
            message_type=MessageType.CONTROL_RESPONSE,
            payload={"control_id": 7, "result": [{"iter": 1, "num_active_requests": 2}]},
        )
        assert roundtrip(response) == response

    def test_handshake_round_trip(self):
        message = handshake_reply(
            capabilities={"endpoints": ["chat", "completions"]},
            readiness_state=ReadinessState.READY,
            model_context={"model": "m", "tokenizer_dir": "/models/m", "max_seq_len": 4096},
        )
        decoded = roundtrip(message)
        assert decoded == message
        assert decoded.payload["readiness_state"] == "ready"

    def test_error_message_round_trip(self):
        message = error_message(EngineErrorCode.SLOW_CONSUMER, "client too slow", "req-9")
        decoded = roundtrip(message)
        assert decoded == message
        assert decoded.payload["error_code"] == "slow_consumer"


class TestCodecRejections:
    def test_callable_fails_encoding(self):
        message = WireMessage(
            message_type=MessageType.CONTROL_RESPONSE, payload={"cb": lambda: None}
        )
        with pytest.raises(ContractViolationError, match="callable"):
            encode_message(message)

    def test_arbitrary_object_fails_encoding(self):
        class Opaque:
            pass

        message = WireMessage(message_type=MessageType.SUBMIT, payload={"state": Opaque()})
        with pytest.raises(ContractViolationError, match="Opaque"):
            encode_message(message)

    def test_tensor_fails_encoding(self):
        message = WireMessage(message_type=MessageType.EVENT, payload={"logits": torch.zeros(2)})
        with pytest.raises(ContractViolationError, match="Tensor"):
            encode_message(message)

    def test_request_with_python_extension_rejected(self):
        request = make_request(
            python_extension=PythonExtension(logits_processor=lambda ids, logits: logits)
        )
        with pytest.raises(ContractViolationError, match="side channel"):
            engine_request_to_payload(request)

    def test_event_with_tensor_payload_rejected(self):
        event = make_event(
            tensor_payload=TensorAuxiliaryPayload(tensors={"logits": torch.zeros(1)})
        )
        with pytest.raises(ContractViolationError, match="side channel"):
            engine_event_to_payload(event)

    def test_no_pickle_fallback(self):
        """The wire never contains pickle: encoding is pure msgpack."""
        message = WireMessage(
            message_type=MessageType.SUBMIT,
            request_id="req-1",
            payload=engine_request_to_payload(make_request()),
        )
        encoded = encode_message(message)
        assert b"\x80\x04" not in encoded[:2]  # no pickle protocol header
        import msgpack

        assert msgpack.unpackb(encoded, raw=False, strict_map_key=False)


class TestEnvelopeValidation:
    def test_undeclared_message_type_rejected(self):
        import msgpack

        data = msgpack.packb(
            {"protocol_version": PROTOCOL_VERSION, "message_type": "teleport", "payload": {}}
        )
        with pytest.raises(ProtocolViolationError, match="undeclared message_type"):
            decode_message(data)

    def test_garbage_bytes_rejected(self):
        with pytest.raises(ProtocolViolationError, match="wire message"):
            decode_message(b"\x00\x01\x02not-msgpack" * 3)

    def test_missing_version_rejected(self):
        import msgpack

        data = msgpack.packb({"message_type": "event", "payload": {}})
        with pytest.raises(ProtocolViolationError, match="protocol_version"):
            decode_message(data)

    def test_version_mismatch_fails_handshake_typed(self):
        peer = WireMessage(
            message_type=MessageType.HANDSHAKE, protocol_version=PROTOCOL_VERSION + 1
        )
        with pytest.raises(ProtocolViolationError, match="protocol_version mismatch"):
            check_handshake_version(peer)

    def test_all_envelope_fields_survive(self):
        message = WireMessage(
            message_type=MessageType.EVENT,
            request_id="req-2",
            payload={"sequence_index": 1, "event_index": 4, "terminal_kind": "finished"},
        )
        decoded = roundtrip(message)
        assert decoded.protocol_version == PROTOCOL_VERSION
        assert decoded.request_id == "req-2"
        assert decoded.payload["sequence_index"] == 1
        assert decoded.payload["event_index"] == 4
        assert decoded.payload["terminal_kind"] == "finished"

    def test_wire_fields_cover_specified_envelope(self):
        """Every field the protocol spec names is representable."""
        event = make_event()
        payload = engine_event_to_payload(event)
        for name in ("sequence_index", "event_index", "terminal_kind"):
            assert name in payload
        handshake = handshake_reply({}, ReadinessState.INITIALIZING)
        assert {"capabilities", "readiness_state"} <= set(handshake.payload)
        error = error_message(EngineErrorCode.INTERNAL_ERROR, "x")
        assert {"error_code", "error_message"} <= set(error.payload)
        fields = {f.name for f in dataclasses.fields(WireMessage)}
        assert {"protocol_version", "message_type", "request_id", "payload"} <= fields
