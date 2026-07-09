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
"""Streaming/abort/error semantics over the boundary socket (CPU-only)."""

import threading
import time

import pytest
import zmq
from fake_engine import FakeEngine, exploding_script, flood_script, infinite_script

from tensorrt_llm.engine_api import (
    EngineClientError,
    EngineErrorCode,
    EngineRequest,
    EventOrderingChecker,
    ProtocolViolationError,
    RuntimeSamplingConfig,
    TerminalKind,
)
from tensorrt_llm.engine_api.protocol import (
    MessageType,
    ReadinessState,
    WireMessage,
    decode_message,
    encode_message,
)
from tensorrt_llm.engine_api.socket_transport import (
    EngineSocketServer,
    LocalProcessEngineClient,
    SocketEngineClient,
)


def make_request(request_id="req-1", num_tokens=3) -> EngineRequest:
    return EngineRequest(
        request_id=request_id,
        prompt_token_ids=list(range(1, num_tokens + 1)),
        sampling=RuntimeSamplingConfig(max_tokens=8),
        streaming=True,
    )


@pytest.fixture
def engine():
    return FakeEngine()


def start_stack(engine, **server_kwargs):
    client = LocalProcessEngineClient(engine, **server_kwargs)
    yield_obj = client
    return yield_obj


@pytest.fixture
def stack(engine):
    client = LocalProcessEngineClient(engine)
    yield client
    client.shutdown()


class TestHandshake:
    def test_handshake_negotiates_capabilities_and_readiness(self, stack):
        client = stack.client
        assert client.readiness is ReadinessState.READY
        assert client.capabilities["generation"]["streaming"] is True
        client.require_capability("generation", "streaming")

    def test_unadvertised_capability_rejected_frontend_side(self, stack):
        with pytest.raises(EngineClientError) as excinfo:
            stack.client.require_capability("generation", "multimodal")
        assert excinfo.value.error.code is EngineErrorCode.UNSUPPORTED_CAPABILITY

    def test_model_context_from_handshake(self, engine):
        client = LocalProcessEngineClient(
            engine, model_context={"model": "m", "tokenizer_dir": "/models/m"}
        )
        try:
            assert client.client.model_context == {
                "model": "m",
                "tokenizer_dir": "/models/m",
            }
        finally:
            client.shutdown()

    def test_version_mismatch_fails_typed_not_hang(self, engine):
        server = EngineSocketServer(engine)
        server.start()
        try:
            ctx = zmq.Context.instance()
            dealer = ctx.socket(zmq.DEALER)
            dealer.setsockopt(zmq.LINGER, 0)
            dealer.connect(server.endpoint)
            bad_handshake = WireMessage(message_type=MessageType.HANDSHAKE, protocol_version=99)
            dealer.send(encode_message(bad_handshake))
            assert dealer.poll(5000), "no reply to mismatched handshake"
            reply = decode_message(dealer.recv())
            assert reply.message_type is MessageType.ERROR
            assert reply.payload["error_code"] == "protocol_version_mismatch"
            dealer.close(linger=0)
        finally:
            server.shutdown()

    def test_frontend_with_no_engine_fails_fast(self):
        started = time.monotonic()
        with pytest.raises(EngineClientError) as excinfo:
            SocketEngineClient("tcp://127.0.0.1:19", handshake_timeout_seconds=0.5)
        assert excinfo.value.error.code is EngineErrorCode.ENGINE_UNAVAILABLE
        assert time.monotonic() - started < 5.0


class TestStreaming:
    def test_stream_round_trip_with_ordering(self, stack):
        handle = stack.submit(make_request(num_tokens=4))
        checker = EventOrderingChecker()
        events = []
        for event in handle.events():
            checker.observe(event)
            events.append(event)
        assert len(events) == 4
        assert events[0].prompt_token_ids == [1, 2, 3, 4]
        assert events[-1].terminal_kind is TerminalKind.FINISHED
        assert [e.event_index for e in events] == [0, 1, 2, 3]

    def test_concurrent_interleaved_streams_stay_correlated(self, stack):
        results = {}
        errors = []

        def consume(request_id, num_tokens):
            try:
                handle = stack.submit(make_request(request_id, num_tokens))
                checker = EventOrderingChecker()
                collected = []
                for event in handle.events():
                    checker.observe(event)
                    assert event.request_id == request_id
                    collected.append(event)
                results[request_id] = collected
            except Exception as e:  # pragma: no cover - surfaced via errors list
                errors.append(e)

        threads = [threading.Thread(target=consume, args=(f"req-{i}", 3 + i)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert not errors
        assert set(results) == {"req-0", "req-1", "req-2", "req-3"}
        for i in range(4):
            assert len(results[f"req-{i}"]) == 3 + i

    def test_engine_exception_surfaces_as_typed_error_event(self, engine):
        stack = LocalProcessEngineClient(FakeEngine(script=exploding_script))
        try:
            handle = stack.submit(make_request())
            events = list(handle.events())
            terminal = events[-1]
            assert terminal.terminal_kind is TerminalKind.ERROR
            assert terminal.error.code is EngineErrorCode.INTERNAL_ERROR
            assert "engine exploded" in terminal.error.message
            assert "Traceback" not in terminal.error.message
        finally:
            stack.shutdown()


class TestAbortRaceMatrix:
    def test_abort_before_submit_typed_unknown_request(self, stack):
        with pytest.raises(EngineClientError) as excinfo:
            stack.abort("never-submitted")
        assert excinfo.value.error.code is EngineErrorCode.UNKNOWN_REQUEST

    def test_abort_during_stream_terminates_with_aborted(self, engine):
        stack = LocalProcessEngineClient(FakeEngine(script=infinite_script))
        try:
            handle = stack.submit(make_request())
            events = []
            for event in handle.events():
                events.append(event)
                if len(events) == 2:
                    handle.abort()
            assert events[-1].terminal_kind is TerminalKind.ABORTED
            assert events[-1].finish_reason == "cancelled"
        finally:
            stack.shutdown()

    def test_abort_after_terminal_is_idempotent_noop(self, stack):
        handle = stack.submit(make_request())
        list(handle.events())
        handle.abort()  # no exception, no effect

    def test_duplicate_abort_is_idempotent(self, engine):
        stack = LocalProcessEngineClient(FakeEngine(script=infinite_script))
        try:
            handle = stack.submit(make_request())
            events = []
            for event in handle.events():
                events.append(event)
                if len(events) == 2:
                    handle.abort()
                    handle.abort()
            terminals = [e for e in events if e.is_terminal]
            assert len(terminals) == 1
            assert terminals[0].terminal_kind is TerminalKind.ABORTED
        finally:
            stack.shutdown()

    def test_abort_unknown_id_is_typed_error_not_silence(self, stack):
        with pytest.raises(EngineClientError) as excinfo:
            stack.abort("ghost-request")
        assert excinfo.value.error.code is EngineErrorCode.UNKNOWN_REQUEST


class TestReadinessGating:
    def test_submit_before_ready_rejected_client_side(self, engine):
        stack = LocalProcessEngineClient(engine, ready=False)
        try:
            assert stack.client.readiness is ReadinessState.INITIALIZING
            with pytest.raises(EngineClientError) as excinfo:
                stack.submit(make_request())
            assert excinfo.value.error.code is EngineErrorCode.ENGINE_UNAVAILABLE
        finally:
            stack.shutdown()

    def test_submit_during_shutdown_gets_typed_error(self, engine):
        stack = LocalProcessEngineClient(engine)
        try:
            stack.server.set_readiness(ReadinessState.SHUTTING_DOWN)
            # Force a submission past the client-side gate to exercise the
            # server-side guard.
            stack.client.readiness = ReadinessState.READY
            handle = stack.submit(make_request())
            with pytest.raises(EngineClientError) as excinfo:
                list(handle.events())
            assert excinfo.value.error.code is EngineErrorCode.ENGINE_SHUTDOWN
        finally:
            stack.shutdown()


class TestSlowConsumerPolicy:
    def test_blocked_send_grace_timeout_disconnect_and_abort(self):
        engine = FakeEngine(script=flood_script(200000))
        server = EngineSocketServer(engine, send_buffer_size=4, slow_consumer_grace_seconds=0.3)
        server.start()
        try:
            ctx = zmq.Context.instance()
            dealer = ctx.socket(zmq.DEALER)
            dealer.setsockopt(zmq.LINGER, 0)
            dealer.setsockopt(zmq.RCVHWM, 2)
            dealer.connect(server.endpoint)
            dealer.send(encode_message(WireMessage(message_type=MessageType.HANDSHAKE)))
            assert dealer.poll(5000)
            decode_message(dealer.recv())  # handshake reply
            request = make_request("slow-req")
            from tensorrt_llm.engine_api.protocol import engine_request_to_payload

            dealer.send(
                encode_message(
                    WireMessage(
                        message_type=MessageType.SUBMIT,
                        request_id="slow-req",
                        payload=engine_request_to_payload(request),
                    )
                )
            )
            # Never read events: the engine-side buffer fills, the grace
            # timer expires, the connection is dropped, and the in-flight
            # request is aborted engine-side under a typed error.
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and not server.slow_consumer_drops:
                time.sleep(0.05)
            assert server.slow_consumer_drops, "slow consumer was never dropped"
            assert server.slow_consumer_drops[0].code is EngineErrorCode.SLOW_CONSUMER
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and "slow-req" not in engine.aborted_request_ids:
                time.sleep(0.05)
            assert "slow-req" in engine.aborted_request_ids
            dealer.close(linger=0)
        finally:
            server.shutdown()


class TestClientOrderingChecker:
    """Crafted wire messages must be rejected by the client's checker."""

    @staticmethod
    def _event_message(request_id, event_index, terminal=False, prompt=False):
        payload = {
            "sequence_index": 0,
            "event_index": event_index,
            "token_ids": [5],
            "cumulative": False,
        }
        if terminal:
            payload["terminal_kind"] = "finished"
            payload["finish_reason"] = "length"
        if prompt:
            payload["prompt_token_ids"] = [1, 2]
        return WireMessage(message_type=MessageType.EVENT, request_id=request_id, payload=payload)

    def _submitted_client(self, stack):
        engine_request = make_request("crafted")
        handle = stack.submit(engine_request)
        return stack.client, handle

    def test_out_of_order_event_index_rejected(self, engine):
        stack = LocalProcessEngineClient(FakeEngine(script=infinite_script))
        try:
            client, handle = self._submitted_client(stack)
            client._handle_wire_message(self._event_message("crafted", 5))
            with pytest.raises(ProtocolViolationError, match="out-of-order"):
                next(handle.events())
        finally:
            stack.shutdown()

    def test_duplicate_terminal_rejected(self, engine):
        stack = LocalProcessEngineClient(FakeEngine(script=infinite_script))
        try:
            client, handle = self._submitted_client(stack)
            client._handle_wire_message(self._event_message("crafted", 0, terminal=True))
            client._handle_wire_message(self._event_message("crafted", 1, terminal=True))
            events = handle.events()
            first = next(events)
            assert first.is_terminal
        finally:
            stack.shutdown()

    def test_post_terminal_event_rejected(self, engine):
        stack = LocalProcessEngineClient(FakeEngine(script=infinite_script))
        try:
            client, handle = self._submitted_client(stack)
            checker = client._checkers["crafted"]
            client._handle_wire_message(self._event_message("crafted", 0, terminal=True))
            with pytest.raises(ProtocolViolationError, match="after terminal"):
                checker.observe(
                    __import__("tensorrt_llm.engine_api", fromlist=["EngineEvent"]).EngineEvent(
                        request_id="crafted", event_index=1, token_ids=[5]
                    )
                )
        finally:
            stack.shutdown()


class TestControlPlaneOverSocket:
    def test_stats_and_kv_events_cross_the_socket(self, stack):
        assert stack.get_stats(timeout=0.1) == [{"iter": 1, "num_active_requests": 0}]
        assert stack.get_kv_events(timeout=0.1) == [{"event_id": 0, "data": {"type": "created"}}]

    def test_health_reflects_backend_and_readiness(self, engine, stack):
        assert stack.check_health() is True
        engine.healthy = False
        assert stack.check_health() is False

    def test_unknown_control_method_typed_error(self, stack):
        with pytest.raises(EngineClientError) as excinfo:
            stack.client._control("collective_rpc")
        assert excinfo.value.error.code is EngineErrorCode.UNSUPPORTED_CAPABILITY
