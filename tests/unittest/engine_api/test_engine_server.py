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
"""CPU-only lifecycle tests for the headless EngineServer (fake-engine backend)."""

import threading
import time
from types import SimpleNamespace

import pytest
import zmq
from fake_engine import FakeEngine, infinite_script

from tensorrt_llm.engine_api import (
    EngineClientError,
    EngineErrorCode,
    EngineRequest,
    RuntimeSamplingConfig,
    TerminalKind,
)
from tensorrt_llm.engine_api.engine_server import (
    EngineServer,
    RemoteEngineClient,
    validate_headless_launch_args,
)
from tensorrt_llm.engine_api.protocol import (
    MessageType,
    ReadinessState,
    WireMessage,
    decode_message,
    encode_message,
    engine_request_to_payload,
)


def make_request(request_id="req-1", num_tokens=3) -> EngineRequest:
    return EngineRequest(
        request_id=request_id,
        prompt_token_ids=list(range(1, num_tokens + 1)),
        sampling=RuntimeSamplingConfig(max_tokens=8),
        streaming=True,
    )


def instant_factory(engine=None):
    engine = engine or FakeEngine()

    def factory():
        return engine, {"model": "fake", "tokenizer_dir": "/fake"}

    return factory, engine


class TestLifecycle:
    def test_frontend_refuses_traffic_until_engine_ready(self):
        release = threading.Event()
        engine = FakeEngine()

        def slow_factory():
            release.wait(timeout=30)
            return engine, {"model": "fake"}

        server = EngineServer(slow_factory)
        server.start(wait=False)
        try:
            client = RemoteEngineClient(server.endpoint)
            assert client.readiness is ReadinessState.INITIALIZING
            with pytest.raises(EngineClientError) as excinfo:
                client.submit(make_request())
            assert excinfo.value.error.code is EngineErrorCode.ENGINE_UNAVAILABLE
            client.shutdown()

            release.set()
            assert server.wait_ready(timeout=30)
            ready_client = RemoteEngineClient(server.endpoint)
            events = list(ready_client.submit(make_request()).events())
            assert events[-1].terminal_kind is TerminalKind.FINISHED
            ready_client.shutdown()
        finally:
            release.set()
            server.shutdown()

    def test_backend_init_failure_marks_unhealthy_and_raises(self):
        def broken_factory():
            raise RuntimeError("model weights missing")

        server = EngineServer(broken_factory)
        with pytest.raises(RuntimeError, match="model weights missing"):
            server.start(wait=True)
        try:
            assert server.readiness is ReadinessState.UNHEALTHY
        finally:
            server.shutdown()

    def test_engine_death_yields_typed_errors_and_degraded_health(self):
        factory, _engine = instant_factory()
        server = EngineServer(factory)
        server.start()
        client = RemoteEngineClient(server.endpoint, control_timeout_seconds=1.0)
        assert client.check_health() is True
        server.shutdown()
        assert client.check_health() is False
        with pytest.raises(EngineClientError) as excinfo:
            client.get_stats(timeout=0.1)
        assert excinfo.value.error.code is EngineErrorCode.ENGINE_UNAVAILABLE
        client.shutdown()

    def test_frontend_death_leaves_engine_serving(self):
        engine = FakeEngine(script=infinite_script)
        server = EngineServer(
            instant_factory(engine)[0],
            send_buffer_size=4,
            slow_consumer_grace_seconds=0.3,
        )
        server.start()
        try:
            # A frontend connects, submits, and dies without consuming.
            ctx = zmq.Context.instance()
            dealer = ctx.socket(zmq.DEALER)
            dealer.setsockopt(zmq.LINGER, 0)
            dealer.setsockopt(zmq.RCVHWM, 2)
            dealer.connect(server.endpoint)
            dealer.send(encode_message(WireMessage(message_type=MessageType.HANDSHAKE)))
            assert dealer.poll(5000)
            decode_message(dealer.recv())
            dealer.send(
                encode_message(
                    WireMessage(
                        message_type=MessageType.SUBMIT,
                        request_id="dead-frontend-req",
                        payload=engine_request_to_payload(make_request("dead-frontend-req")),
                    )
                )
            )
            time.sleep(0.2)
            dealer.close(linger=0)

            deadline = time.monotonic() + 20
            while (
                time.monotonic() < deadline
                and "dead-frontend-req" not in engine.aborted_request_ids
            ):
                time.sleep(0.05)
            assert "dead-frontend-req" in engine.aborted_request_ids

            # The engine keeps serving new frontends.
            survivor_engine_request = make_request("survivor")
            client = RemoteEngineClient(server.endpoint)
            handle = client.submit(survivor_engine_request)
            events = []
            for event in handle.events():
                events.append(event)
                if len(events) == 2:
                    handle.abort()
            assert events[-1].terminal_kind is TerminalKind.ABORTED
            client.shutdown()
        finally:
            server.shutdown()

    def test_late_result_after_abort_dropped_cleanly(self):
        factory, _engine = instant_factory()
        server = EngineServer(factory)
        server.start()
        client = RemoteEngineClient(server.endpoint)
        try:
            # A late event for a request the client no longer tracks must be
            # dropped without crashing the IO loop.
            client._handle_wire_message(
                WireMessage(
                    message_type=MessageType.EVENT,
                    request_id="finished-long-ago",
                    payload={"sequence_index": 0, "event_index": 7, "token_ids": [5]},
                )
            )
            assert client.check_health() is True
        finally:
            client.shutdown()
            server.shutdown()


class TestReadinessRefresh:
    def test_same_client_becomes_ready_after_engine_initializes(self):
        """A client created during init must serve after readiness flips.

        No reconstruction: submit refreshes the handshake.
        """
        release = threading.Event()
        engine = FakeEngine()

        def slow_factory():
            release.wait(timeout=30)
            return engine, {"model": "fake", "tokenizer_dir": "/fake"}

        server = EngineServer(slow_factory)
        server.start(wait=False)
        client = None
        try:
            client = RemoteEngineClient(server.endpoint)
            assert client.readiness is ReadinessState.INITIALIZING
            release.set()
            assert server.wait_ready(timeout=30)
            # Same client, no reconstruction: submit refreshes readiness.
            events = list(client.submit(make_request()).events())
            assert events[-1].terminal_kind is TerminalKind.FINISHED
            assert client.readiness is ReadinessState.READY
            assert client.model_context.get("model") == "fake"
        finally:
            release.set()
            if client is not None:
                client.shutdown()
            server.shutdown()

    def test_refresh_handshake_updates_capabilities_and_context(self):
        release = threading.Event()
        engine = FakeEngine()

        def slow_factory():
            release.wait(timeout=30)
            return engine, {"model": "fake", "tokenizer_dir": "/fake"}

        server = EngineServer(slow_factory)
        server.start(wait=False)
        client = None
        try:
            client = RemoteEngineClient(server.endpoint)
            assert client.model_context == {}
            release.set()
            server.wait_ready(timeout=30)
            state = client.refresh_handshake(timeout=10)
            assert state is ReadinessState.READY
            assert client.model_context["model"] == "fake"
            assert client.capabilities["generation"]["streaming"] is True
        finally:
            release.set()
            if client is not None:
                client.shutdown()
            server.shutdown()


class TestFailFast:
    def test_runtime_factory_fails_fast_before_model_construction(self):
        """Unsupported configs raise before any runtime import or LLM build.

        The factory call itself raises; construction is never reached.
        """
        from tensorrt_llm.engine_api.engine_server import build_runtime_backend_factory

        with pytest.raises(ValueError, match="num_postprocess_workers=2"):
            build_runtime_backend_factory("some/model", {"num_postprocess_workers": 2})
        with pytest.raises(ValueError, match="pytorch"):
            build_runtime_backend_factory("some/model", {"backend": "tensorrt"})
        with pytest.raises(ValueError, match="orchestrator_type"):
            build_runtime_backend_factory("some/model", {"orchestrator_type": "ray"})

    def test_headless_launch_with_postproc_workers_fails_fast(self):
        args = SimpleNamespace(num_postprocess_workers=2, backend="pytorch", orchestrator_type=None)
        with pytest.raises(ValueError, match="num_postprocess_workers=2"):
            validate_headless_launch_args(args)

    def test_headless_launch_requires_pytorch_backend(self):
        args = SimpleNamespace(
            num_postprocess_workers=0, backend="tensorrt", orchestrator_type=None
        )
        with pytest.raises(ValueError, match="pytorch"):
            validate_headless_launch_args(args)

    def test_headless_launch_requires_default_orchestration(self):
        args = SimpleNamespace(
            num_postprocess_workers=0, backend="pytorch", orchestrator_type="ray"
        )
        with pytest.raises(ValueError, match="orchestrator_type"):
            validate_headless_launch_args(args)

    def test_frontend_with_no_engine_fails_fast(self):
        started = time.monotonic()
        with pytest.raises(EngineClientError) as excinfo:
            RemoteEngineClient("tcp://127.0.0.1:19", handshake_timeout_seconds=0.5)
        assert excinfo.value.error.code is EngineErrorCode.ENGINE_UNAVAILABLE
        assert time.monotonic() - started < 5.0
