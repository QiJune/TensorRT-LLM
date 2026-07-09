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
"""Control plane over the boundary socket, exercised through the detached app."""

import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "tests" / "unittest" / "engine_api"))

from fake_engine import FakeEngine  # noqa: E402

from tensorrt_llm.engine_api.contracts import EngineClientError, EngineErrorCode  # noqa: E402
from tensorrt_llm.engine_api.socket_transport import EngineSocketServer  # noqa: E402
from tensorrt_llm.serve.frontend.detached_app import (  # noqa: E402
    DetachedFrontend,
    create_detached_app,
)


class MinimalTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True, **kwargs):
        return [len(word) for word in text.split()]

    def decode(self, token_ids, **kwargs):
        if isinstance(token_ids, int):
            return f"<{token_ids}>"
        return "".join(f"<{t}>" for t in token_ids)


@pytest.fixture
def stack():
    engine = FakeEngine()
    server = EngineSocketServer(
        engine,
        endpoint=f"ipc:///tmp/runtime_control_{uuid.uuid4().hex}.sock",
        model_context={"model": "control-model", "tokenizer_dir": None},
    )
    server.start()
    frontend = DetachedFrontend(server.endpoint, tokenizer=MinimalTokenizer())
    client = TestClient(create_detached_app(frontend))
    yield engine, server, frontend, client
    frontend.shutdown()
    server.shutdown()


class TestControlEndpointsDetached:
    def test_health_returns_live_engine_state(self, stack):
        engine, _server, _frontend, client = stack
        assert client.get("/health").status_code == 200
        engine.healthy = False
        assert client.get("/health").status_code == 503

    def test_stats_cross_the_socket(self, stack):
        _engine, _server, _frontend, client = stack
        response = client.get("/iteration_stats")
        assert response.status_code == 200
        assert response.json() == [{"iter": 1, "num_active_requests": 0}]

    def test_kv_cache_events_cross_the_socket(self, stack):
        _engine, _server, _frontend, client = stack
        response = client.get("/kv_cache_events")
        assert response.status_code == 200
        assert response.json() == [{"event_id": 0, "data": {"type": "created"}}]

    def test_version_and_models(self, stack):
        _engine, _server, _frontend, client = stack
        assert "version" in client.get("/version").json()
        assert client.get("/v1/models").json()["data"][0]["id"] == "control-model"

    def test_collective_rpc_endpoints_return_typed_unsupported(self, stack):
        _engine, _server, _frontend, client = stack
        for route in ("/release_memory", "/resume_memory", "/update_weights"):
            response = client.post(route)
            assert response.status_code == 501
            body = response.json()["error"]
            assert body["code"] == EngineErrorCode.UNSUPPORTED_CAPABILITY.value
            assert body["type"] == "unsupported_capability"

    def test_collective_rpc_rejected_at_the_protocol_layer_too(self, stack):
        _engine, _server, frontend, _client = stack
        with pytest.raises(EngineClientError) as excinfo:
            frontend.client._control("collective_rpc", kwargs={"method": "sleep"})
        assert excinfo.value.error.code is EngineErrorCode.UNSUPPORTED_CAPABILITY


class TestDetachedFrontendLifecycle:
    def test_frontend_constructed_during_initialization_serves_after_ready(self):
        """A DetachedFrontend created while the engine initializes must wait
        for readiness and then serve — no reconstruction."""
        import threading

        from tensorrt_llm.engine_api.engine_server import EngineServer

        release = threading.Event()
        engine = FakeEngine()

        def slow_factory():
            release.wait(timeout=30)
            return engine, {"model": "late-model", "tokenizer_dir": None}

        server = EngineServer(slow_factory)
        server.start(wait=False)
        # Flip the engine to ready shortly after frontend construction begins.
        releaser = threading.Timer(0.5, release.set)
        releaser.start()
        frontend = None
        try:
            frontend = DetachedFrontend(
                server.endpoint,
                handshake_timeout_seconds=30,
                tokenizer=MinimalTokenizer(),
            )
            assert frontend.model_context.model == "late-model"
            client = TestClient(create_detached_app(frontend))
            assert client.get("/health").status_code == 200
            response = client.post(
                "/v1/completions",
                json={"model": "late-model", "prompt": "hi there", "max_tokens": 4},
            )
            assert response.status_code == 200
        finally:
            releaser.cancel()
            release.set()
            if frontend is not None:
                frontend.shutdown()
            server.shutdown()

    def test_frontend_fails_typed_when_engine_never_becomes_ready(self):
        from tensorrt_llm.engine_api.engine_server import EngineServer

        import threading

        release = threading.Event()

        def blocked_factory():
            release.wait(timeout=60)
            return FakeEngine(), {"model": "never"}

        server = EngineServer(blocked_factory)
        server.start(wait=False)
        try:
            with pytest.raises(EngineClientError) as excinfo:
                DetachedFrontend(
                    server.endpoint,
                    handshake_timeout_seconds=1.5,
                    tokenizer=MinimalTokenizer(),
                )
            assert excinfo.value.error.code is EngineErrorCode.ENGINE_UNAVAILABLE
            assert "did not become ready" in excinfo.value.error.message
        finally:
            release.set()
            server.shutdown()


class TestControlAfterEngineDeath:
    def test_control_call_after_engine_death_is_typed_and_health_degrades(self):
        engine = FakeEngine()
        server = EngineSocketServer(
            engine,
            endpoint=f"ipc:///tmp/runtime_control_{uuid.uuid4().hex}.sock",
            model_context={"model": "control-model", "tokenizer_dir": None},
        )
        server.start()
        frontend = DetachedFrontend(server.endpoint, tokenizer=MinimalTokenizer())
        frontend.client._control_timeout = 1.0
        client = TestClient(create_detached_app(frontend))
        try:
            assert client.get("/health").status_code == 200
            server.shutdown()
            assert client.get("/health").status_code == 503
            response = client.get("/iteration_stats")
            assert response.status_code == 503
            assert response.json()["error"]["code"] == EngineErrorCode.ENGINE_UNAVAILABLE.value
        finally:
            frontend.shutdown()
            server.shutdown()
