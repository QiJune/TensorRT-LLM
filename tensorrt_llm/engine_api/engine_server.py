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
"""Headless engine process: owns the runtime and terminates the boundary socket.

``EngineServer`` binds the boundary socket immediately (so frontends can
handshake and observe ``initializing``), then builds the engine backend —
in production the model runtime whose executor proxy owns MPI session
startup, the result-dispatch and error-monitor threads, and the RPC control
client — and flips readiness to ``ready``. Shutdown walks the reverse
order: readiness goes to ``shutting_down`` (new submissions get typed
errors), the socket closes, and the backend's shutdown discipline runs
(the executor proxy's pre-shutdown/shutdown/atexit sequence is preserved
because the proxy machinery is reused unmodified underneath the adapter).

Frontend-death policy: the engine keeps serving. A dead or non-consuming
frontend connection is dropped by the slow-consumer policy and its
in-flight requests are aborted engine-side; other connections and future
connections are unaffected.

``RemoteEngineClient`` is the frontend counterpart for a detached process;
it never constructs a model, an executor, or a proxy.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from tensorrt_llm.engine_api.contracts import (
    EngineClient,
    EngineClientError,
    EngineError,
    EngineErrorCode,
    EngineRequest,
    RequestHandle,
)
from tensorrt_llm.engine_api.protocol import ReadinessState
from tensorrt_llm.engine_api.socket_transport import EngineSocketServer, SocketEngineClient
from tensorrt_llm.logger import logger

__all__ = ["EngineServer", "RemoteEngineClient", "validate_headless_launch_args"]

# The backend plus the plain-data model context advertised in handshakes.
BackendFactory = Callable[[], tuple[EngineClient, dict[str, Any]]]


def validate_headless_launch_args(llm_args: Any) -> None:
    """Fail fast on configurations the headless engine cannot serve.

    Raises:
        ValueError: When the launch configuration is unsupported (postproc
            worker processes, non-pytorch backend, non-default orchestrator).
    """
    num_postprocess_workers = getattr(llm_args, "num_postprocess_workers", 0) or 0
    if num_postprocess_workers > 0:
        raise ValueError(
            "headless engine mode does not support postprocess worker processes; "
            f"got num_postprocess_workers={num_postprocess_workers}. Output "
            "formatting is owned by the frontend on this path."
        )
    backend = getattr(llm_args, "backend", None)
    if backend != "pytorch":
        raise ValueError(f"headless engine mode requires the pytorch backend, got {backend!r}")
    orchestrator_type = getattr(llm_args, "orchestrator_type", None)
    if orchestrator_type is not None:
        raise ValueError(
            "headless engine mode requires the default MPI/IPC orchestration, "
            f"got orchestrator_type={orchestrator_type!r}"
        )


class _PendingBackend(EngineClient):
    """Placeholder backend while the engine initializes; everything is typed-unavailable."""

    def _unavailable(self) -> EngineClientError:
        return EngineClientError(
            EngineError(
                code=EngineErrorCode.ENGINE_UNAVAILABLE,
                message="engine is still initializing",
            )
        )

    def submit(self, request: EngineRequest) -> RequestHandle:
        raise self._unavailable()

    def abort(self, request_id: str) -> None:
        raise self._unavailable()

    def get_capabilities(self) -> dict[str, Any]:
        return {}

    def check_health(self) -> bool:
        return False

    def get_stats(self, timeout: float) -> list[dict[str, Any]]:
        raise self._unavailable()

    def get_kv_events(self, timeout: float) -> list[dict[str, Any]]:
        raise self._unavailable()

    def shutdown(self) -> None:
        pass


class EngineServer:
    """Engine-side process component: socket + backend lifecycle ownership.

    Args:
        backend_factory: Builds the engine backend (and its handshake model
            context). In production this constructs the model runtime —
            spawning the executor proxy with its MPI session, dispatch and
            error-monitor threads, and RPC control client — and wraps it in
            the legacy engine-client adapter.
        endpoint: Boundary socket endpoint (default: random loopback port).
        socket_kwargs: Forwarded to :class:`EngineSocketServer` (buffer
            sizes, slow-consumer grace, ...).
    """

    def __init__(
        self,
        backend_factory: BackendFactory,
        *,
        endpoint: Optional[str] = None,
        **socket_kwargs: Any,
    ) -> None:
        self._backend_factory = backend_factory
        self._backend: Optional[EngineClient] = None
        self._socket = EngineSocketServer(
            _PendingBackend(), endpoint=endpoint, ready=False, **socket_kwargs
        )
        self._init_thread: Optional[threading.Thread] = None
        self._init_error: Optional[BaseException] = None
        self._shutdown_lock = threading.Lock()
        self._shut_down = False

    @property
    def endpoint(self) -> str:
        return self._socket.endpoint

    @property
    def readiness(self) -> ReadinessState:
        return self._socket.readiness

    @property
    def socket_server(self) -> EngineSocketServer:
        return self._socket

    def start(self, wait: bool = True) -> None:
        """Bind the socket, build the backend, then flip readiness to ready.

        Args:
            wait: Block until the backend finished building (or failed).
        """
        self._socket.start()
        self._init_thread = threading.Thread(
            target=self._initialize, name="engine_server_init", daemon=True
        )
        self._init_thread.start()
        if wait:
            self._init_thread.join()
            if self._init_error is not None:
                raise self._init_error

    def _initialize(self) -> None:
        try:
            backend, model_context = self._backend_factory()
        except BaseException as e:
            self._init_error = e
            self._socket.set_readiness(ReadinessState.UNHEALTHY)
            logger.error(f"engine backend initialization failed: {e}")
            return
        # shutdown() may have run while the factory was still constructing the
        # backend; at that point _backend was still None, so shutdown skipped
        # backend cleanup. Decide under the lock whether to install or tear
        # down the late backend so we never leave a live LLM/executor behind
        # (or flip readiness back to READY after shutdown).
        with self._shutdown_lock:
            if self._shut_down:
                late_backend = backend
            else:
                self._backend = backend
                self._socket.set_backend(backend)
                self._socket.set_model_context(model_context)
                self._socket.set_readiness(ReadinessState.READY)
                late_backend = None
        if late_backend is not None:
            logger.warning(
                "engine backend finished initializing after shutdown; "
                "tearing it down instead of installing it"
            )
            late_backend.shutdown()
            return
        logger.info(f"engine server ready on {self.endpoint}")

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        if self._init_thread is not None:
            self._init_thread.join(timeout)
        return self.readiness is ReadinessState.READY

    def serve_forever(self) -> None:
        """Block until shut down (for CLI entrypoints)."""
        try:
            while not self._shut_down:
                threading.Event().wait(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shut_down:
                return
            self._shut_down = True
        # Readiness first: in-flight handshakes/submissions observe
        # shutting_down and receive typed errors, not hangs.
        self._socket.set_readiness(ReadinessState.SHUTTING_DOWN)
        self._socket.shutdown()
        if self._backend is not None:
            # Runs the proxy's historical pre-shutdown/shutdown discipline.
            self._backend.shutdown()
            self._backend = None


def build_runtime_backend_factory(
    model: str,
    llm_args_extra: Optional[dict[str, Any]] = None,
) -> BackendFactory:
    """Backend factory that builds the model runtime (GPU path).

    Constructing the runtime spawns the executor proxy — MPI session
    startup, result-dispatch and error-monitor threads, RPC control client —
    exactly as the in-process path does; the adapter then exposes it over
    the boundary contract.

    Unsupported launch configurations fail fast here — before any runtime
    import or model construction; a second validation after construction
    guards args normalized by defaults.

    Raises:
        ValueError: Immediately, for unsupported launch configurations.
    """
    import types

    launch_extra = dict(llm_args_extra or {})
    launch_extra.setdefault("backend", "pytorch")
    validate_headless_launch_args(
        types.SimpleNamespace(
            backend=launch_extra.get("backend"),
            orchestrator_type=launch_extra.get("orchestrator_type"),
            num_postprocess_workers=launch_extra.get("num_postprocess_workers", 0),
        )
    )

    def factory() -> tuple[EngineClient, dict[str, Any]]:
        from tensorrt_llm.engine_api.legacy_adapter import LegacyEngineClientAdapter
        from tensorrt_llm.llmapi.llm import LLM

        extra = dict(llm_args_extra or {})
        extra.setdefault("backend", "pytorch")
        llm = LLM(model=model, **extra)
        validate_headless_launch_args(llm.args)
        adapter = LegacyEngineClientAdapter(llm._executor)
        model_context = {
            "model": str(model),
            "tokenizer_dir": str(getattr(llm, "_hf_model_dir", None) or model),
            "max_seq_len": getattr(llm.args, "max_seq_len", None),
            "reasoning_parser": getattr(llm.args, "reasoning_parser", None),
            "return_perf_metrics": bool(getattr(llm.args, "return_perf_metrics", False)),
            "stream_interval": getattr(llm.args, "stream_interval", 1),
            # So the detached frontend can detect chat paths it cannot serve
            # (Harmony/gpt_oss) and load a tokenizer matching the engine's.
            "model_type": getattr(getattr(llm, "_hf_model_config", None), "model_type", None),
            "trust_remote_code": bool(getattr(llm.args, "trust_remote_code", False)),
            "tokenizer_mode": getattr(llm.args, "tokenizer_mode", "auto"),
        }
        return _LlmOwningBackend(adapter, llm), model_context

    return factory


class _LlmOwningBackend(EngineClient):
    """Adapter wrapper that also owns the runtime's lifetime."""

    def __init__(self, adapter: EngineClient, llm: Any) -> None:
        self._adapter = adapter
        self._llm = llm

    def submit(self, request: EngineRequest) -> RequestHandle:
        return self._adapter.submit(request)

    def abort(self, request_id: str) -> None:
        self._adapter.abort(request_id)

    def get_capabilities(self) -> dict[str, Any]:
        return self._adapter.get_capabilities()

    def check_health(self) -> bool:
        return self._adapter.check_health()

    def get_stats(self, timeout: float) -> list[dict[str, Any]]:
        return self._adapter.get_stats(timeout)

    def get_kv_events(self, timeout: float) -> list[dict[str, Any]]:
        return self._adapter.get_kv_events(timeout)

    def shutdown(self) -> None:
        self._llm.shutdown()


class RemoteEngineClient(SocketEngineClient):
    """Engine client for a frontend process detached from the engine.

    Identical protocol behavior to :class:`SocketEngineClient`; the name
    marks the deployment: the owning process never constructs a model, an
    executor, or a proxy — everything it knows about the engine comes from
    the handshake (``capabilities``, ``readiness``, ``model_context``).
    """
