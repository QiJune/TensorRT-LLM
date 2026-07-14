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
"""In-process ``EngineClient``: setup gates, pre-submit checks, async streams.

``LocalProcessEngineClient`` is the V0 (in-process) implementation of the
engine-client interface: it attaches an ``EngineFrameRouter`` to the live
``GenerationExecutorProxy``, translates ``EngineRequest`` compositionally
into the worker-facing request, and exposes the typed frame stream as an
async-first ``FrameStream``. Nothing on this surface leaks in-process
affordances: liveness and failure signaling are behind the interface, and
every value that crosses it is a wire type or a typed control-plane result.
"""

import asyncio
import dataclasses
import json
import os
from typing import Optional

from .contract import (ENGINE_CONTRACT_VERSION, ContractError,
                       EngineCapabilities, EngineHealth, EngineRequest,
                       ErrorFrame, FrontendOutputConfig, IterationStatsBatch,
                       KvCacheEventsBatch, RequestComplete, Terminal)
from .conversion import (ConversionError, RequestIneligibleError,
                         derive_required_features)
from .router import RequestBinding, RouterError

__all__ = [
    "ENGINE_CLIENT_FLAG_ENV",
    "V0_CAPABILITY_FEATURES",
    "EngineClientConfigError",
    "RequestRejectedError",
    "EngineClientConfig",
    "FrameStream",
    "LocalProcessEngineClient",
    "engine_client_flag_enabled",
]

ENGINE_CLIENT_FLAG_ENV = "TLLM_EXPERIMENTAL_ENGINE_CLIENT"

V0_CAPABILITY_FEATURES = ("streaming", "logprobs", "prompt_logprobs",
                          "stop_token_sequences", "abort", "usage")


class EngineClientConfigError(ContractError):
    """Unsupported configuration detected at client construction."""


class RequestRejectedError(ContractError):
    """Typed pre-submit rejection (capability, duplicate, or malformed)."""


def resolve_engine_client_flag(args_value: bool = False) -> bool:
    """Resolve the effective flag: the environment wins in both directions.

    Presence, not truthiness, decides the override: unset env defers to the
    configured value; ``"1"``/``"0"`` force the path on/off regardless of
    configuration; anything else fails closed with a typed config error.
    """
    env = os.environ.get(ENGINE_CLIENT_FLAG_ENV)
    if env is None:
        return bool(args_value)
    if env == "1":
        return True
    if env == "0":
        return False
    raise EngineClientConfigError(
        f"flag: {ENGINE_CLIENT_FLAG_ENV} must be '0' or '1', got {env!r}")


def engine_client_flag_enabled() -> bool:
    return resolve_engine_client_flag(False)


@dataclasses.dataclass(frozen=True)
class EngineClientConfig:
    """Deployment facts the setup gates validate.

    The caller (the serving layer) fills this from its own configuration;
    the client rejects anything outside the GPU-validated V0 envelope with
    a typed config error at construction — never mid-run.
    """

    backend: Optional[str] = None
    transport: str = "ipc_proxy"
    num_postprocess_workers: int = 0
    post_processor_hook_set: bool = False
    speculative_config_set: bool = False
    early_first_token_mode: bool = False
    world_size: int = 1
    tokenizer_trust_remote_code: bool = False
    flag_enabled: Optional[bool] = None  # None -> read the env var


class FrameStream:
    """Async single-consumer stream of ``OutputFrame`` for one request.

    Frames buffer from submit time, so nothing emitted before the first
    ``__anext__`` is lost. The stream ends after the request-ending frame.
    ``aclose()`` (or ``async with``) triggers the abort-if-incomplete
    obligation explicitly — never rely on ``GeneratorExit``/GC timing.
    """

    def __init__(self, client: "LocalProcessEngineClient", binding: RequestBinding):
        self._client = client
        self._binding = binding
        self._finished = False
        self._closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._finished or self._closed:
            raise StopAsyncIteration
        loop = asyncio.get_running_loop()
        while True:
            frame, ready = self._binding.delivery.pop_nowait()
            if ready:
                if isinstance(frame, (RequestComplete, ErrorFrame)):
                    self._finished = True
                    # Consumed to the ending frame: retire the delivery so
                    # the request id becomes reusable and router state
                    # returns to its bounds.
                    self._client._router.retire_delivery(
                        self._binding.request_id)
                return frame
            if self._binding.delivery.closed:
                self._finished = True
                raise StopAsyncIteration
            future = loop.create_future()
            if not self._binding.delivery.register_waiter(loop, future):
                continue  # data arrived while registering
            await future

    def pop_ready(self) -> list:
        """Pop all immediately-available frames without awaiting.

        Used by consumers that drain-then-render so a final ``TokenDelta``
        and its ``Terminal`` (enqueued together) are processed as one batch.
        """
        frames = []
        if self._finished or self._closed:
            return frames
        while True:
            frame, ready = self._binding.delivery.pop_nowait()
            if not ready:
                return frames
            frames.append(frame)
            if isinstance(frame, (RequestComplete, ErrorFrame)):
                self._finished = True
                self._client._router.retire_delivery(self._binding.request_id)
                return frames

    async def aclose(self) -> None:
        """Close the stream; aborts the runtime request if incomplete."""
        if self._closed:
            return
        self._closed = True
        if not self._finished:
            try:
                self._client._abort_binding(self._binding)
            finally:
                self._binding.delivery.close()
        # An explicit close retires the delivery either way (the abort's
        # eventual ending is absorbed; the id becomes reusable once ended).
        self._client._router.retire_delivery(self._binding.request_id)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()


class LocalProcessEngineClient:
    """V0 in-process engine client over the IPC-proxy executor."""

    def __init__(self, executor, config: EngineClientConfig):
        self._validate_config(config)
        if getattr(executor, "attach_engine_service", None) is None:
            raise EngineClientConfigError(
                "transport: executor does not expose the engine-service hooks "
                "(IPC GenerationExecutorProxy required)")
        self._executor = executor
        self._config = config
        self._capabilities = EngineCapabilities(features=V0_CAPABILITY_FEATURES)
        # The service owns contract execution (authoritative routing: contract
        # responses never reach legacy delivery); this client is the thin
        # in-process transport adapter in front of it.
        from .service import EngineService
        self._service = EngineService(executor)
        self._router = self._service.router
        self._closed = False
        executor.attach_engine_service(self._service)

    @staticmethod
    def _validate_config(config: EngineClientConfig) -> None:
        # The environment wins in both directions, even over an explicit
        # flag_enabled value (see resolve_engine_client_flag).
        flag = resolve_engine_client_flag(bool(config.flag_enabled))
        if not flag:
            raise EngineClientConfigError(
                f"flag: set {ENGINE_CLIENT_FLAG_ENV}=1 to enable the "
                "experimental engine client")
        # Gate on the explicit backend name: `_is_pytorch_backend`-style
        # checks are also true for the transitional _autodeploy backend,
        # which is explicitly rejected here.
        if config.backend != "pytorch":
            raise EngineClientConfigError(
                f"backend: {config.backend!r} is not supported (pytorch only; "
                "the transitional _autodeploy backend is explicitly rejected)")
        if config.transport != "ipc_proxy":
            raise EngineClientConfigError(
                f"transport: {config.transport!r} is not supported (IPC proxy only)")
        if config.num_postprocess_workers != 0:
            raise EngineClientConfigError(
                "postproc_workers: num_postprocess_workers must be 0")
        if config.post_processor_hook_set:
            raise EngineClientConfigError(
                "post_processor_hook: the global output hook is not supported")
        if config.speculative_config_set:
            raise EngineClientConfigError(
                "speculative_config: speculative decoding is not validated")
        if config.early_first_token_mode:
            raise EngineClientConfigError(
                "early_first_token: early first-token response mode is not supported")
        if config.world_size != 1:
            raise EngineClientConfigError(
                f"topology: world_size={config.world_size} exceeds the GPU-validated "
                "set (TP1 over the IPC proxy)")
        if config.tokenizer_trust_remote_code:
            raise EngineClientConfigError(
                "trust_remote_code_tokenizer: tokenizer provenance cannot be pinned")

    # ------------------------------------------------------------------ #
    # EngineClient interface
    # ------------------------------------------------------------------ #

    def capabilities(self) -> EngineCapabilities:
        return self._capabilities

    def submit(self, engine_request: EngineRequest,
               output_config: Optional[FrontendOutputConfig] = None) -> str:
        """Submit an eligible request; returns its request id.

        Pre-submit checks (typed, before the engine sees anything):
        protocol version, re-derived ``required_features`` vs the caller's,
        capability subset, duplicate id. ``output_config`` never crosses the
        contract; it only supplies the ordered stop-reason association the
        router uses to resolve ``Terminal.stop_reason``.
        """
        if self._closed:
            raise RequestRejectedError("client is closed")
        if self._router.fatal_error is not None:
            raise RequestRejectedError(
                f"engine failed: {self._router.fatal_error}")
        if not isinstance(engine_request, EngineRequest):
            raise RequestRejectedError(
                f"expected EngineRequest, got {type(engine_request).__name__}")
        if engine_request.protocol_version > ENGINE_CONTRACT_VERSION:
            raise RequestRejectedError(
                f"protocol_version {engine_request.protocol_version} is newer than "
                f"supported {ENGINE_CONTRACT_VERSION}")
        derived = derive_required_features(engine_request)
        if tuple(engine_request.required_features) != derived:
            raise RequestRejectedError(
                f"required_features {engine_request.required_features!r} do not match "
                f"the request's own fields (derived {derived!r})")
        missing = set(derived) - set(self._capabilities.features)
        if missing:
            raise RequestRejectedError(
                f"capabilities not supported by this engine: {sorted(missing)}")

        stop_reasons = (output_config.stop_sequence_reasons
                        if output_config is not None else ())
        try:
            self._executor.submit_contract(engine_request,
                                           stop_reasons=stop_reasons)
        except (RequestIneligibleError, ConversionError):
            raise
        except RouterError as e:
            raise RequestRejectedError(str(e)) from e
        return engine_request.request_id

    def stream(self, request_id: str) -> FrameStream:
        """Open the single consumer stream for a submitted request.

        Frames buffer from submit time, so a delayed open loses nothing;
        opening twice is a typed error (single consumer).
        """
        try:
            binding = self._router.open_stream_binding(request_id)
        except RouterError as e:
            raise RequestRejectedError(str(e)) from e
        return FrameStream(self, binding)

    def abort(self, request_id: str) -> None:
        self._router.abort(request_id)

    def get_stats(self, timeout: float = 2.0) -> IterationStatsBatch:
        return self._service.get_stats(timeout=timeout)

    def get_kv_events(self, timeout: float = 2.0) -> KvCacheEventsBatch:
        return self._service.get_kv_events(timeout=timeout)

    def health(self) -> EngineHealth:
        return self._service.health()

    def close_client(self) -> None:
        """Detach from the engine: poison this client's streams AND abort its
        in-flight engine work (best effort). Does NOT shut the shared engine
        down; see ``shutdown_engine``.
        """
        if self._closed:
            return
        self._closed = True
        self._service.close_client()

    def shutdown_engine(self) -> None:
        """Privileged: shut the engine itself down (V0 in-process owner only)."""
        self.close_client()
        self._executor.shutdown()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _abort_binding(self, binding: RequestBinding) -> None:
        try:
            self._router.abort(binding.request_id)
        except RouterError:
            pass  # already ended
