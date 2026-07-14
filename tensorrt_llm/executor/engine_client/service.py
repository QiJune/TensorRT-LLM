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
"""Server-side engine service: the contract execution endpoint.

``EngineService`` owns everything a contract request needs on the engine
side and nothing the frontend owns: it validates/translates
``EngineRequest`` into the worker-facing request, submits through the
proxy's shared client-id allocator and request queue **without** a rank-0
``GenerationResult``, exclusively routes contract responses off the
dispatch loop, and implements abort / client-detach / ``fail_all`` with
best-effort runtime cancellation. It holds no tokenizer, HTTP request,
formatter callback, or event loop.

A later remote detach puts a transport terminator in front of this class
and changes nothing inside: every operation already takes/returns wire
types or primitives (see ``E5B_CUTOVER_DESIGN.md``).
"""

import json
from typing import Optional, Tuple

from tensorrt_llm.logger import logger

from .contract import (EngineHealth, EngineRequest, IterationStatsBatch,
                       KvCacheEventsBatch)
from .conversion import engine_request_to_generation_request
from .router import EngineFrameRouter, RequestBinding, RouterError

__all__ = ["EngineService", "EngineServiceError"]


class EngineServiceError(RouterError):
    """Contract submission failed at the service (engine-side) boundary."""


class EngineService:
    """Contract-native execution endpoint over the IPC proxy runtime.

    The proxy supplies the narrow runtime surface this service depends on:
    ``_start_dispatch_threads``, ``_get_next_client_id``, ``request_queue``,
    ``abort_request``, ``_handle_background_error``, ``get_stats`` /
    ``get_kv_events`` / ``check_health``, ``doing_shutdown``, and the
    submission/lifecycle lock (``_submission_lock``). It never touches ZMQ
    socket construction or MPI process management.
    """

    def __init__(self, proxy):
        self._proxy = proxy
        self.router = EngineFrameRouter(abort_fn=proxy.abort_request)
        self._closed = False
        # The shared proxy allocator is monotonic within a proxy lifetime;
        # wrap or collision must fail submission rather than reuse an id.
        self._max_client_id: int = 0

    # ------------------------------------------------------------------ #
    # Submission
    # ------------------------------------------------------------------ #

    def submit_contract(self, engine_request: EngineRequest,
                        stop_reasons: Tuple = ()) -> int:
        """Submit a contract request; returns the allocated proxy client id.

        No rank-0 ``GenerationResult`` is constructed or registered: the
        router binding IS the rank-0 request state. Order (per the E5b
        design note): translate first (a rejection leaves no engine-side
        state), then — under the proxy submission/lifecycle lock — allocate
        the shared client id, bind, and enqueue, so shutdown can never
        interleave between binding and enqueue and a response can never
        beat the binding.
        """
        # 1. Translate before any allocation: conversion failures are
        #    state-free.
        generation_request = engine_request_to_generation_request(engine_request)
        proxy = self._proxy
        proxy._start_dispatch_threads()
        with proxy._submission_lock:
            if proxy.doing_shutdown or getattr(proxy, "_fatal_error", None) \
                    or self.router.fatal_error is not None or self._closed:
                raise EngineServiceError(
                    "engine is shutting down or failed; submission rejected")
            client_id = proxy._get_next_client_id()
            if client_id in proxy._results:
                raise EngineServiceError(
                    f"client id {client_id} collides with an active legacy "
                    "request; refusing to reuse an id in this proxy lifetime")
            self.observe_legacy_allocation(client_id)
            generation_request.set_id(client_id)
            binding = self.router.bind(engine_request.request_id, client_id,
                                       engine_request.prompt_token_ids,
                                       stop_reasons)
            try:
                proxy.request_queue.put(generation_request)
            except Exception:
                self.router.on_submit_enqueue_failed(client_id)
                raise
        proxy._handle_background_error()
        return client_id

    def observe_legacy_allocation(self, client_id: int) -> None:
        """Monotonic-allocation guard shared by BOTH populations.

        Called (submission lock held) for every id the shared proxy
        allocator hands out — contract submissions above and legacy
        submissions via ``GenerationExecutorProxy.submit``. The raw
        allocator wraps modulo 2**64; while contract routing is exclusive,
        a wrapped or regressed id must fail the submission it was handed
        to, whichever population received it, before it can collide with a
        contract-owned or recently retired id.
        """
        if client_id <= self._max_client_id:
            raise EngineServiceError(
                f"client id allocator wrapped or regressed ({client_id} <= "
                f"{self._max_client_id}); refusing to reuse an id")
        self._max_client_id = client_id

    # ------------------------------------------------------------------ #
    # Response routing (dispatch thread)
    # ------------------------------------------------------------------ #

    def route_response(self, raw) -> bool:
        """Exclusively claim a contract-owned response; see the router."""
        return self.router.route_response(raw)

    # ------------------------------------------------------------------ #
    # Abort / detach / failure
    # ------------------------------------------------------------------ #

    def abort(self, request_id: str) -> None:
        self.router.abort(request_id)

    def fail_all(self, reason: str) -> None:
        """Poison every contract stream typed AND cancel runtime work.

        The active client ids are snapshotted before the bindings are
        pruned; runtime cancellation is best effort (the worker may already
        be gone), poisoning is guaranteed. Idempotent.
        """
        # Latch under the proxy submission lock so a concurrent
        # submit_contract cannot pass its closed/fatal check and bind after
        # the poisoning sweep (the lock is reentrant for shutdown callers).
        with self._proxy._submission_lock:
            active_client_ids = self.router.fail_all_and_collect(reason)
        for client_id in active_client_ids:
            try:
                self._proxy.abort_request(client_id)
            except Exception as e:
                logger.debug(f"engine-service: best-effort runtime cancel of "
                             f"client_id={client_id} failed: {e!r}")

    def close_client(self, reason: str = "client closed") -> None:
        """Detach this client: poison its streams and abort its engine work.

        Does NOT shut the shared engine down.
        """
        with self._proxy._submission_lock:
            if self._closed:
                return
            self._closed = True
        self.fail_all(reason)

    # ------------------------------------------------------------------ #
    # Typed control plane
    # ------------------------------------------------------------------ #

    def get_stats(self, timeout: float = 2.0) -> IterationStatsBatch:
        entries = self._proxy.get_stats(timeout=timeout)
        return IterationStatsBatch(entries=tuple(
            entry if isinstance(entry, str) else json.dumps(entry)
            for entry in entries))

    def get_kv_events(self, timeout: float = 2.0) -> KvCacheEventsBatch:
        entries = self._proxy.get_kv_events(timeout=timeout)
        return KvCacheEventsBatch(entries=tuple(
            entry if isinstance(entry, str) else json.dumps(entry)
            for entry in entries))

    def health(self) -> EngineHealth:
        if self.router.fatal_error is not None:
            return EngineHealth(healthy=False, detail=self.router.fatal_error)
        try:
            healthy = bool(self._proxy.check_health())
        except Exception as e:
            return EngineHealth(healthy=False, detail=repr(e))
        return EngineHealth(healthy=healthy,
                            detail="" if healthy else "executor unhealthy")

    def get_binding(self, request_id: str) -> Optional[RequestBinding]:
        return self.router.get_binding(request_id)
