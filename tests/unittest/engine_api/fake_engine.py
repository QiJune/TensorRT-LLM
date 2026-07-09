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
"""CPU-only scripted fake engine for boundary protocol and lifecycle tests.

Implements the ``EngineClient`` interface with scripted event streams — no
GPU, no model, no runtime. Scripts control per-request behavior: bounded
streams, infinite streams (until aborted), mid-stream engine exceptions,
and floods for slow-consumer tests.
"""

from __future__ import annotations

import threading
import time
from typing import Any, AsyncIterator, Callable, Iterator, Optional

from tensorrt_llm.engine_api.contracts import (
    EngineClient,
    EngineClientError,
    EngineError,
    EngineErrorCode,
    EngineEvent,
    EngineRequest,
    RequestHandle,
    TerminalKind,
)

ScriptFn = Callable[[EngineRequest, "FakeEngine"], Iterator[EngineEvent]]


def default_script(request: EngineRequest, engine: "FakeEngine") -> Iterator[EngineEvent]:
    """Echo one event per prompt token, then finish by length."""
    total = max(len(request.prompt_token_ids), 1)
    for index in range(total):
        is_last = index == total - 1
        yield EngineEvent(
            request_id=request.request_id,
            event_index=index,
            token_ids=[1000 + index],
            prompt_token_ids=list(request.prompt_token_ids) if index == 0 else None,
            terminal_kind=TerminalKind.FINISHED if is_last else None,
            finish_reason="length" if is_last else None,
        )


def infinite_script(request: EngineRequest, engine: "FakeEngine") -> Iterator[EngineEvent]:
    """Stream forever (with a small delay) until the request is aborted."""
    index = 0
    while True:
        yield EngineEvent(
            request_id=request.request_id,
            event_index=index,
            token_ids=[1000 + index],
            prompt_token_ids=list(request.prompt_token_ids) if index == 0 else None,
        )
        index += 1
        time.sleep(0.005)


def exploding_script(request: EngineRequest, engine: "FakeEngine") -> Iterator[EngineEvent]:
    """Emit one event, then raise an engine-side exception."""
    yield EngineEvent(
        request_id=request.request_id,
        event_index=0,
        token_ids=[1000],
        prompt_token_ids=list(request.prompt_token_ids),
    )
    raise RuntimeError("engine exploded mid-stream")


def flood_script(count: int) -> ScriptFn:
    """Emit ``count`` events as fast as possible, then finish."""

    def script(request: EngineRequest, engine: "FakeEngine") -> Iterator[EngineEvent]:
        for index in range(count):
            is_last = index == count - 1
            yield EngineEvent(
                request_id=request.request_id,
                event_index=index,
                token_ids=list(range(2048)),
                prompt_token_ids=list(request.prompt_token_ids) if index == 0 else None,
                terminal_kind=TerminalKind.FINISHED if is_last else None,
                finish_reason="length" if is_last else None,
            )

    return script


class _FakeHandle(RequestHandle):
    def __init__(self, engine: "FakeEngine", request: EngineRequest, script: ScriptFn):
        self._engine = engine
        self._request = request
        self._script = script

    @property
    def request_id(self) -> str:
        return self._request.request_id

    def events(self) -> Iterator[EngineEvent]:
        emitted_index = 0
        for event in self._script(self._request, self._engine):
            if self._engine.is_aborted(self._request.request_id):
                yield EngineEvent(
                    request_id=self._request.request_id,
                    sequence_index=event.sequence_index,
                    event_index=event.event_index,
                    terminal_kind=TerminalKind.ABORTED,
                    finish_reason="cancelled",
                )
                return
            emitted_index = event.event_index
            yield event
            if event.is_terminal:
                return
        # Script exhausted without a terminal: close the stream cleanly.
        yield EngineEvent(
            request_id=self._request.request_id,
            event_index=emitted_index + 1,
            terminal_kind=TerminalKind.FINISHED,
            finish_reason="length",
        )

    async def aevents(self) -> AsyncIterator[EngineEvent]:
        for event in self.events():
            yield event

    def abort(self) -> None:
        self._engine.abort(self._request.request_id)


class FakeEngine(EngineClient):
    """Scripted CPU-only engine backend."""

    def __init__(self, script: Optional[ScriptFn] = None) -> None:
        self._script = script or default_script
        self._aborted: set[str] = set()
        self._known: set[str] = set()
        self._lock = threading.Lock()
        self.submitted: list[EngineRequest] = []
        self.aborted_request_ids: list[str] = []
        self.healthy = True
        self.stats: list[dict[str, Any]] = [{"iter": 1, "num_active_requests": 0}]
        self.kv_events: list[dict[str, Any]] = [{"event_id": 0, "data": {"type": "created"}}]

    def submit(self, request: EngineRequest) -> RequestHandle:
        with self._lock:
            self._known.add(request.request_id)
            self.submitted.append(request)
        return _FakeHandle(self, request, self._script)

    def abort(self, request_id: str) -> None:
        with self._lock:
            if request_id not in self._known:
                raise EngineClientError(
                    EngineError(
                        code=EngineErrorCode.UNKNOWN_REQUEST,
                        message=f"unknown request_id {request_id!r}",
                        request_id=request_id,
                    )
                )
            self._aborted.add(request_id)
            self.aborted_request_ids.append(request_id)

    def is_aborted(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._aborted

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "generation": {"streaming": True, "logprobs": True},
            "control": {"health": True, "stats": True, "kv_events": True},
        }

    def check_health(self) -> bool:
        return self.healthy

    def get_stats(self, timeout: float) -> list[dict[str, Any]]:
        return list(self.stats)

    def get_kv_events(self, timeout: float) -> list[dict[str, Any]]:
        return list(self.kv_events)

    def shutdown(self) -> None:
        pass
