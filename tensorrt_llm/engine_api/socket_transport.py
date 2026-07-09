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
"""ZMQ socket transport for the engine boundary protocol.

``EngineSocketServer`` terminates the boundary on the engine side: it binds
a ROUTER socket, speaks the versioned msgpack protocol of ``PROTOCOL.md``,
and drives any :class:`~tensorrt_llm.engine_api.contracts.EngineClient`
backend (the legacy executor adapter in production; a scripted fake engine
in CPU-only tests).

``SocketEngineClient`` is the frontend side: a DEALER socket with the
version/capability/readiness handshake, per-request event streams validated
by the ordering checker, typed errors, and the control plane.

``LocalProcessEngineClient`` co-locates both ends in one process over a
loopback endpoint so the full codec and protocol are exercised without a
process split.

Slow-consumer policy (engine side, per connection): outbound events buffer
up to ``send_buffer_size`` messages; when a connection stays over that
watermark (or its ZMQ pipe stays full) beyond
``slow_consumer_grace_seconds``, the connection is dropped and its in-flight
requests are aborted engine-side under a typed ``slow_consumer`` error.
"""

from __future__ import annotations

import asyncio
import collections
import itertools
import queue
import threading
import time
import uuid
from typing import Any, AsyncIterator, Iterator, Optional

import zmq

from tensorrt_llm.engine_api.contracts import (
    EngineClient,
    EngineClientError,
    EngineError,
    EngineErrorCode,
    EngineEvent,
    EngineRequest,
    EventOrderingChecker,
    RequestHandle,
    TerminalKind,
)
from tensorrt_llm.engine_api.protocol import (
    PROTOCOL_VERSION,
    MessageType,
    ReadinessState,
    WireMessage,
    decode_message,
    encode_message,
    engine_event_from_payload,
    engine_event_to_payload,
    engine_request_from_payload,
    engine_request_to_payload,
    error_message,
    handshake_reply,
)
from tensorrt_llm.logger import logger

__all__ = [
    "EngineSocketServer",
    "LocalProcessEngineClient",
    "SocketEngineClient",
]

_POLL_INTERVAL_MS = 20
_CONTROL_METHODS = ("get_capabilities", "check_health", "get_stats", "get_kv_events")


class _Connection:
    """Per-frontend-connection outbound state on the server side."""

    def __init__(self, identity: bytes) -> None:
        self.identity = identity
        self.outbox: collections.deque[bytes] = collections.deque()
        self.blocked_since: Optional[float] = None
        self.dropped = False
        self.request_ids: set[str] = set()


class EngineSocketServer:
    """Engine-side protocol server over a ZMQ ROUTER socket.

    Args:
        backend: The engine boundary implementation this server exposes.
        endpoint: ZMQ endpoint to bind; defaults to a random loopback TCP
            port (localhost-only per the protocol threat note).
        capabilities: Capability map advertised in handshakes; defaults to
            the backend's ``get_capabilities()``.
        model_context: Plain-data model context advertised in handshakes
            (tokenizer source, model metadata, limits).
        send_buffer_size: Per-connection outbound watermark (messages).
        slow_consumer_grace_seconds: How long a connection may stay blocked
            over the watermark before it is dropped and its requests aborted.
        ready: Initial readiness; use ``set_readiness`` to change later.
    """

    def __init__(
        self,
        backend: EngineClient,
        *,
        endpoint: Optional[str] = None,
        capabilities: Optional[dict[str, Any]] = None,
        model_context: Optional[dict[str, Any]] = None,
        send_buffer_size: int = 256,
        slow_consumer_grace_seconds: float = 10.0,
        ready: bool = True,
    ) -> None:
        self._backend = backend
        self._capabilities = capabilities
        self._model_context = model_context or {}
        self._send_buffer_size = send_buffer_size
        self._grace = slow_consumer_grace_seconds
        self._readiness = ReadinessState.READY if ready else ReadinessState.INITIALIZING

        self._ctx = zmq.Context.instance()
        self._router = self._ctx.socket(zmq.ROUTER)
        self._router.setsockopt(zmq.LINGER, 0)
        self._router.setsockopt(zmq.SNDHWM, max(send_buffer_size, 8))
        self._router.setsockopt(zmq.ROUTER_MANDATORY, 1)
        if endpoint is None:
            port = self._router.bind_to_random_port("tcp://127.0.0.1")
            endpoint = f"tcp://127.0.0.1:{port}"
        else:
            self._router.bind(endpoint)
        self.endpoint = endpoint

        self._connections: dict[bytes, _Connection] = {}
        self._active: dict[str, bytes] = {}  # request_id -> connection identity
        self._handles: dict[str, RequestHandle] = {}
        self._lock = threading.Lock()
        self._running = False
        self._loop_thread: Optional[threading.Thread] = None
        self.slow_consumer_drops: list[EngineError] = []

    # --- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop_thread = threading.Thread(
            target=self._serve_loop, name="engine_socket_server", daemon=True
        )
        self._loop_thread.start()

    def shutdown(self) -> None:
        self.set_readiness(ReadinessState.SHUTTING_DOWN)
        self._running = False
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5)
            self._loop_thread = None
        self._router.close(linger=0)

    def set_readiness(self, state: ReadinessState) -> None:
        self._readiness = state

    @property
    def readiness(self) -> ReadinessState:
        return self._readiness

    # --- serve loop -------------------------------------------------------------

    def _serve_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._router, zmq.POLLIN)
        while self._running:
            try:
                events = dict(poller.poll(_POLL_INTERVAL_MS))
                if self._router in events:
                    identity, payload = self._router.recv_multipart()
                    self._dispatch(identity, payload)
                self._flush_outboxes()
            except zmq.ZMQError:
                if self._running:
                    logger.error("engine socket server ZMQ error", exc_info=True)
            except Exception:
                logger.error("engine socket server dispatch error", exc_info=True)

    def _connection(self, identity: bytes) -> _Connection:
        connection = self._connections.get(identity)
        if connection is None:
            connection = _Connection(identity)
            self._connections[identity] = connection
        return connection

    def _dispatch(self, identity: bytes, payload: bytes) -> None:
        connection = self._connection(identity)
        if connection.dropped:
            return
        try:
            message = decode_message(payload)
        except Exception as e:
            self._enqueue(connection, error_message(EngineErrorCode.PROTOCOL_VIOLATION, str(e)))
            return

        if message.message_type is MessageType.HANDSHAKE:
            self._handle_handshake(connection, message)
        elif message.message_type is MessageType.SUBMIT:
            self._handle_submit(connection, message)
        elif message.message_type is MessageType.ABORT:
            self._handle_abort(connection, message)
        elif message.message_type is MessageType.CONTROL_REQUEST:
            self._handle_control(connection, message)
        else:
            self._enqueue(
                connection,
                error_message(
                    EngineErrorCode.PROTOCOL_VIOLATION,
                    f"unexpected {message.message_type.value} message on the engine side",
                ),
            )

    def _handle_handshake(self, connection: _Connection, message: WireMessage) -> None:
        if message.protocol_version != PROTOCOL_VERSION:
            self._enqueue(
                connection,
                error_message(
                    EngineErrorCode.PROTOCOL_VERSION_MISMATCH,
                    f"engine speaks protocol {PROTOCOL_VERSION}, "
                    f"frontend spoke {message.protocol_version}",
                ),
            )
            return
        capabilities = self._capabilities
        if capabilities is None:
            capabilities = self._backend.get_capabilities()
        self._enqueue(
            connection, handshake_reply(capabilities, self._readiness, self._model_context)
        )

    def _handle_submit(self, connection: _Connection, message: WireMessage) -> None:
        request_id = message.request_id
        if self._readiness is not ReadinessState.READY:
            code = (
                EngineErrorCode.ENGINE_SHUTDOWN
                if self._readiness is ReadinessState.SHUTTING_DOWN
                else EngineErrorCode.ENGINE_UNAVAILABLE
            )
            self._enqueue(
                connection,
                error_message(code, f"engine is {self._readiness.value}", request_id),
            )
            return
        try:
            request = engine_request_from_payload(message.payload)
            handle = self._backend.submit(request)
        except EngineClientError as e:
            self._enqueue(connection, error_message(e.error.code, e.error.message, request_id))
            return
        except Exception as e:
            self._enqueue(
                connection,
                error_message(EngineErrorCode.INVALID_REQUEST, str(e), request_id),
            )
            return
        with self._lock:
            self._active[request.request_id] = connection.identity
            self._handles[request.request_id] = handle
            connection.request_ids.add(request.request_id)
        pump = threading.Thread(
            target=self._pump_events,
            args=(connection, request.request_id, handle),
            name=f"engine_socket_pump_{request.request_id}",
            daemon=True,
        )
        pump.start()

    def _pump_events(self, connection: _Connection, request_id: str, handle: RequestHandle) -> None:
        try:
            for event in handle.events():
                self._enqueue(
                    connection,
                    WireMessage(
                        message_type=MessageType.EVENT,
                        request_id=request_id,
                        payload=engine_event_to_payload(event),
                    ),
                )
                if connection.dropped:
                    break
        except EngineClientError as e:
            self._send_error_event(connection, request_id, e.error.code, e.error.message)
        except Exception as e:
            # Engine-side exceptions surface as typed error events; never a
            # pickled exception or stack trace.
            self._send_error_event(connection, request_id, EngineErrorCode.INTERNAL_ERROR, str(e))
        finally:
            with self._lock:
                self._active.pop(request_id, None)
                self._handles.pop(request_id, None)
                connection.request_ids.discard(request_id)

    def _send_error_event(
        self,
        connection: _Connection,
        request_id: str,
        code: EngineErrorCode,
        message: str,
    ) -> None:
        event = EngineEvent(
            request_id=request_id,
            event_index=0,
            terminal_kind=TerminalKind.ERROR,
            error=EngineError(code=code, message=message, request_id=request_id),
        )
        # Error terminals are position-independent: the client accepts them
        # regardless of the per-sequence event_index cursor.
        self._enqueue(
            connection,
            WireMessage(
                message_type=MessageType.EVENT,
                request_id=request_id,
                payload=engine_event_to_payload(event),
            ),
        )

    def _handle_abort(self, connection: _Connection, message: WireMessage) -> None:
        request_id = message.request_id
        with self._lock:
            known = request_id in self._active
        self._enqueue(
            connection,
            WireMessage(
                message_type=MessageType.ABORT_ACK,
                request_id=request_id,
                payload={"known": known},
            ),
        )
        if known:
            try:
                self._backend.abort(request_id)
            except EngineClientError:
                pass

    def _handle_control(self, connection: _Connection, message: WireMessage) -> None:
        control_id = message.payload.get("control_id")
        method = message.payload.get("method")
        kwargs = message.payload.get("kwargs") or {}
        payload: dict[str, Any] = {"control_id": control_id}
        if method not in _CONTROL_METHODS:
            payload["error_code"] = EngineErrorCode.UNSUPPORTED_CAPABILITY.value
            payload["error_message"] = f"unknown control method {method!r}"
        else:
            try:
                result = getattr(self._backend, method)(**kwargs)
                if method == "check_health":
                    result = {
                        "healthy": bool(result) and self._readiness is ReadinessState.READY,
                        "readiness_state": self._readiness.value,
                    }
                payload["result"] = result
            except EngineClientError as e:
                payload["error_code"] = e.error.code.value
                payload["error_message"] = e.error.message
            except Exception as e:
                payload["error_code"] = EngineErrorCode.INTERNAL_ERROR.value
                payload["error_message"] = str(e)
        self._enqueue(
            connection,
            WireMessage(
                message_type=MessageType.CONTROL_RESPONSE,
                request_id=message.request_id,
                payload=payload,
            ),
        )

    # --- outbound with slow-consumer policy ------------------------------------

    def _enqueue(self, connection: _Connection, message: WireMessage) -> None:
        if connection.dropped:
            return
        connection.outbox.append(encode_message(message))

    def _flush_outboxes(self) -> None:
        now = time.monotonic()
        for connection in list(self._connections.values()):
            if connection.dropped:
                continue
            while connection.outbox:
                try:
                    self._router.send_multipart(
                        [connection.identity, connection.outbox[0]], flags=zmq.NOBLOCK
                    )
                    connection.outbox.popleft()
                except zmq.ZMQError:
                    # Peer pipe full (or peer gone): stop sending this cycle.
                    break
            # The blocked clock runs while the buffer stays over the
            # watermark; it clears only when the consumer catches up.
            if len(connection.outbox) > self._send_buffer_size:
                if connection.blocked_since is None:
                    connection.blocked_since = now
            else:
                connection.blocked_since = None
            if (
                connection.blocked_since is not None
                and now - connection.blocked_since > self._grace
            ):
                self._drop_slow_consumer(connection)

    def _drop_slow_consumer(self, connection: _Connection) -> None:
        connection.dropped = True
        connection.outbox.clear()
        with self._lock:
            in_flight = list(connection.request_ids)
        error = EngineError(
            code=EngineErrorCode.SLOW_CONSUMER,
            message="frontend connection dropped: events not consumed within the "
            "slow-consumer grace period",
        )
        self.slow_consumer_drops.append(error)
        logger.warning(f"engine socket server: {error.message}")
        for request_id in in_flight:
            try:
                self._backend.abort(request_id)
            except EngineClientError:
                pass


class _SocketRequestHandle(RequestHandle):
    """Client-side handle: events arrive via the client IO thread.

    The stream completes when every sequence of the request has produced its
    terminal event (``expected_terminals`` = ``best_of``/``n``); a
    position-independent error terminal completes it immediately.
    """

    def __init__(
        self, client: "SocketEngineClient", request_id: str, expected_terminals: int = 1
    ) -> None:
        self._client = client
        self._request_id = request_id
        self._queue: queue.Queue = queue.Queue()
        self._finished = False
        self._expected_terminals = max(expected_terminals, 1)
        self._seen_terminals = 0

    @property
    def request_id(self) -> str:
        return self._request_id

    def _deliver(self, item: Any) -> None:
        self._queue.put(item)

    def _next_item(self) -> Any:
        item = self._queue.get()
        if isinstance(item, Exception):
            self._finished = True
            raise item
        if item.is_terminal:
            if item.terminal_kind is TerminalKind.ERROR:
                self._finished = True
            else:
                self._seen_terminals += 1
                if self._seen_terminals >= self._expected_terminals:
                    self._finished = True
        return item

    def events(self) -> Iterator[EngineEvent]:
        while not self._finished:
            yield self._next_item()
        self._client._forget(self._request_id)

    async def aevents(self) -> AsyncIterator[EngineEvent]:
        while not self._finished:
            yield await asyncio.to_thread(self._next_item)
        self._client._forget(self._request_id)

    def abort(self) -> None:
        if self._finished:
            return  # idempotent no-op after the terminal event
        self._client.abort(self._request_id)


class SocketEngineClient(EngineClient):
    """Frontend-side protocol client over a ZMQ DEALER socket.

    Fails fast when no engine answers the handshake within
    ``handshake_timeout_seconds``.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        handshake_timeout_seconds: float = 10.0,
        control_timeout_seconds: float = 30.0,
    ) -> None:
        self._endpoint = endpoint
        self._control_timeout = control_timeout_seconds
        self._ctx = zmq.Context.instance()
        self._dealer = self._ctx.socket(zmq.DEALER)
        self._dealer.setsockopt(zmq.LINGER, 0)
        self._dealer.connect(endpoint)

        self._handles: dict[str, _SocketRequestHandle] = {}
        self._checkers: dict[str, EventOrderingChecker] = {}
        self._control_futures: dict[int, queue.Queue] = {}
        self._control_ids = itertools.count(1)
        self._outbound: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._running = True
        self._fatal: Optional[EngineError] = None

        self.capabilities: dict[str, Any] = {}
        self.readiness: Optional[ReadinessState] = None
        self.model_context: dict[str, Any] = {}

        self._io_thread = threading.Thread(
            target=self._io_loop, name="socket_engine_client_io", daemon=True
        )
        self._io_thread.start()
        self._handshake(handshake_timeout_seconds)

    # --- handshake -------------------------------------------------------------

    def _handshake(self, timeout: float) -> None:
        reply_queue: queue.Queue = queue.Queue()
        with self._lock:
            self._handshake_queue = reply_queue
        self._send(WireMessage(message_type=MessageType.HANDSHAKE, payload={"client_info": {}}))
        try:
            reply = reply_queue.get(timeout=timeout)
        except queue.Empty:
            self.shutdown()
            raise EngineClientError(
                EngineError(
                    code=EngineErrorCode.ENGINE_UNAVAILABLE,
                    message=f"no engine answered the handshake at {self._endpoint} "
                    f"within {timeout}s",
                )
            ) from None
        if isinstance(reply, EngineClientError):
            self.shutdown()
            raise reply
        self.capabilities = reply.payload.get("capabilities") or {}
        self.readiness = ReadinessState(reply.payload.get("readiness_state"))
        self.model_context = reply.payload.get("model_context") or {}

    def require_capability(self, *path: str) -> None:
        """Reject frontend-side when the engine did not advertise a capability."""
        node: Any = self.capabilities
        for key in path:
            if not isinstance(node, dict) or key not in node:
                raise EngineClientError(
                    EngineError(
                        code=EngineErrorCode.UNSUPPORTED_CAPABILITY,
                        message="engine did not advertise capability " + ".".join(path),
                    )
                )
            node = node[key]
        if node is False:
            raise EngineClientError(
                EngineError(
                    code=EngineErrorCode.UNSUPPORTED_CAPABILITY,
                    message="engine did not advertise capability " + ".".join(path),
                )
            )

    # --- IO --------------------------------------------------------------------

    def _send(self, message: WireMessage) -> None:
        self._outbound.put(encode_message(message))

    def _io_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._dealer, zmq.POLLIN)
        while self._running:
            try:
                while True:
                    try:
                        data = self._outbound.get_nowait()
                    except queue.Empty:
                        break
                    self._dealer.send(data)
                events = dict(poller.poll(_POLL_INTERVAL_MS))
                if self._dealer in events:
                    self._handle_wire_message(decode_message(self._dealer.recv()))
            except zmq.ZMQError:
                if self._running:
                    logger.error("socket engine client ZMQ error", exc_info=True)
            except Exception:
                logger.error("socket engine client IO error", exc_info=True)

    def _handle_wire_message(self, message: WireMessage) -> None:
        if message.message_type is MessageType.HANDSHAKE:
            handshake_queue = getattr(self, "_handshake_queue", None)
            if handshake_queue is not None:
                handshake_queue.put(message)
            return
        if message.message_type is MessageType.EVENT:
            self._handle_event(message)
            return
        if message.message_type is MessageType.CONTROL_RESPONSE:
            control_id = message.payload.get("control_id")
            future = self._control_futures.pop(control_id, None)
            if future is not None:
                future.put(message)
            return
        if message.message_type is MessageType.ABORT_ACK:
            return  # aborts are fire-and-forget; terminal events close streams
        if message.message_type is MessageType.ERROR:
            self._handle_error_message(message)
            return

    def _handle_event(self, message: WireMessage) -> None:
        request_id = message.request_id
        handle = self._handles.get(request_id)
        if handle is None:
            return  # late event after local completion: dropped per protocol
        event = engine_event_from_payload(request_id, message.payload)
        if event.terminal_kind is TerminalKind.ERROR:
            # Position-independent error terminal: closes the stream
            # regardless of the per-sequence event_index cursor.
            handle._deliver(event)
            return
        checker = self._checkers.get(request_id)
        try:
            if checker is not None:
                checker.observe(event)
        except Exception as violation:
            handle._deliver(violation)
            return
        handle._deliver(event)

    def _handle_error_message(self, message: WireMessage) -> None:
        error = EngineError(
            code=EngineErrorCode(message.payload.get("error_code", "internal_error")),
            message=message.payload.get("error_message", ""),
            request_id=message.request_id,
        )
        if message.request_id is not None:
            handle = self._handles.get(message.request_id)
            if handle is not None:
                handle._deliver(EngineClientError(error))
            return
        if error.code is EngineErrorCode.PROTOCOL_VERSION_MISMATCH:
            handshake_queue = getattr(self, "_handshake_queue", None)
            if handshake_queue is not None:
                handshake_queue.put(EngineClientError(error))
                return
        self._fatal = error

    # --- data plane --------------------------------------------------------------

    def submit(self, request: EngineRequest) -> RequestHandle:
        if self._fatal is not None:
            raise EngineClientError(self._fatal)
        if self.readiness is not ReadinessState.READY:
            raise EngineClientError(
                EngineError(
                    code=EngineErrorCode.ENGINE_UNAVAILABLE,
                    message=f"engine readiness is {self.readiness}",
                    request_id=request.request_id,
                )
            )
        payload = engine_request_to_payload(request)
        expected_terminals = request.sampling.best_of or request.sampling.n
        handle = _SocketRequestHandle(self, request.request_id, expected_terminals)
        with self._lock:
            if request.request_id in self._handles:
                raise EngineClientError(
                    EngineError(
                        code=EngineErrorCode.INVALID_REQUEST,
                        message=f"duplicate request_id {request.request_id!r}",
                        request_id=request.request_id,
                    )
                )
            self._handles[request.request_id] = handle
            self._checkers[request.request_id] = EventOrderingChecker()
        self._send(
            WireMessage(
                message_type=MessageType.SUBMIT,
                request_id=request.request_id,
                payload=payload,
            )
        )
        return handle

    def abort(self, request_id: str) -> None:
        with self._lock:
            known = request_id in self._handles
        if not known:
            raise EngineClientError(
                EngineError(
                    code=EngineErrorCode.UNKNOWN_REQUEST,
                    message=f"unknown request_id {request_id!r}",
                    request_id=request_id,
                )
            )
        self._send(WireMessage(message_type=MessageType.ABORT, request_id=request_id))

    def _forget(self, request_id: str) -> None:
        with self._lock:
            self._handles.pop(request_id, None)
            self._checkers.pop(request_id, None)

    # --- control plane -----------------------------------------------------------

    def _control(self, method: str, reply_timeout: Optional[float] = None, **kwargs: Any) -> Any:
        control_id = next(self._control_ids)
        future: queue.Queue = queue.Queue()
        self._control_futures[control_id] = future
        self._send(
            WireMessage(
                message_type=MessageType.CONTROL_REQUEST,
                payload={"control_id": control_id, "method": method, "kwargs": kwargs},
            )
        )
        try:
            reply = future.get(timeout=reply_timeout or self._control_timeout)
        except queue.Empty:
            self._control_futures.pop(control_id, None)
            raise EngineClientError(
                EngineError(
                    code=EngineErrorCode.ENGINE_UNAVAILABLE,
                    message=f"control call {method!r} timed out; engine unreachable",
                )
            ) from None
        if reply.payload.get("error_code") is not None:
            raise EngineClientError(
                EngineError(
                    code=EngineErrorCode(reply.payload["error_code"]),
                    message=reply.payload.get("error_message", ""),
                )
            )
        return reply.payload.get("result")

    def get_capabilities(self) -> dict[str, Any]:
        return dict(self.capabilities)

    def check_health(self) -> bool:
        try:
            result = self._control("check_health")
        except EngineClientError:
            return False
        if isinstance(result, dict):
            return bool(result.get("healthy"))
        return bool(result)

    def get_stats(self, timeout: float) -> list[dict[str, Any]]:
        return self._control("get_stats", timeout=timeout) or []

    def get_kv_events(self, timeout: float) -> list[dict[str, Any]]:
        return self._control("get_kv_events", timeout=timeout) or []

    def shutdown(self) -> None:
        self._running = False
        if self._io_thread.is_alive() and threading.current_thread() is not self._io_thread:
            self._io_thread.join(timeout=5)
        self._dealer.close(linger=0)


class LocalProcessEngineClient(EngineClient):
    """Both protocol ends co-located in one process over a loopback socket.

    Exercises the full msgpack codec and protocol state machine without a
    process split — the stepping stone between the in-process adapter and a
    remote engine.
    """

    def __init__(self, backend: EngineClient, **server_kwargs: Any) -> None:
        self._server = EngineSocketServer(backend, **server_kwargs)
        self._server.start()
        self._client = SocketEngineClient(self._server.endpoint)

    @property
    def server(self) -> EngineSocketServer:
        return self._server

    @property
    def client(self) -> SocketEngineClient:
        return self._client

    def submit(self, request: EngineRequest) -> RequestHandle:
        return self._client.submit(request)

    def abort(self, request_id: str) -> None:
        self._client.abort(request_id)

    def get_capabilities(self) -> dict[str, Any]:
        return self._client.get_capabilities()

    def check_health(self) -> bool:
        return self._client.check_health()

    def get_stats(self, timeout: float) -> list[dict[str, Any]]:
        return self._client.get_stats(timeout)

    def get_kv_events(self, timeout: float) -> list[dict[str, Any]]:
        return self._client.get_kv_events(timeout)

    def shutdown(self) -> None:
        self._client.shutdown()
        self._server.shutdown()


def local_endpoint() -> str:
    """A unique loopback IPC endpoint for tests."""
    return f"ipc:///tmp/trtllm_engine_api_{uuid.uuid4().hex}.sock"
