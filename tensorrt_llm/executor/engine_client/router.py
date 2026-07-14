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
"""Per-request exactly-once terminal state machine over the proxy tap.

The proxy natively pops each request's result on the final response and
drops late/duplicate/unknown-id frames; abort is asynchronous best-effort.
This router builds the opposite guarantees on top of that behavior: every
runtime-started sequence gets exactly one ``Terminal``, every bound request
gets exactly one request-ending frame (``RequestComplete``, or a standalone
``ErrorFrame`` iff nothing runtime-started), nothing is emitted after the
ending frame, and a stream never hangs.

Concurrency specification (normative for this module):

- **State owner.** All binding, tombstone, and per-request lifecycle state
  is owned by ``EngineFrameRouter`` and mutated only under its single
  ``_lock``. Threads involved: the proxy submit thread (``observe_submit``,
  ``on_submit_enqueue_failed``), the proxy dispatch thread
  (``on_response``), consumer event-loop threads (``open_stream`` /
  ``abort`` / ``FrameStream.aclose``), and the proxy shutdown path
  (``fail_all``). Any of these may run concurrently.
- **Lock hierarchy.** ``EngineFrameRouter._lock`` is the outermost and only
  router lock; ``_Delivery`` has its own inner lock and never calls back
  into the router while holding it. Never acquire ``_lock`` while holding a
  delivery lock.
- **Reentrancy.** Frame emission never recursively invokes abort/failure
  paths: runtime aborts triggered by router logic (overflow, early close,
  router-internal failure) are collected as deferred actions under the lock
  and executed after it is released, so a failure while aborting cannot
  corrupt state transitions.
- **Atomic bind→enqueue.** Binding happens on the submit thread after
  client-id assignment and strictly before the request is enqueued
  (``observe_submit`` is called by the proxy between those points);
  ``on_submit_enqueue_failed`` rolls the binding back into a standalone
  ``ErrorFrame`` ending, so a request can never be in-flight without a
  binding or bound without an ending path.
- **Abort-vs-final races.** ``abort`` marks the state under the lock and
  issues the runtime abort outside it. Whatever arrives afterwards — a
  ``CANCELLED`` final, a normal final that won the race, or nothing until
  ``fail_all`` — resolves through the same single transition point
  (``_finish_from_envelope`` / ``_fail_locked``), which is idempotent per
  request: the first ending wins, later signals are absorbed and counted.
- **State lifetimes.** Router execution state is pruned on the ending
  frame; the delivery queue survives until consumed, closed, or expired by
  the client's policy; ended request ids move to a bounded tombstone map
  (``abort`` on a tombstone is a no-op, on an unknown id a typed error).
"""

import threading
from collections import OrderedDict, deque
from typing import Callable, Dict, List, Optional, Tuple

from tensorrt_llm.logger import logger

from .contract import (CACHED_TOKENS_METRIC_KEY, ContractError, ErrorFrame,
                       RequestComplete, Terminal, TokenDelta)
from .envelope import (RuntimeResponseEnvelope, normalize_response,
                       resolve_held_prompt_logprobs)

__all__ = [
    "RouterError",
    "UnknownRequestError",
    "RequestBinding",
    "EngineFrameRouter",
    "DEFAULT_DELIVERY_LIMIT",
    "DEFAULT_TOMBSTONE_LIMIT",
]

DEFAULT_DELIVERY_LIMIT = 8192
DEFAULT_TOMBSTONE_LIMIT = 4096


class RouterError(ContractError):
    """Router-level typed error."""


class UnknownRequestError(RouterError):
    """The request id is neither active nor recently ended."""


class _Delivery:
    """Bounded, thread-safe frame buffer with an out-of-band ending slot.

    Frames buffer from submit time so nothing is lost before the first
    consumer arrives. Ending frames (``Terminal`` / ``RequestComplete`` /
    ``ErrorFrame``) go to a dedicated slot so typed termination never
    depends on space in the bounded data buffer.
    """

    def __init__(self, limit: int = DEFAULT_DELIVERY_LIMIT):
        self._lock = threading.Lock()
        self._buffer = deque()
        self._end_frames: List = []
        self._limit = limit
        self._waiter = None  # (loop, future) of the single blocked consumer
        self._closed = False
        self.overflowed = False
        self.dropped_frames = 0

    def _wake_locked(self):
        if self._waiter is not None:
            loop, future = self._waiter
            self._waiter = None

            def _set():
                if not future.done():
                    future.set_result(None)

            try:
                loop.call_soon_threadsafe(_set)
            except RuntimeError:
                # The consumer's event loop is already closed; the frames
                # stay buffered for a later pop, nothing to wake.
                pass

    def put(self, frame) -> bool:
        """Queue a data frame. Returns False on overflow (frame dropped)."""
        with self._lock:
            if self._closed:
                return False
            if len(self._buffer) >= self._limit:
                self.overflowed = True
                self.dropped_frames += 1
                return False
            self._buffer.append(frame)
            self._wake_locked()
            return True

    def put_ending(self, frames) -> None:
        """Queue ending frames via the out-of-band slot (never blocked)."""
        with self._lock:
            if self._closed:
                return
            self._end_frames.extend(frames)
            self._wake_locked()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._buffer.clear()
            self._wake_locked()

    def pop_nowait(self):
        """Pop one frame if available: data first, then ending frames.

        Returns (frame, True) or (None, False).
        """
        with self._lock:
            if self._buffer:
                return self._buffer.popleft(), True
            if self._end_frames:
                return self._end_frames.pop(0), True
            return None, False

    def register_waiter(self, loop, future) -> bool:
        """Register the single consumer waiter; False if data already ready."""
        with self._lock:
            if self._buffer or self._end_frames or self._closed:
                return False
            self._waiter = (loop, future)
            return True

    @property
    def closed(self) -> bool:
        return self._closed


class RequestBinding:
    """Per-request execution state (owned by the router, guarded by its lock)."""

    __slots__ = ("request_id", "client_id", "prompt_token_ids", "stop_reasons",
                 "delivery", "runtime_started", "terminal_emitted", "ended",
                 "abort_requested", "event_seq", "completion_tokens",
                 "cached_tokens", "prompt_logprobs_sent", "held_prompt_logprobs",
                 "held_raw_prompt_entries",
                 "recent_tokens", "pending_final_metrics", "final_status",
                 "stream_opened", "retire_when_ended")

    def __init__(self, request_id: str, prompt_token_ids: Tuple[int, ...],
                 stop_reasons: Tuple, delivery_limit: int):
        self.request_id = request_id
        self.client_id: Optional[int] = None
        self.prompt_token_ids = tuple(prompt_token_ids)
        # Ordered (stop_token_sequence, user_visible_reason) association;
        # configuration order, first match wins. Validated primitive-only:
        # this crosses into engine-side state and must stay encodable.
        validated = []
        for pair in stop_reasons:
            sequence, reason = pair
            sequence = tuple(sequence)
            if not sequence or not all(
                    isinstance(t, int) and not isinstance(t, bool)
                    for t in sequence):
                raise RouterError(
                    f"stop_reasons: invalid stop sequence {sequence!r}")
            if isinstance(reason, bool) or not isinstance(reason, (int, str)):
                raise RouterError(
                    f"stop_reasons: reason must be int or str, got "
                    f"{type(reason).__name__}")
            validated.append((sequence, reason))
        self.stop_reasons = tuple(validated)
        self.delivery = _Delivery(delivery_limit)
        self.runtime_started = False
        self.terminal_emitted = False
        self.ended = False
        self.abort_requested = False
        self.event_seq = 0
        self.completion_tokens = 0
        self.cached_tokens: Optional[int] = None
        self.prompt_logprobs_sent = False
        self.held_prompt_logprobs: Optional[tuple] = None
        self.held_raw_prompt_entries: Optional[tuple] = None
        tail = max((len(seq) for seq, _ in self.stop_reasons), default=0)
        self.recent_tokens = deque(maxlen=max(tail, 1))
        self.pending_final_metrics: Optional[dict] = None
        self.final_status: Optional[str] = None
        self.stream_opened = False
        # Set when the consumer retires (closes) the delivery before the
        # request has ended: the ending response must complete the
        # retirement so the request id does not stay retained until
        # tombstone eviction.
        self.retire_when_ended = False

    def next_seq(self) -> int:
        seq = self.event_seq
        self.event_seq += 1
        return seq


class EngineFrameRouter:
    """Proxy-attached fork target producing the typed engine frame stream.

    ``abort_fn(client_id)`` issues the runtime abort (typically
    ``executor.abort_request``); it is always invoked outside the router
    lock (deferred-action rule of the concurrency specification).
    """

    def __init__(self,
                 abort_fn: Optional[Callable[[int], None]] = None,
                 delivery_limit: int = DEFAULT_DELIVERY_LIMIT,
                 tombstone_limit: int = DEFAULT_TOMBSTONE_LIMIT):
        self._lock = threading.Lock()
        self._abort_fn = abort_fn
        self._delivery_limit = delivery_limit
        self._tombstone_limit = tombstone_limit
        # Pending registrations keyed by GenerationRequest object identity,
        # matched by observe_submit on the submit thread.
        self._pending: Dict[int, RequestBinding] = {}
        self._by_client: Dict[int, RequestBinding] = {}
        self._by_request: Dict[str, RequestBinding] = {}
        # Delivery retention: a request's buffered frames stay reachable for
        # a (delayed) stream open even after the request ends, until evicted
        # alongside its tombstone.
        self._delivery_index: "OrderedDict[str, RequestBinding]" = OrderedDict()
        self._tombstones: "OrderedDict[str, str]" = OrderedDict()
        # BOUNDED recently-retired proxy client ids, used only for
        # late/duplicate accounting. Safety does not depend on this memory:
        # proxy client ids are never reused within a proxy lifetime (the
        # allocator is monotonic and wrap/collision is rejected at submit),
        # so a very late frame whose id aged out of this structure falls
        # through to the legacy lookup, finds nothing, and is dropped exactly
        # as legacy late frames are dropped today. Frontend request ids
        # become REUSABLE once the prior request ended and its delivery is
        # retired (consumed to the ending frame, explicitly closed, or
        # evicted by the bounded tombstone policy) — per plan AC-4.
        self._retired_client_ids: "OrderedDict[int, bool]" = OrderedDict()
        self._retired_limit = tombstone_limit
        self._fatal_error: Optional[str] = None
        # Observability counters (read by the serving layer).
        self.counters = {
            "late_or_duplicate_absorbed": 0,
            "router_failures": 0,
            "synthesized_terminals": 0,
            "overflow_aborts": 0,
        }

    # ------------------------------------------------------------------ #
    # Registration (client side, submit thread)
    # ------------------------------------------------------------------ #

    def register_pending(self, generation_request, request_id: str,
                         prompt_token_ids, stop_reasons) -> RequestBinding:
        """Register a contract request about to be submitted to the proxy.

        Called by the client immediately before ``proxy.submit``; the proxy's
        ``observe_submit`` hook completes the binding with the assigned
        client id before the request is enqueued (same thread, race-free).
        """
        binding = RequestBinding(request_id, prompt_token_ids, stop_reasons,
                                 self._delivery_limit)
        with self._lock:
            self._check_request_id_free_locked(request_id)
            self._pending[id(generation_request)] = binding
            self._by_request[request_id] = binding
            self._delivery_index[request_id] = binding
        return binding

    def _check_request_id_free_locked(self, request_id: str) -> None:
        """Reject a duplicate id while the previous use is still live.

        A frontend request id is reusable once its prior request ended AND
        its delivery is retired; it is a typed duplicate only while the
        previous binding is active or its delivery is still retained.
        """
        if request_id in self._by_request:
            raise RouterError(f"duplicate request_id {request_id!r} "
                              "(previous request still active)")
        if request_id in self._delivery_index:
            raise RouterError(f"duplicate request_id {request_id!r} "
                              "(previous delivery not yet retired)")

    def bind(self, request_id: str, client_id: int, prompt_token_ids,
             stop_reasons) -> RequestBinding:
        """Directly bind a contract request to its allocated client id.

        The contract-native submit path (no pending/object-identity interval):
        the caller allocates the client id and enqueues only after this
        returns, so a worker response can never beat the binding.
        """
        binding = RequestBinding(request_id, prompt_token_ids, stop_reasons,
                                 self._delivery_limit)
        binding.client_id = client_id
        with self._lock:
            self._check_request_id_free_locked(request_id)
            if client_id in self._by_client or client_id in self._retired_client_ids:
                raise RouterError(f"client id {client_id} already contract-owned")
            self._by_request[request_id] = binding
            self._by_client[client_id] = binding
            self._delivery_index[request_id] = binding
        return binding

    def observe_submit(self, generation_request) -> None:
        """Proxy hook: bind the assigned client id (before enqueue)."""
        with self._lock:
            binding = self._pending.pop(id(generation_request), None)
            if binding is None:
                return  # legacy (non-contract) submission
            binding.client_id = generation_request.id
            self._by_client[generation_request.id] = binding

    def on_submit_enqueue_failed(self, client_id: int) -> None:
        """Proxy hook: the request was bound but never reached the worker."""
        with self._lock:
            binding = self._by_client.get(client_id)
            if binding is None or binding.ended:
                return
            # Nothing runtime-started: standalone ErrorFrame ending.
            self._end_locked(binding, [
                ErrorFrame(request_id=binding.request_id, error_code="enqueue_failed",
                           message="request could not be enqueued to the engine",
                           event_seq=binding.next_seq())
            ])

    def discard_pending(self, generation_request) -> None:
        """Roll back a registration whose proxy.submit never ran."""
        with self._lock:
            binding = self._pending.pop(id(generation_request), None)
            if binding is not None:
                self._by_request.pop(binding.request_id, None)
                self._delivery_index.pop(binding.request_id, None)
                binding.delivery.close()

    # ------------------------------------------------------------------ #
    # Response path (proxy dispatch thread)
    # ------------------------------------------------------------------ #

    def route_response(self, raw) -> bool:
        """Exclusively claim and consume a contract-owned response.

        Returns True when the client id belongs to the contract population
        (the response was consumed or deliberately absorbed) — the caller
        must then skip all legacy delivery. False means never contract-owned.
        """
        client_id = getattr(raw, "client_id", None)
        with self._lock:
            if (client_id not in self._by_client
                    and client_id not in self._retired_client_ids):
                # Never (or no longer) contract-tracked. Falling through is
                # safe: proxy ids are never reused, so the legacy lookup
                # finds nothing and drops the frame like any legacy late
                # frame.
                return False
        self.on_response(raw)
        return True

    def on_response(self, raw) -> None:
        """Proxy hook: observe one raw dispatch item (never raises)."""
        deferred: List[Callable[[], None]] = []
        try:
            client_id = getattr(raw, "client_id", None)
            with self._lock:
                binding = self._by_client.get(client_id)
                if binding is None:
                    if client_id in self._retired_client_ids:
                        # Recently retired contract id: absorb and count.
                        self.counters["late_or_duplicate_absorbed"] += 1
                    return  # unbound legacy traffic
                if binding.ended:
                    self.counters["late_or_duplicate_absorbed"] += 1
                    return
            try:
                envelope = normalize_response(raw, binding.prompt_token_ids)
            except Exception as e:
                self.counters["router_failures"] += 1
                self._fail_request(binding, "router_error",
                                   f"response normalization failed: {e}", deferred)
                logger.error(f"engine-frame router: normalization failure for "
                             f"client_id={client_id}: {e!r}")
                return
            with self._lock:
                if binding.ended:
                    self.counters["late_or_duplicate_absorbed"] += 1
                    return
                self._process_envelope_locked(binding, envelope)
        except Exception as e:  # never propagate into the dispatch thread
            self.counters["router_failures"] += 1
            logger.error(f"engine-frame router: internal error: {e!r}")
            try:
                if 'binding' in locals() and binding is not None:
                    self._fail_request(binding, "router_error", str(e), deferred)
            except Exception:
                pass
        finally:
            for action in deferred:
                try:
                    action()
                except Exception as e:
                    logger.error(f"engine-frame router: deferred action failed: {e!r}")

    def _process_envelope_locked(self, binding: RequestBinding,
                                 envelope: RuntimeResponseEnvelope) -> None:
        if envelope.error_msg is not None:
            self._end_with_error_locked(binding, "request_error", envelope.error_msg)
            return
        if envelope.sequence_index != 0:
            # V0 lifecycle state is sequence-0 only (n=1); a multi-sequence
            # response must fail the request rather than mix sequence state.
            self._end_with_error_locked(
                binding, "router_error",
                f"sequence_index {envelope.sequence_index} unsupported in V0")
            return

        binding.runtime_started = True

        if not envelope.new_token_ids and not binding.prompt_logprobs_sent:
            # Hold for the next token-carrying delta (empty TokenDeltas do
            # not exist by construction). Map-shaped entries arrive raw: the
            # first generated token supplies the last lookup key.
            if envelope.prompt_logprobs is not None:
                binding.held_prompt_logprobs = envelope.prompt_logprobs
            elif envelope.raw_prompt_logprob_entries is not None:
                binding.held_raw_prompt_entries = envelope.raw_prompt_logprob_entries
        # Prompt logprobs re-attached AFTER delivery (the worker caches and
        # re-sends them, including on a terminal-only final) are dropped
        # quietly: they were already emitted exactly once.

        if envelope.new_token_ids:
            prompt_logprobs = None
            if not binding.prompt_logprobs_sent:
                prompt_logprobs = envelope.prompt_logprobs or binding.held_prompt_logprobs
                if prompt_logprobs is None and binding.held_raw_prompt_entries is not None:
                    try:
                        prompt_logprobs = resolve_held_prompt_logprobs(
                            binding.held_raw_prompt_entries,
                            binding.prompt_token_ids,
                            envelope.new_token_ids[0], binding.client_id)
                    except Exception as e:
                        self._end_with_error_locked(
                            binding, "router_error",
                            f"held prompt-logprob resolution failed: {e}")
                        return
                if prompt_logprobs is not None:
                    binding.prompt_logprobs_sent = True
                    binding.held_prompt_logprobs = None
                    binding.held_raw_prompt_entries = None
            metrics = dict(envelope.metrics) if envelope.metrics else {}
            if envelope.cached_tokens is not None:
                metrics[CACHED_TOKENS_METRIC_KEY] = float(envelope.cached_tokens)
                binding.cached_tokens = envelope.cached_tokens
            binding.completion_tokens += len(envelope.new_token_ids)
            binding.recent_tokens.extend(envelope.new_token_ids)
            delta = TokenDelta(request_id=binding.request_id,
                               sequence_id=envelope.sequence_index,
                               new_token_ids=envelope.new_token_ids,
                               logprobs=envelope.logprobs,
                               prompt_logprobs=prompt_logprobs,
                               metrics=metrics or None,
                               event_seq=binding.next_seq())
            if not binding.delivery.put(delta):
                if binding.delivery.closed:
                    # The consumer closed the stream (abort already issued by
                    # the close path); absorb the delta — but fall through so
                    # a token-carrying FINAL still ends (and, with retirement
                    # pending, retires) the binding instead of leaking it.
                    self.counters["late_or_duplicate_absorbed"] += 1
                else:
                    self._overflow_locked(binding)
                    return
        else:
            if envelope.is_final and envelope.metrics:
                binding.pending_final_metrics = dict(envelope.metrics)
            if envelope.cached_tokens is not None and binding.cached_tokens is None:
                # A zero-token request may report cached tokens only on its
                # final response (contract divergence note 5). When a delta
                # already recorded a value, leave it for the finish-time
                # consistency check instead of overwriting.
                binding.cached_tokens = envelope.cached_tokens

        if envelope.is_final:
            if (binding.held_prompt_logprobs is not None
                    or binding.held_raw_prompt_entries is not None):
                logger.warning(
                    f"engine-frame router: dropping prompt logprobs for "
                    f"request {binding.request_id!r}: the request ended before a "
                    f"token-carrying delta arrived")
                binding.held_prompt_logprobs = None
                binding.held_raw_prompt_entries = None
            self._finish_from_envelope_locked(binding, envelope)

    def _resolve_stop_reason(self, binding: RequestBinding):
        tail = tuple(binding.recent_tokens)
        for sequence, reason in binding.stop_reasons:
            if len(sequence) <= len(tail) and tail[-len(sequence):] == sequence:
                return reason
        return None

    def _finish_from_envelope_locked(self, binding: RequestBinding,
                                     envelope: RuntimeResponseEnvelope) -> None:
        name = envelope.finish_reason_name
        stop_reason = None
        if name == "END_ID":
            finish_reason = "stop"
        elif name == "STOP_WORDS":
            finish_reason = "stop"
            stop_reason = self._resolve_stop_reason(binding)
        elif name == "LENGTH":
            finish_reason = "length"
        elif name == "CANCELLED":
            finish_reason = "abort"
        elif name == "TIMED_OUT":
            finish_reason = "error"
            stop_reason = "timeout"
        elif name is None or name == "NOT_FINISHED":
            if binding.abort_requested:
                finish_reason = "abort"
            else:
                finish_reason = "error"
                stop_reason = "not_finished"
        else:
            finish_reason = "error"
            stop_reason = name
        status = {"stop": "ok", "length": "ok", "abort": "aborted",
                  "error": "failed"}[finish_reason]
        if (envelope.cached_tokens is not None
                and binding.cached_tokens is not None
                and envelope.cached_tokens != binding.cached_tokens):
            # Consistency rule (divergence note 5): the final value must
            # equal the last per-delta value — a mismatch is a typed failure,
            # not a warning.
            self._end_with_error_locked(
                binding, "cached_tokens_mismatch",
                f"final cached_tokens {envelope.cached_tokens} != last delta "
                f"{binding.cached_tokens}")
            return
        if envelope.cached_tokens is not None:
            binding.cached_tokens = envelope.cached_tokens
        frames = []
        if not binding.terminal_emitted:
            binding.terminal_emitted = True
            frames.append(Terminal(request_id=binding.request_id,
                                   sequence_id=envelope.sequence_index,
                                   finish_reason=finish_reason,
                                   stop_reason=stop_reason,
                                   event_seq=binding.next_seq()))
        metrics = binding.pending_final_metrics
        if envelope.metrics and not envelope.new_token_ids:
            metrics = dict(envelope.metrics)
        frames.append(
            RequestComplete(request_id=binding.request_id, status=status,
                            prompt_tokens=len(binding.prompt_token_ids),
                            completion_tokens=binding.completion_tokens,
                            cached_tokens=binding.cached_tokens,
                            metrics=metrics,
                            event_seq=binding.next_seq()))
        self._end_locked(binding, frames)

    def _end_with_error_locked(self, binding: RequestBinding, error_code: str,
                               message: str) -> None:
        frames = []
        if binding.runtime_started:
            if not binding.terminal_emitted:
                binding.terminal_emitted = True
                self.counters["synthesized_terminals"] += 1
                frames.append(Terminal(request_id=binding.request_id, sequence_id=0,
                                       finish_reason="error", stop_reason=None,
                                       event_seq=binding.next_seq()))
            frames.append(
                RequestComplete(request_id=binding.request_id, status="failed",
                                prompt_tokens=len(binding.prompt_token_ids),
                                completion_tokens=binding.completion_tokens,
                                cached_tokens=binding.cached_tokens,
                                metrics=binding.pending_final_metrics,
                                event_seq=binding.next_seq()))
        else:
            frames.append(ErrorFrame(request_id=binding.request_id,
                                     error_code=error_code, message=message,
                                     event_seq=binding.next_seq()))
        self._end_locked(binding, frames)

    def _end_locked(self, binding: RequestBinding, frames) -> None:
        binding.ended = True
        binding.delivery.put_ending(frames)
        # The client-id mapping is retained (until delivery retirement) so
        # late/duplicate frames for an ended request are recognized and
        # absorbed rather than mistaken for unbound legacy traffic.
        self._by_request.pop(binding.request_id, None)
        self._tombstones[binding.request_id] = binding.final_status or "ended"
        while len(self._tombstones) > self._tombstone_limit:
            evicted_id, _ = self._tombstones.popitem(last=False)
            evicted = self._delivery_index.pop(evicted_id, None)
            if evicted is not None:
                self._retire_binding_locked(evicted)
        if binding.retire_when_ended:
            # The consumer already closed the stream; complete the deferred
            # retirement now that the ending response has arrived.
            self._retire_binding_locked(binding)

    def _retire_binding_locked(self, binding: RequestBinding) -> None:
        """Retire a binding: close its delivery, free its request id for
        reuse, and move its client id to the bounded recently-retired set."""
        binding.delivery.close()
        self._delivery_index.pop(binding.request_id, None)
        if binding.client_id is not None:
            self._by_client.pop(binding.client_id, None)
            self._retired_client_ids[binding.client_id] = True
            while len(self._retired_client_ids) > self._retired_limit:
                self._retired_client_ids.popitem(last=False)

    def retire_delivery(self, request_id: str) -> None:
        """Client hook: the stream was consumed to its ending frame or
        explicitly closed — retire the binding so the request id becomes
        reusable and per-request state returns to its bounds.

        On a close BEFORE the request ended (early ``aclose``), retirement
        is recorded and deferred to the ending response: the client id must
        stay bound until then so the in-flight cancellation final is
        recognized and absorbed rather than treated as unbound traffic.
        """
        with self._lock:
            binding = self._delivery_index.get(request_id)
            if binding is None:
                return
            if not binding.ended:
                binding.retire_when_ended = True
                return
            self._retire_binding_locked(binding)

    def _overflow_locked(self, binding: RequestBinding) -> None:
        self.counters["overflow_aborts"] += 1
        binding.abort_requested = True
        # Synthesize the typed ending via the out-of-band slot; the runtime
        # abort is issued by the client-facing abort path (slow consumer).
        if not binding.terminal_emitted:
            binding.terminal_emitted = True
            self.counters["synthesized_terminals"] += 1
        self._end_locked(binding, [
            Terminal(request_id=binding.request_id, sequence_id=0,
                     finish_reason="error", stop_reason="delivery_overflow",
                     event_seq=binding.next_seq()),
            RequestComplete(request_id=binding.request_id, status="failed",
                            prompt_tokens=len(binding.prompt_token_ids),
                            completion_tokens=binding.completion_tokens,
                            cached_tokens=binding.cached_tokens,
                            metrics=None,
                            event_seq=binding.next_seq()),
        ])
        client_id = binding.client_id
        if self._abort_fn is not None and client_id is not None:
            # Deferred-action rule: schedule outside the lock via a thread —
            # on_response's finally block cannot see this, so use a plain
            # daemon thread to stay non-reentrant.
            threading.Thread(target=self._safe_abort, args=(client_id, ),
                             daemon=True).start()

    def _safe_abort(self, client_id: int) -> None:
        try:
            self._abort_fn(client_id)
        except Exception as e:
            logger.error(f"engine-frame router: runtime abort failed for "
                         f"client_id={client_id}: {e!r}")

    def _fail_request(self, binding: RequestBinding, error_code: str, message: str,
                      deferred: List[Callable[[], None]]) -> None:
        with self._lock:
            if binding.ended:
                return
            self._end_with_error_locked(binding, error_code, message)
            client_id = binding.client_id
        if self._abort_fn is not None and client_id is not None:
            deferred.append(lambda: self._safe_abort(client_id))

    # ------------------------------------------------------------------ #
    # Client-facing operations
    # ------------------------------------------------------------------ #

    def abort(self, request_id: str) -> None:
        """Request abort: idempotent on ended requests, typed error on unknown."""
        with self._lock:
            binding = self._by_request.get(request_id)
            if binding is None:
                if request_id in self._tombstones:
                    return  # no-op on a completed/ended request
                raise UnknownRequestError(f"unknown request_id {request_id!r}")
            if binding.ended or binding.abort_requested:
                return
            binding.abort_requested = True
            client_id = binding.client_id
        if self._abort_fn is not None and client_id is not None:
            self._safe_abort(client_id)

    def get_binding(self, request_id: str) -> Optional[RequestBinding]:
        with self._lock:
            return self._by_request.get(request_id)

    def open_stream_binding(self, request_id: str) -> RequestBinding:
        """Claim the single consumer stream for a request (typed errors)."""
        with self._lock:
            binding = self._delivery_index.get(request_id)
            if binding is None:
                raise UnknownRequestError(f"unknown request_id {request_id!r}")
            if binding.stream_opened:
                raise RouterError(f"stream for {request_id!r} was already opened")
            binding.stream_opened = True
            return binding

    def is_ended(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._tombstones

    @property
    def fatal_error(self) -> Optional[str]:
        return self._fatal_error

    def fail_all(self, reason: str) -> None:
        """Latch a fatal condition and end every in-flight request typed.

        Wired from the proxy's shutdown/fatal-error path. Non-consuming: it
        reads no queues; it only transitions router state.
        """
        self.fail_all_and_collect(reason)

    def fail_all_and_collect(self, reason: str) -> List[int]:
        """``fail_all`` that also snapshots the active contract client ids.

        The snapshot is taken before the bindings are pruned, so the caller
        can issue best-effort runtime cancellation for every request that
        was still in flight. Idempotent: already-ended bindings contribute
        neither frames nor ids.
        """
        with self._lock:
            self._fatal_error = self._fatal_error or str(reason)
            active_client_ids = []
            bindings = list(self._by_request.values())
            for binding in bindings:
                if not binding.ended:
                    if binding.client_id is not None:
                        active_client_ids.append(binding.client_id)
                    self._end_with_error_locked(binding, "executor_failed",
                                                str(reason))
            for binding in self._pending.values():
                if not binding.ended:
                    self._end_with_error_locked(binding, "executor_failed",
                                                str(reason))
            self._pending.clear()
            return active_client_ids

    def active_request_count(self) -> int:
        with self._lock:
            # Pending bindings are already indexed by request id; count each
            # non-ended binding exactly once.
            return len(self._by_request)
