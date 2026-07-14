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
"""Pass-through stream-shape checker (test utility).

``InvariantCheckingStream`` wraps an async frame iterator and enforces the
contract stream invariants: frames belong to one request; per-sequence
ordering; at most one ``Terminal`` per sequence; no ``TokenDelta`` after
its sequence's ``Terminal``; exactly one request-ending frame
(``RequestComplete`` after all started sequences terminated, or a
standalone ``ErrorFrame`` iff nothing started); nothing after the ending
frame; monotonically increasing ``event_seq``; a stream that ends without
an ending frame is a violation.
"""

from .contract import (ContractError, ErrorFrame, RequestComplete, Terminal,
                       TokenDelta)

__all__ = ["StreamInvariantViolation", "InvariantCheckingStream"]


class StreamInvariantViolation(ContractError):
    """A frame stream violated the contract's stream invariants."""


class InvariantCheckingStream:
    """Async pass-through wrapper enforcing the §2.3 stream invariants."""

    def __init__(self, stream, request_id=None):
        self._stream = stream
        self._request_id = request_id
        self._terminated_sequences = set()
        self._started_sequences = set()
        self._ended = False
        self._last_event_seq = -1
        self._frames_seen = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            frame = await self._stream.__anext__()
        except StopAsyncIteration:
            if not self._ended:
                raise StreamInvariantViolation(
                    f"stream for {self._request_id!r} ended without a "
                    "request-ending frame")
            raise
        self._check(frame)
        return frame

    async def aclose(self):
        close = getattr(self._stream, "aclose", None)
        if close is not None:
            await close()

    def _fail(self, message: str):
        raise StreamInvariantViolation(f"request {self._request_id!r}: {message}")

    def _check(self, frame):
        self._frames_seen += 1
        if self._request_id is None:
            self._request_id = frame.request_id
        if frame.request_id != self._request_id:
            self._fail(f"frame for foreign request {frame.request_id!r}")
        if self._ended:
            self._fail(f"{type(frame).__name__} after the request-ending frame")
        if frame.event_seq <= self._last_event_seq:
            self._fail(f"event_seq {frame.event_seq} not increasing "
                       f"(last {self._last_event_seq})")
        self._last_event_seq = frame.event_seq

        if isinstance(frame, TokenDelta):
            if frame.sequence_id in self._terminated_sequences:
                self._fail(f"TokenDelta after Terminal for sequence "
                           f"{frame.sequence_id}")
            self._started_sequences.add(frame.sequence_id)
        elif isinstance(frame, Terminal):
            if frame.sequence_id in self._terminated_sequences:
                self._fail(f"duplicate Terminal for sequence {frame.sequence_id}")
            self._terminated_sequences.add(frame.sequence_id)
            self._started_sequences.add(frame.sequence_id)
        elif isinstance(frame, RequestComplete):
            unterminated = self._started_sequences - self._terminated_sequences
            if unterminated:
                self._fail(f"RequestComplete before Terminal for sequences "
                           f"{sorted(unterminated)}")
            self._ended = True
        elif isinstance(frame, ErrorFrame):
            if self._started_sequences:
                self._fail("standalone ErrorFrame on a request with started "
                           "sequences (must use Terminal + RequestComplete)")
            self._ended = True
        else:
            self._fail(f"unknown frame type {type(frame).__name__}")
