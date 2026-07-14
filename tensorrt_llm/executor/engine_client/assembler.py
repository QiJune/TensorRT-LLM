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
"""Frontend response assembly: detokenization, stop handling, presentation.

``FrontendResponseAssembler`` consumes the typed frame stream for one
request and produces presentation-ready updates. All formatting lives here,
on the frontend — no callable ever crosses to the engine.

Semantics (legacy-parity by design; the replay oracle is the gate):

- **Incremental detokenization** uses the tokenizer's incremental decoder
  and is O(total tokens) unconditionally: tokenizers without incremental
  support are rejected at setup (never a per-delta full re-decode). The
  single full re-decode after an engine stop-word trim runs at most once
  per request.
- **Stop-string detection** mirrors the legacy algorithm: the ordered stop
  reasons are checked in configuration order and the first match wins; the
  cumulative text is trimmed at the match (honoring
  ``include_stop_str_in_output``), the user-visible finish reason becomes
  ``"stop"`` with the string as ``stop_reason``, and the engine abort is
  issued as an **internal control action** — the user-visible finish reason
  never becomes ``"cancelled"`` for a stop hit, and post-stop frames are
  absorbed. The scan uses a bounded tail window, which yields the same
  result as the legacy full-text scan because every prior window was
  already checked on earlier deltas.
- **Engine stop-word trims**: a ``Terminal(stop)`` whose ``stop_reason``
  matches a configured stop sequence trims the sequence's tokens (and
  re-derives the text) when ``include_stop_str_in_output`` is false —
  mirroring the legacy token-level trim. This requires the final
  ``TokenDelta`` and its ``Terminal`` to be processed in one
  ``process_frames`` batch (the router enqueues them together).
- **Presentation map** (contract draft DEC-9): wire ``Terminal(error,
  stop_reason="timeout")`` renders as user-visible ``finish_reason=
  "timeout"``; wire ``abort`` renders as ``"cancelled"``.
- **Usage**: the wire-level ``RequestComplete.completion_tokens`` stays the
  raw adapter-observed engine count (contract draft DEC-2); the
  **user-visible** ``completion_tokens`` reports the assembled (post-trim)
  token count, matching the legacy accounting that derives usage from the
  trimmed output. This trim-vs-raw split is the documented decision: raw on
  the wire, assembled at presentation.
"""

import dataclasses
from typing import List, Optional, Sequence, Union

from tensorrt_llm.logger import logger

from .contract import (CACHED_TOKENS_METRIC_KEY, ContractError, ErrorFrame,
                       FrontendOutputConfig, RequestComplete, Terminal,
                       TokenDelta)

__all__ = ["AssemblyUpdate", "FrontendResponseAssembler", "AssemblerError"]


class AssemblerError(ContractError):
    """The assembler received frames it cannot present."""


@dataclasses.dataclass
class AssemblyUpdate:
    """One presentation-ready update produced from a frame batch.

    ``kind`` is one of ``"delta"`` (new text/tokens), ``"finish"``
    (user-visible finish reason resolved), ``"complete"`` (usage available),
    ``"error"`` (typed failure).
    """

    kind: str
    text_diff: str = ""
    new_token_ids: tuple = ()
    logprobs: Optional[tuple] = None
    prompt_logprobs: Optional[tuple] = None
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None
    usage: Optional[dict] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    metrics_cached_tokens: Optional[int] = None


class FrontendResponseAssembler:
    """Assembles one request's frame stream into presentation updates."""

    def __init__(self,
                 request_id: str,
                 output_config: FrontendOutputConfig,
                 tokenizer=None,
                 abort_callback=None,
                 stream_interval: int = 1):
        self.request_id = request_id
        self.config = output_config
        if (output_config.detokenize and tokenizer is not None
                and not hasattr(tokenizer, "decode_incrementally")):
            # O(total tokens) is a hard requirement: a tokenizer without
            # incremental decoding would force per-delta full re-decodes
            # (O(n^2)), so it is rejected at setup rather than degraded.
            raise AssemblerError(
                f"tokenizer {type(tokenizer).__name__} does not support "
                "incremental decoding (decode_incrementally); unsupported for "
                "streaming assembly")
        self.tokenizer = tokenizer
        self._abort_callback = abort_callback
        self._stream_interval = stream_interval

        self.token_ids: List[int] = []
        self.text = ""
        self.logprobs: List[float] = []
        self.prompt_logprobs: Optional[tuple] = None
        self._incremental_states = None
        self._flushed_text_len = 0

        self.finish_reason: Optional[str] = None
        self.stop_reason: Union[int, str, None] = None
        self.usage: Optional[dict] = None
        self._stopped_by_string = False
        self._finish_presented = False
        self._completed = False
        max_stop = max((len(s) for s in output_config.stop_strings), default=0)
        self._stop_window = max_stop

    # ------------------------------------------------------------------ #

    def process_frames(self, frames: Sequence) -> List[AssemblyUpdate]:
        """Process one batch of frames atomically.

        Trims implied by a ``Terminal`` in the batch apply before any text
        from the same batch is surfaced, mirroring the legacy behavior where
        final tokens and finish state arrive as one response.
        """
        updates: List[AssemblyUpdate] = []
        for position, frame in enumerate(frames):
            if isinstance(frame, TokenDelta):
                # An engine stop-word finish takes precedence over frontend
                # stop-string scanning for ITS OWN final delta — mirroring
                # legacy ordering, where the final response trims stop tokens
                # before the text scan ever sees them. Earlier deltas in the
                # batch still scan normally.
                next_frame = frames[position + 1] if position + 1 < len(frames) else None
                engine_stops_this_delta = (isinstance(next_frame, Terminal)
                                           and next_frame.finish_reason == "stop")
                update = self._process_delta(frame,
                                             scan_stops=not engine_stops_this_delta)
                if update is not None:
                    updates.append(update)
            elif isinstance(frame, Terminal):
                update = self._process_terminal(frame, updates)
                if update is not None:
                    updates.append(update)
            elif isinstance(frame, RequestComplete):
                updates.append(self._process_complete(frame))
            elif isinstance(frame, ErrorFrame):
                updates.append(
                    AssemblyUpdate(kind="error", error_code=frame.error_code,
                                   error_message=frame.message))
                self._completed = True
            else:
                raise AssemblerError(f"unknown frame type {type(frame).__name__}")
        return updates

    # ------------------------------------------------------------------ #

    def _decode_new_tokens(self, new_token_ids, flush: bool) -> str:
        if not self.config.detokenize or self.tokenizer is None:
            return ""
        kwargs = dict(
            skip_special_tokens=self.config.skip_special_tokens,
            spaces_between_special_tokens=self.config.spaces_between_special_tokens)
        previous_text = self.text
        self.text, self._incremental_states = self.tokenizer.decode_incrementally(
            list(new_token_ids),
            prev_text=previous_text,
            states=self._incremental_states,
            flush=flush,
            stream_interval=self._stream_interval,
            **kwargs)
        return self.text[len(previous_text):]

    def _scan_stop_strings(self) -> Optional[str]:
        if not self.config.stop_strings:
            return None
        search_start = max(0, self._flushed_text_len - self._stop_window + 1)
        window = self.text[search_start:]
        for _, reason in self.config.stop_sequence_reasons:
            if isinstance(reason, str) and reason in window:
                return reason
        for reason in self.config.stop_strings:
            if reason in window:
                return reason
        return None

    def _process_delta(self, frame: TokenDelta,
                       scan_stops: bool = True) -> Optional[AssemblyUpdate]:
        if self._stopped_by_string or self._finish_presented:
            return None  # post-stop frames are absorbed
        self.token_ids.extend(frame.new_token_ids)
        if frame.logprobs is not None:
            self.logprobs.extend(frame.logprobs)
        if frame.prompt_logprobs is not None:
            self.prompt_logprobs = frame.prompt_logprobs
        text_diff = self._decode_new_tokens(frame.new_token_ids, flush=False)

        stop_hit = self._scan_stop_strings() if scan_stops else None
        finish_reason = None
        stop_reason = None
        if stop_hit is not None:
            stop_position = self.text.find(
                stop_hit, max(0, self._flushed_text_len - self._stop_window + 1))
            if stop_position < 0:
                stop_position = self.text.find(stop_hit)
            if self.config.include_stop_str_in_output:
                self.text = self.text[:stop_position + len(stop_hit)]
            else:
                self.text = self.text[:stop_position]
            text_diff = self.text[self._flushed_text_len:]
            self._stopped_by_string = True
            self._finish_presented = True
            self.finish_reason = finish_reason = "stop"
            self.stop_reason = stop_reason = stop_hit
            if self._abort_callback is not None:
                try:
                    # Internal control action: the user-visible finish reason
                    # stays "stop"; the abort only stops background generation.
                    self._abort_callback(self.request_id)
                except Exception as e:
                    logger.warning(f"assembler: engine abort after stop-string hit "
                                   f"failed for {self.request_id!r}: {e!r}")
        cached = None
        if frame.metrics is not None:
            raw_cached = frame.metrics.get(CACHED_TOKENS_METRIC_KEY)
            if raw_cached is not None:
                cached = int(raw_cached)
        self._flushed_text_len += len(text_diff)
        return AssemblyUpdate(kind="delta", text_diff=text_diff,
                              new_token_ids=frame.new_token_ids,
                              logprobs=frame.logprobs,
                              prompt_logprobs=frame.prompt_logprobs,
                              finish_reason=finish_reason, stop_reason=stop_reason,
                              metrics_cached_tokens=cached)

    def _trim_engine_stop_tokens(self, stop_reason,
                                 updates: List[AssemblyUpdate]) -> None:
        """Token-level trim for an engine stop-word finish (legacy parity)."""
        if self.config.include_stop_str_in_output:
            return
        matched_sequence = None
        for sequence, reason in self.config.stop_sequence_reasons:
            if reason == stop_reason:
                matched_sequence = sequence
                break
        if matched_sequence is None:
            return
        n = len(matched_sequence)
        if n == 0 or tuple(self.token_ids[-n:]) != tuple(matched_sequence):
            return
        del self.token_ids[-n:]
        if self.logprobs:
            del self.logprobs[-n:]
        if self.config.detokenize and self.tokenizer is not None:
            # One full re-decode at end-of-request: O(total tokens), once.
            kwargs = dict(
                skip_special_tokens=self.config.skip_special_tokens,
                spaces_between_special_tokens=self.config.
                spaces_between_special_tokens)
            self.text = self.tokenizer.decode(self.token_ids, **kwargs)
        # Adjust the batch's delta updates: the stop sequence may span
        # several buffered deltas, so trim tokens walking backwards, and
        # rebase the batch's text so nothing past the trim is presented.
        delta_updates = [u for u in updates if u.kind == "delta"]
        remaining = n
        for update in reversed(delta_updates):
            if remaining == 0:
                break
            take = min(remaining, len(update.new_token_ids))
            if take:
                keep = len(update.new_token_ids) - take
                update.new_token_ids = tuple(update.new_token_ids[:keep])
                if update.logprobs:
                    update.logprobs = tuple(update.logprobs[:keep])
                remaining -= take
        if delta_updates:
            batch_start = self._flushed_text_len - sum(
                len(u.text_diff) for u in delta_updates)
            # The batch renders together; concentrate the trimmed tail on the
            # first delta and blank the rest so concatenation is exact.
            delta_updates[0].text_diff = self.text[batch_start:]
            for update in delta_updates[1:]:
                update.text_diff = ""
            self._flushed_text_len = batch_start + len(delta_updates[0].text_diff)

    def _process_terminal(self, frame: Terminal,
                          updates: List[AssemblyUpdate]) -> Optional[AssemblyUpdate]:
        if self._finish_presented:
            return None  # stop-string hit already presented the finish
        # Flush any pending detokenization state.
        if self.config.detokenize and self.tokenizer is not None and hasattr(
                self.tokenizer, "decode_incrementally"):
            try:
                text_diff = self._decode_new_tokens((), flush=True)
            except Exception:
                text_diff = ""
            if text_diff:
                self._flushed_text_len += len(text_diff)
                for update in reversed(updates):
                    if update.kind == "delta":
                        update.text_diff += text_diff
                        break
                else:
                    updates.append(AssemblyUpdate(kind="delta", text_diff=text_diff))

        if frame.finish_reason == "stop":
            self._trim_engine_stop_tokens(frame.stop_reason, updates)
            finish_reason = "stop"
        elif frame.finish_reason == "length":
            finish_reason = "length"
        elif frame.finish_reason == "abort":
            finish_reason = "cancelled"
        elif frame.finish_reason == "error" and frame.stop_reason == "timeout":
            finish_reason = "timeout"
        else:
            self._finish_presented = True
            self.finish_reason = "error"
            return AssemblyUpdate(kind="error", error_code="engine_error",
                                  error_message=str(frame.stop_reason or "error"))
        self._finish_presented = True
        self.finish_reason = finish_reason
        # Only a real stop carries a user-visible stop_reason; the wire-level
        # "timeout" marker is a rendering input, not presented (legacy parity).
        presented_stop_reason = frame.stop_reason if finish_reason == "stop" else None
        self.stop_reason = presented_stop_reason
        return AssemblyUpdate(kind="finish", finish_reason=finish_reason,
                              stop_reason=presented_stop_reason)

    def _process_complete(self, frame: RequestComplete) -> AssemblyUpdate:
        self._completed = True
        # User-visible usage counts the assembled (post-trim) tokens for
        # legacy parity; the wire frame keeps the raw engine count. See the
        # module docstring for the documented trim-vs-raw decision.
        completion_tokens = len(self.token_ids)
        self.usage = {
            "prompt_tokens": frame.prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": frame.prompt_tokens + completion_tokens,
        }
        if frame.cached_tokens is not None:
            self.usage["cached_tokens"] = frame.cached_tokens
        if frame.status == "failed" and not self._finish_presented:
            self._finish_presented = True
            return AssemblyUpdate(kind="error", error_code="request_failed",
                                  error_message="request failed", usage=self.usage)
        return AssemblyUpdate(kind="complete", usage=self.usage,
                              finish_reason=self.finish_reason,
                              stop_reason=self.stop_reason)

    @property
    def completed(self) -> bool:
        return self._completed
