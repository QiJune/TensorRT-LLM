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
"""Assembler tests: incremental detok, ordered stops, presentation maps."""

import pytest

from tensorrt_llm.executor.engine_client.assembler import FrontendResponseAssembler
from tensorrt_llm.executor.engine_client.contract import (ErrorFrame,
                                                          FrontendOutputConfig,
                                                          RequestComplete,
                                                          Terminal, TokenDelta)


class CharTokenizer:
    """Deterministic fake: token id N decodes to chr(97 + N % 26)."""

    def decode(self, ids, **kwargs):
        return "".join(chr(97 + i % 26) for i in ids)

    def decode_incrementally(self, token_ids, prev_text=None, states=None, *,
                             flush=False, stream_interval=1, **kwargs):
        prev_text = prev_text or ""
        return prev_text + self.decode(token_ids), states or {}


class HoldingTokenizer(CharTokenizer):
    """Holds the last token pending until flush (exercises terminal flush)."""

    def decode_incrementally(self, token_ids, prev_text=None, states=None, *,
                             flush=False, stream_interval=1, **kwargs):
        prev_text = prev_text or ""
        states = dict(states or {"held": []})
        pending = states["held"] + list(token_ids)
        if flush:
            return prev_text + self.decode(pending), {"held": []}
        if pending:
            states["held"] = pending[-1:]
            return prev_text + self.decode(pending[:-1]), states
        return prev_text, states


def ids_for(text: str) -> list:
    return [ord(ch) - 97 for ch in text]


def make_assembler(stop_strings=(), stop_sequence_reasons=(), include_stop=False,
                   tokenizer=None, abort_calls=None):
    config = FrontendOutputConfig(stop_strings=tuple(stop_strings),
                                  stop_sequence_reasons=tuple(stop_sequence_reasons),
                                  include_stop_str_in_output=include_stop)
    callback = abort_calls.append if abort_calls is not None else None
    return FrontendResponseAssembler("req-1", config,
                                     tokenizer=tokenizer or CharTokenizer(),
                                     abort_callback=callback)


def delta(tokens, event_seq=0, logprobs=None):
    return TokenDelta(request_id="req-1", sequence_id=0,
                      new_token_ids=tuple(tokens), logprobs=logprobs,
                      event_seq=event_seq)


def terminal(finish_reason, stop_reason=None, event_seq=5):
    return Terminal(request_id="req-1", sequence_id=0, finish_reason=finish_reason,
                    stop_reason=stop_reason, event_seq=event_seq)


def complete(status="ok", prompt=3, completion=2, cached=None, event_seq=6):
    return RequestComplete(request_id="req-1", status=status, prompt_tokens=prompt,
                           completion_tokens=completion, cached_tokens=cached,
                           event_seq=event_seq)


class CountingTokenizer(CharTokenizer):
    """Records incremental-decode calls to guard the O(total tokens) bound."""

    def __init__(self):
        self.incremental_calls = []
        self.full_decode_calls = 0

    def decode(self, ids, **kwargs):
        self.full_decode_calls += 1
        return super().decode(ids, **kwargs)

    def decode_incrementally(self, token_ids, prev_text=None, states=None, *,
                             flush=False, stream_interval=1, **kwargs):
        self.incremental_calls.append(list(token_ids))
        prev_text = prev_text or ""
        piece = "".join(chr(97 + i % 26) for i in token_ids)
        return prev_text + piece, states or {}


class TestDetokenizationComplexity:

    def test_non_incremental_tokenizer_rejected_at_setup(self):
        from tensorrt_llm.executor.engine_client.assembler import AssemblerError

        class DecodeOnly:

            def decode(self, ids, **kwargs):
                return ""

        with pytest.raises(AssemblerError):
            make_assembler(tokenizer=DecodeOnly())

    def test_streaming_cost_is_linear_in_tokens(self):
        tokenizer = CountingTokenizer()
        assembler = make_assembler(tokenizer=tokenizer)
        delta_count = 50
        for i in range(delta_count):
            assembler.process_frames([delta(ids_for("ab"), event_seq=i)])
        # One incremental call per delta, each fed ONLY the new ids — never
        # the accumulated sequence, and never a full re-decode.
        assert len(tokenizer.incremental_calls) == delta_count
        assert all(len(call) == 2 for call in tokenizer.incremental_calls)
        assert tokenizer.full_decode_calls == 0


class TestPlainAssembly:

    def test_text_diffs_accumulate(self):
        assembler = make_assembler()
        updates = assembler.process_frames([delta(ids_for("hel"))])
        assert updates[0].text_diff == "hel"
        updates = assembler.process_frames([delta(ids_for("lo"), 1)])
        assert updates[0].text_diff == "lo"
        assert assembler.text == "hello"

    def test_finish_and_usage(self):
        assembler = make_assembler()
        assembler.process_frames([delta(ids_for("hi"))])
        updates = assembler.process_frames(
            [terminal("stop"), complete(cached=2)])
        assert [u.kind for u in updates] == ["finish", "complete"]
        assert updates[0].finish_reason == "stop"
        assert updates[1].usage == {
            "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5,
            "cached_tokens": 2
        }

    def test_logprobs_accumulate(self):
        assembler = make_assembler()
        assembler.process_frames([delta(ids_for("ab"), logprobs=(-0.1, -0.2))])
        assert assembler.logprobs == [-0.1, -0.2]

    def test_terminal_flushes_held_detok_state(self):
        assembler = make_assembler(tokenizer=HoldingTokenizer())
        updates = assembler.process_frames([delta(ids_for("hi"))])
        assert updates[0].text_diff == "h"  # last token held
        updates = assembler.process_frames([terminal("length"), complete()])
        finish = next(u for u in updates if u.kind == "finish")
        deltas = [u for u in updates if u.kind == "delta"]
        assert deltas and deltas[0].text_diff == "i"
        assert finish.finish_reason == "length"
        assert assembler.text == "hi"


class TestStopStrings:

    def test_cross_delta_stop_string_trims_and_aborts(self):
        abort_calls = []
        assembler = make_assembler(stop_strings=("xy", ), abort_calls=abort_calls)
        first = assembler.process_frames([delta(ids_for("ax"))])
        assert first[0].text_diff == "ax"
        second = assembler.process_frames([delta(ids_for("yb"), 1)])
        update = second[0]
        assert update.finish_reason == "stop"
        assert update.stop_reason == "xy"
        # Trimmed before the stop string: only nothing new streams out.
        assert assembler.text == "a"
        assert update.text_diff == ""
        assert abort_calls == ["req-1"]

    def test_include_stop_str_keeps_string(self):
        assembler = make_assembler(stop_strings=("xy", ), include_stop=True)
        assembler.process_frames([delta(ids_for("a"))])
        updates = assembler.process_frames([delta(ids_for("xy"), 1)])
        assert assembler.text == "axy"
        assert updates[0].text_diff == "xy"
        assert updates[0].finish_reason == "stop"

    def test_post_stop_frames_absorbed(self):
        assembler = make_assembler(stop_strings=("x", ))
        assembler.process_frames([delta(ids_for("x"))])
        updates = assembler.process_frames(
            [delta(ids_for("zz"), 1), terminal("abort"), complete("aborted")])
        kinds = [u.kind for u in updates]
        # The extra delta and the abort terminal are absorbed; usage still lands.
        assert kinds == ["complete"]
        assert assembler.finish_reason == "stop"

    def test_config_order_wins_over_position(self):
        assembler = make_assembler(stop_strings=("q", "a"))
        # "a" appears earlier in the text, but "q" is configured first and
        # both are present in the same scan window: config order wins,
        # mirroring the legacy first-match-in-config-order loop.
        updates = assembler.process_frames([delta(ids_for("aq"))])
        assert updates[0].stop_reason == "q"


class TestEngineStopTrim:

    def test_stop_word_tokens_trimmed_in_same_batch(self):
        sequence = tuple(ids_for("xy"))
        assembler = make_assembler(
            stop_sequence_reasons=((sequence, "xy"), ))
        updates = assembler.process_frames([
            delta(ids_for("abxy")),
            terminal("stop", stop_reason="xy"),
            complete(completion=4),
        ])
        by_kind = {u.kind: u for u in updates}
        assert assembler.token_ids == ids_for("ab")
        assert assembler.text == "ab"
        assert by_kind["delta"].text_diff == "ab"
        assert by_kind["finish"].finish_reason == "stop"
        # User-visible usage counts assembled (post-trim) tokens.
        assert by_kind["complete"].usage["completion_tokens"] == 2

    def test_include_stop_str_keeps_engine_stop_tokens(self):
        sequence = tuple(ids_for("xy"))
        assembler = make_assembler(stop_sequence_reasons=((sequence, "xy"), ),
                                   include_stop=True)
        assembler.process_frames([
            delta(ids_for("abxy")),
            terminal("stop", stop_reason="xy"),
        ])
        assert assembler.text == "abxy"


class TestPresentationMap:

    def test_abort_renders_cancelled(self):
        assembler = make_assembler()
        updates = assembler.process_frames([terminal("abort")])
        assert updates[0].finish_reason == "cancelled"

    def test_timeout_renders_timeout(self):
        assembler = make_assembler()
        updates = assembler.process_frames(
            [terminal("error", stop_reason="timeout")])
        assert updates[0].kind == "finish"
        assert updates[0].finish_reason == "timeout"

    def test_other_errors_render_error_update(self):
        assembler = make_assembler()
        updates = assembler.process_frames(
            [terminal("error", stop_reason="not_finished")])
        assert updates[0].kind == "error"
        assert updates[0].error_message == "not_finished"

    def test_error_frame(self):
        assembler = make_assembler()
        updates = assembler.process_frames(
            [ErrorFrame(request_id="req-1", error_code="worker_died", message="x")])
        assert updates[0].kind == "error"
        assert updates[0].error_code == "worker_died"

    def test_failed_complete_without_finish_is_error(self):
        assembler = make_assembler()
        assembler.process_frames([delta(ids_for("a"))])
        updates = assembler.process_frames([complete(status="failed")])
        assert updates[0].kind == "error"
