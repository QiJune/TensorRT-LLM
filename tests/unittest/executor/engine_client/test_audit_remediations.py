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
"""Regression tests for the final conformance-audit remediations."""

from types import SimpleNamespace

import msgpack
import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor.engine_client.assembler import FrontendResponseAssembler
from tensorrt_llm.executor.engine_client.codec import EncodeError, encode
from tensorrt_llm.executor.engine_client.contract import (FrontendOutputConfig,
                                                          RequestComplete,
                                                          Terminal, TokenDelta)
from tensorrt_llm.executor.engine_client.envelope import (EnvelopeError,
                                                          normalize_response)
from tensorrt_llm.executor.engine_client.router import (EngineFrameRouter,
                                                        RouterError)
from tensorrt_llm.executor.result import ResponseWrapper


def make_result(**overrides):
    fields = dict(is_final=False, output_token_ids=[[5]],
                  finish_reasons=[tllm.FinishReason.NOT_FINISHED], log_probs=None,
                  cum_log_probs=None, sequence_index=0)
    fields.update(overrides)
    return SimpleNamespace(**fields)


def response_for(client_id, **overrides):
    return LlmResponse(request_id=client_id, result=make_result(**overrides),
                       client_id=client_id)


def bind_request(router, request_id="req-1", client_id=41, stop_reasons=()):
    binding = router.bind(request_id, client_id, (1, 2), stop_reasons)
    binding.stream_opened = True  # tests drain the delivery directly
    return binding


def drain(binding):
    frames = []
    while True:
        frame, ready = binding.delivery.pop_nowait()
        if not ready:
            return frames
        frames.append(frame)


class TestRouterRemediations:

    def test_nonzero_sequence_index_fails_request(self):
        router = EngineFrameRouter(abort_fn=lambda cid: None)
        binding = bind_request(router)
        router.on_response(response_for(41, sequence_index=1))
        frames = drain(binding)
        assert frames, "expected a typed ending"
        assert frames[-1].error_code == "router_error" or isinstance(
            frames[-1], RequestComplete)
        assert binding.ended

    def test_zero_token_final_carries_cached_tokens(self):
        router = EngineFrameRouter(abort_fn=lambda cid: None)
        binding = bind_request(router)
        router.on_response(
            response_for(41, is_final=True, output_token_ids=[[]],
                         finish_reasons=[tllm.FinishReason.LENGTH],
                         cached_tokens=7))
        frames = drain(binding)
        complete = frames[-1]
        assert isinstance(complete, RequestComplete)
        assert complete.cached_tokens == 7

    def test_closed_delivery_is_not_recorded_as_overflow(self):
        router = EngineFrameRouter(abort_fn=lambda cid: None)
        binding = bind_request(router)
        binding.delivery.close()
        router.on_response(response_for(41))
        assert router.counters["overflow_aborts"] == 0
        assert router.counters["late_or_duplicate_absorbed"] == 1

    def test_malformed_stop_reasons_rejected_at_bind(self):
        router = EngineFrameRouter(abort_fn=lambda cid: None)
        with pytest.raises(RouterError):
            bind_request(router, stop_reasons=(((1, 2), lambda: None), ))
        with pytest.raises(RouterError):
            bind_request(router, request_id="req-2",
                         stop_reasons=((("a", ), "x"), ))


class TestPromptLogprobLifecycle:
    """Cached/map-shaped prompt logprobs across multiple responses (R2)."""

    @staticmethod
    def map_wrapper(client_id, tokens, prompt_entries):
        from tensorrt_llm.executor.result import LogProbsResult
        return ResponseWrapper(
            response_for(client_id, output_token_ids=[list(tokens)]),
            logprobs=LogProbsResult(prompt=prompt_entries, generation=None))

    @staticmethod
    def map_entries():
        from tensorrt_llm.executor.result import Logprob
        # prompt (1, 2): expected lookup keys are prompt[1:] + first token.
        return [{2: Logprob(-1.0, 1)}, {5: Logprob(-2.0, 1)}]

    def test_reattached_prompt_map_on_terminal_only_final_is_benign(self):
        router = EngineFrameRouter(abort_fn=lambda cid: None)
        binding = bind_request(router)
        # First token-carrying response delivers the prompt logprobs.
        router.on_response(self.map_wrapper(41, [5], self.map_entries()))
        # The worker re-attaches the cached map on the terminal-only final.
        final = self.map_wrapper(41, [], self.map_entries())
        final._response.result.is_final = True
        final._response.result.finish_reasons = [tllm.FinishReason.END_ID]
        router.on_response(final)
        frames = drain(binding)
        assert isinstance(frames[-1], RequestComplete)
        assert frames[-1].status == "ok"
        deltas = [f for f in frames if isinstance(f, TokenDelta)]
        assert deltas[0].prompt_logprobs == (-1.0, -2.0)

    def test_map_prompt_logprobs_held_from_no_token_response(self):
        router = EngineFrameRouter(abort_fn=lambda cid: None)
        binding = bind_request(router)
        # Map-shaped prompt logprobs arrive BEFORE any token: held raw.
        router.on_response(self.map_wrapper(41, [], self.map_entries()))
        router.on_response(response_for(41, output_token_ids=[[5]]))
        router.on_response(
            response_for(41, is_final=True, output_token_ids=[[]],
                         finish_reasons=[tllm.FinishReason.END_ID]))
        frames = drain(binding)
        deltas = [f for f in frames if isinstance(f, TokenDelta)]
        assert deltas[0].prompt_logprobs == (-1.0, -2.0)
        assert isinstance(frames[-1], RequestComplete)
        assert frames[-1].status == "ok"


class TestCachedTokenConsistency:
    """Absent / zero / match / mismatch cases (R2)."""

    def run_case(self, delta_cached, final_cached):
        router = EngineFrameRouter(abort_fn=lambda cid: None)
        binding = bind_request(router)
        router.on_response(response_for(41, cached_tokens=delta_cached))
        router.on_response(
            response_for(41, is_final=True, output_token_ids=[[]],
                         finish_reasons=[tllm.FinishReason.END_ID],
                         cached_tokens=final_cached))
        return drain(binding)

    def test_absent(self):
        frames = self.run_case(None, None)
        assert frames[-1].status == "ok" and frames[-1].cached_tokens is None

    def test_zero(self):
        frames = self.run_case(0, 0)
        assert frames[-1].status == "ok" and frames[-1].cached_tokens == 0

    def test_match(self):
        frames = self.run_case(3, 3)
        assert frames[-1].status == "ok" and frames[-1].cached_tokens == 3

    def test_mismatch_is_typed_failure(self):
        frames = self.run_case(3, 9)
        assert isinstance(frames[-1], RequestComplete)
        assert frames[-1].status == "failed"

    def test_final_only_value_accepted(self):
        frames = self.run_case(None, 7)
        assert frames[-1].status == "ok" and frames[-1].cached_tokens == 7


class TestLifecycleRework:
    """Plan AC-4 lifecycle: completed-id reuse + bounded state (R1)."""

    @staticmethod
    def finish_and_retire(router, binding, client_id):
        router.on_response(
            response_for(client_id, is_final=True, output_token_ids=[[5]],
                         finish_reasons=[tllm.FinishReason.END_ID]))
        drain(binding)
        router.retire_delivery(binding.request_id)

    def test_request_id_reusable_after_delivery_retired(self):
        router = EngineFrameRouter(abort_fn=lambda cid: None)
        binding = bind_request(router, "req-reuse", client_id=41)
        # While active or un-retired: duplicate is rejected.
        with pytest.raises(RouterError):
            bind_request(router, "req-reuse", client_id=42)
        self.finish_and_retire(router, binding, 41)
        # After consumption + retirement: the id is reusable (positive case).
        rebound = bind_request(router, "req-reuse", client_id=43)
        self.finish_and_retire(router, rebound, 43)

    def test_state_returns_to_bounds_after_stress(self):
        router = EngineFrameRouter(abort_fn=lambda cid: None)
        for i in range(60):
            binding = bind_request(router, f"req-{i}", client_id=100 + i)
            self.finish_and_retire(router, binding, 100 + i)
        assert router.active_request_count() == 0
        assert len(router._by_request) == 0
        assert len(router._by_client) == 0
        assert len(router._delivery_index) == 0
        assert len(router._tombstones) <= router._tombstone_limit
        assert len(router._retired_client_ids) <= router._retired_limit

    def test_close_before_end_retires_on_cancellation_final(self):
        """AC-4: an early close records retirement intent; the in-flight
        (token-carrying) cancellation final both ENDS the binding and
        completes the retirement, so the request id is immediately
        reusable rather than retained until tombstone eviction."""
        router = EngineFrameRouter(abort_fn=lambda cid: None)
        binding = bind_request(router, "req-close", client_id=61)
        router.on_response(response_for(61))  # mid-stream delta
        # The consumer closes early (FrameStream.aclose: abort issued,
        # delivery closed, retirement recorded).
        binding.delivery.close()
        router.retire_delivery("req-close")
        assert not binding.ended
        assert binding.retire_when_ended
        # The in-flight cancellation final arrives afterwards — token-
        # carrying, against the closed delivery.
        router.on_response(
            response_for(61, is_final=True, output_token_ids=[[5]],
                         finish_reasons=[tllm.FinishReason.CANCELLED]))
        assert binding.ended
        assert 61 not in router._by_client
        assert "req-close" not in router._delivery_index
        # Retirement completed: the id is immediately reusable.
        rebound = bind_request(router, "req-close", client_id=62)
        self.finish_and_retire(router, rebound, 62)

    def test_repeated_close_stress_returns_to_bounds(self):
        router = EngineFrameRouter(abort_fn=lambda cid: None)
        for i in range(60):
            binding = bind_request(router, f"req-c{i}", client_id=300 + i)
            router.on_response(response_for(300 + i))
            binding.delivery.close()
            router.retire_delivery(f"req-c{i}")
            router.on_response(
                response_for(300 + i, is_final=True, output_token_ids=[[7]],
                             finish_reasons=[tllm.FinishReason.CANCELLED]))
        assert len(router._by_request) == 0
        assert len(router._by_client) == 0
        assert len(router._delivery_index) == 0
        assert len(router._tombstones) <= router._tombstone_limit
        assert len(router._retired_client_ids) <= router._retired_limit

    def test_retired_late_frame_absorbed_and_ancient_falls_through(self):
        router = EngineFrameRouter(abort_fn=lambda cid: None,
                                   tombstone_limit=2)
        router._retired_limit = 2
        for i in range(5):
            binding = bind_request(router, f"req-{i}", client_id=200 + i)
            self.finish_and_retire(router, binding, 200 + i)
        # Recently retired: claimed and absorbed.
        assert router.route_response(response_for(204)) is True
        # Ancient (aged out of the bounded set): falls through — safe because
        # proxy ids are never reused, so legacy lookup drops it.
        assert router.route_response(response_for(200)) is False


class TestCounterIncrements:

    def test_router_failures_increments_on_normalization_failure(self):
        router = EngineFrameRouter(abort_fn=lambda cid: None)
        binding = bind_request(router)
        before = router.counters["router_failures"]
        # Mismatched logprob count fails normalization typed.
        router.on_response(
            response_for(41, output_token_ids=[[5, 6]], log_probs=[[-0.5]]))
        assert router.counters["router_failures"] == before + 1
        frames = drain(binding)
        assert frames and binding.ended


class TestCodecRemediation:

    def test_mixed_mapping_keys_typed_error(self):
        delta = TokenDelta(request_id="r", sequence_id=0, new_token_ids=(1, ))
        object.__setattr__(delta, "metrics", {"a": 1.0, 3: 2.0})
        with pytest.raises(EncodeError) as excinfo:
            encode(delta)
        assert excinfo.value.reason == "non_primitive"


class TestEnvelopeRemediation:

    def test_non_finite_metric_typed_error(self):
        wrapper = ResponseWrapper(response_for(41),
                                  request_perf_metrics={"t": float("nan")})
        with pytest.raises(EnvelopeError) as excinfo:
            normalize_response(wrapper)
        assert excinfo.value.reason == "metrics_mismatch"


class CharTokenizer:

    def decode(self, ids, **kwargs):
        return "".join(chr(97 + i % 26) for i in ids)

    def decode_incrementally(self, token_ids, prev_text=None, states=None, *,
                             flush=False, stream_interval=1, **kwargs):
        prev_text = prev_text or ""
        return prev_text + self.decode(token_ids), states or {}


def ids_for(text):
    return [ord(ch) - 97 for ch in text]


class TestAssemblerRemediation:

    def test_engine_stop_trim_spans_multiple_deltas(self):
        sequence = tuple(ids_for("xy"))
        config = FrontendOutputConfig(stop_sequence_reasons=((sequence, "xy"), ))
        assembler = FrontendResponseAssembler("req-1", config,
                                              tokenizer=CharTokenizer())
        # The stop sequence spans two buffered deltas in one batch.
        updates = assembler.process_frames([
            TokenDelta(request_id="req-1", sequence_id=0,
                       new_token_ids=tuple(ids_for("abx")), event_seq=0),
            TokenDelta(request_id="req-1", sequence_id=0,
                       new_token_ids=tuple(ids_for("y")), event_seq=1),
            Terminal(request_id="req-1", sequence_id=0, finish_reason="stop",
                     stop_reason="xy", event_seq=2),
            RequestComplete(request_id="req-1", status="ok", prompt_tokens=2,
                            completion_tokens=4, event_seq=3),
        ])
        assert assembler.token_ids == ids_for("ab")
        assert assembler.text == "ab"
        rendered = "".join(u.text_diff for u in updates if u.kind == "delta")
        assert rendered == "ab"
        delta_tokens = [t for u in updates if u.kind == "delta"
                        for t in u.new_token_ids]
        assert delta_tokens == ids_for("ab")
