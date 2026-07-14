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
"""Full SSE oracle surface: completions endpoint, ordered continuous usage,
SSE logprobs, assembly errors, disconnect abort, [DONE], and every
request-level fallback axis through the real facade."""

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor.engine_client.conversion import ELIGIBILITY_MATRIX
from tensorrt_llm.executor.result import DetokenizedGenerationResultBase
from tensorrt_llm.bindings import executor as tllme
from tensorrt_llm.sampling_params import LogprobMode, SamplingParams
from tensorrt_llm.serve.engine_client_serving import (_ContractStreamError,
                                                      wrap_sse_with_done)
from tensorrt_llm.serve.openai_protocol import CompletionRequest, StreamOptions
from tensorrt_llm.serve.postprocess_handlers import (
    CompletionPostprocArgs, chat_stream_post_processor,
    completion_stream_post_processor)
from tensorrt_llm.executor.utils import ErrorResponse

from test_serving_e5a import (PROMPT_IDS, CharTokenizer, delta_response,
                              final_response, ids_for, make_chat_request,
                              make_postproc_args, make_serving, normalize_events,
                              parse_sse, prepared_sampling_params,
                              run_contract_sse, run_legacy_sse)


def make_completion_request(**overrides):
    kwargs = dict(model="test-model", prompt="hi", stream=True, max_tokens=32)
    kwargs.update(overrides)
    return CompletionRequest(**kwargs)


def make_completion_args(request, tokenizer, num_prompt_tokens=3):
    args = CompletionPostprocArgs.from_request(request)
    args.prompt_idx = 0
    args.tokenizer = tokenizer
    args.num_prompt_tokens = num_prompt_tokens
    return args


def run_legacy_completion_sse(responses, sampling_params, request, tokenizer):
    result = DetokenizedGenerationResultBase(id=7, sampling_params=sampling_params,
                                             tokenizer=tokenizer, streaming=True)
    args = make_completion_args(request, tokenizer)
    pieces = []
    for response in responses:
        result._handle_response(response)
        pieces.extend(completion_stream_post_processor(result, args))
        if result._done:
            break
    return pieces


def run_contract_completion_sse(responses, sampling_params, request,
                                serving=None):
    serving = serving or make_serving()
    args = make_completion_args(request, serving.tokenizer)
    generator = serving.try_stream(
        preprocessed=SimpleNamespace(prompt_token_ids=list(PROMPT_IDS),
                                     query_token_ids=None, multimodal_params=None,
                                     encoder_input_token_ids=None),
        sampling_params=sampling_params,
        post_processor=completion_stream_post_processor, postproc_args=args)
    assert generator is not None
    executor = serving.client._executor

    async def collect():
        pieces = []

        async def puller():
            async for piece in generator:
                pieces.append(piece)

        pull_task = asyncio.ensure_future(puller())
        for response in responses:
            executor.router.on_response(response)
            for _ in range(5):
                await asyncio.sleep(0)
        await asyncio.wait_for(pull_task, 10)
        return pieces

    return asyncio.run(collect()), serving


class TestCompletionsSseParity:

    def assert_parity(self, responses, request=None, sampling_kwargs=None):
        request = request or make_completion_request()
        tokenizer = CharTokenizer()
        legacy = normalize_events(parse_sse(
            run_legacy_completion_sse(
                responses, prepared_sampling_params(**(sampling_kwargs or {})),
                request, tokenizer)))
        pieces, _ = run_contract_completion_sse(
            responses, prepared_sampling_params(**(sampling_kwargs or {})),
            request)
        contract = normalize_events(parse_sse(pieces))
        assert contract == legacy
        return contract

    def test_plain_stream(self):
        responses = [delta_response(1, ids_for("hel")),
                     delta_response(1, ids_for("lo")),
                     final_response(1, tllm.FinishReason.END_ID)]
        result = self.assert_parity(responses)
        assert result["content"] == "hello"
        assert result["finish_reason"] == "stop"

    def test_continuous_usage_sequence(self):
        request = make_completion_request(stream_options=StreamOptions(
            include_usage=True, continuous_usage_stats=True))
        responses = [delta_response(1, ids_for("ab")),
                     delta_response(1, ids_for("cd")),
                     final_response(1, tllm.FinishReason.LENGTH)]
        result = self.assert_parity(responses, request=request)
        assert len(result["usage_sequence"]) >= 2
        assert result["final_usage"]["completion_tokens"] == 4

    def test_stop_string(self):
        responses = [delta_response(1, ids_for("ax")),
                     delta_response(1, ids_for("yb")),
                     final_response(1, tllm.FinishReason.LENGTH)]
        params_kwargs = dict(stop=["xy"])
        result = self.assert_parity(responses, sampling_kwargs=params_kwargs)
        assert result["finish_reason"] == "stop"
        assert result["stop_reason"] == "xy"

    def test_cancelled(self):
        responses = [delta_response(1, ids_for("a")),
                     final_response(1, tllm.FinishReason.CANCELLED)]
        result = self.assert_parity(responses)
        assert result["finish_reason"] == "cancelled"


def extract_logprob_values(pieces):
    values = []
    for event in parse_sse(pieces):
        if event == "DONE":
            continue
        for choice in event.get("choices", []):
            logprobs = choice.get("logprobs")
            if not logprobs:
                continue
            content = logprobs.get("content") or []
            for entry in content:
                values.append(round(entry.get("logprob", 0.0), 6))
            token_logprobs = logprobs.get("token_logprobs") or []
            values.extend(round(v, 6) for v in token_logprobs if v is not None)
    return values


class TestSseLogprobs:

    def test_chat_logprob_chunks_match_legacy(self):
        request = make_chat_request(logprobs=True)
        tokenizer = CharTokenizer()
        # Real legacy streams fuse the last delta with the final response;
        # a logprob-carrying stream therefore ends with a token+logprob
        # final, which both pipelines must render identically.
        responses = [
            delta_response(1, ids_for("ab"), log_probs=[[-0.5, -1.0]]),
            final_response(1, tllm.FinishReason.END_ID, tokens=ids_for("c"),
                           log_probs=[[-0.25]]),
        ]
        legacy_pieces = run_legacy_sse(
            responses, prepared_sampling_params(logprobs=0), request, tokenizer)
        contract_pieces, _ = run_contract_sse(
            responses, prepared_sampling_params(logprobs=0), request)
        assert extract_logprob_values(contract_pieces) == \
            extract_logprob_values(legacy_pieces)
        assert extract_logprob_values(contract_pieces)  # non-empty


class TestAssemblyErrors:

    def test_mid_stream_error_raises_typed(self):
        serving = make_serving()
        request = make_chat_request()
        args = make_postproc_args(request, serving.tokenizer)
        generator = serving.try_stream(
            preprocessed=SimpleNamespace(prompt_token_ids=list(PROMPT_IDS),
                                         query_token_ids=None,
                                         multimodal_params=None,
                                         encoder_input_token_ids=None),
            sampling_params=prepared_sampling_params(),
            post_processor=chat_stream_post_processor, postproc_args=args)
        executor = serving.client._executor
        client_id = executor.submitted[-1].id
        executor.router.on_response(delta_response(client_id, ids_for("a")))
        executor.router.on_response(
            ErrorResponse(client_id=client_id, error_msg="boom", request_id=1))

        async def consume():
            async for _ in generator:
                pass

        with pytest.raises(_ContractStreamError):
            asyncio.run(consume())


class TestDisconnectAbort:

    def test_client_disconnect_triggers_engine_abort(self):
        serving = make_serving()
        request = make_chat_request()
        args = make_postproc_args(request, serving.tokenizer)

        class FakeRawRequest:

            def __init__(self):
                self.disconnected = False
                self.state = SimpleNamespace()

            async def is_disconnected(self):
                return self.disconnected

        raw_request = FakeRawRequest()
        generator = serving.try_stream(
            preprocessed=SimpleNamespace(prompt_token_ids=list(PROMPT_IDS),
                                         query_token_ids=None,
                                         multimodal_params=None,
                                         encoder_input_token_ids=None),
            sampling_params=prepared_sampling_params(),
            post_processor=chat_stream_post_processor, postproc_args=args,
            raw_request=raw_request)
        executor = serving.client._executor
        client_id = executor.submitted[-1].id

        async def scenario():
            executor.router.on_response(delta_response(client_id, ids_for("a")))
            task = asyncio.ensure_future(
                self._collect_with_timeout(generator))
            await asyncio.sleep(0.05)
            raw_request.disconnected = True
            # The watcher polls every second; the abort lands, then the fake
            # engine answers with a CANCELLED final and the stream ends.
            for _ in range(30):
                await asyncio.sleep(0.1)
                if executor.aborted:
                    break
            assert executor.aborted, "disconnect did not trigger an engine abort"
            executor.router.on_response(
                final_response(client_id, tllm.FinishReason.CANCELLED))
            await asyncio.wait_for(task, 10)

        asyncio.run(scenario())

    @staticmethod
    async def _collect_with_timeout(generator):
        async for _ in generator:
            pass


class TestDoneWrapper:

    def test_done_appended_and_first_token_time_set(self):

        async def pieces():
            yield "data: {}\n\n"
            yield "data: {}\n\n"

        raw_request = SimpleNamespace(state=SimpleNamespace())

        async def collect():
            return [p async for p in wrap_sse_with_done(pieces(), raw_request)]

        out = asyncio.run(collect())
        assert out[-1] == "data: [DONE]\n\n"
        assert len(out) == 3
        assert hasattr(raw_request.state, "server_first_token_time")

    def test_empty_stream_still_emits_done(self):

        async def empty():
            return
            yield  # pragma: no cover

        async def collect():
            return [p async for p in wrap_sse_with_done(empty())]

        out = asyncio.run(collect())
        assert out == ["data: [DONE]\n\n"]


class TestCompletionsSseLogprobs:

    def test_completion_logprob_chunks_match_legacy(self):
        request = make_completion_request(logprobs=1)
        tokenizer = CharTokenizer()
        # Legacy-faithful fused final: the last delta rides the final
        # response, carrying both tokens and their logprobs.
        responses = [
            delta_response(1, ids_for("ab"), log_probs=[[-0.5, -1.0]]),
            final_response(1, tllm.FinishReason.END_ID, tokens=ids_for("c"),
                           log_probs=[[-0.25]]),
        ]
        legacy_pieces = run_legacy_completion_sse(
            responses, prepared_sampling_params(logprobs=1), request, tokenizer)
        contract_pieces, _ = run_contract_completion_sse(
            responses, prepared_sampling_params(logprobs=1), request)
        assert extract_logprob_values(contract_pieces) == \
            extract_logprob_values(legacy_pieces)
        assert extract_logprob_values(contract_pieces)  # non-empty


class TestCompletionsAssemblyErrors:

    def test_mid_stream_error_raises_typed(self):
        serving = make_serving()
        request = make_completion_request()
        args = make_completion_args(request, serving.tokenizer)
        generator = serving.try_stream(
            preprocessed=SimpleNamespace(prompt_token_ids=list(PROMPT_IDS),
                                         query_token_ids=None,
                                         multimodal_params=None,
                                         encoder_input_token_ids=None),
            sampling_params=prepared_sampling_params(),
            post_processor=completion_stream_post_processor,
            postproc_args=args)
        executor = serving.client._executor
        client_id = executor.submitted[-1].id
        executor.router.on_response(delta_response(client_id, ids_for("a")))
        executor.router.on_response(
            ErrorResponse(client_id=client_id, error_msg="boom", request_id=1))

        async def consume():
            async for _ in generator:
                pass

        with pytest.raises(_ContractStreamError):
            asyncio.run(consume())


class TestCompletionsDisconnectAbort:

    def test_client_disconnect_triggers_engine_abort(self):
        serving = make_serving()
        request = make_completion_request()
        args = make_completion_args(request, serving.tokenizer)

        class FakeRawRequest:

            def __init__(self):
                self.disconnected = False
                self.state = SimpleNamespace()

            async def is_disconnected(self):
                return self.disconnected

        raw_request = FakeRawRequest()
        generator = serving.try_stream(
            preprocessed=SimpleNamespace(prompt_token_ids=list(PROMPT_IDS),
                                         query_token_ids=None,
                                         multimodal_params=None,
                                         encoder_input_token_ids=None),
            sampling_params=prepared_sampling_params(),
            post_processor=completion_stream_post_processor,
            postproc_args=args, raw_request=raw_request)
        executor = serving.client._executor
        client_id = executor.submitted[-1].id

        async def scenario():
            executor.router.on_response(delta_response(client_id, ids_for("a")))

            async def consume():
                async for _ in generator:
                    pass

            task = asyncio.ensure_future(consume())
            await asyncio.sleep(0.05)
            raw_request.disconnected = True
            for _ in range(30):
                await asyncio.sleep(0.1)
                if executor.aborted:
                    break
            assert executor.aborted, "disconnect did not trigger an engine abort"
            executor.router.on_response(
                final_response(client_id, tllm.FinishReason.CANCELLED))
            await asyncio.wait_for(task, 10)

        asyncio.run(scenario())


class TestActiveRequestCounter:

    def test_active_count_transitions_zero_one_zero(self):
        serving = make_serving()
        assert serving.get_counters()["active_requests"] == 0
        request = make_completion_request()
        responses = [delta_response(1, ids_for("ab")),
                     final_response(1, tllm.FinishReason.END_ID)]
        # Mid-stream observation: submit, deliver one delta, snapshot.
        args = make_completion_args(request, serving.tokenizer)
        generator = serving.try_stream(
            preprocessed=SimpleNamespace(prompt_token_ids=list(PROMPT_IDS),
                                         query_token_ids=None,
                                         multimodal_params=None,
                                         encoder_input_token_ids=None),
            sampling_params=prepared_sampling_params(),
            post_processor=completion_stream_post_processor,
            postproc_args=args)
        executor = serving.client._executor
        client_id = executor.submitted[-1].id
        executor.router.on_response(responses[0])
        assert serving.get_counters()["active_requests"] == 1

        async def finish():
            pieces = []

            async def puller():
                async for piece in generator:
                    pieces.append(piece)

            task = asyncio.ensure_future(puller())
            executor.router.on_response(responses[1])
            await asyncio.wait_for(task, 10)
            return pieces

        asyncio.run(finish())
        assert serving.get_counters()["active_requests"] == 0
        assert serving.counters["contract_requests"] == 1


class TestNoLiveNormalizationReaches:
    """The contract path must normalize ONLY via the context-only boundary:
    neither the legacy-input normalizer nor any live ``LLM`` sampling
    preparation may run for a contract request."""

    def test_contract_stream_with_live_normalizers_poisoned(self, monkeypatch):
        import tensorrt_llm.serve.engine_client_serving as serving_module
        from tensorrt_llm.llmapi.llm import LLM

        def poison(*args, **kwargs):
            raise AssertionError(
                "live sampling normalization reached on the contract path")

        monkeypatch.setattr(serving_module, "prepare_sampling_params", poison,
                            raising=False)
        monkeypatch.setattr(LLM, "_prepare_sampling_params", poison,
                            raising=False)
        request = make_completion_request()
        responses = [delta_response(1, ids_for("hi")),
                     final_response(1, tllm.FinishReason.END_ID)]
        pieces, _ = run_contract_completion_sse(
            responses, prepared_sampling_params(), request)
        assert pieces, "contract stream produced no output"


class TestTokenizerSpecProvenance:

    def test_manifest_hash_mismatch_fails_typed(self, tmp_path):
        from tensorrt_llm.executor.engine_client.contract import TokenizerSpec
        from tensorrt_llm.serve.engine_client_serving import (
            EngineClientConfigError, load_tokenizer_from_spec)
        (tmp_path / "tokenizer.json").write_bytes(b"{}")
        spec = TokenizerSpec(uri=str(tmp_path),
                             files_manifest=(("tokenizer.json", "0" * 64), ))
        with pytest.raises(EngineClientConfigError, match="manifest"):
            load_tokenizer_from_spec(spec)

    def test_missing_manifest_file_fails_typed(self, tmp_path):
        from tensorrt_llm.executor.engine_client.contract import TokenizerSpec
        from tensorrt_llm.serve.engine_client_serving import (
            EngineClientConfigError, load_tokenizer_from_spec)
        spec = TokenizerSpec(uri=str(tmp_path),
                             files_manifest=(("tokenizer.json", "0" * 64), ))
        with pytest.raises(EngineClientConfigError, match="unreadable"):
            load_tokenizer_from_spec(spec)


AXIS_RECIPES = [
    ("echo", {}, dict(echo=True)),
    ("non_streaming", {}, dict(streaming=False)),
    ("logprobs_mode", dict(logprobs=0, logprobs_mode=LogprobMode.PROCESSED), {}),
    ("lookahead_config",
     dict(lookahead_config=tllme.LookaheadDecodingConfig(2, 2, 2)), {}),
    ("n_gt_1", dict(n=2, best_of=2, top_p=0.9), {}),
    ("beam_search", dict(use_beam_search=True, n=1, best_of=1), {}),
    ("top_logprobs", dict(logprobs=5), {}),
    ("prompt_top_logprobs", dict(prompt_logprobs=5), {}),
    ("logits_processor", dict(apply_batched_logits_processor=True), {}),
    ("embedding_bias", dict(embedding_bias=torch.zeros(4)), {}),
    ("bad_words", dict(bad_token_ids=[3]), {}),
    ("ignore_eos", dict(ignore_eos=True), {}),
    ("min_p", dict(min_p=0.2), {}),
    ("top_p_extras", dict(top_p_min=0.5), {}),
    ("no_repeat_ngram", dict(no_repeat_ngram_size=2), {}),
    ("prompt_ignore_length", dict(prompt_ignore_length=1), {}),
    ("return_logits", dict(return_context_logits=True), {}),
    ("exclude_input_from_output", dict(exclude_input_from_output=False), {}),
    ("truncate_prompt_tokens", dict(truncate_prompt_tokens=4), {}),
    ("thinking_token_budget", dict(thinking_token_budget=10), {}),
    ("multimodal", {}, dict(shim_multimodal=True)),
    ("lora", {}, dict(lora_request=object())),
    ("prompt_adapter", {}, dict(prompt_adapter_request=object())),
    ("disaggregated", {}, dict(disaggregated_params=object())),
    ("scheduling_params", {}, dict(scheduling_params=SimpleNamespace(
        agent_hierarchy="x"))),
    ("conversation_params", {}, dict(conversation_params=object())),
    ("trace_headers", {}, dict(trace_headers={"traceparent": "00-abc"})),
    ("cache_salt", {}, dict(cache_salt="salt")),
    ("query_token_ids", {}, dict(shim_query=True)),
    ("encoder_input", {}, dict(shim_encoder=True)),
    ("kv_cache_retention", {}, dict(kv_cache_retention_config=object())),
    ("priority", {}, dict(priority=0.9)),
]


class TestEveryFallbackAxisThroughFacade:

    @pytest.mark.parametrize("axis,param_kwargs,stream_kwargs", AXIS_RECIPES)
    def test_axis_falls_back_with_counter(self, axis, param_kwargs, stream_kwargs):
        serving = make_serving()
        request = make_chat_request()
        args = make_postproc_args(request, serving.tokenizer)
        params = prepared_sampling_params(**param_kwargs)
        stream_kwargs = dict(stream_kwargs)
        shim = SimpleNamespace(
            prompt_token_ids=list(PROMPT_IDS),
            query_token_ids=[1] if stream_kwargs.pop("shim_query", False) else None,
            multimodal_params=object()
            if stream_kwargs.pop("shim_multimodal", False) else None,
            encoder_input_token_ids=[1]
            if stream_kwargs.pop("shim_encoder", False) else None)
        generator = serving.try_stream(
            preprocessed=shim, sampling_params=params,
            post_processor=chat_stream_post_processor, postproc_args=args,
            **stream_kwargs)
        assert generator is None
        assert serving.counters[f"fallback:{axis}"] == 1
        assert serving.counters["contract_requests"] == 0

    def test_every_request_level_axis_is_driven_through_the_facade(self):
        tested = {axis for axis, _, _ in AXIS_RECIPES}
        # The only two axes not drivable via AXIS_RECIPES, each verified by a
        # dedicated facade/enforcement-point test rather than silently waived:
        # - bart_forced_tokens fires from the model context (needs a serving
        #   fixture with model_type="bart") — driven through the real facade
        #   with its counter by TestContextOnlyBoundary.
        # - postproc_params cannot reach the facade at all (both endpoints
        #   construct PostprocParams strictly after the contract branch, and
        #   try_stream exposes no such parameter); its defense-in-depth
        #   rejection at convert_request is exercised by
        #   test_conversion.REJECTION_CASES.
        exempt = {"bart_forced_tokens", "postproc_params"}
        matrix_axes = {rule.axis for rule in ELIGIBILITY_MATRIX
                       if rule.classification == "ineligible"}
        missing = matrix_axes - tested - exempt
        assert not missing, f"axes not driven through the facade: {missing}"
