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
"""E5a serving-path tests: SSE semantic parity, fallback counters, model
context/tokenizer reload, event-loop responsiveness, and contract-path
guards. All CPU-only: the engine is faked at the proxy seam."""

import asyncio
import json
import os
import time
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor.engine_client import conversion as conversion_module
from tensorrt_llm.executor.engine_client.assembler import FrontendResponseAssembler
from tensorrt_llm.executor.engine_client.contract import TokenizerSpec
from tensorrt_llm.executor.engine_client.local_client import (
    EngineClientConfig, LocalProcessEngineClient)
from tensorrt_llm.executor.result import DetokenizedGenerationResultBase
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.engine_client_serving import (ContractStreamView,
                                                      EngineClientServing)
from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                StreamOptions)
from tensorrt_llm.serve.postprocess_handlers import (ChatPostprocArgs,
                                                     chat_stream_post_processor)

PROMPT_IDS = (1, 2, 3)
MODEL_DIR = "/workspace/models/Qwen2.5-0.5B-Instruct"


class CharTokenizer:

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) - 97 for ch in text]

    def decode(self, ids, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(chr(97 + i % 26) for i in ids)

    def decode_incrementally(self, token_ids, prev_text=None, states=None, *,
                             flush=False, stream_interval=1, **kwargs):
        prev_text = prev_text or ""
        return prev_text + self.decode(token_ids), states or {}


def ids_for(text):
    return [ord(ch) - 97 for ch in text]


class FakeExecutor:

    def __init__(self):
        import threading
        self.service = None
        self.router = None
        self.aborted = []
        self.submitted = []
        self._results = {}
        self.doing_shutdown = False
        self._fatal_error = None
        self._submission_lock = threading.RLock()
        self._next_id = 0

    def attach_engine_service(self, service):
        self.service = service
        self.router = service.router

    def submit_contract(self, engine_request, stop_reasons=()):
        return self.service.submit_contract(engine_request,
                                            stop_reasons=stop_reasons)

    class _Queue:

        def __init__(self, outer):
            self.outer = outer

        def put(self, item):
            self.outer.submitted.append(item)

    @property
    def request_queue(self):
        return self._Queue(self)

    def _start_dispatch_threads(self):
        pass

    def _get_next_client_id(self):
        self._next_id += 1
        return self._next_id

    def _handle_background_error(self):
        pass

    def abort_request(self, client_id):
        self.aborted.append(client_id)

    def check_health(self):
        return True

    def get_stats(self, timeout):
        return []

    def get_kv_events(self, timeout):
        return []

    def shutdown(self):
        pass


def make_serving(tokenizer=None, model_type=None,
                 generation_stop_token_ids=()) -> EngineClientServing:
    """Build the serving glue over a fake engine (bypasses the LLM ctor)."""
    from tensorrt_llm.executor.engine_client.contract import (
        EngineCapabilities, FrontendModelContext)
    serving = object.__new__(EngineClientServing)
    serving.client = LocalProcessEngineClient(
        FakeExecutor(), EngineClientConfig(backend="pytorch", flag_enabled=True))
    serving.tokenizer = tokenizer or CharTokenizer()
    serving.counters = {"contract_requests": 0, "capability_rejections": 0}
    serving._stream_interval = 1
    serving._force_return_perf_metrics = False
    serving.model_context = FrontendModelContext(
        tokenizer=TokenizerSpec(uri="/fake"),
        capabilities=EngineCapabilities(features=("streaming", )),
        eos_id=2, pad_id=0,
        generation_stop_token_ids=tuple(generation_stop_token_ids),
        model_type=model_type)
    return serving


def make_result(**overrides):
    fields = dict(is_final=False, output_token_ids=[[]],
                  finish_reasons=[tllm.FinishReason.NOT_FINISHED], log_probs=None,
                  cum_log_probs=None, sequence_index=0, context_phase_params=None,
                  decoding_iter=1, avg_decoded_tokens_per_iter=None,
                  request_perf_metrics=None, generation_logits=None,
                  context_logits=None)
    fields.update(overrides)
    return SimpleNamespace(**fields)


def delta_response(client_id, tokens, **overrides):
    return LlmResponse(request_id=1, client_id=client_id,
                       result=make_result(output_token_ids=[list(tokens)],
                                          **overrides))


def final_response(client_id, reason, tokens=(), **overrides):
    return LlmResponse(request_id=1, client_id=client_id,
                       result=make_result(is_final=True,
                                          output_token_ids=[list(tokens)],
                                          finish_reasons=[reason],
                                          **overrides))


def make_chat_request(**overrides):
    kwargs = dict(model="test-model",
                  messages=[{"role": "user", "content": "hi"}], stream=True,
                  max_tokens=32)
    kwargs.update(overrides)
    return ChatCompletionRequest(**kwargs)


def make_postproc_args(request, tokenizer, num_prompt_tokens=3) -> ChatPostprocArgs:
    args = ChatPostprocArgs.from_request(request)
    args.tokenizer = tokenizer
    args.num_prompt_tokens = num_prompt_tokens
    return args


def prepared_sampling_params(**kwargs) -> SamplingParams:
    params = SamplingParams(max_tokens=32, end_id=2, pad_id=0, **kwargs)
    if params.stop is not None:
        strings = [params.stop] if isinstance(params.stop, str) else params.stop
        params._stop_word_ids = [ids_for(s) for s in strings]
    params._stream_interval = 1
    return params


def parse_sse(pieces):
    events = []
    for piece in pieces:
        piece = piece.strip()
        if not piece.startswith("data: "):
            continue
        body = piece[len("data: "):].strip()
        if body == "[DONE]":
            events.append("DONE")
            continue
        events.append(json.loads(body))
    return events


def normalize_events(events):
    """Semantic normalization: drop ids/timestamps, coalesce content deltas."""
    role = None
    content = ""
    finish_reason = None
    stop_reason = None
    usage_snapshots = []
    for event in events:
        if event == "DONE":
            continue
        usage = event.get("usage")
        if usage:
            usage_snapshots.append({
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
                    "cached_tokens"),
            })
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("role"):
                role = delta["role"]
            if delta.get("content"):
                content += delta["content"]
            if choice.get("text"):  # completions-endpoint chunks
                content += choice["text"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            if choice.get("stop_reason") is not None:
                stop_reason = choice["stop_reason"]
    return {
        "role": role,
        "content": content,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        # The ORDERED usage sequence is semantic (continuous usage chunks
        # must match); chunk-boundary coalescing may merge token deltas but
        # never reorders usage snapshots.
        "usage_sequence": usage_snapshots,
        "final_usage": usage_snapshots[-1] if usage_snapshots else None,
    }


def run_legacy_sse(responses, sampling_params, request, tokenizer):
    result = DetokenizedGenerationResultBase(id=7, sampling_params=sampling_params,
                                             tokenizer=tokenizer, streaming=True)
    args = make_postproc_args(request, tokenizer)
    pieces = []
    for response in responses:
        result._handle_response(response)
        pieces.extend(chat_stream_post_processor(result, args))
        if result._done:
            break
    return pieces


def run_contract_sse(responses, sampling_params, request, serving=None):
    serving = serving or make_serving()
    args = make_postproc_args(request, serving.tokenizer)
    generator = serving.try_stream(
        preprocessed=SimpleNamespace(prompt_token_ids=list(PROMPT_IDS),
                                     query_token_ids=None, multimodal_params=None,
                                     encoder_input_token_ids=None),
        sampling_params=sampling_params,
        post_processor=chat_stream_post_processor,
        postproc_args=args)
    assert generator is not None, "expected the request to be eligible"
    executor = serving.client._executor

    async def collect():
        # Feed responses incrementally so chunk boundaries (and therefore
        # the ordered continuous-usage sequence) match the legacy
        # per-response handler calls one-to-one.
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


class TestSseSemanticParity:

    def assert_sse_parity(self, responses, request=None, sampling_kwargs=None):
        request = request or make_chat_request()
        tokenizer = CharTokenizer()
        legacy_params = prepared_sampling_params(**(sampling_kwargs or {}))
        contract_params = prepared_sampling_params(**(sampling_kwargs or {}))
        legacy = normalize_events(
            parse_sse(run_legacy_sse(responses, legacy_params, request, tokenizer)))
        contract_pieces, _ = run_contract_sse(responses, contract_params, request)
        contract = normalize_events(parse_sse(contract_pieces))
        assert contract == legacy
        return contract

    def test_plain_stream(self):
        responses = [delta_response(1, ids_for("hel")),
                     delta_response(1, ids_for("lo")),
                     final_response(1, tllm.FinishReason.END_ID)]
        result = self.assert_sse_parity(responses)
        assert result["role"] == "assistant"
        assert result["content"] == "hello"
        assert result["finish_reason"] == "stop"

    def test_final_usage_and_continuous_usage(self):
        request = make_chat_request(stream_options=StreamOptions(
            include_usage=True, continuous_usage_stats=True))
        responses = [delta_response(1, ids_for("hi")),
                     final_response(1, tllm.FinishReason.LENGTH)]
        result = self.assert_sse_parity(responses, request=request)
        assert result["final_usage"] == {
            "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5,
            "cached_tokens": 0,
        }
        assert result["finish_reason"] == "length"

    def test_stop_string_parity(self):
        request = make_chat_request()
        responses = [delta_response(1, ids_for("ax")),
                     delta_response(1, ids_for("yb")),
                     final_response(1, tllm.FinishReason.LENGTH)]
        result = self.assert_sse_parity(responses, request=request,
                                        sampling_kwargs=self.stop_kwargs("xy"))
        # The already-streamed "x" prefix leaks before the stop trims — the
        # known legacy cross-chunk behavior, deliberately preserved in V0
        # (a semantic fix is future work). Parity, not improvement, is the gate.
        assert result["content"] == "ax"
        assert result["finish_reason"] == "stop"
        assert result["stop_reason"] == "xy"

    @staticmethod
    def stop_kwargs(stop_string):
        params_kwargs = dict(stop=[stop_string])
        return params_kwargs

    def test_timeout_rendering_parity(self):
        responses = [delta_response(1, ids_for("a")),
                     final_response(1, tllm.FinishReason.TIMED_OUT)]
        result = self.assert_sse_parity(responses)
        assert result["finish_reason"] == "timeout"

    def test_cancelled_rendering_parity(self):
        responses = [delta_response(1, ids_for("a")),
                     final_response(1, tllm.FinishReason.CANCELLED)]
        result = self.assert_sse_parity(responses)
        assert result["finish_reason"] == "cancelled"


class TestFallbackCounters:

    def test_ineligible_axes_fall_back_with_counters(self):
        serving = make_serving()
        request = make_chat_request()
        args = make_postproc_args(request, serving.tokenizer)
        cases = [
            (dict(n=2, best_of=2, top_p=0.9), "n_gt_1"),
            (dict(logprobs=5), "top_logprobs"),
            (dict(ignore_eos=True), "ignore_eos"),
        ]
        for sampling_kwargs, axis in cases:
            generator = serving.try_stream(
                preprocessed=SimpleNamespace(prompt_token_ids=[1, 2],
                                             query_token_ids=None,
                                             multimodal_params=None,
                                             encoder_input_token_ids=None),
                sampling_params=prepared_sampling_params(**sampling_kwargs),
                post_processor=chat_stream_post_processor, postproc_args=args)
            assert generator is None
            assert serving.counters[f"fallback:{axis}"] == 1
        assert serving.counters["contract_requests"] == 0

    def test_request_level_fallbacks(self):
        serving = make_serving()
        request = make_chat_request()
        args = make_postproc_args(request, serving.tokenizer)
        generator = serving.try_stream(
            preprocessed=SimpleNamespace(prompt_token_ids=[1],
                                         query_token_ids=None,
                                         multimodal_params=object(),
                                         encoder_input_token_ids=None),
            sampling_params=prepared_sampling_params(),
            post_processor=chat_stream_post_processor, postproc_args=args)
        assert generator is None
        assert serving.counters["fallback:multimodal"] == 1
        # Missing token ids
        generator = serving.try_stream(
            preprocessed=SimpleNamespace(prompt_token_ids=None),
            sampling_params=prepared_sampling_params(),
            post_processor=chat_stream_post_processor, postproc_args=args)
        assert generator is None
        assert serving.counters["fallback:no_preprocessed_inputs"] == 1

    def test_all_default_scheduling_params_normalized(self):
        serving = make_serving()
        normalized = serving._normalize_scheduling(
            SimpleNamespace(agent_hierarchy=None))
        assert normalized is None
        kept = serving._normalize_scheduling(SimpleNamespace(agent_hierarchy="x"))
        assert kept is not None

    def test_counter_snapshot_includes_router(self):
        serving = make_serving()
        counters = serving.get_counters()
        for key in ("late_or_duplicate_absorbed", "router_failures",
                    "synthesized_terminals", "overflow_aborts",
                    "active_requests", "contract_requests",
                    "capability_rejections"):
            assert key in counters


class TestContractPathGuards:

    def test_contract_path_never_reenters_legacy(self, monkeypatch):
        """The contract path must not call _setup, build PostprocParams, or
        construct a GenerationResult."""
        from tensorrt_llm.executor import result as result_module
        from tensorrt_llm.executor.postproc_worker import PostprocParams

        def forbidden(name):
            def _raise(*args, **kwargs):
                raise AssertionError(f"contract path invoked forbidden {name}")
            return _raise

        monkeypatch.setattr(SamplingParams, "_setup", forbidden("_setup"))
        monkeypatch.setattr(PostprocParams, "__init__",
                            forbidden("PostprocParams"))
        monkeypatch.setattr(result_module.GenerationResult, "__init__",
                            forbidden("GenerationResult"))

        responses = [delta_response(1, ids_for("hi")),
                     final_response(1, tllm.FinishReason.END_ID)]
        request = make_chat_request()
        pieces, serving = run_contract_sse(responses, prepared_sampling_params(),
                                           request)
        assert pieces
        assert serving.counters["contract_requests"] == 1


class TestEventLoopResponsiveness:

    def test_many_concurrent_streams_keep_loop_responsive(self):
        serving = make_serving()
        request = make_chat_request()
        stream_count = 32
        generators = []
        executor = serving.client._executor
        client_ids = []
        for i in range(stream_count):
            args = make_postproc_args(request, serving.tokenizer)
            generator = serving.try_stream(
                preprocessed=SimpleNamespace(prompt_token_ids=list(PROMPT_IDS),
                                             query_token_ids=None,
                                             multimodal_params=None,
                                             encoder_input_token_ids=None),
                sampling_params=prepared_sampling_params(),
                post_processor=chat_stream_post_processor, postproc_args=args)
            assert generator is not None
            generators.append(generator)
            client_ids.append(executor.submitted[-1].id)

        async def scenario():
            max_gap = 0.0

            async def heartbeat(stop_event):
                nonlocal max_gap
                last = time.monotonic()
                while not stop_event.is_set():
                    await asyncio.sleep(0.005)
                    now = time.monotonic()
                    max_gap = max(max_gap, now - last)
                    last = now

            stop_event = asyncio.Event()
            beat = asyncio.create_task(heartbeat(stop_event))

            def feed():
                for step in range(20):
                    for client_id in client_ids:
                        executor.router.on_response(
                            delta_response(client_id, [step % 26]))
                for client_id in client_ids:
                    executor.router.on_response(
                        final_response(client_id, tllm.FinishReason.END_ID))

            feeder = asyncio.get_running_loop().run_in_executor(None, feed)

            async def drain(generator):
                async for _ in generator:
                    pass

            await asyncio.gather(*(drain(g) for g in generators))
            await feeder
            stop_event.set()
            await beat
            # The event loop must never stall meaningfully while 32 streams
            # and a background feeder are active.
            assert max_gap < 0.5, f"event loop stalled for {max_gap:.3f}s"

        asyncio.run(scenario())


class TestContextOnlyBoundary:
    """The contract path consumes only the data-only model context (R3)."""

    def test_no_live_model_objects_retained(self):
        serving = make_serving()
        # The glue holds no live LLM/model-config/generation-config handles:
        # its normalization inputs are the frozen context + spec tokenizer.
        assert not hasattr(serving, "_normalization")
        forbidden = ("_hf_model_config", "_generation_config", "_llm")
        for name in forbidden:
            assert name not in vars(serving)

    def unprepared_params(self, **kwargs):
        params = SamplingParams(max_tokens=8, **kwargs)  # end_id unset
        return params

    def try_unprepared(self, serving, params):
        request = make_chat_request()
        args = make_postproc_args(request, serving.tokenizer)
        return serving.try_stream(
            preprocessed=SimpleNamespace(prompt_token_ids=list(PROMPT_IDS),
                                         query_token_ids=None,
                                         multimodal_params=None,
                                         encoder_input_token_ids=None),
            sampling_params=params,
            post_processor=chat_stream_post_processor, postproc_args=args)

    def test_context_defaults_end_and_pad_ids(self):
        serving = make_serving()
        params = self.unprepared_params()
        generator = self.try_unprepared(serving, params)
        assert generator is not None
        submitted = serving.client._executor.submitted[-1]
        assert submitted.sampling_params.end_id == 2
        assert submitted.sampling_params.pad_id == 0

    def test_stop_strings_tokenized_with_spec_tokenizer(self):
        serving = make_serving()
        params = self.unprepared_params(stop=["xy"])
        generator = self.try_unprepared(serving, params)
        assert generator is not None
        submitted = serving.client._executor.submitted[-1]
        assert submitted.stop_token_sequences == [ids_for("xy")]

    def test_generation_stop_ids_merged_from_context(self):
        serving = make_serving(generation_stop_token_ids=(32000, ))
        params = self.unprepared_params()
        generator = self.try_unprepared(serving, params)
        assert generator is not None
        submitted = serving.client._executor.submitted[-1]
        assert 32000 in submitted.sampling_params.stop_token_ids

    def test_bart_model_type_ineligible_via_context(self):
        serving = make_serving(model_type="bart")
        generator = self.try_unprepared(serving, self.unprepared_params())
        assert generator is None
        assert serving.counters["fallback:bart_forced_tokens"] == 1

    def test_empty_trace_headers_are_eligible(self):
        serving = make_serving()
        request = make_chat_request()
        args = make_postproc_args(request, serving.tokenizer)
        generator = serving.try_stream(
            preprocessed=SimpleNamespace(prompt_token_ids=list(PROMPT_IDS),
                                         query_token_ids=None,
                                         multimodal_params=None,
                                         encoder_input_token_ids=None),
            sampling_params=prepared_sampling_params(),
            post_processor=chat_stream_post_processor, postproc_args=args,
            trace_headers={})
        assert generator is not None  # {} means "tracing on, no header"
        generator2 = serving.try_stream(
            preprocessed=SimpleNamespace(prompt_token_ids=list(PROMPT_IDS),
                                         query_token_ids=None,
                                         multimodal_params=None,
                                         encoder_input_token_ids=None),
            sampling_params=prepared_sampling_params(),
            post_processor=chat_stream_post_processor, postproc_args=args,
            trace_headers={"traceparent": "00-abc"})
        assert generator2 is None
        assert serving.counters["fallback:trace_headers"] == 1

    def test_capability_rejection_counts_and_falls_back(self):
        from tensorrt_llm.sampling_params import GuidedDecodingParams
        serving = make_serving()
        params = prepared_sampling_params(
            guided_decoding=GuidedDecodingParams(json_object=True))
        request = make_chat_request()
        args = make_postproc_args(request, serving.tokenizer)
        generator = serving.try_stream(
            preprocessed=SimpleNamespace(prompt_token_ids=list(PROMPT_IDS),
                                         query_token_ids=None,
                                         multimodal_params=None,
                                         encoder_input_token_ids=None),
            sampling_params=params,
            post_processor=chat_stream_post_processor, postproc_args=args)
        assert generator is None
        assert serving.counters["capability_rejections"] == 1


class TestFailClosedStartup:

    class FakeArgsLLM(SimpleNamespace):
        pass

    def make_llm(self, **arg_overrides):
        args = SimpleNamespace(experimental_engine_client=True,
                               backend="tensorrt", trust_remote_code=False)
        for key, value in arg_overrides.items():
            setattr(args, key, value)
        return SimpleNamespace(args=args, _executor=SimpleNamespace(),
                               tokenizer=None)

    def test_flag_off_returns_none(self, monkeypatch):
        monkeypatch.delenv("TLLM_EXPERIMENTAL_ENGINE_CLIENT", raising=False)
        llm = self.make_llm(experimental_engine_client=False)
        assert EngineClientServing.create_if_enabled(llm) is None

    def test_malformed_env_fails_closed(self, monkeypatch):
        from tensorrt_llm.executor.engine_client.local_client import \
            EngineClientConfigError
        monkeypatch.setenv("TLLM_EXPERIMENTAL_ENGINE_CLIENT", "yes")
        with pytest.raises(EngineClientConfigError):
            EngineClientServing.create_if_enabled(self.make_llm())

    def test_config_gate_violation_fails_closed(self, monkeypatch):
        from tensorrt_llm.executor.engine_client.local_client import \
            EngineClientConfigError
        monkeypatch.setenv("TLLM_EXPERIMENTAL_ENGINE_CLIENT", "1")
        # Wrong backend while enablement is requested: typed startup error,
        # never a silent legacy fallback.
        with pytest.raises(EngineClientConfigError):
            EngineClientServing.create_if_enabled(
                self.make_llm(backend="tensorrt"))


@pytest.mark.skipif(not os.path.isdir(MODEL_DIR),
                    reason="local tokenizer artifacts unavailable")
class TestModelContextReload:

    def test_tokenizer_reload_parity(self):
        from tensorrt_llm.serve.engine_client_serving import \
            load_tokenizer_from_spec
        from tensorrt_llm.tokenizer.tokenizer import TransformersTokenizer
        spec = TokenizerSpec(uri=MODEL_DIR)
        reloaded = load_tokenizer_from_spec(spec)
        reference = TransformersTokenizer.from_pretrained(MODEL_DIR)
        corpus = [
            "The capital of France is Paris.",
            "def main():\n    return 0",
            "Unicode: héllo ☂ 你好 — em-dash",
            "  leading spaces and\ttabs\n",
            "",
        ]
        for text in corpus:
            assert reloaded.encode(text) == reference.encode(text)
        sample_ids = reference.encode("Hello world, streaming détok!")
        assert reloaded.decode(sample_ids) == reference.decode(sample_ids)

    def test_model_context_manifest(self):
        from tensorrt_llm.serve.engine_client_serving import _file_manifest
        manifest = _file_manifest(MODEL_DIR)
        names = [name for name, _ in manifest]
        assert "tokenizer_config.json" in names
        for _, digest in manifest:
            assert len(digest) == 64
