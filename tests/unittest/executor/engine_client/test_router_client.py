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
"""Replay-matrix tests for the exactly-once router and the local client."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor.engine_client.contract import (EngineRequest,
                                                          EngineSamplingConfig,
                                                          ErrorFrame,
                                                          FrontendOutputConfig,
                                                          GuidedDecodingSpec,
                                                          RequestComplete,
                                                          Terminal, TokenDelta)
from tensorrt_llm.executor.engine_client.invariants import (
    InvariantCheckingStream, StreamInvariantViolation)
from tensorrt_llm.executor.engine_client.local_client import (
    EngineClientConfig, EngineClientConfigError, LocalProcessEngineClient,
    RequestRejectedError)
from tensorrt_llm.executor.engine_client.router import (DEFAULT_TOMBSTONE_LIMIT,
                                                        EngineFrameRouter,
                                                        UnknownRequestError)
from tensorrt_llm.executor.result import LogProbsResult, ResponseWrapper
from tensorrt_llm.executor.utils import ErrorResponse

STREAM_DEADLINE = 5.0


class FakeRequestQueue:

    def __init__(self):
        self.items = []
        self.raise_on_put = False

    def put(self, item):
        if self.raise_on_put:
            raise RuntimeError("enqueue failed after binding")
        self.items.append(item)


class FakeExecutor:
    """Mimics the proxy's contract-native surface for the client/service."""

    def __init__(self):
        self.service = None
        self.router = None
        self.aborted = []
        self.request_queue = FakeRequestQueue()
        self._results = {}
        self.doing_shutdown = False
        self._fatal_error = None
        self._submission_lock = threading.RLock()
        self._next_id = 0
        self.shutdown_called = False
        self.raise_on_submit = False  # fails before any binding exists

    @property
    def submitted(self):
        return self.request_queue.items

    @property
    def raise_after_observe(self):
        return self.request_queue.raise_on_put

    @raise_after_observe.setter
    def raise_after_observe(self, value):
        self.request_queue.raise_on_put = value

    def attach_engine_service(self, service):
        if self.service is not None:
            raise RuntimeError("service already attached")
        self.service = service
        self.router = service.router

    def attach_engine_frame_router(self, router):
        self.router = router

    def submit_contract(self, engine_request, stop_reasons=()):
        return self.service.submit_contract(engine_request,
                                            stop_reasons=stop_reasons)

    def _start_dispatch_threads(self):
        if self.raise_on_submit:
            raise RuntimeError("submit failed before binding")

    def _get_next_client_id(self):
        self._next_id += 1
        return self._next_id

    def _handle_background_error(self):
        pass

    def abort_request(self, client_id):
        self.aborted.append(client_id)

    def get_stats(self, timeout):
        return [{"iteration": 1}]

    def get_kv_events(self, timeout):
        return ['{"event": 1}']

    def check_health(self):
        return True

    def shutdown(self):
        self.shutdown_called = True


def make_config(**overrides) -> EngineClientConfig:
    kwargs = dict(backend="pytorch", flag_enabled=True)
    kwargs.update(overrides)
    return EngineClientConfig(**kwargs)


def make_client(executor=None) -> LocalProcessEngineClient:
    return LocalProcessEngineClient(executor or FakeExecutor(), make_config())


def make_engine_request(request_id="req-1", **sampling_overrides) -> EngineRequest:
    sampling_kwargs = dict(max_new_tokens=16, end_id=2, pad_id=0)
    sampling_kwargs.update(sampling_overrides)
    return EngineRequest(request_id=request_id, prompt_token_ids=(1, 2, 3),
                         sampling=EngineSamplingConfig(**sampling_kwargs))


def make_result(**overrides) -> SimpleNamespace:
    fields = dict(is_final=False, output_token_ids=[[5, 6]], finish_reasons=None,
                  log_probs=None, cum_log_probs=None, sequence_index=0)
    fields.update(overrides)
    return SimpleNamespace(**fields)


def response_for(client_id, **result_overrides) -> LlmResponse:
    return LlmResponse(request_id=client_id, result=make_result(**result_overrides),
                       client_id=client_id)


def delta_response(client_id, tokens, **overrides):
    return response_for(client_id, output_token_ids=[list(tokens)], **overrides)


def final_response(client_id, reason, tokens=(), **overrides):
    return response_for(client_id, is_final=True, output_token_ids=[list(tokens)],
                        finish_reasons=[reason], **overrides)


async def collect_frames(client, request_id, check_invariants=True):
    stream = client.stream(request_id)
    if check_invariants:
        stream = InvariantCheckingStream(stream, request_id)
    frames = []
    while True:
        try:
            frame = await asyncio.wait_for(stream.__anext__(), STREAM_DEADLINE)
        except StopAsyncIteration:
            break
        frames.append(frame)
        if isinstance(frame, (RequestComplete, ErrorFrame)):
            break
    return frames


def submitted_client_id(executor) -> int:
    return executor.submitted[-1].id


class TestPlainStreaming:

    def test_deltas_terminal_complete(self):
        executor = FakeExecutor()
        client = make_client(executor)
        client.submit(make_engine_request())
        client_id = submitted_client_id(executor)
        router = executor.router
        router.on_response(delta_response(client_id, [5, 6]))
        router.on_response(delta_response(client_id, [7]))
        router.on_response(final_response(client_id, tllm.FinishReason.END_ID))

        frames = asyncio.run(collect_frames(client, "req-1"))
        kinds = [type(f).__name__ for f in frames]
        assert kinds == ["TokenDelta", "TokenDelta", "Terminal", "RequestComplete"]
        assert frames[0].new_token_ids == (5, 6)
        assert frames[2].finish_reason == "stop" and frames[2].stop_reason is None
        complete = frames[3]
        assert complete.status == "ok"
        assert complete.prompt_tokens == 3
        assert complete.completion_tokens == 3
        assert [f.event_seq for f in frames] == [0, 1, 2, 3]

    def test_final_with_last_tokens(self):
        executor = FakeExecutor()
        client = make_client(executor)
        client.submit(make_engine_request())
        client_id = submitted_client_id(executor)
        executor.router.on_response(
            final_response(client_id, tllm.FinishReason.LENGTH, tokens=[9]))
        frames = asyncio.run(collect_frames(client, "req-1"))
        assert [type(f).__name__ for f in frames
                ] == ["TokenDelta", "Terminal", "RequestComplete"]
        assert frames[1].finish_reason == "length"
        assert frames[2].completion_tokens == 1


class TestFinishReasonTable:

    def run_final(self, reason, abort_first=False, stop_reasons=(), tokens=()):
        executor = FakeExecutor()
        client = make_client(executor)
        request = make_engine_request()
        output_config = FrontendOutputConfig(stop_sequence_reasons=stop_reasons)
        client.submit(request, output_config=output_config)
        client_id = submitted_client_id(executor)
        if abort_first:
            client.abort("req-1")
        executor.router.on_response(final_response(client_id, reason, tokens=tokens))
        frames = asyncio.run(collect_frames(client, "req-1"))
        terminal = next(f for f in frames if isinstance(f, Terminal))
        complete = next(f for f in frames if isinstance(f, RequestComplete))
        return terminal, complete

    def test_end_id(self):
        terminal, complete = self.run_final(tllm.FinishReason.END_ID)
        assert (terminal.finish_reason, complete.status) == ("stop", "ok")

    def test_stop_words_resolves_ordered_reason(self):
        stop_reasons = (((13, ), 13), ((7, 8), "STOP"))
        terminal, complete = self.run_final(tllm.FinishReason.STOP_WORDS,
                                            stop_reasons=stop_reasons,
                                            tokens=[5, 7, 8])
        assert terminal.finish_reason == "stop"
        assert terminal.stop_reason == "STOP"
        assert complete.status == "ok"

    def test_length(self):
        terminal, complete = self.run_final(tllm.FinishReason.LENGTH)
        assert (terminal.finish_reason, complete.status) == ("length", "ok")

    def test_cancelled(self):
        terminal, complete = self.run_final(tllm.FinishReason.CANCELLED)
        assert (terminal.finish_reason, complete.status) == ("abort", "aborted")

    def test_timed_out_maps_to_error_timeout(self):
        terminal, complete = self.run_final(tllm.FinishReason.TIMED_OUT)
        assert terminal.finish_reason == "error"
        assert terminal.stop_reason == "timeout"
        assert complete.status == "failed"

    def test_not_finished_without_abort(self):
        terminal, complete = self.run_final(tllm.FinishReason.NOT_FINISHED)
        assert terminal.finish_reason == "error"
        assert terminal.stop_reason == "not_finished"

    def test_not_finished_with_requested_abort(self):
        terminal, complete = self.run_final(tllm.FinishReason.NOT_FINISHED,
                                            abort_first=True)
        assert (terminal.finish_reason, complete.status) == ("abort", "aborted")


class TestAbort:

    def test_abort_before_first_token(self):
        executor = FakeExecutor()
        client = make_client(executor)
        client.submit(make_engine_request())
        client_id = submitted_client_id(executor)
        client.abort("req-1")
        assert executor.aborted == [client_id]
        executor.router.on_response(
            final_response(client_id, tllm.FinishReason.CANCELLED))
        frames = asyncio.run(collect_frames(client, "req-1"))
        assert [type(f).__name__ for f in frames] == ["Terminal", "RequestComplete"]
        assert frames[0].finish_reason == "abort"
        assert frames[1].status == "aborted"
        assert frames[1].completion_tokens == 0

    def test_abort_after_completion_is_noop(self):
        executor = FakeExecutor()
        client = make_client(executor)
        client.submit(make_engine_request())
        client_id = submitted_client_id(executor)
        executor.router.on_response(
            final_response(client_id, tllm.FinishReason.END_ID))
        client.abort("req-1")  # no-op, no error
        assert executor.aborted == []

    def test_abort_unknown_is_typed_error(self):
        client = make_client()
        with pytest.raises(UnknownRequestError):
            client.abort("nope")

    def test_stream_close_aborts_incomplete_request(self):
        executor = FakeExecutor()
        client = make_client(executor)
        client.submit(make_engine_request())
        client_id = submitted_client_id(executor)
        executor.router.on_response(delta_response(client_id, [5]))

        async def scenario():
            stream = client.stream("req-1")
            frame = await asyncio.wait_for(stream.__anext__(), STREAM_DEADLINE)
            assert isinstance(frame, TokenDelta)
            await stream.aclose()

        asyncio.run(scenario())
        assert executor.aborted == [client_id]


class TestLateAndDuplicate:

    def test_late_duplicate_finals_absorbed(self):
        executor = FakeExecutor()
        client = make_client(executor)
        client.submit(make_engine_request())
        client_id = submitted_client_id(executor)
        router = executor.router
        router.on_response(final_response(client_id, tllm.FinishReason.END_ID))
        router.on_response(final_response(client_id, tllm.FinishReason.END_ID))
        router.on_response(delta_response(client_id, [9]))
        assert router.counters["late_or_duplicate_absorbed"] == 2
        frames = asyncio.run(collect_frames(client, "req-1"))
        assert [type(f).__name__ for f in frames] == ["Terminal", "RequestComplete"]

    def test_unbound_legacy_traffic_ignored(self):
        executor = FakeExecutor()
        make_client(executor)
        executor.router.on_response(
            ErrorResponse(client_id=999, error_msg="legacy", request_id=1))
        assert executor.router.active_request_count() == 0


class TestFailurePaths:

    def test_error_response_before_start_is_standalone_error_frame(self):
        executor = FakeExecutor()
        client = make_client(executor)
        client.submit(make_engine_request())
        client_id = submitted_client_id(executor)
        executor.router.on_response(
            ErrorResponse(client_id=client_id, error_msg="admission failed",
                          request_id=1))
        frames = asyncio.run(collect_frames(client, "req-1"))
        assert [type(f).__name__ for f in frames] == ["ErrorFrame"]
        assert frames[0].error_code == "request_error"

    def test_error_after_start_ends_terminal_failed(self):
        executor = FakeExecutor()
        client = make_client(executor)
        client.submit(make_engine_request())
        client_id = submitted_client_id(executor)
        router = executor.router
        router.on_response(delta_response(client_id, [5]))
        router.on_response(
            ErrorResponse(client_id=client_id, error_msg="boom", request_id=1))
        frames = asyncio.run(collect_frames(client, "req-1"))
        assert [type(f).__name__ for f in frames
                ] == ["TokenDelta", "Terminal", "RequestComplete"]
        assert frames[1].finish_reason == "error"
        assert frames[2].status == "failed"

    def test_fail_all_nothing_started_is_error_frame(self):
        executor = FakeExecutor()
        client = make_client(executor)
        client.submit(make_engine_request())
        executor.router.fail_all("worker crashed")
        frames = asyncio.run(collect_frames(client, "req-1"))
        assert [type(f).__name__ for f in frames] == ["ErrorFrame"]
        assert frames[0].error_code == "executor_failed"

    def test_fail_all_started_synthesizes_terminal(self):
        executor = FakeExecutor()
        client = make_client(executor)
        client.submit(make_engine_request())
        client_id = submitted_client_id(executor)
        executor.router.on_response(delta_response(client_id, [5]))
        executor.router.fail_all("worker crashed")
        frames = asyncio.run(collect_frames(client, "req-1"))
        assert [type(f).__name__ for f in frames
                ] == ["TokenDelta", "Terminal", "RequestComplete"]
        assert frames[2].status == "failed"
        assert executor.router.counters["synthesized_terminals"] == 1

    def test_fatal_error_latch_blocks_new_submits(self):
        executor = FakeExecutor()
        client = make_client(executor)
        executor.router.fail_all("fatal")
        assert not client.health().healthy
        with pytest.raises(RequestRejectedError):
            client.submit(make_engine_request(request_id="req-2"))

    def test_enqueue_failure_after_binding(self):
        executor = FakeExecutor()
        executor.raise_after_observe = True
        client = make_client(executor)
        with pytest.raises(RuntimeError):
            client.submit(make_engine_request())
        frames = asyncio.run(collect_frames(client, "req-1"))
        assert [type(f).__name__ for f in frames] == ["ErrorFrame"]
        assert frames[0].error_code == "enqueue_failed"

    def test_submit_failure_before_binding_rolls_back(self):
        executor = FakeExecutor()
        executor.raise_on_submit = True
        client = make_client(executor)
        with pytest.raises(RuntimeError):
            client.submit(make_engine_request())
        assert executor.router.active_request_count() == 0
        # The id is reusable: the registration was rolled back entirely.
        executor.raise_on_submit = False
        client.submit(make_engine_request())


class TestStreamLifecycle:

    def test_submit_without_stream_then_delayed_open(self):
        executor = FakeExecutor()
        client = make_client(executor)
        client.submit(make_engine_request())
        client_id = submitted_client_id(executor)
        router = executor.router
        router.on_response(delta_response(client_id, [5]))
        router.on_response(final_response(client_id, tllm.FinishReason.END_ID))
        # Open only after the request fully ended: nothing is lost.
        frames = asyncio.run(collect_frames(client, "req-1"))
        assert [type(f).__name__ for f in frames
                ] == ["TokenDelta", "Terminal", "RequestComplete"]

    def test_double_open_is_typed_error(self):
        executor = FakeExecutor()
        client = make_client(executor)
        client.submit(make_engine_request())
        client.stream("req-1")
        with pytest.raises(RequestRejectedError):
            client.stream("req-1")

    def test_unknown_stream_is_typed_error(self):
        client = make_client()
        with pytest.raises(RequestRejectedError):
            client.stream("ghost")

    def test_request_id_reuse_after_completion_rejected(self):
        executor = FakeExecutor()
        client = make_client(executor)
        client.submit(make_engine_request())
        client_id = submitted_client_id(executor)
        executor.router.on_response(
            final_response(client_id, tllm.FinishReason.END_ID))
        with pytest.raises(RequestRejectedError):
            client.submit(make_engine_request())

    def test_duplicate_active_id_rejected(self):
        executor = FakeExecutor()
        client = make_client(executor)
        client.submit(make_engine_request())
        with pytest.raises(RequestRejectedError):
            client.submit(make_engine_request())


class TestOverflow:

    def test_overflow_ends_typed_via_out_of_band_path(self):
        executor = FakeExecutor()
        router = EngineFrameRouter(abort_fn=executor.abort_request,
                                   delivery_limit=2)
        executor.attach_engine_frame_router(router)
        generation_request = SimpleNamespace(id=None)

        def set_id(value):
            generation_request.id = value
            return generation_request

        generation_request.set_id = set_id
        binding = router.register_pending(generation_request, "req-o", (1, 2), ())
        generation_request.set_id(41)
        router.observe_submit(generation_request)
        for i in range(4):
            router.on_response(delta_response(41, [i]))
        assert binding.delivery.overflowed or binding.ended
        assert router.counters["overflow_aborts"] == 1

        async def drain():
            frames = []
            stream_binding = router.open_stream_binding("req-o")
            while True:
                frame, ready = stream_binding.delivery.pop_nowait()
                if not ready:
                    break
                frames.append(frame)
            return frames

        frames = asyncio.run(drain())
        assert isinstance(frames[-1], RequestComplete)
        assert frames[-1].status == "failed"
        terminal = next(f for f in frames if isinstance(f, Terminal))
        assert terminal.stop_reason == "delivery_overflow"


class TestPromptLogprobsAndMetrics:

    def test_held_prompt_logprobs_attach_to_next_delta(self):
        executor = FakeExecutor()
        client = make_client(executor)
        client.submit(make_engine_request())
        client_id = submitted_client_id(executor)
        router = executor.router
        # A no-token response carrying prompt logprobs is held...
        wrapper = ResponseWrapper(
            response_for(client_id, output_token_ids=[[]]),
            logprobs=LogProbsResult(prompt=[-1.0, -2.0], generation=None))
        router.on_response(wrapper)
        # ...and attached to the next token-carrying delta, exactly once.
        router.on_response(delta_response(client_id, [5]))
        router.on_response(delta_response(client_id, [6]))
        router.on_response(final_response(client_id, tllm.FinishReason.END_ID))
        frames = asyncio.run(collect_frames(client, "req-1"))
        deltas = [f for f in frames if isinstance(f, TokenDelta)]
        assert deltas[0].prompt_logprobs == (-1.0, -2.0)
        assert deltas[1].prompt_logprobs is None

    def test_cached_tokens_flow(self):
        executor = FakeExecutor()
        client = make_client(executor)
        client.submit(make_engine_request())
        client_id = submitted_client_id(executor)
        router = executor.router
        router.on_response(delta_response(client_id, [5], cached_tokens=2))
        router.on_response(final_response(client_id, tllm.FinishReason.END_ID))
        frames = asyncio.run(collect_frames(client, "req-1"))
        delta = frames[0]
        assert delta.metrics["cached_tokens"] == 2.0
        assert frames[-1].cached_tokens == 2


class TestSetupGates:

    @pytest.mark.parametrize("overrides,fragment", [
        (dict(flag_enabled=False), "flag"),
        (dict(backend="_autodeploy"), "backend"),
        (dict(backend="tensorrt"), "backend"),
        (dict(transport="rpc"), "transport"),
        (dict(num_postprocess_workers=2), "postproc"),
        (dict(post_processor_hook_set=True), "post_processor_hook"),
        (dict(speculative_config_set=True), "speculative"),
        (dict(early_first_token_mode=True), "early_first_token"),
        (dict(world_size=2), "topology"),
        (dict(tokenizer_trust_remote_code=True), "trust_remote_code"),
    ])
    def test_config_gate(self, overrides, fragment):
        with pytest.raises(EngineClientConfigError) as excinfo:
            LocalProcessEngineClient(FakeExecutor(), make_config(**overrides))
        assert fragment in str(excinfo.value)

    def test_flag_from_env(self, monkeypatch):
        monkeypatch.delenv("TLLM_EXPERIMENTAL_ENGINE_CLIENT", raising=False)
        with pytest.raises(EngineClientConfigError):
            LocalProcessEngineClient(FakeExecutor(), make_config(flag_enabled=None))
        monkeypatch.setenv("TLLM_EXPERIMENTAL_ENGINE_CLIENT", "1")
        LocalProcessEngineClient(FakeExecutor(), make_config(flag_enabled=None))

    def test_executor_without_hooks_rejected(self):
        with pytest.raises(EngineClientConfigError):
            LocalProcessEngineClient(SimpleNamespace(), make_config())


class TestPreSubmitChecks:

    def test_newer_protocol_rejected(self):
        client = make_client()
        request = make_engine_request()
        object.__setattr__(request, "protocol_version", 99)
        with pytest.raises(RequestRejectedError):
            client.submit(request)

    def test_required_features_rederived_not_trusted(self):
        client = make_client()
        request = make_engine_request()
        object.__setattr__(request, "required_features", ("guided_decoding", ))
        with pytest.raises(RequestRejectedError):
            client.submit(request)

    def test_capability_gated_feature_rejected_pre_submit(self):
        client = make_client()
        request = EngineRequest(
            request_id="req-g", prompt_token_ids=(1, ),
            sampling=EngineSamplingConfig(max_new_tokens=4, end_id=2),
            guided_decoding=GuidedDecodingSpec(mode="json_object"),
            required_features=("guided_decoding", ))
        with pytest.raises(RequestRejectedError) as excinfo:
            client.submit(request)
        assert "guided_decoding" in str(excinfo.value)


class TestControlPlane:

    def test_typed_delegation(self):
        client = make_client()
        stats = client.get_stats()
        assert stats.entries and stats.entries[0].startswith("{")
        events = client.get_kv_events()
        assert events.entries == ('{"event": 1}', )
        assert client.health().healthy

    def test_close_client_vs_shutdown_engine(self):
        executor = FakeExecutor()
        client = make_client(executor)
        client.submit(make_engine_request())
        client.close_client()
        assert not executor.shutdown_called
        frames = asyncio.run(collect_frames(client, "req-1"))
        assert isinstance(frames[-1], ErrorFrame)
        executor2 = FakeExecutor()
        client2 = make_client(executor2)
        client2.shutdown_engine()
        assert executor2.shutdown_called


class TestInvariantChecker:

    def test_detects_missing_ending(self):

        async def scenario():
            async def frames():
                yield TokenDelta(request_id="r", sequence_id=0, new_token_ids=(1, ),
                                 event_seq=0)

            stream = InvariantCheckingStream(frames(), "r")
            with pytest.raises(StreamInvariantViolation):
                async for _ in stream:
                    pass

        asyncio.run(scenario())

    def test_detects_frames_after_ending(self):

        async def scenario():
            async def frames():
                yield ErrorFrame(request_id="r", error_code="x", event_seq=0)
                yield TokenDelta(request_id="r", sequence_id=0, new_token_ids=(1, ),
                                 event_seq=1)

            stream = InvariantCheckingStream(frames(), "r")
            with pytest.raises(StreamInvariantViolation):
                async for _ in stream:
                    pass

        asyncio.run(scenario())


class TestStressBaseline:

    def test_concurrent_requests_return_to_baseline(self):
        executor = FakeExecutor()
        client = make_client(executor)
        router = executor.router
        request_count = 40

        for i in range(request_count):
            client.submit(make_engine_request(request_id=f"req-{i}"))

        def respond(request):
            client_id = request.id
            router.on_response(delta_response(client_id, [5, 6]))
            router.on_response(final_response(client_id, tllm.FinishReason.END_ID))

        threads = [
            threading.Thread(target=respond, args=(request, ))
            for request in executor.submitted
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        async def consume_all():
            for i in range(request_count):
                frames = await collect_frames(client, f"req-{i}")
                assert isinstance(frames[-1], RequestComplete)

        asyncio.run(consume_all())
        assert router.active_request_count() == 0

    def test_tombstone_eviction_closes_deliveries(self):
        executor = FakeExecutor()
        router = EngineFrameRouter(abort_fn=executor.abort_request,
                                   tombstone_limit=4)
        executor.attach_engine_frame_router(router)
        bindings = []
        for i in range(8):
            request = SimpleNamespace(id=None)
            request.set_id = lambda v, r=request: setattr(r, "id", v) or r
            binding = router.register_pending(request, f"req-{i}", (1, ), ())
            request.set_id(100 + i)
            router.observe_submit(request)
            router.on_response(final_response(100 + i, tllm.FinishReason.END_ID))
            bindings.append(binding)
        # Oldest deliveries were evicted alongside their tombstones.
        assert bindings[0].delivery.closed
        assert not bindings[-1].delivery.closed
        assert DEFAULT_TOMBSTONE_LIMIT >= 4
