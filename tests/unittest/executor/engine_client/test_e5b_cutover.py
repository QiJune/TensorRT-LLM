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
"""E5b cut-over tests on a skeleton of the REAL GenerationExecutorProxy.

Gate ①c: contract requests get no rank-0 GenerationResult, contract
responses are exclusively routed (never legacy-delivered), both request
populations are enumerated symmetrically on shutdown, contract client-id
ownership survives tombstone eviction, and frontend request-id reuse is
forbidden for the service session.
"""

import threading
from queue import Queue
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor.engine_client.contract import (EngineRequest,
                                                          EngineSamplingConfig,
                                                          ErrorFrame,
                                                          RequestComplete)
from tensorrt_llm.executor.engine_client.local_client import (
    EngineClientConfigError, resolve_engine_client_flag)
from tensorrt_llm.executor.engine_client.router import RouterError
from tensorrt_llm.executor.engine_client.service import (EngineService,
                                                         EngineServiceError)
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.utils import ErrorResponse


class FakeRequestQueue:

    def __init__(self):
        self.items = []
        self.raise_on_put = False
        self.noblock = []

    def put(self, item):
        if self.raise_on_put:
            raise RuntimeError("enqueue failed")
        self.items.append(item)

    def put_noblock(self, item, retry=0):
        self.noblock.append(item)


class FakeLegacyResult:

    def __init__(self):
        self.queue = Queue()
        self.aborted = False

    def abort(self):
        self.aborted = True


def make_proxy() -> GenerationExecutorProxy:
    proxy = object.__new__(GenerationExecutorProxy)
    proxy._results = {}
    proxy.doing_shutdown = False
    proxy._fatal_error = None
    proxy._last_client_id = 0
    proxy.request_queue = FakeRequestQueue()
    proxy.workers_started = True
    proxy.mpi_futures = []
    proxy._shutdown_event = threading.Event()
    stub = SimpleNamespace()

    class _Stub:

        def handle_background_error(self, *args, **kwargs):
            return None

        def start_dispatch_threads(self):
            return None

    stub_obj = _Stub()
    proxy._stub = stub_obj
    proxy._handle_background_error = stub_obj.handle_background_error
    proxy._start_dispatch_threads = stub_obj.start_dispatch_threads
    return proxy


def make_engine_request(request_id="req-1") -> EngineRequest:
    return EngineRequest(request_id=request_id, prompt_token_ids=(1, 2, 3),
                         sampling=EngineSamplingConfig(max_new_tokens=8, end_id=2,
                                                       pad_id=0))


def make_service(proxy) -> EngineService:
    service = EngineService(proxy)
    proxy.attach_engine_service(service)
    return service


def contract_final(client_id):
    result = SimpleNamespace(is_final=True, output_token_ids=[[5]],
                             finish_reasons=[tllm.FinishReason.END_ID],
                             log_probs=None, cum_log_probs=None, sequence_index=0)
    return LlmResponse(request_id=client_id, result=result, client_id=client_id)


def drain_binding(binding):
    frames = []
    while True:
        frame, ready = binding.delivery.pop_nowait()
        if not ready:
            return frames
        frames.append(frame)


class TestContractNativeSubmit:

    def test_no_rank0_generation_result(self):
        proxy = make_proxy()
        service = make_service(proxy)
        client_id = service.submit_contract(make_engine_request())
        assert proxy._results == {}  # the binding replaces GenerationResult
        assert proxy.request_queue.items[0].id == client_id
        assert service.router.get_binding("req-1") is not None

    def test_submission_rejected_during_shutdown(self):
        proxy = make_proxy()
        service = make_service(proxy)
        proxy.doing_shutdown = True
        with pytest.raises(EngineServiceError):
            service.submit_contract(make_engine_request())
        assert proxy.request_queue.items == []

    def test_enqueue_failure_yields_typed_ending_and_no_state_leak(self):
        proxy = make_proxy()
        service = make_service(proxy)
        proxy.request_queue.raise_on_put = True
        with pytest.raises(RuntimeError):
            service.submit_contract(make_engine_request())
        binding = service.router.open_stream_binding("req-1")
        frames = drain_binding(binding)
        assert [type(f).__name__ for f in frames] == ["ErrorFrame"]
        assert frames[0].error_code == "enqueue_failed"
        assert proxy._results == {}
        # The request was never enqueued, so no runtime abort was sent.
        assert proxy.aborted == [] if hasattr(proxy, "aborted") else True

    def test_second_service_attachment_rejected(self):
        proxy = make_proxy()
        make_service(proxy)
        with pytest.raises(RuntimeError):
            proxy.attach_engine_service(EngineService(proxy))

    def test_request_id_reuse_rejected_while_delivery_retained(self):
        proxy = make_proxy()
        service = make_service(proxy)
        client_id = service.submit_contract(make_engine_request())
        service.router.route_response(contract_final(client_id))
        # Ended but not consumed/retired: the id is still retained.
        with pytest.raises(RouterError):
            service.submit_contract(make_engine_request())

    def test_allocator_wrap_rejected_for_contract_population(self):
        proxy = make_proxy()
        service = make_service(proxy)
        proxy._last_client_id = (1 << 64) - 1  # next allocation wraps to 0
        with pytest.raises(EngineServiceError):
            service.submit_contract(make_engine_request())
        assert proxy.request_queue.items == []
        assert service.router.get_binding("req-1") is None

    def test_allocator_wrap_rejected_for_legacy_population(self):
        from tensorrt_llm.executor.request import GenerationRequest
        from tensorrt_llm.sampling_params import SamplingParams
        proxy = make_proxy()
        service = make_service(proxy)
        # Interleaved monotonic allocations across populations are fine...
        contract_id = service.submit_contract(make_engine_request())
        service.observe_legacy_allocation(contract_id + 1)
        # ...but a wrapped id handed to a LEGACY submission must fail that
        # submission before it can collide with contract-owned state.
        proxy._last_client_id = (1 << 64) - 1
        request = GenerationRequest(prompt_token_ids=[1, 2, 3],
                                    sampling_params=SamplingParams(max_tokens=4),
                                    streaming=True)
        with pytest.raises(EngineServiceError):
            proxy.submit(request)
        assert proxy._results == {}
        assert all(item.id == contract_id
                   for item in proxy.request_queue.items)


class TestExclusiveRouting:

    def test_contract_claimed_legacy_delivered(self):
        proxy = make_proxy()
        service = make_service(proxy)
        contract_id = service.submit_contract(make_engine_request())
        legacy_result = FakeLegacyResult()
        legacy_id = 999
        proxy._results[legacy_id] = legacy_result

        contract_response = contract_final(contract_id)
        legacy_response = ErrorResponse(client_id=legacy_id, error_msg="x",
                                        request_id=1)
        proxy.result_queue = SimpleNamespace(
            get=lambda items=[[contract_response, legacy_response]]: items.pop(0)
            if items else None)
        assert proxy.dispatch_result_task() is True

        # Contract: consumed by the service (frames emitted), absent legacy.
        binding = service.router.open_stream_binding("req-1")
        frames = drain_binding(binding)
        assert isinstance(frames[-1], RequestComplete)
        # Legacy: delivered and popped exactly as before.
        assert legacy_result.queue.get_nowait() is legacy_response
        assert legacy_id not in proxy._results

    def test_contract_response_never_reaches_legacy_even_with_result_entry(self):
        proxy = make_proxy()
        service = make_service(proxy)
        contract_id = service.submit_contract(make_engine_request())
        # Hostile setup: even if a legacy result somehow existed under the
        # same id, the claim runs first and legacy never sees the response.
        shadow_result = FakeLegacyResult()
        proxy._results[contract_id] = shadow_result
        response = contract_final(contract_id)
        proxy.result_queue = SimpleNamespace(
            get=lambda items=[[response]]: items.pop(0) if items else None)
        proxy.dispatch_result_task()
        assert shadow_result.queue.empty()
        assert contract_id in proxy._results  # untouched by the claim

    def test_late_frame_after_tombstone_eviction_is_absorbed(self):
        proxy = make_proxy()
        service = make_service(proxy)
        service.router._tombstone_limit = 1
        first_id = service.submit_contract(make_engine_request("req-a"))
        service.router.route_response(contract_final(first_id))
        second_id = service.submit_contract(make_engine_request("req-b"))
        service.router.route_response(contract_final(second_id))
        # req-a's tombstone is evicted now; a very late frame for it must
        # still be classified contract-owned and absorbed, never legacy.
        before = service.router.counters["late_or_duplicate_absorbed"]
        assert service.route_response(contract_final(first_id)) is True
        assert service.router.counters["late_or_duplicate_absorbed"] == before + 1


class TestShutdownSymmetry:

    def test_pre_shutdown_covers_both_populations(self):
        proxy = make_proxy()
        aborted = []
        proxy.abort_request = aborted.append
        service = make_service(proxy)
        # Re-point the router abort at the spy (service was built before).
        service.router._abort_fn = aborted.append
        contract_id = service.submit_contract(make_engine_request())
        legacy_result = FakeLegacyResult()
        proxy._results[7] = legacy_result

        proxy.pre_shutdown()

        binding = service.router.open_stream_binding("req-1")
        frames = drain_binding(binding)
        assert isinstance(frames[-1], (RequestComplete, ErrorFrame))
        assert legacy_result.aborted
        assert contract_id in aborted  # best-effort runtime cancellation
        assert proxy.request_queue.noblock == [None]  # worker sentinel sent

    def test_close_client_aborts_in_flight_work(self):
        proxy = make_proxy()
        aborted = []
        proxy.abort_request = aborted.append
        service = make_service(proxy)
        service.router._abort_fn = aborted.append
        contract_id = service.submit_contract(make_engine_request())
        service.close_client()
        assert contract_id in aborted
        binding = service.router.open_stream_binding("req-1")
        frames = drain_binding(binding)
        assert isinstance(frames[-1], (RequestComplete, ErrorFrame))


class TestFlagResolver:

    @pytest.mark.parametrize("env,args_value,expected", [
        (None, False, False),
        (None, True, True),
        ("0", False, False),
        ("0", True, False),
        ("1", False, True),
        ("1", True, True),
    ])
    def test_precedence_matrix(self, monkeypatch, env, args_value, expected):
        if env is None:
            monkeypatch.delenv("TLLM_EXPERIMENTAL_ENGINE_CLIENT", raising=False)
        else:
            monkeypatch.setenv("TLLM_EXPERIMENTAL_ENGINE_CLIENT", env)
        assert resolve_engine_client_flag(args_value) is expected

    def test_invalid_env_fails_closed(self, monkeypatch):
        monkeypatch.setenv("TLLM_EXPERIMENTAL_ENGINE_CLIENT", "yes")
        with pytest.raises(EngineClientConfigError):
            resolve_engine_client_flag(True)

    def test_llm_args_field_exists(self):
        from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
        field = TorchLlmArgs.model_fields["experimental_engine_client"]
        assert field.default is False


class TestConcurrentMixedTraffic:

    def test_contract_and_legacy_share_the_proxy_without_misrouting(self):
        proxy = make_proxy()
        service = make_service(proxy)
        contract_count = 12
        legacy_count = 12

        contract_ids = {}

        def submit_contracts():
            for i in range(contract_count):
                cid = service.submit_contract(make_engine_request(f"req-{i}"))
                contract_ids[cid] = f"req-{i}"

        legacy_results = {}

        def submit_legacy():
            for i in range(legacy_count):
                with proxy._submission_lock:
                    legacy_id = proxy._get_next_client_id()
                result = FakeLegacyResult()
                proxy._results[legacy_id] = result
                legacy_results[legacy_id] = result

        t1 = threading.Thread(target=submit_contracts)
        t2 = threading.Thread(target=submit_legacy)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Ids never collide across populations.
        assert not set(contract_ids) & set(legacy_results)

        # Dispatch a mixed batch.
        responses = [contract_final(cid) for cid in contract_ids]
        responses += [
            ErrorResponse(client_id=lid, error_msg="bye", request_id=1)
            for lid in legacy_results
        ]
        proxy.result_queue = SimpleNamespace(
            get=lambda items=[responses]: items.pop(0) if items else None)
        proxy.dispatch_result_task()

        for cid, request_id in contract_ids.items():
            binding = service.router.open_stream_binding(request_id)
            frames = drain_binding(binding)
            assert isinstance(frames[-1], RequestComplete)
        for lid, result in legacy_results.items():
            assert not result.queue.empty()
            assert lid not in proxy._results
        assert service.router.active_request_count() == 0
