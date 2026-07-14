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
"""Proxy hook and worker prompt-logprob-guard tests.

These tests exercise the engine-client hooks on a skeleton
GenerationExecutorProxy (no MPI workers) to prove the hooks are inert with
no router attached, observe every raw item before the legacy drop logic,
and contain router failures.
"""

from queue import Queue
from types import SimpleNamespace

import pytest

from tensorrt_llm.executor.base_worker import _compute_pytorch_prompt_logprobs
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.utils import ErrorResponse


class RecordingRouter:

    def __init__(self, raise_on_response=False):
        self.observed_submits = []
        self.observed_responses = []
        self.failed_submits = []
        self.fail_all_calls = []
        self.raise_on_response = raise_on_response

    def observe_submit(self, request):
        self.observed_submits.append(request)

    def on_response(self, response):
        self.observed_responses.append(response)
        if self.raise_on_response:
            raise RuntimeError("router bug")

    def on_submit_enqueue_failed(self, client_id):
        self.failed_submits.append(client_id)

    def fail_all(self, reason):
        self.fail_all_calls.append(reason)


class FakeResultQueue:

    def __init__(self, items):
        self._items = list(items)

    def get(self):
        return self._items.pop(0) if self._items else None


def make_skeleton_proxy() -> GenerationExecutorProxy:
    proxy = object.__new__(GenerationExecutorProxy)
    proxy._results = {}
    return proxy


def make_legacy_result() -> SimpleNamespace:
    return SimpleNamespace(queue=Queue())


class TestDispatchHook:

    def run_dispatch(self, proxy, items):
        proxy.result_queue = FakeResultQueue([items])
        return proxy.dispatch_result_task()

    def test_inert_without_router(self):
        proxy = make_skeleton_proxy()
        result = make_legacy_result()
        proxy._results = {3: result}
        response = ErrorResponse(client_id=3, error_msg="x", request_id=1)
        assert self.run_dispatch(proxy, [response]) is True
        # Legacy behavior: delivered then popped on final/error.
        assert result.queue.get_nowait() is response
        assert 3 not in proxy._results

    def test_router_sees_items_before_legacy_drop(self):
        proxy = make_skeleton_proxy()
        router = RecordingRouter()
        proxy.attach_engine_frame_router(router)
        # Unknown client id: the legacy path drops it silently, the router
        # must still observe it (late/duplicate frame absorption).
        unknown = ErrorResponse(client_id=99, error_msg="late", request_id=1)
        assert self.run_dispatch(proxy, [unknown]) is True
        assert router.observed_responses == [unknown]

    def test_router_sees_batched_list_items(self):
        proxy = make_skeleton_proxy()
        router = RecordingRouter()
        proxy.attach_engine_frame_router(router)
        items = [ErrorResponse(client_id=1, error_msg="a", request_id=1),
                 ErrorResponse(client_id=2, error_msg="b", request_id=2)]
        assert self.run_dispatch(proxy, items) is True
        assert router.observed_responses == items

    def test_router_exception_does_not_break_dispatch(self):
        proxy = make_skeleton_proxy()
        router = RecordingRouter(raise_on_response=True)
        proxy.attach_engine_frame_router(router)
        result = make_legacy_result()
        proxy._results = {3: result}
        response = ErrorResponse(client_id=3, error_msg="x", request_id=1)
        assert self.run_dispatch(proxy, [response]) is True
        # Legacy delivery still happened despite the router raising.
        assert result.queue.get_nowait() is response


class TestSubmitHook:

    class _Stub:
        """Provides bound methods (WeakMethod-compatible) for the skeleton."""

        def handle_background_error(self, *args, **kwargs):
            return None

        def start_dispatch_threads(self):
            return None

        def get_logprob_params(self, request):
            return None

    def make_submit_proxy(self, put_raises=False):
        proxy = make_skeleton_proxy()
        proxy._last_client_id = 0
        proxy.dispatch_result_thread = object()  # skip thread start
        proxy._error_queue = Queue()
        proxy.doing_shutdown = False
        stub = self._Stub()
        proxy._stub_ref = stub  # keep alive for WeakMethod
        proxy._handle_background_error = stub.handle_background_error
        proxy._start_dispatch_threads = stub.start_dispatch_threads
        proxy._get_logprob_params = stub.get_logprob_params

        class FakeRequestQueue:

            def __init__(self):
                self.items = []

            def put(self, item):
                if put_raises:
                    raise RuntimeError("zmq down")
                self.items.append(item)

        proxy.request_queue = FakeRequestQueue()
        return proxy

    def make_request(self):
        from tensorrt_llm.executor.request import GenerationRequest
        from tensorrt_llm.sampling_params import SamplingParams
        return GenerationRequest(prompt_token_ids=[1, 2],
                                 sampling_params=SamplingParams(max_tokens=4, end_id=2),
                                 streaming=True)

    def test_observe_submit_binds_before_enqueue(self):
        proxy = self.make_submit_proxy()
        router = RecordingRouter()
        proxy.attach_engine_frame_router(router)
        request = self.make_request()
        proxy.submit(request)
        assert router.observed_submits == [request]
        assert request.id is not None
        assert proxy.request_queue.items == [request]

    def test_enqueue_failure_rolls_back(self):
        proxy = self.make_submit_proxy(put_raises=True)
        router = RecordingRouter()
        proxy.attach_engine_frame_router(router)
        request = self.make_request()
        with pytest.raises(RuntimeError):
            proxy.submit(request)
        assert router.failed_submits == [request.id]

    def test_submit_without_router_unchanged(self):
        proxy = self.make_submit_proxy()
        request = self.make_request()
        result = proxy.submit(request)
        assert proxy.request_queue.items == [request]
        assert proxy._results[request.id] is result


class TestPromptLogprobGuard:

    def make_generation_result(self, prompt_ids=(1, 2, 3)):
        return SimpleNamespace(
            _logprob_params=SimpleNamespace(prompt_logprobs=0,
                                            prompt_logprobs_simple_format=True),
            _streaming=True,
            _generation_request=SimpleNamespace(prompt_token_ids=list(prompt_ids)))

    def make_response(self, output_token_ids, context_logits):
        result = SimpleNamespace(output_token_ids=output_token_ids)
        inner = SimpleNamespace(context_logits=context_logits,
                                get_result=lambda: result)
        return SimpleNamespace(result=inner, client_id=5)

    def test_no_token_final_degrades_gracefully(self):
        generation_result = self.make_generation_result()
        response = self.make_response(output_token_ids=[[]], context_logits=None)
        assert _compute_pytorch_prompt_logprobs(generation_result, response) is None

    def test_missing_context_logits_degrades_gracefully(self):
        generation_result = self.make_generation_result()
        response = self.make_response(output_token_ids=[[5]], context_logits=None)
        assert _compute_pytorch_prompt_logprobs(generation_result, response) is None

    def test_cached_prompt_logprobs_short_circuit_still_works(self):
        generation_result = self.make_generation_result()
        generation_result._cached_prompt_logprobs = [-1.0]
        response = self.make_response(output_token_ids=[[]], context_logits=None)
        logprobs_result = _compute_pytorch_prompt_logprobs(generation_result, response)
        assert logprobs_result is not None
        assert logprobs_result.prompt == [-1.0]
