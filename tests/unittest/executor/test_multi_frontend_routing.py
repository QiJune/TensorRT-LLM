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
"""CPU-only tests for classic-path multi-frontend serving.

Multi-frontend serving (num_serve_frontends) on the classic IPC
executor path. Covers client-id namespacing, worker-side response-lane routing, and the
attached-frontend proxy lifecycle over real ipc:// sockets.
"""

import os
import tempfile
import time
from types import SimpleNamespace

import pytest
from test_common.executor.fakes import executor_requests, fake_rank0

from tensorrt_llm.executor.utils import (
    FRONTEND_COUNTER_MASK,
    MAX_NUM_FRONTENDS,
    bucket_responses_by_frontend,
    frontend_lane_index,
    get_frontend_id,
    namespace_client_id,
)

pytestmark = pytest.mark.cpu_only


class TestClientIdNamespacing:
    def test_frontend_zero_keeps_legacy_ids(self):
        for client_id in (1, 42, FRONTEND_COUNTER_MASK):
            assert namespace_client_id(0, client_id) == client_id

    def test_roundtrip(self):
        for frontend_id in (0, 1, 7, MAX_NUM_FRONTENDS - 1):
            for counter in (1, 12345, FRONTEND_COUNTER_MASK):
                client_id = namespace_client_id(frontend_id, counter)
                assert get_frontend_id(client_id) == frontend_id
                assert client_id & FRONTEND_COUNTER_MASK == counter
                # Bit 63 stays clear: ids remain positive as signed int64.
                assert 0 < client_id < (1 << 63)

    def test_knob_cap_matches_encoding(self):
        from tensorrt_llm.llmapi.llm_args import BaseLlmArgs

        meta = BaseLlmArgs.model_fields["num_serve_frontends"].metadata
        assert any(getattr(m, "le", None) == MAX_NUM_FRONTENDS for m in meta)

    def test_counter_wraparound_stays_in_namespace(self):
        # A counter overflowing its field must not leak into the frontend bits.
        client_id = namespace_client_id(3, FRONTEND_COUNTER_MASK + 5)
        assert get_frontend_id(client_id) == 3
        assert client_id & FRONTEND_COUNTER_MASK == 4

    def test_non_int_client_id_routes_to_launcher(self):
        assert get_frontend_id(None) == 0

    def test_lane_index_clamps_to_launcher(self):
        assert frontend_lane_index(namespace_client_id(2, 7), 3) == 2
        assert frontend_lane_index(namespace_client_id(7, 1), 2) == 0
        assert frontend_lane_index(None, 4) == 0


def _response(client_id):
    return SimpleNamespace(client_id=client_id)


class TestClassicResponseBucketing:
    """The classic IPC path routes ErrorResponse batches with bucket_responses_by_frontend."""

    def test_buckets_by_namespace(self):
        responses = [
            _response(namespace_client_id(0, 1)),
            _response(namespace_client_id(1, 1)),
            _response(namespace_client_id(1, 2)),
            _response(namespace_client_id(2, 1)),
        ]
        buckets = bucket_responses_by_frontend(responses, 3)
        assert [len(b) for b in buckets] == [1, 2, 1]
        assert all(get_frontend_id(r.client_id) == 1 for r in buckets[1])

    def test_none_client_id_routes_to_launcher(self):
        # ADP dummy responses carry client_id=None: they must land in the
        # launcher's lane (which silently discards them), not raise.
        buckets = bucket_responses_by_frontend([_response(None)], 4)
        assert len(buckets[0]) == 1
        assert all(not b for b in buckets[1:])

    def test_out_of_range_frontend_routes_to_launcher(self):
        buckets = bucket_responses_by_frontend([_response(namespace_client_id(7, 1))], 2)
        assert len(buckets[0]) == 1 and not buckets[1]


class TestClassicLaneRouting:
    """Rank 0 sends each response down the lane of the frontend its client id names.

    engine_server.publish puts a batch's rows in one StepFrame per frontend lane;
    AwaitResponseHelper.handle_for_ipc_batched sends the ErrorResponses no request
    entry waits for in one list per lane.
    """

    @staticmethod
    def _submit(worker, client_id, **sampling):
        """Submit a request to rank 0 as the frontend ``client_id`` names sends it."""
        from tensorrt_llm.executor import engine_server
        from tensorrt_llm.executor.request import GenerationRequest
        from tensorrt_llm.llmapi.engine_client import proto_wire
        from tensorrt_llm.sampling_params import SamplingParams

        request = GenerationRequest([1, 2, 3], SamplingParams(**sampling), streaming=True)
        request.set_id(client_id)
        engine_server.submit(worker, proto_wire.encode(request, postprocess=False))
        return request

    def test_rows_route_to_origin_lane(self):
        from tensorrt_llm._torch.pyexecutor.response_utils import prepare_response
        from tensorrt_llm.executor.wire import StepFrame
        from tensorrt_llm.llmapi.engine_client import proto_wire
        from tensorrt_llm.llmapi.engine_client.result import GenerationResult

        worker = fake_rank0(lanes=3)
        # Frontend 7 has no lane among three, so its rows go to the launcher's.
        client_ids_by_lane = {
            0: [namespace_client_id(0, 1), namespace_client_id(7, 1)],
            1: [namespace_client_id(1, 1), namespace_client_id(1, 2)],
            2: [namespace_client_id(2, 1)],
        }
        results = {}
        for client_ids in client_ids_by_lane.values():
            for client_id in client_ids:
                results[client_id] = GenerationResult(self._submit(worker, client_id))

        # every request's first token
        responses = []
        for (request,) in executor_requests(worker.engine.enqueued):
            request.add_new_token(5, 0)
            responses.append(prepare_response(request))
        worker._await_response_helper.handle_for_ipc_batched(responses)

        for lane, client_ids in client_ids_by_lane.items():
            # One StepFrame per lane for the whole batch.
            ((frame,),) = worker.frontend_result_queues[lane].items
            assert isinstance(frame, StepFrame)
            rows = proto_wire.decode(frame, results)
            assert sorted(row.client_id for row in rows) == sorted(client_ids)
        # The requests go on, so rank 0 keeps their entries.
        assert worker.popped == []
        assert worker._results.keys() == results.keys()

    def test_error_response_routes_to_origin_lane(self):
        from tensorrt_llm.executor.utils import ErrorResponse

        worker = fake_rank0(lanes=3)
        client_id = namespace_client_id(2, 7)
        # Rank 0 refuses the request, so no request entry waits for its ErrorResponse.
        self._submit(worker, client_id, max_tokens=0)
        assert not worker._results

        worker._await_response_helper.process_and_handle_responses([])

        assert [len(q.items) for q in worker.frontend_result_queues] == [0, 0, 1]
        ((error,),) = worker.frontend_result_queues[2].items
        assert isinstance(error, ErrorResponse)
        assert error.client_id == client_id
        assert worker.popped == [client_id]

    def test_none_client_id_routes_to_launcher_lane(self):
        from tensorrt_llm.executor.utils import ErrorResponse

        worker = fake_rank0(lanes=2)
        worker._await_response_helper.handle_for_ipc_batched([ErrorResponse(None, "adp dummy", 1)])
        assert len(worker.frontend_result_queues[0].items) == 1
        assert not worker.frontend_result_queues[1].items


class TestClassicFrontendProxyEndToEnd:
    """GenerationExecutorFrontendProxy against a fake rank0 worker.

    Real ipc:// sockets: namespaced submit, cancel-on-shutdown, and --
    critically -- that an attached frontend NEVER emits the None
    engine-shutdown sentinel.
    """

    @staticmethod
    def _make_proxy_and_fake_worker(tmpdir, frontend_id=1, num_frontends=2):
        import zmq

        from tensorrt_llm.executor.ipc import IpcQueue
        from tensorrt_llm.llmapi.engine_client.proxy import GenerationExecutorFrontendProxy

        hmac_key = os.urandom(32)
        request_addr = f"ipc://{os.path.join(tmpdir, 'request.sock')}"
        result_addrs = [
            f"ipc://{os.path.join(tmpdir, f'result_{i}.sock')}" for i in range(num_frontends)
        ]
        # The fake rank0 worker binds the request ingress (PULL), exactly
        # like worker_main does in multi-frontend mode.
        worker_ingress = IpcQueue(
            (request_addr, hmac_key),
            is_server=True,
            socket_type=zmq.PULL,
            name="fake_worker_request_queue",
        )
        proxy = GenerationExecutorFrontendProxy(
            {
                "mode": "classic",
                "request_addr": request_addr,
                "result_addrs": result_addrs,
                "hmac_key": hmac_key.hex(),
            },
            frontend_id=frontend_id,
        )
        return proxy, worker_ingress, hmac_key, result_addrs

    @staticmethod
    def _stop_dispatch_thread(proxy, result_addrs, hmac_key):
        """End the dispatch thread the way the worker does at engine teardown.

        The worker fans a per-lane None sentinel to every frontend (see
        notify_proxy_threads_to_quit). Frontend shutdown deliberately leaves
        the thread to process teardown, which the threadleak checker would
        report as a leak.
        """
        import zmq

        from tensorrt_llm.executor.ipc import FusedIpcQueue

        worker_lane = FusedIpcQueue(
            (result_addrs[proxy._frontend_id], hmac_key),
            is_server=False,
            fuse_message=False,
            socket_type=zmq.PUSH,
            name="fake_worker_shutdown_lane",
        )
        worker_lane.put(None)
        proxy.dispatch_result_thread.join(timeout=10)
        assert not proxy.dispatch_result_thread.is_alive()

    def test_submit_namespaces_and_shutdown_never_sends_sentinel(self):
        from tensorrt_llm.executor.request import CancellingRequest, GenerationRequest
        from tensorrt_llm.executor.wire import SubmitFrame
        from tensorrt_llm.sampling_params import SamplingParams

        with tempfile.TemporaryDirectory() as tmpdir:
            proxy, worker_ingress, hmac_key, result_addrs = self._make_proxy_and_fake_worker(tmpdir)

            # Attributes read by OpenAIServer at init must exist (a missing
            # _resource_governor_queue crashed all siblings in the first e2e).
            assert proxy.resource_governor_queue is None

            result = proxy.submit(GenerationRequest([1, 2, 3], SamplingParams()))
            assert get_frontend_id(result.request_id) == 1
            assert worker_ingress.poll(5)
            received = worker_ingress.get()
            assert isinstance(received, SubmitFrame)
            assert received.client_ids == (result.request_id,)

            # Frontend shutdown aborts its in-flight requests (cancel) but
            # must NOT emit the None engine-shutdown sentinel.
            proxy.shutdown()
            assert worker_ingress.poll(5)
            cancel = worker_ingress.get()
            assert isinstance(cancel, CancellingRequest)
            assert cancel.id == result.request_id
            assert not worker_ingress.poll(1), (
                "an attached frontend must never send the engine-shutdown sentinel"
            )
            self._stop_dispatch_thread(proxy, result_addrs, hmac_key)

    def test_check_health_and_submit_reflect_fatal_error(self):
        from tensorrt_llm.executor.request import GenerationRequest
        from tensorrt_llm.executor.utils import EngineDeadError
        from tensorrt_llm.sampling_params import SamplingParams

        with tempfile.TemporaryDirectory() as tmpdir:
            proxy, _, _, _ = self._make_proxy_and_fake_worker(tmpdir)

            # Regression: this state lives in GenerationExecutorProxy.__init__,
            # which an attached frontend deliberately skips, so it must be
            # initialized explicitly (a missing _engine_dead broke the first
            # submit(); a missing _worker_process_monitor broke /health).
            assert proxy.check_health()
            assert proxy.model_world_size == 1

            proxy._set_fatal_error(RuntimeError("rank0 worker died"))
            assert not proxy.check_health()
            with pytest.raises(EngineDeadError):
                proxy.submit(GenerationRequest([1], SamplingParams()))

    def test_dispatch_routes_own_lane_responses(self):
        import zmq

        from tensorrt_llm.executor.ipc import FusedIpcQueue
        from tensorrt_llm.executor.request import GenerationRequest
        from tensorrt_llm.executor.utils import ErrorResponse
        from tensorrt_llm.sampling_params import SamplingParams

        with tempfile.TemporaryDirectory() as tmpdir:
            proxy, _, hmac_key, result_addrs = self._make_proxy_and_fake_worker(tmpdir)
            result = proxy.submit(GenerationRequest([1, 2, 3], SamplingParams()))
            client_id = result.request_id
            assert client_id in proxy._results

            # The fake worker pushes an ErrorResponse down this frontend's
            # result lane; the dispatcher must deliver it and retire the
            # request.
            worker_lane = FusedIpcQueue(
                (result_addrs[1], hmac_key),
                is_server=False,
                fuse_message=False,
                socket_type=zmq.PUSH,
                name="fake_worker_result_lane",
            )
            worker_lane.put(ErrorResponse(client_id, "boom", 1))
            deadline = time.time() + 5
            while client_id in proxy._results and time.time() < deadline:
                time.sleep(0.01)
            assert client_id not in proxy._results

            self._stop_dispatch_thread(proxy, result_addrs, hmac_key)
