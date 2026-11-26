# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for proxy late-response handling after a hook terminate.

When a post-processor hook returns ``terminate`` the result is
marked done and popped from the
proxy's ``_results`` map, but the engine may still have in-flight responses for
the same ``client_id`` (abort is async). ``GenerationExecutorProxy.dispatch_result_task``
must drop those late responses via ``self._results.get(client_id)`` rather than
``pop``-ing a missing key and crashing the dispatch thread with a ``KeyError``.
"""

from types import SimpleNamespace

from test_common.executor.fakes import Rank0

from tensorrt_llm._torch.pyexecutor.llm_request import FinishReason
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.llmapi.engine_client.proxy import GenerationExecutorProxy
from tensorrt_llm.sampling_params import SamplingParams


class _FakeResultQueue:
    """Returns one queued item per ``get()`` call (mirrors the IPC queue)."""

    def __init__(self, items):
        self._items = list(items)

    def get(self):
        return self._items.pop(0)


class _RecordingQueue:
    """Stand-in for a result's delivery queue (non-``_SyncQueue`` branch)."""

    def __init__(self):
        self.delivered = []

    def put(self, res):
        self.delivered.append(res)


def _make_proxy():
    # Avoid GenerationExecutorProxy.__init__ (it spawns workers); we only
    # exercise the pure dispatch logic with hand-set attributes.
    proxy = object.__new__(GenerationExecutorProxy)
    # GC-time __del__ -> shutdown() reads these; __init__ is bypassed here, so
    # seed them to keep teardown a clean no-op.
    proxy.workers_started = False
    proxy._multi_frontend_ipc_dir = None
    return proxy


def test_late_response_after_terminate_is_dropped_without_keyerror():
    """A response for an already-popped client_id must be dropped, not crash."""
    proxy = _make_proxy()
    proxy._results = {}  # terminate already finalized + popped this client_id

    late = SimpleNamespace(client_id=999, result=SimpleNamespace(is_final=True))
    proxy.result_queue = _FakeResultQueue([late])

    # Must not raise KeyError; the dispatch loop stays alive (returns True).
    assert proxy.dispatch_result_task() is True


def test_final_response_is_delivered_and_popped():
    """A live, final response is delivered and removed from ``_results``.

    This establishes the exact condition under which a later duplicate becomes
    the 'late response' that the drop above guards against.
    """
    request = GenerationRequest([1, 2, 3], SamplingParams(max_tokens=1))
    request.set_id(7)
    rank0 = Rank0()
    (sequence,) = rank0.submit(request)
    sequence.add_new_token(5, 0)
    sequence.finish_by(FinishReason.LENGTH, 0)
    # the request's last row, in the StepFrame rank 0 sends the proxy
    (message,) = rank0.send(sequence)

    proxy = _make_proxy()
    queue = _RecordingQueue()
    # decoding reads what the request asked for from its result
    proxy._results = {7: SimpleNamespace(queue=queue, sampling_params=request.sampling_params)}
    proxy.result_queue = _FakeResultQueue([message])

    assert proxy.dispatch_result_task() is True
    (row,) = queue.delivered
    assert row.client_id == 7 and row.result.is_final
    assert 7 not in proxy._results  # popped on is_final
