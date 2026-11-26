# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exports of a response that is dropped on its way to the client.

A multimodal encoder's response carries its embeddings and MRoPE positions as exported
handles, which only a consumer that opens them, or ``release_dict``, consumes. The
exports here are CPU tensors: each lives in a ``/dev/shm`` file until its handles are
consumed. Rank 0 drops engine responses; a frontend or a postprocess worker drops rows.
"""

import base64
import gc
from pathlib import Path

import pytest
import torch
from test_common.executor.fakes import (
    executor_requests,
    fake_postprocessor,
    fake_rank0,
    run_postprocessor,
)

from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse, LlmResult, PyResult
from tensorrt_llm._torch.pyexecutor.response_utils import prepare_response
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor import engine_server
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.utils import ErrorResponse
from tensorrt_llm.executor.wire import StepFrame
from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalParams
from tensorrt_llm.llmapi.engine_client import proto_wire
from tensorrt_llm.llmapi.engine_client.executor import GenerationExecutor
from tensorrt_llm.llmapi.engine_client.proxy import GenerationExecutorProxy
from tensorrt_llm.llmapi.engine_client.rpc_proxy_mixin import RpcExecutorMixin
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only

CLIENT_ID = 7


def _export(py_result: PyResult) -> set[Path]:
    """Export an encoder part's embeddings and MRoPE positions from ``py_result``; their files."""
    py_result.append_mm_embeddings(torch.ones(2, 1024), [1, 1])
    py_result.set_mrope_position(torch.zeros(3, 1024), torch.zeros(1024))
    handles = [
        *py_result.mm_embedding_handles,
        py_result.mrope_position_ids_handle,
        py_result.mrope_position_deltas_handle,
    ]
    return {
        Path("/dev/shm") / base64.b64decode(handle["storage_handle"]).decode().lstrip("/")
        for handle in handles
    }


def _exporting_response() -> tuple[LlmResponse, set[Path]]:
    py_result = PyResult(prompt_len=1, max_new_tokens=1)
    files = _export(py_result)
    response = LlmResponse(
        request_id=CLIENT_ID,
        result=LlmResult(b"", py_result, is_final=True),
        client_id=CLIENT_ID,
    )
    return response, files


def _drop_at_worker(response):
    worker = fake_rank0()
    worker._refused_client_ids.add(CLIENT_ID)
    assert engine_server.drop_refused_response(worker, response)


def _replace_at_worker_on_background_error(response):
    worker = fake_rank0()
    worker._error_queue.put(RuntimeError("background error"))
    worker._await_response_helper.handle_for_ipc_batched([response])
    assert worker.result_queue.items == [[ErrorResponse(CLIENT_ID, "background error", CLIENT_ID)]]


def _drop_unclaimed_at_worker(response):
    # No RequestEntry waits for the response: it is a step of a request that already ended,
    # or that rank 0 never started.
    worker = fake_rank0()
    assert engine_server.publish(worker, [response], postprocess=False) == []
    assert worker.result_queue.items == []


@pytest.mark.parametrize(
    "drop",
    [_drop_at_worker, _replace_at_worker_on_background_error, _drop_unclaimed_at_worker],
    ids=["refused_at_worker", "errored_at_worker", "unclaimed_at_worker"],
)
def test_a_dropped_response_releases_its_exports(drop):
    response, files = _exporting_response()
    gc.collect()
    assert all(path.exists() for path in files)

    drop(response)
    gc.collect()

    assert not any(path.exists() for path in files)


def _exporting_frame() -> tuple[StepFrame, set[Path]]:
    """A ``StepFrame`` whose one row carries an encoder part's exports, as rank 0 publishes it."""
    request = GenerationRequest(
        list(range(1, 9)),
        SamplingParams(max_tokens=1),
        multimodal_params=MultimodalParams(
            multimodal_input=MultimodalInput(
                multimodal_hashes=[list(range(1, 9))],
                multimodal_positions=[2],
                multimodal_lengths=[6],
            ),
        ),
    )
    request.set_id(CLIENT_ID)
    worker = fake_rank0()
    engine_server.submit(worker, proto_wire.encode(request, postprocess=False), raise_errors=True)
    ((encoder,),) = executor_requests(worker.engine.enqueued)
    files = _export(encoder.py_result)
    encoder.add_new_token(5, 0)
    encoder.finish_by(tllm.FinishReason.LENGTH, 0)
    assert engine_server.publish(worker, [prepare_response(encoder)], postprocess=False) == []
    ((frame,),) = worker.result_queue.items
    return frame, files


class _ResultQueue:
    def __init__(self, item):
        self._item = item

    def get(self):
        return self._item


def _drop_late_at_proxy(frame):
    # Without __init__, which spawns workers; the attributes keep teardown a no-op.
    proxy = object.__new__(GenerationExecutorProxy)
    proxy.workers_started = False
    proxy._multi_frontend_ipc_dir = None
    proxy._results = {}
    proxy.result_queue = _ResultQueue([frame])
    assert proxy.dispatch_result_task()


class _RpcProxy(RpcExecutorMixin, GenerationExecutor):
    pass


_RpcProxy.__abstractmethods__ = frozenset()


def _drop_unknown_at_rpc_proxy(frame):
    proxy = _RpcProxy.__new__(_RpcProxy)
    proxy._results = {}
    proxy.handle_responses([[frame]])


def _drop_late_at_postproc_worker(frame):
    # The row's request has no record (its last row went out, or a `terminate` verdict ended
    # it), and the row brings no frontend context to build one from.
    assert run_postprocessor(fake_postprocessor(), [[frame]]) == []


@pytest.mark.parametrize(
    "drop",
    [_drop_late_at_proxy, _drop_unknown_at_rpc_proxy, _drop_late_at_postproc_worker],
    ids=["late_at_proxy", "unknown_at_rpc_proxy", "late_at_postproc_worker"],
)
def test_a_dropped_row_releases_its_exports(drop):
    frame, files = _exporting_frame()
    gc.collect()
    assert all(path.exists() for path in files)

    drop(frame)
    gc.collect()

    assert not any(path.exists() for path in files)
