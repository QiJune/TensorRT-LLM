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
"""Self-tests of the executor test seam: the fakes and snapshots executor tests build on."""

import datetime
import hashlib
import json

import numpy as np
import pytest
import torch
from test_common.executor import snapshots
from test_common.executor.fakes import (
    Rank0,
    executor_requests,
    fake_postprocessor,
    fake_rank0,
    run_postprocessor,
)

from tensorrt_llm._torch.pyexecutor.llm_request import FinishReason, LlmRequestState
from tensorrt_llm._torch.pyexecutor.response_utils import prepare_response
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.executor import engine_server
from tensorrt_llm.executor.base_worker import AwaitResponseHelper
from tensorrt_llm.executor.logprobs import Logprob
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.llmapi.engine_client import proto_wire
from tensorrt_llm.llmapi.engine_client.postproc_worker import PostprocWorker
from tensorrt_llm.llmapi.engine_client.result import GenerationResult
from tensorrt_llm.sampling_params import LogprobMode, SamplingParams

pytestmark = pytest.mark.cpu_only

HandlerKind = AwaitResponseHelper.HandlerKind


def _request(streaming=True, **sampling) -> GenerationRequest:
    request = GenerationRequest(
        [1, 2, 3], SamplingParams(max_tokens=2, **sampling), streaming=streaming
    )
    request.set_id(7)
    return request


def _keep_logits(req_id, logits, token_ids, stream_ptr, client_id):
    """A logits processor a user passes by reference."""


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


def test_fake_rank0_keeps_what_rank0_enqueues():
    worker = fake_rank0()
    engine_server.submit(
        worker, proto_wire.encode(_request(), postprocess=False), raise_errors=True
    )
    (enqueued,) = worker.engine.enqueued
    assert enqueued.request.client_id == 7
    assert isinstance(worker._results[7], engine_server.RequestEntry)


def test_fake_rank0_has_the_queues_it_is_asked_for():
    one = fake_rank0()
    assert one.result_queue is not None and one.frontend_result_queues is None
    assert one._await_response_helper._resolve_handler_kind() is HandlerKind.ipc_batched

    lanes = fake_rank0(lanes=2)
    assert lanes.result_queue is None and len(lanes.frontend_result_queues) == 2

    postprocess = fake_rank0(postprocess_workers=2)
    assert len(postprocess.postproc_queues) == 2 and postprocess.result_queue is not None
    assert postprocess._await_response_helper.enable_postprocprocess_parallel

    here = fake_rank0(in_frontend_process=True)
    assert here.result_queue is None and here.postproc_queues is None
    kind = here._await_response_helper._resolve_handler_kind()
    assert kind is HandlerKind.single_process_worker


@pytest.mark.parametrize("beam_search", [False, True])
def test_executor_requests_are_a_family_per_request(beam_search):
    worker = fake_rank0()
    request = _request(n=2, best_of=2, use_beam_search=beam_search, temperature=1.0)
    engine_server.submit(worker, proto_wire.encode(request, postprocess=False), raise_errors=True)
    (family,) = executor_requests(worker.engine.enqueued)
    # a beam search is one request; sampled sequences are a request and its children
    assert [member.py_request_id for member in family] == ([7] if beam_search else [7, 71])
    assert all(member.py_client_id == 7 for member in family)
    assert all(member.state == LlmRequestState.GENERATION_IN_PROGRESS for member in family)


def test_executor_requests_keep_logits_on_the_host():
    worker = fake_rank0()
    request = _request(
        n=2, temperature=1.0, return_context_logits=True, return_generation_logits=True
    )
    engine_server.submit(worker, proto_wire.encode(request, postprocess=False), raise_errors=True)
    ((parent, child),) = executor_requests(worker.engine.enqueued)
    for member in (parent, child):
        member.py_result.append_context_logits(torch.zeros(3, 8))
        member.py_result.append_generation_logits(torch.zeros(1, 1, 8))
        assert member.py_result.context_logits.device.type == "cpu"
        assert member.py_result.generation_logits.device.type == "cpu"


def test_rank0_hands_rows_to_their_results():
    rank0 = Rank0()
    request = _request()
    result = GenerationResult(request)
    (sequence,) = rank0.submit(request, result)

    sequence.add_new_token(5, 0)
    (row,) = rank0.publish(sequence)
    assert row.client_id == 7 and not row.result.is_final

    sequence.add_new_token(6, 0)
    sequence.finish_by(FinishReason.LENGTH, 0)
    (row,) = rank0.publish(sequence)
    assert row.result.is_final
    assert result.finished and result.outputs[0].token_ids == [5, 6]


def test_rank0_sends_nothing_for_a_request_with_nothing_to_say():
    rank0 = Rank0()
    (sequence,) = rank0.submit(_request(streaming=False))
    sequence.add_new_token(5, 0)
    # a non-streaming request responds when it ends
    assert rank0.send(sequence) == []
    sequence.finish_by(FinishReason.LENGTH, 0)
    (row,) = rank0.publish(sequence)
    assert row.result.is_final


def test_run_postprocessor_gives_what_a_postprocess_worker_sends():
    worker = fake_rank0(postprocess_workers=1)
    engine_server.submit(worker, proto_wire.encode(_request(), postprocess=True), raise_errors=True)
    ((sequence,),) = executor_requests(worker.engine.enqueued)
    sequence.add_new_token(5, 0)
    sequence.finish_by(FinishReason.LENGTH, 0)
    worker._await_response_helper.process_and_handle_responses([prepare_response(sequence)])

    (output,) = run_postprocessor(fake_postprocessor(), worker.postproc_queues[0].items)
    assert isinstance(output, PostprocWorker.Output)
    assert output.client_id == 7 and output.is_final
    assert output.res.token_ids == [5]


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------


def test_encode_gives_plain_json():
    tensor = torch.arange(4, dtype=torch.int32)
    digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    encoded = snapshots.encode(
        {
            "mode": LogprobMode.PROCESSED,
            150: Logprob(logprob=-0.5, rank=2),
            "params": DisaggregatedParams(
                request_type="context_only",
                opaque_state=b"\x00s",
                schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
            ),
            "tensor": tensor,
            "array": np.arange(4, dtype=np.int32),
            "wait": datetime.timedelta(milliseconds=500),
            "inf": float("inf"),
            "pair": (1, (2, 3)),
        }
    )
    assert encoded == {
        "mode": "PROCESSED",
        "150": [-0.5, 2],
        # a record leaves out its fields that are None
        "params": {
            "request_type": "context_only",
            "opaque_state": "0073",
            "schedule_style": "GENERATION_FIRST",
        },
        "tensor": {"dtype": "torch.int32", "shape": [4], "sha256": digest},
        "array": {"dtype": "int32", "shape": [4], "sha256": digest},
        "wait": 0.5,
        "inf": "inf",
        "pair": [1, [2, 3]],
    }
    assert json.loads(json.dumps(encoded)) == encoded


def test_encode_refuses_what_it_has_no_encoding_for():
    with pytest.raises(TypeError, match="no snapshot encoding"):
        snapshots.encode(object())


def test_users_view_is_what_users_read_by_field():
    rank0 = Rank0()
    request = _request(streaming=False)
    result = GenerationResult(request)
    (sequence,) = rank0.submit(request, result)
    sequence.add_new_token(5, 0)
    sequence.finish_by(FinishReason.LENGTH, 0)
    rank0.publish(sequence)

    view = snapshots.users_view({7: result})
    assert list(view) == ["7"]
    assert view["7"]["done"] is True and "error" not in view["7"]
    (output,) = view["7"]["outputs"]
    assert output["token_ids"] == [5] and output["finish_reason"] == "length"


def test_request_state_is_what_the_engine_reads():
    worker = fake_rank0(kv_cache_transceiver=True)
    request = _request(logits_processor=_keep_logits)
    request.disaggregated_params = DisaggregatedParams(
        request_type="generation_only",
        ctx_request_id=3,
        first_gen_tokens=[9],
        ctx_usage={
            "prompt_tokens": 32,
            "completion_tokens": 0,
            "total_tokens": 32,
            "prompt_tokens_details": {"cached_tokens": 16},
        },
    )
    engine_server.submit(worker, proto_wire.encode(request, postprocess=False), raise_errors=True)
    (enqueued,) = worker.engine.enqueued

    state = snapshots.request_state(enqueued)
    assert state["inputs"]["logits_post_processors"] == ["_keep_logits"]
    # of a ctx_usage, the engine reads the prompt and cached token counts
    assert state["inputs"]["disaggregated_params"]["ctx_usage"] == [32, 16]
    assert isinstance(state["request"], list)
    assert json.loads(json.dumps(state)) == state


def test_step_changes_hold_what_each_step_changed():
    views = [
        {"7": {"outputs": [{"token_ids": [5]}, {"token_ids": [6]}], "done": False}},
        {
            "7": {
                "outputs": [{"token_ids": [5, 8], "finish_reason": "length"}, {"token_ids": [6]}],
                "done": True,
            }
        },
        {"7": {"outputs": [{"token_ids": [5, 8]}, {"token_ids": [6]}], "done": True}},
    ]
    assert snapshots.step_changes(views) == [
        views[0],
        {"7": {"outputs": [{"token_ids": [5, 8], "finish_reason": "length"}, {}], "done": True}},
        # a field that became None is null
        {"7": {"outputs": [{"finish_reason": None}, {}]}},
    ]


def test_assert_recorded_names_the_first_difference():
    snapshots.assert_recorded({"a": [1, 2]}, {"a": [1, 2]}, "same")
    first_difference = r"x differs from its snapshot at \['a'\]\[1\]: 3 != 2"
    with pytest.raises(AssertionError, match=first_difference):
        snapshots.assert_recorded({"a": [1, 3]}, {"a": [1, 2]}, "x")
    # dict order counts
    with pytest.raises(AssertionError, match=r"keys \['a', 'b'\] != \['b', 'a'\]"):
        snapshots.assert_recorded({"a": 1, "b": 2}, {"b": 2, "a": 1}, "order")


def test_dump_writes_one_snapshot_a_line_that_load_reads(tmp_path):
    entries = {"submit/plain": {"request": [1, None]}, "refusals/zero_max_tokens": "error"}
    path = tmp_path / "snapshots.jsonl"
    snapshots.dump(path, entries)
    assert path.read_text().splitlines() == [
        '{"submit/plain":{"request":[1,null]}}',
        '{"refusals/zero_max_tokens":"error"}',
    ]
    assert snapshots.load(path) == entries
