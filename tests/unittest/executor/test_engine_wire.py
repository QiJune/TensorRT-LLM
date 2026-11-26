# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The executor's wire (proto_wire.py, engine_server.py) against its snapshots.

For the cases of engine_wire_cases.py, the requests rank 0 builds from what a frontend sends, the
errors it refuses requests with, and what users read from their results after every step,
directly and through postprocess workers, are compared with the snapshots in
engine_wire_snapshots.jsonl, which test_common/executor/snapshots.py encodes. Every frame crosses
the harness pickled, as it crosses the queues between processes.

A changed snapshot is a change of behaviour, to be reviewed as one. To record the snapshots from
the code as it stands:

    python tests/unittest/executor/test_engine_wire.py --record
"""

import argparse
import datetime
import functools
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState, LlmResponse, PerfTimingInfo
from tensorrt_llm._torch.pyexecutor.response_utils import prepare_response
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor import engine_server, wire
from tensorrt_llm.executor.logprobs import Logprob
from tensorrt_llm.executor.request import GenerationRequest, LoRARequest
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.llmapi.engine_client import proto_wire
from tensorrt_llm.llmapi.engine_client.result import GenerationResult

if __name__ == "__main__":
    # run as a script to record: put tests/ on the path, as pytest.ini's pythonpath does for pytest
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

__extra_import_path__ = ["."]
import engine_wire_cases as cases  # noqa: E402
from test_common.executor import snapshots  # noqa: E402
from test_common.executor.fakes import (
    executor_requests,
    fake_postprocessor,
    fake_rank0,
    run_postprocessor,
)

pytestmark = pytest.mark.cpu_only

SNAPSHOTS = Path(__file__).with_name("engine_wire_snapshots.jsonl")


@functools.cache
def _snapshots() -> dict:
    return snapshots.load(SNAPSHOTS)


def _recorded(key: str):
    recorded = _snapshots()
    assert key in recorded, f"no snapshot of {key}: record them as the module docstring says"
    return recorded[key]


def _crossed(frame):
    """``frame`` as the other process gets it: pickled, with the objects it refers to."""
    return pickle.loads(pickle.dumps(frame))


def _submit_request(name: str) -> GenerationRequest:
    """``SUBMIT_CASES[name]``'s request, the first-token logprobs it hands over as Logprobs."""
    sampling, change = cases.SUBMIT_CASES[name]
    request = cases.base_request(7, **sampling)
    if change is not None:
        change(request)
    params = request.disaggregated_params
    if params is not None and params.first_gen_log_probs is not None:
        params.first_gen_log_probs = [
            {token: Logprob(*pair) for token, pair in entry.items()}
            for entry in params.first_gen_log_probs
        ]
    return request


def _submitted(request: GenerationRequest, worker=None) -> dict:
    """The request rank 0 builds from ``request``'s frame and enqueues, as a snapshot."""
    worker = worker or fake_rank0(kv_cache_transceiver=True)
    engine_server.submit(worker, _crossed(proto_wire.encode(request, postprocess=False)))
    (enqueued,) = worker.engine.enqueued
    assert isinstance(worker._results[request.id], engine_server.RequestEntry)
    return snapshots.request_state(enqueued)


@pytest.mark.parametrize("name", list(cases.SUBMIT_CASES))
def test_submit_builds_the_recorded_request(name):
    got = _submitted(_submit_request(name))
    snapshots.assert_recorded(got, _recorded(f"submit/{name}"), name)


def test_submit_reads_the_llm_apis_steady_clock_arrival_time():
    """The arrival time the LLM API stamps, a timedelta, reaches rank 0 as its seconds."""
    request = cases.base_request(7)
    request.arrival_time = datetime.timedelta(seconds=12345, microseconds=678901)
    assert _submitted(request)["inputs"]["arrival_time"] == 12345.678901


class _LoraManager:
    """Rank 0's LoRA manager; it loads every adapter it is asked for."""

    def __init__(self):
        self.cpp_lora_config = {}
        self.cpp_lora_weights = {}
        self.loaded = []

    def load_from_ckpt(self, paths, model_config, uids, ckpt_source):
        self.loaded.append((tuple(paths), tuple(uids), ckpt_source))
        for uid in uids:
            self.cpp_lora_config[uid] = torch.tensor([[0, 1, 8]], dtype=torch.int32)


def _lora_rank0():
    worker = fake_rank0(kv_cache_transceiver=True)
    worker._lora_manager = _LoraManager()
    return worker


def _lora_request(source: str) -> GenerationRequest:
    request = cases.base_request(7)
    request.lora_request = LoRARequest(*cases.LORA_ADAPTER, source)
    return request


@pytest.mark.parametrize("source", cases.LORA_SOURCES)
def test_submit_loads_lora_and_builds_the_recorded_request(source):
    worker = _lora_rank0()
    got = _submitted(_lora_request(source), worker)
    snapshots.assert_recorded(got, _recorded(f"submit/lora_{source}"), source)
    _, adapter_id, path = cases.LORA_ADAPTER
    assert worker._lora_manager.loaded == [((path,), (str(adapter_id),), source)]


def _refusal(worker, request: GenerationRequest) -> str:
    """The error rank 0 refuses ``request`` with."""
    engine_server.submit(worker, _crossed(proto_wire.encode(request, postprocess=False)))
    error = worker._await_response_helper.temp_error_responses.get_nowait()
    assert error.client_id == request.id
    assert not worker.engine.enqueued and not worker._results
    return error.error_msg


def _case_refusal(name: str) -> str:
    """The error rank 0 refuses ``REFUSAL_CASES[name]``'s request with."""
    case = cases.REFUSAL_CASES[name]
    worker = fake_rank0(
        max_seq_len=case.get("max_seq_len", 4096),
        kv_cache_transceiver=case.get("kv_cache_transceiver", False),
    )
    request = cases.base_request(7, **case.get("sampling", {}))
    if (change := case.get("change")) is not None:
        change(request)
    return _refusal(worker, request)


@pytest.mark.parametrize("name", list(cases.REFUSAL_CASES))
def test_submit_refuses_with_the_recorded_error(name):
    snapshots.assert_recorded(_case_refusal(name), _recorded(f"refusals/{name}"), name)


def _no_transceiver_error() -> str:
    """What a disaggregated request fails with on a rank 0 without a KV cache transceiver."""
    request = cases.base_request(7)
    request.disaggregated_params = DisaggregatedParams(
        request_type="context_only", disagg_request_id=3
    )
    frame = _crossed(proto_wire.encode(request, postprocess=False))
    with pytest.raises(AssertionError) as raised:
        engine_server.submit(fake_rank0(), frame)
    return str(raised.value)


def test_disagg_without_a_transceiver_fails_with_the_recorded_error():
    error = _no_transceiver_error()
    snapshots.assert_recorded(error, _recorded("refusals/no_transceiver"), "no transceiver")


def _lora_load_error() -> str:
    """The error rank 0 refuses a request with whose LoRA adapter does not load."""

    def failing_load(*args, **kwargs):
        raise OSError("no such adapter")

    worker = _lora_rank0()
    worker._lora_manager.load_from_ckpt = failing_load
    name, adapter_id, _ = cases.LORA_ADAPTER
    request = cases.base_request(7)
    request.lora_request = LoRARequest(name, adapter_id, "/adapters/missing", "hf")
    return _refusal(worker, request)


def test_lora_load_failure_is_refused_with_the_recorded_error():
    snapshots.assert_recorded(_lora_load_error(), _recorded("refusals/lora_load"), "lora load")


def test_a_logits_processor_rides_beside_the_message():
    request = cases.base_request(7, logits_processor=cases.keep_logits)
    frame = proto_wire.encode(request, postprocess=False)
    assert frame.objects == [cases.keep_logits]
    worker = fake_rank0()
    engine_server.submit(worker, _crossed(frame))
    (enqueued,) = worker.engine.enqueued
    assert enqueued.inputs.logits_post_processors == [cases.keep_logits]


def test_numpy_inputs_ride_beside_the_message():
    request = cases.base_request(9)
    request.multimodal_params = MultimodalParams(multimodal_data={"sizes": np.array([1, 2])})
    worker = fake_rank0()
    engine_server.submit(worker, _crossed(proto_wire.encode(request, postprocess=False)))
    (enqueued,) = worker.engine.enqueued
    sizes = enqueued.inputs.multimodal_data["sizes"]
    assert type(sizes) is np.ndarray and sizes.tolist() == [1, 2]


VOCAB = 256


def _logits(rows, seed):
    """float32 logits, as the model gives them.

    Each row is a permutation of the same values (37 is coprime with VOCAB), so no two tokens tie.
    """
    tokens = torch.arange(VOCAB) * 37 + torch.arange(rows).unsqueeze(1) * 11 + seed
    return (tokens % VOCAB).to(torch.float32) / 16


def _sample(r, step, case, finish):
    """One step of the sampler for ``r``: tokens, logprobs, counters, and the end."""
    if step == 0 and case.get("usage_from_kv"):
        # generation-first: the decode part learns its prefill part's usage with the KV
        r.py_disaggregated_params.ctx_usage = wire.decode_prefill_usage(30, 12)
    beams = case.get("beams", 1)
    tokens = case.get("tokens", 1)
    topk = case.get("topk")
    if step == 0 and r.py_return_context_logits:
        r.py_result.append_context_logits(_logits(len(cases.PROMPT), r.py_client_id))
    if r.py_return_generation_logits:
        r.py_result.append_generation_logits(_logits(tokens, step).unsqueeze(1))
    if r.py_additional_outputs:
        r.py_result.append_additional_generation_outputs("hidden", _logits(tokens, step + 7)[:, :4])
    # the k-th child of an n > 1 request samples its own tokens when the case asks
    child = 0 if r.py_request_id == r.py_client_id else r.py_request_id - 10 * r.py_client_id
    shift = 50 * child if case.get("children_differ") else 0
    new = [[100 + 10 * beam + step + j + shift for j in range(tokens)] for beam in range(beams)]
    for beam in range(beams):
        for token in new[beam]:
            r.add_new_token(token, beam)
    if case.get("values_only"):
        r.py_result.append_log_probs([[cases.SAMPLED] * tokens for _ in range(beams)])
    elif topk is not None:
        # a top-k candidate ranked 2 with none ranked 1, which no logprob columns can say
        odd = case.get("odd_logprobs_at") == step and r.py_client_id == 100
        top = {200 + t: Logprob(logprob=cases.TOPK[t], rank=t + 1 + odd) for t in range(topk)}
        # the sampler's dict: the sampled token first, then the top-k
        r.py_result.append_log_probs(
            [
                [
                    {token: Logprob(logprob=cases.SAMPLED, rank=topk + 1), **top}
                    for token in new[beam]
                ]
                for beam in range(beams)
            ]
        )
    if case.get("drafting") and step > 0:
        width = len(r.py_per_pos_drafted)
        r.py_per_pos_drafted = [step] * 2 + [0] * (width - 2)
        r.py_per_pos_accepted = [step, step // 2] + [0] * (width - 2)
        r.py_total_draft_tokens = 2 * step
        r.py_total_accepted_draft_tokens = step + step // 2
    if finish and case.get("routed_experts"):
        # as route_capture attaches them when the request ends
        pyr = r.py_result
        if pyr._additional_generation_outputs is None:
            pyr._additional_generation_outputs = {}
        pyr._additional_generation_outputs.setdefault("routed_experts", [])
        pyr.append_additional_generation_outputs(
            "routed_experts", torch.arange(step * 4 + 4, dtype=torch.int32).reshape(-1, 2)
        )
    if finish and case.get("context_only"):
        # the prefill part ends: its handoff instead of a finish reason
        r.context_phase_params = tllm.ContextPhaseParams(
            [new[beam][-1] for beam in range(beams)],
            r.py_request_id,
            None,
            [7, 8],
            1,
            "tcp://ctx:9",
        )
        if case.get("remote_tail"):
            r.py_csa2_remote_tail_mode = "source"
            r.py_csa2_remote_tail_start = 16
        if case.get("first_gen_beam"):
            r.py_result.set_first_gen_log_probs(
                [
                    {new[beam][-1]: Logprob(logprob=cases.TOPK[beam], rank=None)}
                    for beam in range(beams)
                ]
            )
        r.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE
        return
    if finish and case.get("encoder_handoff"):
        # an encoder part's embeddings and mRoPE positions, behind shared handles
        r.py_result._mm_embeddings = [cases.shared_handle(70)]
        r.py_result._mrope_position_ids = cases.shared_handle(80)
        r.py_result._mrope_position_deltas = cases.shared_handle(81)
    if finish:
        if case.get("perf"):
            r.py_perf_timing = PerfTimingInfo(
                step_metrics=[
                    dict(
                        forward_start_time=1000.000125,
                        forward_end_time=1000.0025,
                        sample_start_time=1000.0025,
                        sample_end_time=1000.003,
                        gpu_forward_time=2.25,
                        gpu_sample_time=0.5,
                        token_time=1000.003125,
                        iter=1,
                    ),
                    dict(
                        forward_start_time=1000.01,
                        forward_end_time=1000.0125,
                        sample_start_time=1000.0125,
                        sample_end_time=1000.013,
                        gpu_forward_time=0,
                        gpu_sample_time=0,
                        token_time=1000.013,
                        iter=2,
                    ),
                ],
                ctx_chunk_metrics=[
                    dict(
                        forward_start_time=999.5,
                        forward_end_time=999.75,
                        sample_start_time=999.75,
                        sample_end_time=999.8,
                        gpu_forward_time=180.5,
                        gpu_sample_time=1.25,
                        token_time=999.8,
                    )
                ],
                ctx_gpu_forward_time=180.5,
                ctx_gpu_sample_time=1.25,
            )
        r.state = LlmRequestState.GENERATION_COMPLETE
        for beam in range(beams):
            r.set_finished_reason(tllm.FinishReason.LENGTH, beam)


def _take(lane) -> list:
    """The messages rank 0 put on ``lane``, as the other process gets them; the lane is emptied."""
    messages = [_crossed(message) for message in lane.items]
    lane.items.clear()
    return messages


def _flow(case: dict, postprocess: bool, requests: int = 3) -> tuple[list, set, bool]:
    """Run requests through submission, the case's sampler steps on rank 0, and the frontend.

    The postprocess workers run too when asked. Returns the users' view after every step, the
    types of what rank 0 sent, and whether rank 0 refused the requests.
    """
    workers = 2 if postprocess else 0
    worker = fake_rank0(postprocess_workers=workers, kv_cache_transceiver="disagg" in case)
    postprocessors = [fake_postprocessor() for _ in range(workers)]
    frontend = {}
    streaming = case.get("streaming", True)
    refused = False
    for i in range(requests):
        client_id = 100 + i
        request = cases.base_request(client_id, **case.get("sampling", {}))
        request.streaming = i % 2 == 0 if streaming == "mixed" else streaming
        request.arrival_time = 1000.0 + i / 8
        if "disagg" in case:
            request.disaggregated_params = case["disagg"](client_id)
        if "multimodal" in case:
            request.multimodal_params = case["multimodal"]()
        frontend[client_id] = GenerationResult(
            request, disaggregated_params=request.disaggregated_params
        )
        frame = proto_wire.encode(request, postprocess=postprocess)
        engine_server.submit(worker, _crossed(frame))
        refused = frame.refusal is not None
        if not refused:
            # rows say the decoding counters for streaming requests; sequences' ends for the others
            assert worker._results[client_id].decoding_stats == request.streaming
    live = {family[0].py_client_id: family for family in executor_requests(worker.engine.enqueued)}
    steps = case.get("steps", 3)
    sent_types, views, held = set(), [], []
    for step in range(steps):
        responses = []
        for client_id, members in list(live.items()):
            if step == 1 and client_id == 101 and case.get("error"):
                responses.append(
                    LlmResponse(request_id=client_id, error_msg="boom", client_id=client_id)
                )
                del live[client_id]
                continue
            if step == 0 and client_id == 101 and case.get("cancel_first"):
                # cancelled before its first token: nothing was sampled for it
                for r in members:
                    r.state = LlmRequestState.GENERATION_COMPLETE
                    for beam in range(case.get("beams", 1)):
                        r.set_finished_reason(tllm.FinishReason.CANCELLED, beam)
                    if (response := prepare_response(r)) is not None:
                        responses.append(response)
                del live[client_id]
                continue
            for k, r in enumerate(list(members)):
                # children end one step apart, the last at the last step
                finish = step == steps - 1 - (len(members) - 1 - k)
                _sample(r, step, case, finish)
                if (response := prepare_response(r)) is not None:
                    responses.append(response)
                if finish:
                    members.remove(r)
            if not members:
                del live[client_id]
        if case.get("pair_steps") and step % 2 == 0 and step + 1 < steps:
            # the executor's responses of two steps reach the await thread together
            held = responses
            views.append(snapshots.users_view(frontend))
            continue
        # with rank 0's refusals, as the await thread takes them
        worker._await_response_helper.process_and_handle_responses(held + responses)
        held = []
        received = [item for message in _take(worker.result_queue) for item in message]
        sent_types.update(type(item).__name__ for item in received)
        for lane, postprocessor in zip(worker.postproc_queues or (), postprocessors):
            messages = _take(lane)
            sent_types.update(type(item).__name__ for message in messages for item in message)
            received += run_postprocessor(postprocessor, messages)
        for item in received:
            rows = [item]
            if isinstance(item, wire.StepFrame):
                rows = proto_wire.decode(item, frontend)
            for row in rows:
                frontend[row.client_id]._handle_response(row)
        views.append(snapshots.users_view(frontend))
    assert not worker._results
    return views, sent_types, refused


def _users_see_the_recorded_views(name: str, postprocess: bool) -> None:
    views, sent_types, refused = _flow(cases.RESPONSE_CASES[name], postprocess)
    if refused:
        assert sent_types == {"ErrorResponse"}, sent_types
    else:
        # rows, and the errors rank 0 answers requests with
        assert "StepFrame" in sent_types, sent_types
        assert sent_types <= {"StepFrame", "ErrorResponse"}, sent_types
    section = "postprocess" if postprocess else "responses"
    recorded = _recorded(f"{section}/{name}")
    snapshots.assert_recorded(snapshots.step_changes(views), recorded, name)


@pytest.mark.parametrize("name", list(cases.RESPONSE_CASES))
def test_rows_give_users_the_recorded_views(name):
    _users_see_the_recorded_views(name, postprocess=False)


@pytest.mark.parametrize("name", cases.POSTPROCESS_CASES)
def test_postprocess_workers_give_users_the_recorded_views(name):
    _users_see_the_recorded_views(name, postprocess=True)


def test_a_response_the_wire_cannot_say_ends_its_request():
    # request 100's second step has a top-k logprob ranked 2 with none ranked 1
    case = dict(cases.RESPONSE_CASES["two_steps_per_batch"], odd_logprobs_at=1)
    views, sent_types, _ = _flow(case, postprocess=False)
    assert sent_types == {"StepFrame"}
    failed = views[-1]["100"]
    assert failed["done"] and "cannot send the response" in failed["error"]

    def others(changes):
        return [{cid: v for cid, v in step.items() if cid != "100"} for step in changes]

    recorded = _recorded("responses/two_steps_per_batch")
    snapshots.assert_recorded(
        others(snapshots.step_changes(views)), others(recorded), "the other requests"
    )


def test_a_malformed_step_output_raises():
    # one handle and no offsets or finish column
    frame = wire.StepFrame(b"\x1a\x08" + (7).to_bytes(8, "little"))
    with pytest.raises(RuntimeError, match="malformed StepOutput"):
        proto_wire.decode(_crossed(frame), {})


def test_an_unreadable_submit_frame_rejects_its_requests():
    worker = fake_rank0()
    # a Request whose bytes stop short of its length
    engine_server.submit(worker, _crossed(wire.SubmitFrame(b"\x0a\x05\xff", (5,))))
    error = worker._await_response_helper.temp_error_responses.get_nowait()
    assert error.client_id == 5 and "cannot read the request" in error.error_msg
    assert not worker._results


def _record() -> dict:
    """Every snapshot, from the code as it stands, by key."""
    entries = {f"submit/{name}": _submitted(_submit_request(name)) for name in cases.SUBMIT_CASES}
    for source in cases.LORA_SOURCES:
        entries[f"submit/lora_{source}"] = _submitted(_lora_request(source), _lora_rank0())
    for name in cases.REFUSAL_CASES:
        entries[f"refusals/{name}"] = _case_refusal(name)
    entries["refusals/no_transceiver"] = _no_transceiver_error()
    entries["refusals/lora_load"] = _lora_load_error()
    for name, case in cases.RESPONSE_CASES.items():
        entries[f"responses/{name}"] = snapshots.step_changes(_flow(case, postprocess=False)[0])
    for name in cases.POSTPROCESS_CASES:
        views, _, _ = _flow(cases.RESPONSE_CASES[name], postprocess=True)
        entries[f"postprocess/{name}"] = snapshots.step_changes(views)
    return entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--record",
        action="store_true",
        required=True,
        help=f"write {SNAPSHOTS.name} from the code as it stands",
    )
    parser.parse_args()
    snapshots.dump(SNAPSHOTS, _record())
