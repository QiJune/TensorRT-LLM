# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The proto wire (executor/engine_client.py, engine_server.py) against the pickled path it stands beside."""

import asyncio
import dataclasses
import datetime
import pickle
import struct
import threading
import types
from types import MethodType

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import (
    REQUEST_TYPE_MAPPING,
    LlmRequest,
    LlmRequestState,
    LlmResponse,
    PerfTimingInfo,
)
from tensorrt_llm._torch.pyexecutor.response_utils import prepare_response
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.conversation_params import ConversationParams
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.executor import engine_client, engine_server, wire
from tensorrt_llm.executor.base_worker import AwaitResponseHelper, BaseWorker
from tensorrt_llm.executor.postproc_worker import PostprocWorker
from tensorrt_llm.executor.request import GenerationRequest, LoRARequest
from tensorrt_llm.executor.result import GenerationResult, Logprob
from tensorrt_llm.executor.utils import RequestError
from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalInputLayout, MultimodalParams
from tensorrt_llm.kv_hints.protocol import KvHintsSnapshot
from tensorrt_llm.llmapi.thinking_budget import ThinkingBudgetLogitsProcessor
from tensorrt_llm.sampling_params import GuidedDecodingParams, SamplingParams
from tensorrt_llm.scheduling_params import SchedulingParams

pytestmark = pytest.mark.cpu_only

PROMPT = list(range(10, 22))
# float32 values, as the sampler's tensors give; their float64 sums are not float32
SAMPLED = struct.unpack("f", struct.pack("f", -0.3))[0]
TOPK = [struct.unpack("f", struct.pack("f", -1.1 - t))[0] for t in range(8)]


class _Engine:
    kv_cache_transceiver = None
    is_generation_model = True

    def __init__(self):
        self.enqueued = []

    def enqueue_request(self, request, result_wait_queue=None):
        self.enqueued.append(request)
        return len(self.enqueued)


class _Worker:
    pass


def _worker():
    w = _Worker()
    w.enable_postprocess_parallel = False
    w._refused_client_ids = set()
    w._engine_wire_active = False
    w._prepared_client_ids = set()
    w.frontend_result_queues = None
    w.postproc_queues = None
    w._results = {}
    w._client_id_to_request_id = {}
    w.rank = 0
    w._lora_manager = None
    w.engine = _Engine()
    w.mapping = types.SimpleNamespace(cp_size=1)
    w.max_seq_len = 4096
    w.postproc_config = types.SimpleNamespace(num_postprocess_workers=0)
    w.start = lambda: None
    # GenerationResult keeps a WeakMethod of it
    w._handle_background_error = MethodType(lambda self, *args: None, w)
    w._has_background_error = lambda: False
    for name in (
        "_pop_result",
        "_enqueue_request",
        "_get_logprob_params",
        "_load_lora_config",
        "submit",
    ):
        setattr(w, name, MethodType(getattr(BaseWorker, name), w))
    w._await_response_helper = AwaitResponseHelper(w)
    sent = []
    w.result_queue = types.SimpleNamespace(put=sent.append)
    return w, sent


def _request(client_id, **sampling):
    params = dict(max_tokens=64, temperature=0.7, top_k=50, top_p=0.9, seed=3, end_id=2, pad_id=0)
    params.update(sampling)
    request = GenerationRequest(
        list(PROMPT), SamplingParams(**params), streaming=True, priority=0.25
    )
    request.set_id(client_id)
    return request


def _state(value):
    """A bound object's pickled state, nested bound objects and tensors included, for comparison."""
    if isinstance(value, (list, tuple)):
        return type(value)(_state(v) for v in value)
    if isinstance(value, dict):
        return {k: _state(v) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        return "tensor", value.dtype, tuple(value.shape), value.cpu().tolist()
    if type(value).__module__.startswith("tensorrt_llm.bindings"):
        getstate = getattr(value, "__getstate__", None)
        if getstate is not None:
            return type(value).__name__, _state(getstate())
    return value


def _disagg_view(params):
    if params is None:
        return None
    return {f.name: _state(getattr(params, f.name)) for f in dataclasses.fields(params)}


def _inputs(inputs):
    """PythonRequestInputs' fields, logits processors by type and state.

    Of a ctx_usage, the engine reads the prompt and cached token counts. Of an arrival time, it
    reads the steady clock's time point, which a timedelta and its seconds as a float both name.
    """
    out = {f.name: getattr(inputs, f.name) for f in dataclasses.fields(inputs)}
    if isinstance(out["arrival_time"], datetime.timedelta):
        out["arrival_time"] = out["arrival_time"].total_seconds()
    if out["logits_post_processors"]:
        out["logits_post_processors"] = [(type(p), vars(p)) for p in out["logits_post_processors"]]
    disagg = _disagg_view(out["disaggregated_params"])
    if disagg is not None and disagg["ctx_usage"] is not None:
        disagg["ctx_usage"] = wire.encode_prefill_usage(disagg["ctx_usage"])
    out["disaggregated_params"] = disagg
    out["multimodal_data"] = _state(out["multimodal_data"])
    return out


def _retention():
    token_range = tllm.KvCacheRetentionConfig.TokenRangeRetentionConfig
    return tllm.KvCacheRetentionConfig(
        [token_range(0, 16, 80, datetime.timedelta(milliseconds=500)), token_range(16, None, 10)],
        decode_retention_priority=20,
        decode_duration_ms=datetime.timedelta(milliseconds=30),
    )


def _set(**attrs):
    def apply(request):
        for name, value in attrs.items():
            setattr(request, name, value)

    return apply


def _encoder_decoder(request):
    request.encoder_input_token_ids = [11, 12, 13]
    request.sampling_params._decoder_output_token_prefix = (0,)


# (sampling params, a change to the request): what each field the proto carries looks like
SUBMIT_CASES = {
    "plain": ({}, None),
    "logprobs_stop_bad": (
        dict(logprobs=3, stop_token_ids=[13, 14], bad_token_ids=[7], min_tokens=4),
        None,
    ),
    "ignore_eos_penalties": (
        dict(ignore_eos=True, max_tokens=None, repetition_penalty=1.1, frequency_penalty=0.2),
        None,
    ),
    "guided_json": (dict(guided_decoding=GuidedDecodingParams(json={"type": "object"})), None),
    "guided_regex": (dict(guided_decoding=GuidedDecodingParams(regex="[0-9]+")), None),
    "guided_grammar": (dict(guided_decoding=GuidedDecodingParams(grammar='root ::= "a"')), None),
    "guided_json_object": (dict(guided_decoding=GuidedDecodingParams(json_object=True)), None),
    "guided_structural_tag": (
        dict(guided_decoding=GuidedDecodingParams(structural_tag='{"type": "t"}')),
        None,
    ),
    "kv_retention": ({}, _set(kv_cache_retention_config=_retention())),
    "scheduling": (
        {},
        _set(
            scheduling_params=SchedulingParams(
                attention_dp_rank=1,
                attention_dp_relax=False,
                agent_hierarchy=[("planner", 0), ("worker", 2)],
                subagent_affinity_id="s1",
            )
        ),
    ),
    "conversation": ({}, _set(conversation_params=ConversationParams(conversation_id="c1"))),
    "kv_hints": ({}, _set(kv_hints=KvHintsSnapshot(b'{"version": 1}'))),
    "encoder_decoder": (dict(max_tokens=10), _encoder_decoder),
    "dense_bias_list": (dict(embedding_bias=[0.5, -1.25, 2.0]), None),
    "dense_bias_tensor": (
        dict(embedding_bias=torch.tensor([0.5, -1.0], dtype=torch.bfloat16)),
        None,
    ),
    "sparse_bias": (dict(embedding_bias=((3, 0.1), (9, -2.5))), None),
    "watermark_off": (dict(watermarking=False), None),
    "watermark_on": (dict(watermarking=True), None),
    "empty_cache_salt": ({}, _set(cache_salt="")),
    "cache_salt": ({}, _set(cache_salt="tenant-a")),
    "arrival_time": ({}, _set(arrival_time=12345.678901234)),
    # what the LLM API stamps a request with: bindings.steady_clock_now()
    "steady_clock_arrival_time": (
        {},
        _set(arrival_time=datetime.timedelta(seconds=12345, microseconds=678901)),
    ),
    "thinking_budget": (
        dict(logits_processor=ThinkingBudgetLogitsProcessor(16, [5, 6], [7])),
        None,
    ),
    "disagg_context_only": (
        {},
        _set(
            disaggregated_params=DisaggregatedParams(
                request_type="context_only",
                disagg_request_id=77,
                opaque_state=b"\x00state",
                ctx_info_endpoint="",
                schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
                orig_elapsed_s=0.125,
            )
        ),
    ),
    "disagg_generation_only": ({}, lambda r: _set(disaggregated_params=_handoff())(r)),
    "disagg_generation_only_prompt_tail": (
        {},
        lambda r: (_set(disaggregated_params=_handoff())(r), _set(_prompt_tail_tokens=8)(r)),
    ),
    "disagg_both": (
        {},
        _set(
            disaggregated_params=DisaggregatedParams(
                request_type="context_and_generation", disagg_request_id=5
            )
        ),
    ),
    "disagg_usage_from_serve": (
        {},
        lambda r: _set(
            disaggregated_params=_handoff(
                ctx_usage={
                    "prompt_tokens": 32,
                    "completion_tokens": 1,
                    "total_tokens": 33,
                    "prompt_tokens_details": {"cached_tokens": 8},
                    "completion_tokens_details": None,
                }
            )
        )(r),
    ),
    "disagg_generation_only_mrope": (
        {},
        lambda r: _set(
            disaggregated_params=_handoff(
                mrope_position_ids_handle=_handle(8), mrope_position_deltas_handle=_handle(9)
            )
        )(r),
    ),
    "epd_prefill": (
        {},
        lambda r: _set(
            disaggregated_params=DisaggregatedParams(
                request_type="context_only",
                disagg_request_id=9,
                multimodal_embedding_handles=[_handle(7)],
                multimodal_hashes=[list(range(1, 9))],
                multimodal_positions=[2],
                multimodal_lengths=[6],
                mrope_position_ids_handle=_handle(8),
                mrope_position_deltas_handle=_handle(9),
                multimodal_input_layout=MultimodalInputLayout(
                    prompt_token_ids=list(PROMPT),
                    multimodal_lengths=[6],
                    multimodal_positions=[2],
                    multimodal_embedding_lengths=[6],
                ),
            )
        )(r),
    ),
    "multimodal": ({}, lambda r: _set(multimodal_params=_multimodal())(r)),
    "multimodal_runs": ({}, lambda r: _set(multimodal_params=_multimodal(runs=True))(r)),
    "multimodal_data_only": (
        {},
        _set(multimodal_params=MultimodalParams(multimodal_data={"special_token_offsets": [3]})),
    ),
}


def _handle(token):
    """A shared tensor handle as a frontend sends it (a process-local one, for the test)."""
    return {"method_key": 3, "local_token": token, "producer_pid": 1}


def _multimodal(runs=False):
    """A multimodal request's params as the LLM API sends them: its tensors behind shared handles."""
    layout = dict(
        multimodal_hashes=[list(range(1, 9)), list(range(9, 17))],
        multimodal_positions=[2, 9],
        multimodal_lengths=[6, 3],
        multimodal_uuids=["img-a", None],
    )
    if runs:
        layout.update(
            multimodal_item_run_cu_offsets=[0, 2, 3],
            multimodal_run_positions=[2, 5, 9],
            multimodal_run_lengths=[2, 4, 3],
        )
    params = MultimodalParams(
        multimodal_input=MultimodalInput(**layout),
        multimodal_data={
            "image": {
                "pixel_values": torch.arange(6.0).reshape(2, 3),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            },
            "special_token_offsets": [1, 5],
            "processed": True,
            "scale": 0.5,
            "name": "qwen",
            "shape": (2, 3),
        },
    )
    params.to_handle("multimodal_data")
    return params


def _handoff(**changes):
    """A decode part's params: what the prefill part handed over."""
    params = dict(
        request_type="generation_only",
        ctx_request_id=41,
        ctx_info_endpoint="tcp://ctx:1",
        # opaque_state stays None here: the C++ ContextPhaseParams parses it as a transceiver state
        ctx_dp_rank=2,
        opaque_state=None,
        first_gen_tokens=[150],
        draft_tokens=[151, 152],
        first_gen_log_probs=[
            {150: Logprob(logprob=SAMPLED, rank=1), 3: Logprob(logprob=TOPK[1], rank=2)}
        ],
        first_gen_logits=[torch.arange(8, dtype=torch.float32).reshape(1, 8)],
        remote_tail_start=None,
        ctx_usage={
            "prompt_tokens": 32,
            "completion_tokens": 0,
            "total_tokens": 32,
            "prompt_tokens_details": {"cached_tokens": 16},
        },
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
        orig_elapsed_s=0.5,
    )
    params.update(changes)
    return DisaggregatedParams(**params)


def _case_request(case, client_id=7):
    sampling, change = SUBMIT_CASES[case]
    request = _request(client_id, **sampling)
    if change is not None:
        change(request)
    return request


def _disagg_worker():
    w, sent = _worker()
    w.engine.kv_cache_transceiver = object()
    return w, sent


def _submit_both(make_request, worker=_disagg_worker):
    """The request ``make_request`` gives, through _enqueue_request and through the proto.

    The first goes through the queue's pickling.
    """
    pickled, _ = worker()
    pickled._enqueue_request(pickle.loads(pickle.dumps(make_request())))
    proto, _ = worker()
    frame = engine_client.encode(make_request(), postprocess=False)
    assert frame is not None
    engine_server.submit(proto, frame)
    (want,), (got,) = pickled.engine.enqueued, proto.engine.enqueued
    assert _state(got.request) == _state(want.request)
    assert _inputs(got.inputs) == _inputs(want.inputs)
    assert isinstance(proto._results[7], engine_server.RequestEntry)
    return proto


@pytest.mark.parametrize("case", list(SUBMIT_CASES))
def test_submit_builds_what_enqueue_request_builds(case):
    _submit_both(lambda: _case_request(case))


class _LoraManager:
    def __init__(self):
        self.cpp_lora_config = {}
        self.cpp_lora_weights = {}
        self.loaded = []

    def load_from_ckpt(self, paths, model_config, uids, ckpt_source):
        self.loaded.append((tuple(paths), tuple(uids), ckpt_source))
        for uid in uids:
            self.cpp_lora_config[uid] = torch.tensor([[0, 1, 8]], dtype=torch.int32)


def _lora_worker():
    w, sent = _disagg_worker()
    w._lora_manager = _LoraManager()
    w._lora_model_config = None
    w._lora_adapter_id_locks_guard = threading.Lock()
    w._lora_adapter_id_locks = {}
    w._lora_adapter_identities = {}
    w._load_lora_adapter = MethodType(BaseWorker._load_lora_adapter, w)
    return w, sent


@pytest.mark.parametrize("source", ["hf", "nemo"])
def test_submit_loads_lora_as_enqueue_request_does(source):
    def make_request():
        request = _request(7)
        request.lora_request = LoRARequest("adapter", 3, "/adapters/a", source)
        return request

    proto = _submit_both(make_request, worker=_lora_worker)
    assert proto._lora_manager.loaded == [(("/adapters/a",), ("3",), source)]


@pytest.mark.parametrize(
    "case",
    [
        dict(max_tokens=0),
        dict(max_tokens=-3),
        dict(max_seq_len=8),
        dict(max_seq_len=None, max_tokens=None),
        dict(encoder_decoder_max_tokens=1),
    ],
)
def test_submit_refuses_what_enqueue_request_refuses(case):
    case = dict(case)
    max_seq_len = case.pop("max_seq_len", 4096)
    prefix_max_tokens = case.pop("encoder_decoder_max_tokens", None)

    def make_request():
        if prefix_max_tokens is None:
            return _request(7, **case)
        request = _request(7, max_tokens=prefix_max_tokens)
        _encoder_decoder(request)
        return request

    pickled, _ = _worker()
    pickled.max_seq_len = max_seq_len
    with pytest.raises(RequestError) as want:
        pickled._enqueue_request(make_request())
    proto, _ = _worker()
    proto.max_seq_len = max_seq_len
    engine_server.submit(proto, engine_client.encode(make_request(), postprocess=False))
    error = proto._await_response_helper.temp_error_responses.get_nowait()
    assert (error.client_id, str(error.error_msg)) == (7, str(want.value))
    assert not proto.engine.enqueued and not proto._results


def test_disagg_without_a_transceiver_fails_as_enqueue_request_does():
    def make_request():
        request = _request(7)
        request.disaggregated_params = DisaggregatedParams(
            request_type="context_only", disagg_request_id=3
        )
        return request

    pickled, _ = _worker()
    with pytest.raises(AssertionError) as want:
        pickled._enqueue_request(make_request())
    proto, _ = _worker()
    with pytest.raises(AssertionError) as got:
        engine_server.submit(proto, engine_client.encode(make_request(), postprocess=False))
    assert str(got.value) == str(want.value)


def test_lora_load_failure_is_refused_as_enqueue_request_refuses_it():
    def failing_load(*args, **kwargs):
        raise OSError("no such adapter")

    def make_request():
        request = _request(7)
        request.lora_request = LoRARequest("adapter", 3, "/adapters/missing", "hf")
        return request

    pickled, _ = _lora_worker()
    pickled._lora_manager.load_from_ckpt = failing_load
    with pytest.raises(RequestError) as want:
        pickled._enqueue_request(make_request())
    proto, _ = _lora_worker()
    proto._lora_manager.load_from_ckpt = failing_load
    engine_server.submit(proto, engine_client.encode(make_request(), postprocess=False))
    error = proto._await_response_helper.temp_error_responses.get_nowait()
    assert str(error.error_msg) == str(want.value)


def test_requests_the_proto_cannot_carry_keep_the_pickled_path():
    # a Python callable is code, not data
    request = _request(7, logits_processor=lambda *args: None)
    assert engine_client.encode(request, postprocess=False) is None
    request = _request(9)
    request.multimodal_params = MultimodalParams(multimodal_data={"sizes": np.array([1, 2])})
    assert engine_client.encode(request, postprocess=False) is None


def test_postprocess_workers_keep_the_pickled_path_for_logits_and_prompt_in_output():
    for sampling in (
        dict(return_context_logits=True),
        dict(return_generation_logits=True),
        dict(exclude_input_from_output=False),
    ):
        assert engine_client.encode(_request(8, **sampling), postprocess=True) is None
    assert engine_client.encode(_request(8), postprocess=True) is not None


def _llm_requests(enqueued, children):
    """The executor's requests for what a path enqueued.

    One per request, plus a child per extra sampled sequence.
    """
    out = []
    for item in enqueued:
        request = item.request
        output = request.output_config
        r = LlmRequest(
            request_id=request.client_id,
            max_new_tokens=request.max_tokens,
            input_tokens=list(PROMPT),
            sampling_config=request.sampling_config,
            is_streaming=request.streaming,
            client_id=request.client_id,
            return_log_probs=output.return_log_probs,
            return_perf_metrics=output.return_perf_metrics,
            exclude_input_from_output=output.exclude_input_from_output,
            logprobs_simple_format=item.inputs.logprobs_simple_format,
            arrival_time=item.inputs.arrival_time,
            return_context_logits=output.return_context_logits,
            context_logits_explicit=item.inputs.context_logits_explicit,
            prompt_logprobs=item.inputs.prompt_logprobs,
            prompt_logprobs_simple_format=item.inputs.prompt_logprobs_simple_format,
            return_generation_logits=output.return_generation_logits,
            additional_outputs=[o.name for o in output.additional_model_outputs]
            if output.additional_model_outputs
            else None,
            return_logits_device_memory=False,
            use_chunked_generation_logits=False,
            return_routed_experts=item.inputs.return_routed_experts,
            llm_request_type=REQUEST_TYPE_MAPPING[request.request_type],
            context_phase_params=request.context_phase_params,
        )
        r.py_disaggregated_params = item.inputs.disaggregated_params
        family = [r]
        for k in range(1, children):
            r.create_child_request(request.client_id * 10 + k)
            family.append(r.child_requests[-1])
        for member in family:
            member.state = LlmRequestState.GENERATION_IN_PROGRESS
        out.append(family)
    return out


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
    if step == 0 and r.py_result._context_logits is not None:
        r.py_result.append_context_logits(_logits(len(PROMPT), r.py_client_id))
    if r.py_result._generation_logits is not None:
        r.py_result.append_generation_logits(_logits(tokens, step).unsqueeze(1))
    if r.py_result._additional_generation_outputs is not None:
        r.py_result.append_additional_generation_outputs("hidden", _logits(tokens, step + 7)[:, :4])
    # the k-th child of an n > 1 request samples its own tokens when the case asks
    child = 0 if r.py_request_id == r.py_client_id else r.py_request_id - 10 * r.py_client_id
    shift = 50 * child if case.get("children_differ") else 0
    new = [[100 + 10 * beam + step + j + shift for j in range(tokens)] for beam in range(beams)]
    for beam in range(beams):
        for token in new[beam]:
            r.add_new_token(token, beam)
    if case.get("values_only"):
        r.py_result.append_log_probs([[SAMPLED] * tokens for _ in range(beams)])
    elif topk is not None:
        # a candidate ranked 2 with none ranked 1: a shape only the pickled path carries
        odd = case.get("odd_logprobs_at") == step and r.py_client_id == 100
        top = {200 + t: Logprob(logprob=TOPK[t], rank=t + 1 + odd) for t in range(topk)}
        # the sampler's dict: the sampled token first, then the top-k
        r.py_result.append_log_probs(
            [
                [{token: Logprob(logprob=SAMPLED, rank=topk + 1), **top} for token in new[beam]]
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
                [{new[beam][-1]: Logprob(logprob=TOPK[beam], rank=None)} for beam in range(beams)]
            )
        r.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE
        return
    if finish and case.get("encoder_handoff"):
        # an encoder part's embeddings and mRoPE positions, behind shared handles
        r.py_result._mm_embeddings = [_handle(70)]
        r.py_result._mrope_position_ids = _handle(80)
        r.py_result._mrope_position_deltas = _handle(81)
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


def _tensors(outputs):
    return None if outputs is None else {name: _state(t) for name, t in outputs.items()}


def _users_view(frontend):
    """What users read from each GenerationResult."""
    view = {}
    for cid, g in sorted(frontend.items()):
        outputs = [
            (
                list(o.token_ids),
                [d if isinstance(d, float) else dict(d) for d in o.logprobs],
                _disagg_view(o.disaggregated_params),
                o.cumulative_logprob,
                o.finish_reason,
                o._spec_dec_counters,
                _state(o.request_perf_metrics),
                None
                if o.prompt_logprobs is None
                else [d if isinstance(d, float) else dict(d) for d in o.prompt_logprobs],
                _state(o.generation_logits),
                _tensors(o.additional_context_outputs),
                _tensors(o.additional_generation_outputs),
            )
            for o in g._outputs
        ]
        view[cid] = (
            outputs,
            [o.index for o in g.outputs],
            _state(g.context_logits),
            _disagg_view(g.disaggregated_params),
            g._done,
            g._error_msg,
            g.decoding_iter,
            g.avg_decoded_tokens_per_iter,
            g.cached_tokens,
            g.ctx_computed_tokens,
            g.ctx_first_begin,
            g.ctx_num_chunks,
            g.per_pos_drafted,
            g.per_pos_accepted,
            g.spec_dec_totals,
            g.time_breakdown_metrics,
            dict(g.metrics_dict),
        )
    return view


class _Pipe:
    """A postprocess worker's input queue."""

    def __init__(self):
        self.items = []

    async def get_async(self):
        return self.items.pop(0)


def _postprocessor():
    worker = object.__new__(PostprocWorker)
    worker._records = {}
    worker._record_creator = PostprocWorker.default_record_creator
    worker._tokenizer = None
    worker._post_processor_hook = None
    worker._pull_pipe = _Pipe()
    return worker


def _postprocess(worker, items):
    """What a postprocess worker sends the proxy for ``items``."""
    worker._pull_pipe.items = list(items) + [None]
    worker._to_stop = asyncio.Event()
    out = []

    async def run():
        async for batch in worker._mainloop():
            if batch is None:
                break
            out.extend(batch)

    asyncio.run(run())
    return out


def _flow(proto, case, steps=3, requests=3, postprocess=False):
    """Run requests through submission, ``steps`` sampler steps on rank 0, and the frontend.

    The postprocess workers run too when asked. Returns the users' view after every step, and
    the types rank 0 sent.
    """
    wire.ENABLED = proto
    worker, sent = _worker()
    lanes = []
    if postprocess:
        worker.enable_postprocess_parallel = True
        worker.postproc_config = types.SimpleNamespace(num_postprocess_workers=2, enabled=True)
        lanes = [[], []]
        worker.postproc_queues = [types.SimpleNamespace(put=lane.append) for lane in lanes]
        postprocessors = [_postprocessor() for _ in lanes]
    frontend = {}
    sampling = dict(case.get("sampling", {}))
    streaming = case.get("streaming", True)
    for i in range(requests):
        client_id = 100 + i
        generation = _request(client_id, **sampling)
        generation.streaming = i % 2 == 0 if streaming == "mixed" else streaming
        generation.arrival_time = 1000.0 + i / 8
        if "disagg" in case:
            generation.disaggregated_params = case["disagg"](client_id)
            worker.engine.kv_cache_transceiver = object()
        if "multimodal" in case:
            generation.multimodal_params = case["multimodal"]()
        frontend[client_id] = GenerationResult(
            generation, disaggregated_params=generation.disaggregated_params
        )
        if proto:
            engine_server.submit(worker, engine_client.encode(generation, postprocess=bool(lanes)))
            # rows say the decoding counters for streaming requests; sequences' ends for the others
            assert worker._results[client_id].decoding_stats == generation.streaming
        else:
            worker.submit(generation)
    sp = SamplingParams(**{**dict(max_tokens=64, temperature=0.7, end_id=2), **sampling})
    children = sp.best_of if not sp.use_beam_search else 1
    families = _llm_requests(worker.engine.enqueued, children)
    helper = AwaitResponseHelper(worker)
    sent_types, views = set(), []
    live = {family[0].py_client_id: list(family) for family in families}
    held = []
    for step in range(steps):
        responses = []
        for cid, members in list(live.items()):
            if step == 1 and cid == 101 and case.get("error"):
                responses.append(LlmResponse(request_id=cid, error_msg="boom", client_id=cid))
                del live[cid]
                continue
            if step == 0 and cid == 101 and case.get("cancel_first"):
                # cancelled before its first token: nothing was sampled for it
                for r in members:
                    r.state = LlmRequestState.GENERATION_COMPLETE
                    for beam in range(case.get("beams", 1)):
                        r.set_finished_reason(tllm.FinishReason.CANCELLED, beam)
                    if (rsp := prepare_response(r)) is not None:
                        responses.append(rsp)
                del live[cid]
                continue
            for k, r in enumerate(list(members)):
                # children end one step apart, the last at the last step
                finish = step == steps - 1 - (len(members) - 1 - k)
                _sample(r, step, case, finish)
                if (rsp := prepare_response(r)) is not None:
                    responses.append(rsp)
                if finish:
                    members.remove(r)
            if not members:
                del live[cid]
        if case.get("pair_steps") and step % 2 == 0 and step + 1 < steps:
            # the executor's responses of two steps reach the await thread together
            held = responses
            views.append(_users_view(frontend))
            continue
        responses, held = held + responses, []
        helper.handle_for_ipc_batched(responses)
        for lane, postprocessor in zip(lanes, postprocessors if lanes else ()):
            sent_types.update(type(item).__name__ for item in lane)
            sent.append(_postprocess(postprocessor, lane))
            lane.clear()
        for message in sent:
            for item in message if isinstance(message, list) else [message]:
                if not lanes:
                    sent_types.add(type(item).__name__)
                rows = (
                    engine_client.decode(item, frontend)
                    if isinstance(item, wire.StepFrame)
                    else [item]
                )
                for row in rows:
                    frontend[row.client_id]._handle_response(row)
        sent.clear()
        views.append(_users_view(frontend))
    assert not worker._results
    return views, sent_types


RESPONSE_CASES = {
    **{
        f"top{topk}_tokens{tokens}": dict(sampling=dict(logprobs=topk), topk=topk, tokens=tokens)
        for topk in (None, 0, 1, 5)
        for tokens in (1, 3)
    },
    "values_only": dict(
        sampling=dict(logprobs=0, logprobs_simple_format=True), values_only=True, tokens=2
    ),
    "drafting": dict(drafting=True, tokens=2),
    "drafting_non_streaming": dict(drafting=True, tokens=2, streaming=False),
    "drafting_mixed_streaming": dict(drafting=True, tokens=2, streaming="mixed"),
    "perf_metrics": dict(sampling=dict(return_perf_metrics=True), perf=True),
    "error": dict(sampling=dict(logprobs=1), topk=1, error=True),
    "cancelled_before_first_token": dict(
        sampling=dict(logprobs=1), topk=1, cancel_first=True, pickled=True
    ),
    "two_steps_per_batch": dict(
        sampling=dict(logprobs=1, prompt_logprobs=2), topk=1, pair_steps=True, steps=4
    ),
    "two_steps_per_batch_one_pickled": dict(
        sampling=dict(logprobs=1, prompt_logprobs=2),
        topk=1,
        pair_steps=True,
        odd_logprobs_at=1,
        steps=4,
        pickled=True,
    ),
    "n2": dict(sampling=dict(n=2, logprobs=1), topk=1),
    "n2_non_streaming": dict(sampling=dict(n=2), drafting=True, streaming=False),
    "n2_of_best_of3": dict(sampling=dict(n=2, best_of=3, logprobs=0), topk=0, steps=4),
    "beams2": dict(
        sampling=dict(use_beam_search=True, n=2, best_of=2, logprobs=1), topk=1, beams=2
    ),
    "beams2_no_logprobs": dict(
        sampling=dict(use_beam_search=True, n=2, best_of=2), beams=2, tokens=2
    ),
    "non_streaming": dict(sampling=dict(logprobs=1), topk=1, streaming=False),
    "with_prompt": dict(sampling=dict(exclude_input_from_output=False), streaming=False, tokens=2),
    "prompt_logprobs_2": dict(sampling=dict(prompt_logprobs=2, logprobs=1), topk=1),
    "prompt_logprobs_0": dict(sampling=dict(prompt_logprobs=0)),
    "prompt_logprobs_values_only": dict(
        sampling=dict(prompt_logprobs=0, prompt_logprobs_simple_format=True)
    ),
    "prompt_logprobs_non_streaming": dict(sampling=dict(prompt_logprobs=3), streaming=False),
    # the second sequence repeats the prompt logprobs the first computed, first token and all
    "n2_prompt_logprobs_children_differ": dict(
        sampling=dict(n=2, prompt_logprobs=2), children_differ=True
    ),
    "prompt_logprobs_and_context_logits": dict(
        sampling=dict(prompt_logprobs=1, return_context_logits=True)
    ),
    "context_logits": dict(sampling=dict(return_context_logits=True)),
    "generation_logits": dict(sampling=dict(return_generation_logits=True), tokens=2),
    "generation_logits_non_streaming": dict(
        sampling=dict(return_generation_logits=True), streaming=False
    ),
    "additional_outputs": dict(sampling=dict(additional_model_outputs=["hidden"])),
    "routed_experts": dict(sampling=dict(return_routed_experts=True), routed_experts=True),
    "disagg_context_only": dict(
        sampling=dict(logprobs=1),
        topk=1,
        context_only=True,
        disagg=lambda cid: DisaggregatedParams(request_type="context_only", disagg_request_id=cid),
    ),
    "disagg_context_only_remote_tail": dict(
        context_only=True,
        remote_tail=True,
        disagg=lambda cid: DisaggregatedParams(request_type="context_only", disagg_request_id=cid),
    ),
    "disagg_context_only_beams": dict(
        sampling=dict(use_beam_search=True, n=2, best_of=2),
        beams=2,
        context_only=True,
        first_gen_beam=True,
        disagg=lambda cid: DisaggregatedParams(request_type="context_only", disagg_request_id=cid),
    ),
    # a postprocess worker reads the multimodal spans from the request's disaggregated params
    "disagg_context_only_multimodal": dict(
        context_only=True,
        disagg=lambda cid: DisaggregatedParams(request_type="context_only", disagg_request_id=cid),
        multimodal=lambda: MultimodalParams(
            multimodal_input=MultimodalInput(
                multimodal_hashes=[list(range(1, 9))],
                multimodal_positions=[2],
                multimodal_lengths=[6],
            ),
        ),
    ),
    "disagg_context_only_generation_logits": dict(
        sampling=dict(return_generation_logits=True),
        context_only=True,
        disagg=lambda cid: DisaggregatedParams(request_type="context_only", disagg_request_id=cid),
    ),
    "disagg_generation_first_usage_from_kv": dict(
        usage_from_kv=True,
        disagg=lambda cid: _handoff(
            ctx_request_id=cid,
            first_gen_tokens=[100],
            first_gen_log_probs=None,
            first_gen_logits=None,
            schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
        ),
    ),
    "multimodal_encoder_handoff": dict(
        encoder_handoff=True,
        multimodal=lambda: MultimodalParams(
            multimodal_input=MultimodalInput(
                multimodal_hashes=[list(range(1, 9))],
                multimodal_positions=[2],
                multimodal_lengths=[6],
            ),
            multimodal_data={"special_token_offsets": [1]},
        ),
    ),
    "disagg_generation_first_ctx_usage": dict(
        sampling=dict(logprobs=0),
        topk=0,
        disagg=lambda cid: _handoff(
            ctx_request_id=cid,
            first_gen_tokens=[100],
            first_gen_log_probs=None,
            first_gen_logits=None,
            schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
        ),
    ),
}


@pytest.mark.parametrize("name", list(RESPONSE_CASES))
def test_rows_give_users_what_pickled_responses_give(name):
    case = RESPONSE_CASES[name]
    steps = case.get("steps", 3)
    enabled = wire.ENABLED
    try:
        want, _ = _flow(False, case, steps)
        got, proto_types = _flow(True, case, steps)
    finally:
        wire.ENABLED = enabled
    # a case marked "pickled" sends some responses down the pickled path on purpose
    assert "StepFrame" in proto_types and (case.get("pickled") or proto_types == {"StepFrame"}), (
        proto_types
    )
    for step, (g, w) in enumerate(zip(got, want)):
        assert g == w, f"step {step}: {_first_difference(g, w)}"


@pytest.mark.parametrize(
    "name",
    [
        "top1_tokens3",
        "values_only",
        "drafting",
        "drafting_mixed_streaming",
        "perf_metrics",
        "error",
        "n2",
        "beams2",
        "non_streaming",
        "prompt_logprobs_2",
        "additional_outputs",
        "cancelled_before_first_token",
        "two_steps_per_batch_one_pickled",
        "disagg_generation_first_ctx_usage",
        "disagg_context_only_multimodal",
        "n2_prompt_logprobs_children_differ",
    ],
)
def test_postprocess_workers_give_users_what_pickled_responses_give(name):
    case = RESPONSE_CASES[name]
    steps = case.get("steps", 3)
    enabled = wire.ENABLED
    try:
        want, _ = _flow(False, case, steps, postprocess=True)
        got, proto_types = _flow(True, case, steps, postprocess=True)
    finally:
        wire.ENABLED = enabled
    # a case marked "pickled" sends some responses down the pickled path on purpose
    assert "StepFrame" in proto_types and (case.get("pickled") or proto_types == {"StepFrame"}), (
        proto_types
    )
    for step, (g, w) in enumerate(zip(got, want)):
        assert g == w, f"step {step}: {_first_difference(g, w)}"


def _first_difference(got, want, path=""):
    """Where two views first differ, for a readable failure."""
    if isinstance(got, dict) and isinstance(want, dict) and got.keys() == want.keys():
        for key in got:
            if got[key] != want[key]:
                return _first_difference(got[key], want[key], f"{path}[{key!r}]")
    if isinstance(got, (list, tuple)) and isinstance(want, (list, tuple)) and len(got) == len(want):
        for i, (g, w) in enumerate(zip(got, want)):
            if g != w:
                return _first_difference(g, w, f"{path}[{i}]")
    return f"{path}: {got!r} != {want!r}"


def test_a_malformed_step_output_raises():
    # one handle and no offsets or finish column
    frame = wire.StepFrame(b"\x1a\x08" + (7).to_bytes(8, "little"))
    with pytest.raises(RuntimeError, match="malformed StepOutput"):
        engine_client.decode(frame, {})


def test_an_unreadable_submit_frame_rejects_its_requests():
    worker, _ = _worker()
    # a Request whose bytes stop short of its length
    engine_server.submit(worker, wire.SubmitFrame(b"\x0a\x05\xff", (5,)))
    error = worker._await_response_helper.temp_error_responses.get_nowait()
    assert error.client_id == 5 and "cannot read the request" in str(error.error_msg)
    assert not worker._results
