# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cases test_engine_wire.py runs through the executor's wire, as data.

- ``SUBMIT_CASES``: requests rank 0 builds an executor request from, by what of the engine
  contract they carry.
- ``REFUSAL_CASES``: requests rank 0 refuses, with an error.
- ``LORA_SOURCES``: the checkpoint sources of a request's LoRA adapter (``LORA_ADAPTER``).
- ``RESPONSE_CASES``: what the sampler gives three requests over a few steps.
- ``POSTPROCESS_CASES``: the response cases that also run through postprocess workers.

Every request starts from ``base_request``. A ``Logprob`` here is ``(logprob, rank)``: the module
imports only public modules, not where the executor defines its types, so a harness of another
version of the executor runs the same cases.
"""

import datetime
import struct
import types

import torch

from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.conversation_params import ConversationParams
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalInputLayout, MultimodalParams
from tensorrt_llm.kv_hints.protocol import KvHintsSnapshot
from tensorrt_llm.llmapi.thinking_budget import ThinkingBudgetLogitsProcessor
from tensorrt_llm.sampling_params import GuidedDecodingParams, SamplingParams
from tensorrt_llm.scheduling_params import SchedulingParams

PROMPT = list(range(10, 22))
# float32 values, as the sampler's tensors give; their float64 sums are not float32
SAMPLED = struct.unpack("f", struct.pack("f", -0.3))[0]
TOPK = [struct.unpack("f", struct.pack("f", -1.1 - t))[0] for t in range(8)]


def base_request(client_id: int, **sampling) -> GenerationRequest:
    """The streaming request every case starts from; ``sampling`` changes its sampling params."""
    params = dict(max_tokens=64, temperature=0.7, top_k=50, top_p=0.9, seed=3, end_id=2, pad_id=0)
    params.update(sampling)
    request = GenerationRequest(
        list(PROMPT), SamplingParams(**params), streaming=True, priority=0.25
    )
    request.set_id(client_id)
    return request


def keep_logits(req_id, logits, token_ids, stream_ptr, client_id):
    """A logits processor a user passes by reference: code, which no Value can say."""


def _set(**attrs):
    def apply(request):
        for name, value in attrs.items():
            setattr(request, name, value)

    return apply


def _retention():
    token_range = tllm.KvCacheRetentionConfig.TokenRangeRetentionConfig
    return tllm.KvCacheRetentionConfig(
        [token_range(0, 16, 80, datetime.timedelta(milliseconds=500)), token_range(16, None, 10)],
        decode_retention_priority=20,
        decode_duration_ms=datetime.timedelta(milliseconds=30),
    )


def _encoder_decoder(request):
    request.encoder_input_token_ids = [11, 12, 13]
    request.sampling_params._decoder_output_token_prefix = (0,)


def shared_handle(token):
    """A shared tensor handle, as the frontend and the engine pass them (a process-local one)."""
    return {"method_key": 3, "local_token": token, "producer_pid": 1}


def _multimodal(runs=False):
    """A multimodal request's params as the LLM API sends them, tensors behind shared handles."""
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
        first_gen_log_probs=[{150: (SAMPLED, 1), 3: (TOPK[1], 2)}],
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
    "thinking_budget": (
        dict(logits_processor=ThinkingBudgetLogitsProcessor(16, [5, 6], [7])),
        None,
    ),
    # code rides beside the message, by reference
    "logits_processor": (dict(logits_processor=[keep_logits]), None),
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
                mrope_position_ids_handle=shared_handle(8),
                mrope_position_deltas_handle=shared_handle(9),
            )
        )(r),
    ),
    "epd_prefill": (
        {},
        lambda r: _set(
            disaggregated_params=DisaggregatedParams(
                request_type="context_only",
                disagg_request_id=9,
                multimodal_embedding_handles=[shared_handle(7)],
                multimodal_hashes=[list(range(1, 9))],
                multimodal_positions=[2],
                multimodal_lengths=[6],
                mrope_position_ids_handle=shared_handle(8),
                mrope_position_deltas_handle=shared_handle(9),
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


def _unknown_request_type(request):
    # the constructor checks the type; a caller can still change it afterwards
    request.disaggregated_params = DisaggregatedParams(request_type="context_and_generation")
    request.disaggregated_params.request_type = "context_and_decode"


# sampling: the request's sampling params; change: a change to the request; max_seq_len: rank 0's,
# None when it cannot be deduced (4096 when not given); kv_cache_transceiver: whether rank 0 has one
REFUSAL_CASES = {
    "zero_max_tokens": dict(sampling=dict(max_tokens=0)),
    "negative_max_tokens": dict(sampling=dict(max_tokens=-3)),
    "prompt_longer_than_max_seq_len": dict(max_seq_len=8),
    "no_max_tokens_nor_max_seq_len": dict(sampling=dict(max_tokens=None), max_seq_len=None),
    "decoder_prefix_takes_every_token": dict(sampling=dict(max_tokens=1), change=_encoder_decoder),
    "prompt_adapter": dict(
        change=_set(prompt_adapter_request=types.SimpleNamespace(adapter_id=1)),
        kv_cache_transceiver=True,
    ),
    "embedding_bias_of_unknown_type": dict(
        change=lambda r: setattr(r.sampling_params, "embedding_bias", {"not": "a bias"}),
        kv_cache_transceiver=True,
    ),
    "unknown_request_type": dict(change=_unknown_request_type, kv_cache_transceiver=True),
}

# a request's LoRA adapter (name, id, path), loaded from each checkpoint source
LORA_ADAPTER = ("adapter", 3, "/adapters/a")
LORA_SOURCES = ("hf", "nemo")

# What the sampler gives requests 100, 101 and 102 at each of ``steps`` (3) steps, the last of
# which ends them (a beam's or a child's sequence each). sampling: their sampling params;
# streaming: True, False or "mixed" (only the even ones); tokens: new tokens a step; topk: the
# top-k logprobs a token comes with (None: no logprobs); values_only: a logprob value a token;
# drafting: speculative decoding counters from the second step; perf: time breakdown at the end;
# error: request 101 fails at the second step; cancel_first: request 101 is cancelled before its
# first token; pair_steps: the responses of two steps reach rank 0's await thread together;
# beams: beam width; children_differ: an n > 1 request's children sample their own tokens;
# routed_experts, context_only (the prefill part ends with its handoff; remote_tail,
# first_gen_beam), encoder_handoff, usage_from_kv: what the executor attaches at the end;
# disagg: the request's disaggregated params, by client id; multimodal: its multimodal params.
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
    "cancelled_before_first_token": dict(sampling=dict(logprobs=1), topk=1, cancel_first=True),
    "two_steps_per_batch": dict(
        sampling=dict(logprobs=1, prompt_logprobs=2), topk=1, pair_steps=True, steps=4
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

POSTPROCESS_CASES = [
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
    "disagg_generation_first_ctx_usage",
    "context_logits",
    "with_prompt",
    "disagg_context_only_multimodal",
    "n2_prompt_logprobs_children_differ",
]
