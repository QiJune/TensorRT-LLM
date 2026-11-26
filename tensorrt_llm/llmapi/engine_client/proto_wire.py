# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The frontend's half of the executor's wire: requests to the engine contract, its rows back.

A frontend sends rank 0 each request as a ``SubmitBatch`` (``encode``), and the frontend and the
postprocess workers read the rows of the ``StepOutput`` rank 0 sends them each step (``decode``)
into objects ``GenerationResult`` reads. Python objects a message refers to (logits processors,
model inputs no Value can say) travel beside it in the frame. A request rank 0 refuses gets its
error from rank 0.
"""

import functools
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from ...executor.logprobs import Logprob, LogProbsResult
from ...executor.wire import (
    DISAGG_PHASES,
    DTYPES,
    LOGPROB_MODES,
    FrontendContext,
    StepFrame,
    SubmitFrame,
    decode_frontend_context,
    device_tensor,
    encode_disagg,
    encode_frontend_context,
    encode_multimodal,
    native,
    tensor_ref,
)

if TYPE_CHECKING:
    from ...executor.request import GenerationRequest


class StepRowResult:
    """A row's result, read by ``GenerationResult._handle_step_row``.

    One per response, so a beam search row has a list per beam. ``decode`` fills the slots;
    ``_handle_step_row`` sets the ``per_pos_*`` and ``spec_dec_totals`` slots that
    ``_handle_sequence`` reads. A row without extras is this class; its class attributes are the
    values that ``StepRowExtras`` and ``StepRowOutputs`` hold in slots.
    """

    __slots__ = (
        "is_final",
        "decoding_iter",
        "avg_decoded_tokens_per_iter",
        "finish_reasons",
        "output_token_ids",
        "log_probs",
        "cum_log_probs",
        "per_pos_drafted",
        "per_pos_accepted",
        "spec_dec_totals",
    )
    sequence_index = 0
    counters = None
    spec = None
    request_perf_metrics = None
    time_breakdown_metrics = None
    context_logits = None
    generation_logits = None
    additional_context_outputs = None
    additional_generation_outputs = None
    kv_export = None
    encoder_export = None
    prefill_usage = None


class StepRowExtras(StepRowResult):
    """A row's result with extras.

    ``counters`` is (cached_tokens, ctx_computed_tokens, ctx_first_begin, ctx_num_chunks) and
    ``spec`` is (per_pos_drafted, per_pos_accepted, spec_dec_totals), each None where they have
    not changed.
    """

    __slots__ = (
        "sequence_index",
        "counters",
        "spec",
        "request_perf_metrics",
        "time_breakdown_metrics",
    )


class StepRowOutputs(StepRowExtras):
    """A row's result that also has prompt logprobs, tensors or handoffs.

    ``decode`` fills ``prompt_logprobs``, ``tensors`` ({name: (dtype, shape, bytes, device)}),
    ``output_groups``, the handoffs ``kv_export`` (a prefill part's, by DisaggregatedParams field)
    and ``encoder_export`` (an encoder part's shared tensor handles: ``mm_embedding_handles``,
    ``mrope_position_ids_handle`` and ``mrope_position_deltas_handle``), and ``prefill_usage``
    ((prompt_tokens, cached_tokens)); ``take_outputs`` turns the tensors into the logits and
    additional outputs ``_handle_sequence`` reads.
    """

    __slots__ = (
        "prompt_logprobs",
        "tensors",
        "output_groups",
        "context_logits",
        "generation_logits",
        "additional_context_outputs",
        "additional_generation_outputs",
        "kv_export",
        "encoder_export",
        "prefill_usage",
    )


def take_outputs(row: StepRowOutputs) -> Optional[LogProbsResult]:
    """Set ``row``'s logits and additional outputs from its tensors.

    Returns the row's prompt logprobs as a ``LogProbsResult``, or None.
    """
    row.context_logits = row.generation_logits = None
    row.additional_context_outputs = row.additional_generation_outputs = None
    for group in row.output_groups:
        setattr(row, group, {})
    for key, ref in row.tensors.items():
        tensor = device_tensor(*ref)
        group, _, name = key.partition("/")
        if name:
            getattr(row, group)[name] = tensor
        else:
            setattr(row, key, tensor)
    if row.prompt_logprobs is None:
        return None
    return LogProbsResult(prompt=row.prompt_logprobs)


class StepRow:
    """One response of a ``StepFrame``, dispatched to its ``GenerationResult``."""

    __slots__ = ("client_id", "request_id", "result")
    error_msg = None


class StepRowError(StepRow):
    """A row that ends its request with ``error_msg``."""

    __slots__ = ("error_msg",)


def release_undelivered(response) -> None:
    """Consume the shared tensor handles of a response that no result will receive.

    An encoder part's row exports its embeddings and mRoPE positions, which their producer frees
    only once a consumer opens the handles or they are consumed here; a postprocess worker's output
    carries them in its disaggregated params. Error responses export nothing.
    """
    if isinstance(response, StepRow):
        if (export := response.result.encoder_export) is None:
            return
        handles = (
            export.get("mm_embedding_handles"),
            export.get("mrope_position_ids_handle"),
            export.get("mrope_position_deltas_handle"),
        )
    else:
        # a postprocess worker's output; an ErrorResponse has no params
        if (params := getattr(response, "disaggregated_params", None)) is None:
            return
        handles = (
            params.multimodal_embedding_handles,
            params.mrope_position_ids_handle,
            params.mrope_position_deltas_handle,
        )
    # tensorrt_llm._torch imports the LLM API, which imports this module
    from ..._torch.shared_tensor import release_exported_handles

    release_exported_handles(*handles)


def _thinking_budget(processors) -> Optional[tuple]:
    """(budget, start ids, end ids) when ``processors`` is a thinking budget alone, else None."""
    from ..thinking_budget import ThinkingBudgetLogitsProcessor

    if isinstance(processors, list) and len(processors) == 1:
        processors = processors[0]
    if type(processors) is not ThinkingBudgetLogitsProcessor:
        return None
    return (
        processors.thinking_token_budget,
        processors.reasoning_start_token_ids,
        processors.reasoning_end_token_ids,
    )


def _refusal(request: "GenerationRequest", postprocess: bool) -> Optional[str]:
    """The error rank 0 refuses ``request`` with on sight, as it refuses it, or None.

    ``postprocess`` is whether postprocess workers read the request's rows.
    """
    sp = request.sampling_params
    if request.prompt_adapter_request is not None:
        return "Prompt adapters are not supported by the PyTorch executor."
    if postprocess and (sp.return_context_logits or sp.return_generation_logits):
        return (
            "Postprocessing error: Context logits or generation logits are not supposed to be "
            "sent to postprocessing workers."
        )
    bias = sp.embedding_bias
    if bias is not None and not isinstance(bias, (list, tuple)):
        if not isinstance(bias, torch.Tensor):
            return "Unsupported embedding bias type"
        if bias.dtype not in DTYPES:
            return f"Unsupported embedding bias dtype {bias.dtype}"
    if (params := request.disaggregated_params) is not None:
        if params.request_type not in DISAGG_PHASES:
            try:
                params.get_request_type()
            except ValueError as e:
                return str(e)
        if (
            params.request_type == "generation_only"
            and params.disagg_request_id is None
            and params.ctx_request_id is None
        ):
            return "A generation-only request needs disagg_request_id or ctx_request_id"
    return None


def _cold(request: "GenerationRequest", postprocess: bool) -> Optional[dict]:
    """The fields few requests set, for ``encode_requests``; None when there are none."""
    sp = request.sampling_params
    cold = {}
    if (params := request.disaggregated_params) is not None:
        cold["disagg"] = encode_disagg(params)
    if request.multimodal_params is not None:
        cold["multimodal"] = encode_multimodal(request.multimodal_params)
    if postprocess:
        from .result import with_multimodal_spans

        # what a postprocess worker builds its record from, and the prompt for outputs that
        # start with it
        prompt = None
        if not sp.exclude_input_from_output and not request.streaming:
            prompt = request.prompt_token_ids
        cold["frontend_context"] = encode_frontend_context(
            FrontendContext(
                sp,
                request.postproc_params,
                request.streaming,
                with_multimodal_spans(request.disaggregated_params, request),
                prompt,
            )
        )
    if sp.guided_decoding is not None:
        guide = sp._get_guided_decoding_params()
        if guide is not None:
            cold["guided"] = guide
    if request.kv_cache_retention_config is not None:
        cold["retention"] = request.kv_cache_retention_config
    if request.encoder_input_token_ids is not None:
        cold["encoder_input"] = request.encoder_input_token_ids
    bias = sp.embedding_bias
    if isinstance(bias, tuple):
        cold["token_bias"] = bias
    elif bias is not None:
        if isinstance(bias, list):
            bias = torch.tensor(bias, dtype=torch.float32)
        cold["embedding_bias"] = tensor_ref(bias)
    if (s := request.scheduling_params) is not None:
        cold["scheduling"] = (
            s.attention_dp_rank,
            s.attention_dp_relax,
            s.agent_hierarchy,
            s.subagent_affinity_id,
        )
    if request.conversation_params is not None:
        cold["conversation_id"] = request.conversation_params.conversation_id
    if request.kv_hints is not None:
        cold["kv_hints"] = request.kv_hints.wire_json
    if (lora := request.lora_request) is not None:
        cold["lora"] = (lora.adapter_id, lora.path, lora.ckpt_source)
    if sp.watermarking is not None:
        cold["watermark"] = sp.watermarking
    if sp.return_perf_metrics:
        cold["perf_metrics"] = True
    if sp.logprobs_simple_format:
        cold["logprobs_values_only"] = True
    if sp.prompt_logprobs is not None:
        cold["prompt_logprobs"] = sp.prompt_logprobs
    if sp.prompt_logprobs_simple_format:
        cold["prompt_logprobs_values_only"] = True
    if sp.return_context_logits:
        cold["context_logits"] = True
    if sp.return_generation_logits:
        cold["generation_logits"] = True
    if sp.additional_model_outputs:
        cold["additional_outputs"] = list(sp.additional_model_outputs)
    if sp.return_routed_experts:
        cold["routed_experts"] = True
    if sp.return_encoder_output:
        cold["encoder_output"] = True
    if processors := sp.logits_processor:
        if (budget := _thinking_budget(processors)) is not None:
            cold["thinking_budget"] = budget
        else:
            cold["logits_processors"] = processors if isinstance(processors, list) else [processors]
    return cold or None


def encode(request: "GenerationRequest", postprocess: bool) -> SubmitFrame:
    """``request`` as a ``SubmitFrame``; one that rank 0 refuses, when it would.

    ``postprocess`` is whether postprocess workers read the request's rows.
    """
    if (refusal := _refusal(request, postprocess)) is not None:
        return SubmitFrame(b"", (request.id,), refusal=refusal)
    sp = request.sampling_params
    cold = _cold(request, postprocess)
    prompt = request.__dict__.get("_prompt_token_ids_i32")
    if prompt is None:
        prompt = request.__dict__.get("_prompt_token_ids")
    if tail := request._prompt_tail_tokens:
        # a decode part's frontend sends only the prompt's tail (prompt_tail.py)
        trimmed = len(prompt) - tail
        prompt = np.ascontiguousarray(prompt[trimmed:], dtype=np.int32)
        cold = cold or {}
        cold["prompt_trimmed_prefix"] = trimmed
    max_tokens = sp.max_tokens
    if max_tokens is not None:
        # the forced decoder prefix is already in the prompt
        max_tokens -= len(sp._decoder_output_token_prefix)
    item = (
        request.id,
        prompt,
        request.streaming,
        sp._get_sampling_config(),
        max_tokens,
        sp.end_id,
        sp.ignore_eos,
        [] if sp.ignore_eos else sp._get_stop_words(),
        sp._get_bad_words(),
        sp.logprobs,
        LOGPROB_MODES.index(sp.logprobs_mode),
        request.priority,
        request.arrival_time,
        request.cache_salt,
        cold,
    )
    objects = []
    codec = native()
    try:
        data = codec.encode_requests([item], objects)
    # a token id or value the executor's Request would not take either
    except (TypeError, codec.NotCarried) as e:
        return SubmitFrame(b"", (request.id,), refusal=str(e))
    return SubmitFrame(data, (request.id,), objects or None)


@functools.cache
def _finish_reasons() -> tuple:
    """The executor's finish reasons, by engine.proto's FinishReason.

    An error row has no finish reason, and a prefill part ends as the executor's Result says it:
    not finished.
    """
    from ...bindings import executor as tllm

    reasons = tllm.FinishReason
    return (
        reasons.NOT_FINISHED,
        reasons.END_ID,
        reasons.STOP_WORDS,
        reasons.LENGTH,
        reasons.TIMED_OUT,
        reasons.CANCELLED,
        None,
        reasons.NOT_FINISHED,
    )


def frontend_contexts(frame: StepFrame) -> dict[int, FrontendContext]:
    """The frontend contexts of the requests that start in ``frame``, by client id.

    A postprocess worker builds its records from them.
    """
    return {
        client_id: decode_frontend_context(data)
        for client_id, data in native().frontend_contexts(frame.data)
    }


def decode(frame: StepFrame, results: dict, more_results: Optional[dict] = None) -> list[StepRow]:
    """A ``StepFrame``'s rows, for the frontend or a postprocess worker.

    ``results``, then ``more_results``, hold by client id what says what each request asked for
    (its ``sampling_params``): a ``GenerationResult``, a postprocess worker's record, or a
    ``FrontendContext``.
    """
    return native().decode_rows(
        frame.data,
        frame.buffers or [],
        results,
        more_results,
        (StepRow, StepRowError),
        (StepRowResult, StepRowExtras, StepRowOutputs),
        _finish_reasons(),
        Logprob,
    )
