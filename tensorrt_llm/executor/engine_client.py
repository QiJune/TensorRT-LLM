# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The frontend's half of the executor's proto wire: requests to the engine contract, its rows back.

The proxy sends rank 0 a request the proto carries as a ``SubmitBatch`` (``encode``). The proxy
and the postprocess workers read the rows of the ``StepOutput`` rank 0 sends them each step
(``decode``) into objects ``GenerationResult`` reads.
"""

import functools
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from .wire import (
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
    encode_prefill_usage,
    native,
    tensor_ref,
)

if TYPE_CHECKING:
    from .request import GenerationRequest
    from .result import LogProbsResult


class StepRowResult:
    """A row's result, read by ``GenerationResult._handle_step_row`` as it reads an ``LlmResult``.

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
    and ``encoder_export`` (an encoder part's, by ``LlmResult`` attribute), and
    ``prefill_usage`` ((prompt_tokens, cached_tokens)); ``take_outputs`` turns the tensors into
    the logits and additional outputs ``_handle_sequence`` reads.
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


def take_outputs(row: StepRowOutputs) -> Optional["LogProbsResult"]:
    """Frontend: set ``row``'s logits and additional outputs from its tensors.

    They are set as a pickled ``LlmResult`` has them. Returns the row's prompt logprobs as a
    ``LogProbsResult``, or None.
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
    from .result import LogProbsResult

    return LogProbsResult(prompt=row.prompt_logprobs)


class StepRow:
    """One response of a ``StepFrame``, dispatched to its ``GenerationResult`` like an ``LlmResponse``."""

    __slots__ = ("client_id", "request_id", "result")
    error_msg = None

    def has_error(self) -> bool:
        return self.error_msg is not None


class StepRowError(StepRow):
    """A row that ends its request with ``error_msg``."""

    __slots__ = ("error_msg",)


def _thinking_budget(processors) -> Optional[tuple]:
    """(budget, start ids, end ids) when ``processors`` is a thinking budget alone, else None."""
    if not processors:
        return None
    from ..llmapi.thinking_budget import ThinkingBudgetLogitsProcessor

    if isinstance(processors, list) and len(processors) == 1:
        processors = processors[0]
    if type(processors) is not ThinkingBudgetLogitsProcessor:
        return None
    return (
        processors.thinking_token_budget,
        processors.reasoning_start_token_ids,
        processors.reasoning_end_token_ids,
    )


def _carried(
    request: "GenerationRequest", postprocess: bool, thinking_budget: Optional[tuple]
) -> bool:
    """Whether the proto carries what rank 0 reads from ``request``.

    It does not carry, in this order: with postprocess workers, rows that need the prompt or carry
    logits (the pickled path handles them); a prompt adapter or a prompt already trimmed; a logits
    processor other than a thinking budget alone (``thinking_budget``); an embedding bias other
    than a dense tensor or list, or sparse (token id, bias) pairs, which ``_enqueue_request``
    takes; a prompt tail without disaggregated params; disaggregated params in a phase it does not
    name, of a decode part without a request id (get_context_phase_params needs one), or with a
    usage other than the engine's two counts; and a multimodal layout ``encode_multimodal``'s
    items cannot say.

    A value engine.proto's Value cannot hold (in multimodal data or handoffs) is left to
    ``encode_requests``, which raises ``NotCarried`` for it.
    """
    sp = request.sampling_params
    if postprocess and (
        sp.return_context_logits or sp.return_generation_logits or not sp.exclude_input_from_output
    ):
        return False
    if request.prompt_adapter_request is not None or request._prompt_trimmed_prefix:
        return False
    if sp.logits_processor and thinking_budget is None:
        return False
    bias = sp.embedding_bias
    if not (
        bias is None
        or isinstance(bias, (list, tuple))
        or (isinstance(bias, torch.Tensor) and bias.dtype in DTYPES)
    ):
        return False
    params = request.disaggregated_params
    if params is None:
        if request._prompt_tail_tokens:
            return False
    elif (
        params.request_type not in DISAGG_PHASES
        or (
            params.request_type == "generation_only"
            and params.disagg_request_id is None
            and params.ctx_request_id is None
        )
        or (params.orig_elapsed_s is not None and type(params.orig_elapsed_s) is not float)
        or (params.ctx_usage is not None and encode_prefill_usage(params.ctx_usage) is None)
    ):
        return False
    mm = None if request.multimodal_params is None else request.multimodal_params.multimodal_input
    if mm is None:
        return True
    uuids = mm.multimodal_uuids
    offsets = mm.multimodal_item_run_cu_offsets
    return (
        len(mm.multimodal_hashes) > 0
        and (uuids is None or any(u is not None for u in uuids))
        and (offsets is None or offsets[-1] != 0)
    )


def _cold(
    request: "GenerationRequest", postprocess: bool, thinking_budget: Optional[tuple]
) -> Optional[dict]:
    """The fields few requests set, for ``encode_requests``; None when there are none."""
    sp = request.sampling_params
    cold = {}
    if (params := request.disaggregated_params) is not None:
        cold["disagg"] = encode_disagg(params)
    if request.multimodal_params is not None:
        cold["multimodal"] = encode_multimodal(request.multimodal_params)
    if postprocess:
        from .result import with_multimodal_spans

        # what a postprocess worker builds its record from, as rank 0 sends a pickled request's
        cold["frontend_context"] = encode_frontend_context(
            FrontendContext(
                sp,
                request.postproc_params,
                request.streaming,
                with_multimodal_spans(request.disaggregated_params, request),
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
    if thinking_budget is not None:
        cold["thinking_budget"] = thinking_budget
    return cold or None


def encode(request: "GenerationRequest", postprocess: bool) -> Optional[SubmitFrame]:
    """Proxy: ``request`` as a ``SubmitFrame``, or None when it keeps the pickled path.

    ``postprocess`` is whether postprocess workers read the request's rows.
    """
    sp = request.sampling_params
    thinking_budget = _thinking_budget(sp.logits_processor)
    if not _carried(request, postprocess, thinking_budget):
        return None
    cold = _cold(request, postprocess, thinking_budget)
    prompt = request.__dict__.get("_prompt_token_ids_i32")
    if tail := request._prompt_tail_tokens:
        # a decode part's frontend sends only the prompt's tail (prompt_tail.py)
        ids = prompt if prompt is not None else request._prompt_token_ids
        prompt = np.ascontiguousarray(ids[len(ids) - tail :], dtype=np.int32)
        cold = cold or {}
        cold["prompt_trimmed_prefix"] = len(ids) - tail
    elif prompt is None:
        prompt = request.prompt_token_ids
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
    try:
        return SubmitFrame(native().encode_requests([item]), (request.id,))
    except native().NotCarried:
        # a value engine.proto's Value cannot hold (in multimodal data or handoffs)
        return None


@functools.cache
def _finish_reasons() -> tuple:
    """The executor's finish reasons, by engine.proto's FinishReason.

    An error row has no finish reason, and a prefill part ends as a pickled Result says it: not
    finished.
    """
    from ..bindings import executor as tllm

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
    """Postprocess worker: the frontend contexts of the requests that start in ``frame``, by client id."""
    return {
        client_id: decode_frontend_context(data)
        for client_id, data in native().frontend_contexts(frame.data)
    }


def decode(frame: StepFrame, results: dict, more_results: Optional[dict] = None) -> list[StepRow]:
    """Frontend and postprocess workers: a ``StepFrame``'s rows.

    ``results``, then ``more_results``, hold by client id what says what each request asked for
    (its ``sampling_params``): a ``GenerationResult``, a postprocess worker's record, or a
    ``FrontendContext``.
    """
    from .result import Logprob

    return native().decode_rows(
        frame.data,
        results,
        more_results,
        (StepRow, StepRowError),
        (StepRowResult, StepRowExtras, StepRowOutputs),
        _finish_reasons(),
        Logprob,
    )
