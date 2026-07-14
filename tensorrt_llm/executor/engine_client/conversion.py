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
"""Conversion between legacy request inputs and the engine contract.

Three responsibilities, in pipeline order:

1. ``prepare_sampling_params`` — the one-shot normalization boundary. EOS/PAD
   defaulting, generation-config stop ids, and stop-string tokenization run
   exactly once here (the documented equivalent of
   ``LLM._prepare_sampling_params`` minus its callable-producing steps, which
   make a request ineligible instead).
2. ``convert_request`` — prepared ``SamplingParams`` → ``EngineRequest`` +
   ``FrontendOutputConfig``. The first capability gate: everything the
   primitive contract cannot represent is rejected here with a typed error
   naming the axis, before the engine could ever see the request.
3. ``engine_request_to_generation_request`` — the compositional runtime
   translation. Consumes ONLY the ``EngineRequest`` (no unencoded sidecar),
   so the worker-facing request is reproducible from the encoded wire form
   alone; pre-tokenized stop sequences ride ``GenerationRequest.
   stop_token_sequences`` and ``SamplingParams._setup`` is never re-run.

``ELIGIBILITY_MATRIX`` is the machine-readable classification of every
``SamplingParams`` field, request-level input, and engine/server config axis;
tests enforce its completeness against the live ``SamplingParams`` surface.
"""

import dataclasses
import json as json_module
from typing import List, Optional, Tuple

from pydantic import BaseModel

from ...sampling_params import GuidedDecodingParams, SamplingParams
from ..request import DEFAULT_REQUEST_PRIORITY, GenerationRequest
from .contract import (ContractError, EngineRequest, EngineSamplingConfig,
                       FrontendOutputConfig, GuidedDecodingSpec)

__all__ = [
    "ConversionError",
    "RequestIneligibleError",
    "EligibilityRule",
    "ELIGIBILITY_MATRIX",
    "GUIDED_DECODING_FEATURE",
    "prepare_sampling_params",
    "derive_required_features",
    "convert_request",
    "engine_request_to_generation_request",
]

GUIDED_DECODING_FEATURE = "guided_decoding"

# Frozen V0 disposition of every OpenAI request field (the union of the
# chat and completions request models). Machine-checked for completeness
# against the live protocol models. Values:
#   "preprocessing"    — consumed by shared frontend preprocessing before
#                        conversion (templates, tokenization, metadata)
#   "supported"        — carried on the contract (directly or via
#                        SamplingParams fields classified in the matrix)
#   "frontend"         — frontend-only presentation/assembly concern
#   "capability_gated" — representable data, rejected pre-submit until the
#                        capability is validated (guided decoding)
#   "axis:<name>"      — maps to that ineligible/config-gate matrix axis
OPENAI_REQUEST_FIELD_DISPOSITION = {
    "model": "preprocessing",
    "messages": "preprocessing",
    "prompt": "preprocessing",
    "prompt_token_ids": "preprocessing",
    "add_generation_prompt": "preprocessing",
    "add_special_tokens": "preprocessing",
    "chat_template": "preprocessing",
    "chat_template_kwargs": "preprocessing",
    "documents": "preprocessing",
    "media_io_kwargs": "axis:multimodal",
    "mm_processor_kwargs": "axis:multimodal",
    "user": "preprocessing",
    "reasoning_effort": "preprocessing",
    "stream": "supported",
    "stream_options": "frontend",
    "suffix": "frontend",
    "echo": "axis:echo",
    "max_tokens": "supported",
    "max_completion_tokens": "supported",
    "temperature": "supported",
    "top_p": "supported",
    "top_k": "supported",
    "seed": "supported",
    "min_tokens": "supported",
    "presence_penalty": "supported",
    "frequency_penalty": "supported",
    "repetition_penalty": "supported",
    "stop": "supported",
    "stop_token_ids": "supported",
    "include_stop_str_in_output": "frontend",
    "detokenize": "frontend",
    "skip_special_tokens": "frontend",
    "spaces_between_special_tokens": "frontend",
    "logprobs": "supported",
    "top_logprobs": "axis:top_logprobs",
    "logit_bias": "axis:embedding_bias",
    "n": "axis:n_gt_1",
    "best_of": "axis:n_gt_1",
    "use_beam_search": "axis:beam_search",
    "early_stopping": "axis:beam_search",
    "length_penalty": "axis:beam_search",
    "min_p": "axis:min_p",
    "top_p_min": "axis:top_p_extras",
    "ignore_eos": "axis:ignore_eos",
    "prompt_ignore_length": "axis:prompt_ignore_length",
    "return_context_logits": "axis:return_logits",
    "truncate_prompt_tokens": "axis:truncate_prompt_tokens",
    "thinking_token_budget": "axis:thinking_token_budget",
    "response_format": "capability_gated",
    # Tools render through the chat template and parse in the frontend
    # formatters; strict-mode tools add constrained decoding, which flows
    # into guided_decoding and hits the capability gate automatically.
    "tools": "frontend",
    "tool_choice": "frontend",
    "lora_request": "axis:lora",
    "disaggregated_params": "axis:disaggregated",
    "conversation_params": "axis:conversation_params",
    "cache_salt": "axis:cache_salt",
    "agent_hierarchy": "axis:scheduling_params",
}


class ConversionError(ContractError):
    """A legacy input cannot be converted onto the contract."""


class RequestIneligibleError(ConversionError):
    """Typed pre-submit rejection naming the eligibility axis.

    The serving facade treats this as "fall back to the legacy path" (and
    counts it per axis); it must be raised before any engine enqueue.
    """

    def __init__(self, axis: str, message: str):
        super().__init__(f"[{axis}] {message}")
        self.axis = axis


@dataclasses.dataclass(frozen=True)
class EligibilityRule:
    """One row of the machine-readable V0 eligibility matrix.

    ``classification`` is one of:
    - ``supported``: representable and carried on the contract.
    - ``normalized``: consumed by the normalization boundary, not carried.
    - ``ineligible``: request-level typed rejection (axis = ``axis``).
    - ``config_gate``: rejected at client construction, not per request.
    """

    axis: str
    classification: str
    sources: tuple
    notes: str = ""


ELIGIBILITY_MATRIX: Tuple[EligibilityRule, ...] = (
    # --- supported sampling surface (crosses as EngineSamplingConfig) ---
    EligibilityRule("core_sampling", "supported",
                    ("max_tokens", "end_id", "pad_id", "stop_token_ids", "min_tokens",
                     "temperature", "top_p", "top_k", "seed", "repetition_penalty",
                     "presence_penalty", "frequency_penalty"),
                    "direct EngineSamplingConfig fields"),
    EligibilityRule("chosen_token_logprobs", "supported", ("logprobs", "prompt_logprobs"),
                    "eligible only when <= 1; higher values are the top_logprobs axis"),
    EligibilityRule("logprob_wire_shape", "supported",
                    ("logprobs_mode", "logprobs_simple_format", "prompt_logprobs_simple_format"),
                    "RAW mode only; both legacy result shapes normalized by the envelope"),
    EligibilityRule("stop_strings", "supported", ("stop", "include_stop_str_in_output"),
                    "strings stay frontend-side; tokenized sequences cross as "
                    "stop_token_sequences with ordered reasons"),
    EligibilityRule("frontend_detok", "supported",
                    ("detokenize", "skip_special_tokens", "spaces_between_special_tokens"),
                    "frontend-only assembly configuration (FrontendOutputConfig)"),
    EligibilityRule("perf_metrics", "supported", ("return_perf_metrics", ),
                    "metrics keys convert enum->str at the envelope"),
    EligibilityRule("guided_decoding_schema", "supported", ("guided_decoding", ),
                    "schema-as-data crosses; capability-gated at submit until the "
                    "engine-path validation, so V0 rejects it pre-submit"),
    # --- consumed by normalization, never carried ---
    EligibilityRule("normalization_inputs", "normalized",
                    ("add_special_tokens", "_stream_interval"),
                    "add_special_tokens is a preprocessing input; stream interval only "
                    "affects legacy chunking / delta granularity"),
    EligibilityRule("derived_candidate_count", "normalized", ("n", "best_of"),
                    "must equal 1 in V0; the n>1 axis rejects everything else"),
    # --- request-level ineligible axes (typed rejection pre-submit) ---
    EligibilityRule("n_gt_1", "ineligible", ("n", "best_of"), "n>1 / best_of>1"),
    EligibilityRule("beam_search", "ineligible",
                    ("use_beam_search", "beam_width_array", "beam_search_diversity_rate",
                     "length_penalty", "early_stopping"),
                    "beam search and beam-only controls"),
    EligibilityRule("top_logprobs", "ineligible", ("logprobs", ), "logprobs > 1"),
    EligibilityRule("prompt_top_logprobs", "ineligible", ("prompt_logprobs", ),
                    "prompt_logprobs > 1"),
    EligibilityRule("logprobs_mode", "ineligible", ("logprobs_mode", ),
                    "non-RAW logprob modes"),
    EligibilityRule("logits_processor", "ineligible",
                    ("logits_processor", "apply_batched_logits_processor"),
                    "per-request callables cannot cross the wire"),
    EligibilityRule("embedding_bias", "ineligible", ("embedding_bias", ),
                    "tensor payloads need the tensor channel (covers OpenAI logit_bias)"),
    EligibilityRule("bad_words", "ineligible", ("bad", "bad_token_ids"), ""),
    EligibilityRule("ignore_eos", "ineligible", ("ignore_eos", ), ""),
    EligibilityRule("min_p", "ineligible", ("min_p", ), "not in EngineSamplingConfig"),
    EligibilityRule("top_p_extras", "ineligible",
                    ("top_p_min", "top_p_reset_ids", "top_p_decay"), ""),
    EligibilityRule("no_repeat_ngram", "ineligible", ("no_repeat_ngram_size", ), ""),
    EligibilityRule("prompt_ignore_length", "ineligible", ("prompt_ignore_length", ), ""),
    EligibilityRule("return_logits", "ineligible",
                    ("return_context_logits", "return_generation_logits",
                     "return_encoder_output", "additional_model_outputs",
                     "_return_log_probs", "_context_logits_auto_enabled",
                     "_generation_logits_auto_enabled"),
                    "raw logits/tensor outputs need the tensor channel"),
    EligibilityRule("exclude_input_from_output", "ineligible", ("exclude_input_from_output", ),
                    "non-default echo-style output"),
    EligibilityRule("truncate_prompt_tokens", "ineligible", ("truncate_prompt_tokens", ), ""),
    EligibilityRule("lookahead_config", "ineligible", ("lookahead_config", ), ""),
    EligibilityRule("thinking_token_budget", "ineligible", ("thinking_token_budget", ),
                    "normalization would attach a logits-processor callable"),
    EligibilityRule("bart_forced_tokens", "ineligible", ("model_type", ),
                    "BART normalization would attach a logits-processor callable"),
    EligibilityRule("echo", "ineligible", ("echo", ),
                    "prompt echo needs legacy prompt-text handling in the "
                    "postprocessors"),
    EligibilityRule("non_streaming", "ineligible", ("streaming", ), "V0 is streaming-only"),
    EligibilityRule("multimodal", "ineligible", ("multimodal_params", ), "request-level input"),
    EligibilityRule("lora", "ineligible", ("lora_request", ), "request-level input"),
    EligibilityRule("prompt_adapter", "ineligible", ("prompt_adapter_request", ),
                    "request-level input"),
    EligibilityRule("disaggregated", "ineligible", ("disaggregated_params", ),
                    "request-level input"),
    EligibilityRule("scheduling_params", "ineligible", ("scheduling_params", ),
                    "request-level input"),
    EligibilityRule("conversation_params", "ineligible", ("conversation_params", ),
                    "request-level input"),
    EligibilityRule("postproc_params", "ineligible", ("postproc_params", ),
                    "formatter callables never cross; assembly is frontend-owned"),
    EligibilityRule("trace_headers", "ineligible", ("trace_headers", ),
                    "no telemetry propagation on the contract yet; rejecting avoids "
                    "silent trace loss"),
    EligibilityRule("cache_salt", "ineligible", ("cache_salt", ), "request-level input"),
    EligibilityRule("query_token_ids", "ineligible", ("query_token_ids", ),
                    "star-attention workflow"),
    EligibilityRule("encoder_input", "ineligible", ("encoder_input_token_ids", ),
                    "encoder-decoder models"),
    EligibilityRule("priority", "ineligible", ("priority", ),
                    "non-default request priority is not carried in V0"),
    EligibilityRule("kv_cache_retention", "ineligible", ("kv_cache_retention_config", ),
                    "request-level input"),
    # --- config gates (client construction; enforced by local_client) ---
    EligibilityRule("backend", "config_gate", ("backend", ),
                    "pytorch only; the transitional _autodeploy is explicitly rejected"),
    EligibilityRule("transport", "config_gate", ("transport", ), "IPC proxy only"),
    EligibilityRule("postproc_workers", "config_gate", ("num_postprocess_workers", ),
                    "must be 0"),
    EligibilityRule("post_processor_hook", "config_gate", ("post_processor_hook", ),
                    "global output hook can rewrite/suppress/terminate output"),
    EligibilityRule("speculative_config", "config_gate", ("speculative_config", ),
                    "unvalidated interaction surface"),
    EligibilityRule("early_first_token", "config_gate",
                    ("enable_early_first_token_response", ),
                    "changes the per-step response shape; unvalidated"),
    EligibilityRule("topology", "config_gate", ("world_size", ),
                    "the setup gate admits exactly the GPU-validated topology (TP1)"),
    EligibilityRule("trust_remote_code_tokenizer", "config_gate", ("trust_remote_code", ),
                    "tokenizer provenance cannot be pinned"),
    EligibilityRule("flag", "config_gate", ("TLLM_EXPERIMENTAL_ENGINE_CLIENT", ),
                    "client construction requires the experimental flag"),
)


def prepare_sampling_params(sampling_params: SamplingParams,
                            *,
                            tokenizer,
                            hf_model_config=None,
                            generation_config=None,
                            model_type: Optional[str] = None,
                            stream_interval: int = 1,
                            force_return_perf_metrics: bool = False) -> SamplingParams:
    """One-shot normalization boundary for the contract path.

    Mirrors ``LLM._prepare_sampling_params`` for the V0-eligible surface:
    EOS/PAD defaulting, generation-config stop ids, and stop-string
    tokenization run exactly once (via ``SamplingParams._setup`` when
    ``end_id`` is unset, identical to the legacy condition). The legacy
    callable-producing steps (BART forced-token processors, thinking-budget
    processors) are NOT replicated: requests that would need them are
    rejected as ineligible instead. After this call the params must never be
    passed through ``_setup`` again — downstream translation carries
    tokenized stop sequences on the request envelope.
    """
    if not isinstance(sampling_params, SamplingParams):
        raise ConversionError(
            f"expected SamplingParams, got {type(sampling_params).__name__}")
    if model_type == "bart" and generation_config is not None and (
            getattr(generation_config, "forced_bos_token_id", None) is not None
            or getattr(generation_config, "forced_eos_token_id", None) is not None):
        raise RequestIneligibleError(
            "bart_forced_tokens",
            "BART forced-token normalization would attach a logits-processor callable")
    if sampling_params.thinking_token_budget is not None:
        raise RequestIneligibleError(
            "thinking_token_budget",
            "thinking-budget normalization would attach a logits-processor callable")
    if sampling_params.end_id is None:
        if tokenizer is None:
            raise ConversionError(
                "tokenizer is required to derive end_id/pad_id when end_id is None")
        sampling_params._setup(tokenizer, hf_model_config, generation_config)
    if sampling_params._stream_interval is None:
        sampling_params._stream_interval = stream_interval
    sampling_params.return_perf_metrics = (sampling_params.return_perf_metrics
                                           or force_return_perf_metrics)
    return sampling_params


def prepare_sampling_params_from_context(sampling_params: SamplingParams,
                                         *,
                                         context,
                                         tokenizer,
                                         stream_interval: int = 1,
                                         force_return_perf_metrics: bool = False
                                         ) -> SamplingParams:
    """Context-only normalization boundary: no live model-config reaches.

    Mirrors ``prepare_sampling_params`` but consumes only the data-only
    ``FrontendModelContext`` (eos/pad ids, generation-config stop ids,
    model type) plus the spec-reloaded tokenizer for stop-string
    tokenization. This is the serving glue's boundary; a remote detach
    swaps the context's delivery without touching this code.
    """
    if not isinstance(sampling_params, SamplingParams):
        raise ConversionError(
            f"expected SamplingParams, got {type(sampling_params).__name__}")
    if context.model_type == "bart":
        # Conservative context-only stand-in for the legacy forced-token
        # check: the BART family normalization attaches callables.
        raise RequestIneligibleError(
            "bart_forced_tokens",
            "BART-family normalization would attach a logits-processor callable")
    if sampling_params.thinking_token_budget is not None:
        raise RequestIneligibleError(
            "thinking_token_budget",
            "thinking-budget normalization would attach a logits-processor callable")
    if sampling_params.end_id is None:
        if context.eos_id is None:
            raise ConversionError(
                "the model context carries no eos id; cannot default end_id")
        sampling_params.end_id = context.eos_id
        sampling_params.pad_id = (context.pad_id
                                  if context.pad_id is not None else context.eos_id)
        # Stop strings are tokenized exactly once, with the spec-reloaded
        # tokenizer (mirroring SamplingParams._setup's encoding call).
        if sampling_params.stop is not None and sampling_params._stop_word_ids is None:
            strings = ([sampling_params.stop]
                       if isinstance(sampling_params.stop, str) else
                       list(sampling_params.stop))
            sampling_params._stop_word_ids = [
                _encode_stop(tokenizer, text) for text in strings
            ]
        # Generation-config stop ids merge (mirrors _setup).
        if context.generation_stop_token_ids:
            stop_ids = list(sampling_params.stop_token_ids or [])
            for token_id in context.generation_stop_token_ids:
                if token_id != sampling_params.end_id and token_id not in stop_ids:
                    stop_ids.append(token_id)
            sampling_params.stop_token_ids = stop_ids or None
    if sampling_params._stream_interval is None:
        sampling_params._stream_interval = stream_interval
    sampling_params.return_perf_metrics = (sampling_params.return_perf_metrics
                                           or force_return_perf_metrics)
    return sampling_params


def _encode_stop(tokenizer, text: str):
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def _check_sampling_eligibility(sp: SamplingParams) -> None:
    if sp.n != 1 or (sp.best_of or sp.n) != 1:
        raise RequestIneligibleError("n_gt_1", f"n={sp.n}, best_of={sp.best_of}")
    if sp.use_beam_search or sp.beam_width_array is not None or \
            sp.beam_search_diversity_rate is not None or sp.length_penalty is not None or \
            sp.early_stopping is not None:
        raise RequestIneligibleError("beam_search", "beam search / beam-only controls set")
    if sp.logprobs is not None and sp.logprobs > 1:
        raise RequestIneligibleError("top_logprobs", f"logprobs={sp.logprobs}")
    if sp.prompt_logprobs is not None and sp.prompt_logprobs > 1:
        raise RequestIneligibleError("prompt_top_logprobs",
                                     f"prompt_logprobs={sp.prompt_logprobs}")
    if getattr(sp.logprobs_mode, "value", sp.logprobs_mode) not in ("raw", "RAW"):
        raise RequestIneligibleError("logprobs_mode", f"logprobs_mode={sp.logprobs_mode}")
    if sp.logits_processor is not None or sp.apply_batched_logits_processor:
        raise RequestIneligibleError("logits_processor",
                                     "per-request logits processors cannot cross the wire")
    if sp.embedding_bias is not None:
        raise RequestIneligibleError("embedding_bias", "embedding/logit bias tensor set")
    if sp.bad is not None or sp.bad_token_ids:
        raise RequestIneligibleError("bad_words", "bad words set")
    if sp.ignore_eos:
        raise RequestIneligibleError("ignore_eos", "ignore_eos set")
    if sp.min_p is not None:
        raise RequestIneligibleError("min_p", "min_p set")
    if sp.top_p_min is not None or sp.top_p_reset_ids is not None or sp.top_p_decay is not None:
        raise RequestIneligibleError("top_p_extras", "top_p decay controls set")
    if sp.no_repeat_ngram_size is not None:
        raise RequestIneligibleError("no_repeat_ngram", "no_repeat_ngram_size set")
    if sp.prompt_ignore_length is not None:
        raise RequestIneligibleError("prompt_ignore_length", "prompt_ignore_length set")
    if sp.return_context_logits or sp.return_generation_logits or sp.return_encoder_output \
            or sp.additional_model_outputs is not None or sp._return_log_probs:
        raise RequestIneligibleError("return_logits", "raw logits/tensor outputs requested")
    if not sp.exclude_input_from_output:
        raise RequestIneligibleError("exclude_input_from_output",
                                     "echo-style output requested")
    if sp.truncate_prompt_tokens is not None:
        raise RequestIneligibleError("truncate_prompt_tokens", "prompt truncation requested")
    if sp.lookahead_config is not None:
        raise RequestIneligibleError("lookahead_config", "lookahead decoding config set")
    if sp.thinking_token_budget is not None:
        raise RequestIneligibleError("thinking_token_budget", "thinking token budget set")


_REQUEST_LEVEL_AXES = (
    ("multimodal_params", "multimodal"),
    ("lora_request", "lora"),
    ("prompt_adapter_request", "prompt_adapter"),
    ("disaggregated_params", "disaggregated"),
    ("scheduling_params", "scheduling_params"),
    ("conversation_params", "conversation_params"),
    ("postproc_params", "postproc_params"),
    ("trace_headers", "trace_headers"),
    ("cache_salt", "cache_salt"),
    ("query_token_ids", "query_token_ids"),
    ("encoder_input_token_ids", "encoder_input"),
    ("kv_cache_retention_config", "kv_cache_retention"),
)


def _guided_decoding_to_spec(params: GuidedDecodingParams) -> GuidedDecodingSpec:
    if params.json_object:
        return GuidedDecodingSpec(mode="json_object")
    if params.json is not None:
        schema = params.json
        if isinstance(schema, BaseModel):
            schema = schema.model_json_schema()
        if isinstance(schema, dict):
            schema = json_module.dumps(schema)
        return GuidedDecodingSpec(mode="json_schema", payload=schema)
    if params.regex is not None:
        return GuidedDecodingSpec(mode="regex", payload=params.regex)
    if params.grammar is not None:
        return GuidedDecodingSpec(mode="grammar", payload=params.grammar)
    if params.structural_tag is not None:
        return GuidedDecodingSpec(mode="structural_tag", payload=params.structural_tag)
    raise ConversionError("guided_decoding set but no guide field populated")


def derive_required_features(engine_request: EngineRequest) -> tuple:
    """Derive the feature set a request actually needs from its own fields.

    The client re-derives and validates this at submit time; caller-supplied
    ``required_features`` are checked against it, never trusted.
    """
    features = []
    if engine_request.guided_decoding is not None:
        features.append(GUIDED_DECODING_FEATURE)
    return tuple(features)


def _split_stop_handling(sp: SamplingParams):
    """Split prepared stop configuration by representation.

    Stop token ids and tokenized stop-string sequences cross to the engine;
    the stop strings themselves stay frontend-side, with an ordered
    ``(sequence, user_visible_reason)`` association mirroring the legacy
    stop-reason resolution (configuration order, first match wins).
    """
    stop_token_ids = tuple(sp.stop_token_ids or ())
    reasons = [((token_id, ), token_id) for token_id in stop_token_ids]
    stop_strings: List[str] = []
    stop_sequences = []
    if sp.stop is not None:
        if sp._stop_word_ids is None:
            raise ConversionError(
                "stop strings present but not tokenized; prepare_sampling_params "
                "must run before conversion")
        stop_strings = [sp.stop] if isinstance(sp.stop, str) else list(sp.stop)
        if len(sp._stop_word_ids) != len(stop_strings):
            raise ConversionError(
                f"tokenized stop sequences ({len(sp._stop_word_ids)}) do not match "
                f"stop strings ({len(stop_strings)})")
        for stop_string, word_ids in zip(stop_strings, sp._stop_word_ids):
            sequence = tuple(word_ids)
            stop_sequences.append(sequence)
            reasons.append((sequence, stop_string))
    return stop_token_ids, tuple(stop_sequences), tuple(stop_strings), tuple(reasons)


def convert_request(request_id: str,
                    prompt_token_ids,
                    sampling_params: SamplingParams,
                    *,
                    streaming: bool = True,
                    multimodal_params=None,
                    lora_request=None,
                    prompt_adapter_request=None,
                    disaggregated_params=None,
                    scheduling_params=None,
                    conversation_params=None,
                    postproc_params=None,
                    trace_headers=None,
                    cache_salt=None,
                    query_token_ids=None,
                    encoder_input_token_ids=None,
                    kv_cache_retention_config=None,
                    priority=None,
                    echo=False):
    """Convert prepared legacy inputs into ``(EngineRequest, FrontendOutputConfig)``.

    This is the first capability gate: every input the V0 contract cannot
    carry raises ``RequestIneligibleError`` (with the axis name) before any
    engine enqueue. ``sampling_params`` must already be normalized by
    ``prepare_sampling_params``.
    """
    if not streaming:
        raise RequestIneligibleError("non_streaming", "V0 is streaming-only")
    if echo:
        raise RequestIneligibleError(
            "echo", "prompt echo needs legacy prompt-text handling in the "
            "postprocessors")
    local_values = dict(multimodal_params=multimodal_params,
                        lora_request=lora_request,
                        prompt_adapter_request=prompt_adapter_request,
                        disaggregated_params=disaggregated_params,
                        scheduling_params=scheduling_params,
                        conversation_params=conversation_params,
                        postproc_params=postproc_params,
                        trace_headers=trace_headers,
                        cache_salt=cache_salt,
                        query_token_ids=query_token_ids,
                        encoder_input_token_ids=encoder_input_token_ids,
                        kv_cache_retention_config=kv_cache_retention_config)
    for kwarg_name, axis in _REQUEST_LEVEL_AXES:
        if local_values[kwarg_name] is not None:
            raise RequestIneligibleError(axis, f"{kwarg_name} is set")
    if priority is not None and priority != DEFAULT_REQUEST_PRIORITY:
        raise RequestIneligibleError("priority", f"non-default priority {priority}")
    _check_sampling_eligibility(sampling_params)
    if sampling_params.end_id is None:
        raise ConversionError(
            "end_id is unset; prepare_sampling_params must run before conversion")

    stop_token_ids, stop_sequences, stop_strings, stop_reasons = \
        _split_stop_handling(sampling_params)

    sampling = EngineSamplingConfig(
        max_new_tokens=sampling_params.max_tokens,
        end_id=sampling_params.end_id,
        pad_id=sampling_params.pad_id,
        stop_token_ids=stop_token_ids,
        stop_token_sequences=stop_sequences,
        min_tokens=sampling_params.min_tokens,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
        seed=sampling_params.seed,
        repetition_penalty=sampling_params.repetition_penalty,
        presence_penalty=sampling_params.presence_penalty,
        frequency_penalty=sampling_params.frequency_penalty,
        num_logprobs=sampling_params.logprobs,
        num_prompt_logprobs=sampling_params.prompt_logprobs,
    )
    guided_decoding = None
    if sampling_params.guided_decoding is not None:
        guided_decoding = _guided_decoding_to_spec(sampling_params.guided_decoding)

    engine_request = EngineRequest(request_id=request_id,
                                   prompt_token_ids=tuple(prompt_token_ids),
                                   sampling=sampling,
                                   guided_decoding=guided_decoding)
    engine_request = dataclasses.replace(
        engine_request, required_features=derive_required_features(engine_request))

    output_config = FrontendOutputConfig(
        detokenize=sampling_params.detokenize,
        skip_special_tokens=sampling_params.skip_special_tokens,
        spaces_between_special_tokens=sampling_params.spaces_between_special_tokens,
        stop_strings=stop_strings,
        include_stop_str_in_output=sampling_params.include_stop_str_in_output,
        stop_sequence_reasons=stop_reasons,
        end_id=sampling_params.end_id,
        num_logprobs=sampling_params.logprobs,
    )
    return engine_request, output_config


def engine_request_to_generation_request(engine_request: EngineRequest) -> GenerationRequest:
    """Compositional runtime translation: encoded ``EngineRequest`` → ``GenerationRequest``.

    Consumes only the ``EngineRequest``: the synthetic ``SamplingParams`` is
    rebuilt purely from ``EngineSamplingConfig`` wire fields (which is why
    ``pad_id`` is on the wire), and tokenized stop sequences ride the
    request's ``stop_token_sequences`` carrier — ``SamplingParams._setup`` is
    never involved, so nothing can re-tokenize the stops.
    """
    if not isinstance(engine_request, EngineRequest):
        raise ConversionError(
            f"expected EngineRequest, got {type(engine_request).__name__}")
    if engine_request.guided_decoding is not None:
        # V0 capability-gates guided decoding at submit; translation refuses it
        # too so a bypassed gate cannot smuggle it into the runtime untested.
        raise ConversionError("guided_decoding is capability-gated in V0")
    sampling = engine_request.sampling
    synthetic = SamplingParams(
        max_tokens=sampling.max_new_tokens,
        end_id=sampling.end_id,
        pad_id=sampling.pad_id,
        stop_token_ids=list(sampling.stop_token_ids) or None,
        min_tokens=sampling.min_tokens,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        seed=sampling.seed,
        repetition_penalty=sampling.repetition_penalty,
        presence_penalty=sampling.presence_penalty,
        frequency_penalty=sampling.frequency_penalty,
        logprobs=sampling.num_logprobs,
        prompt_logprobs=sampling.num_prompt_logprobs,
    )
    stop_token_sequences = [list(seq) for seq in sampling.stop_token_sequences] or None
    return GenerationRequest(prompt_token_ids=list(engine_request.prompt_token_ids),
                             sampling_params=synthetic,
                             streaming=True,
                             stop_token_sequences=stop_token_sequences)
