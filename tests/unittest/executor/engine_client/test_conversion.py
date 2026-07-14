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
"""Conversion, normalization-boundary, and eligibility-matrix tests."""

import dataclasses

import pytest
import torch

from tensorrt_llm.executor.base_worker import _request_stop_words
from tensorrt_llm.executor.engine_client.codec import decode, encode
from tensorrt_llm.executor.engine_client.contract import EngineRequest
from tensorrt_llm.executor.engine_client.conversion import (
    ELIGIBILITY_MATRIX, ConversionError, RequestIneligibleError,
    convert_request, derive_required_features,
    engine_request_to_generation_request, prepare_sampling_params)
from tensorrt_llm.sampling_params import (GuidedDecodingParams, LogprobMode,
                                          SamplingParams)


class FakeTokenizer:
    """Minimal tokenizer surface used by SamplingParams._setup."""

    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        # Deterministic non-trivial encoding: one token per character.
        return [100 + (ord(ch) % 50) for ch in text]


class FakeGenerationConfig:
    eos_token_id = [2, 32000]
    forced_bos_token_id = None
    forced_eos_token_id = None


def prepared_params(**kwargs) -> SamplingParams:
    params = SamplingParams(max_tokens=8, **kwargs)
    return prepare_sampling_params(params, tokenizer=FakeTokenizer(),
                                   generation_config=FakeGenerationConfig())


def convert(params: SamplingParams, **kwargs):
    return convert_request("req-1", [1, 2, 3], params, streaming=True, **kwargs)


class TestNormalizationBoundary:

    def test_eos_pad_defaults_from_tokenizer(self):
        params = prepared_params()
        assert params.end_id == 2
        assert params.pad_id == 0

    def test_generation_config_stop_ids_added_once(self):
        params = prepared_params()
        assert params.stop_token_ids == [32000]

    def test_stop_strings_tokenized_exactly_once(self):
        params = prepared_params(stop=["DONE"])
        first = params._stop_word_ids
        assert first is not None
        # Conversion must not re-tokenize: same object flows through.
        engine_request, output_config = convert(params)
        assert engine_request.sampling.stop_token_sequences == (tuple(first[0]), )
        assert params._stop_word_ids is first

    def test_explicit_end_id_skips_setup(self):
        params = SamplingParams(max_tokens=8, end_id=7, pad_id=7)
        prepare_sampling_params(params, tokenizer=None)
        assert params.end_id == 7
        assert params.stop_token_ids is None

    def test_bart_is_ineligible(self):
        config = FakeGenerationConfig()
        config.forced_bos_token_id = 0
        with pytest.raises(RequestIneligibleError) as excinfo:
            prepare_sampling_params(SamplingParams(max_tokens=8),
                                    tokenizer=FakeTokenizer(),
                                    generation_config=config, model_type="bart")
        assert excinfo.value.axis == "bart_forced_tokens"

    def test_thinking_budget_is_ineligible(self):
        with pytest.raises(RequestIneligibleError) as excinfo:
            prepare_sampling_params(SamplingParams(max_tokens=8, thinking_token_budget=100),
                                    tokenizer=FakeTokenizer())
        assert excinfo.value.axis == "thinking_token_budget"

    def test_stream_interval_and_perf_metrics_defaults(self):
        params = prepare_sampling_params(SamplingParams(max_tokens=8, end_id=2),
                                         tokenizer=None, stream_interval=4,
                                         force_return_perf_metrics=True)
        assert params._stream_interval == 4
        assert params.return_perf_metrics


class TestConvertRequest:

    def test_plain_text_conversion(self):
        params = prepared_params(temperature=0.5, top_p=0.9, top_k=40, seed=11,
                                 stop_token_ids=[13], stop=["STOP"], logprobs=1)
        engine_request, output_config = convert(params)
        sampling = engine_request.sampling
        assert sampling.max_new_tokens == 8
        assert sampling.end_id == 2 and sampling.pad_id == 0
        # stop_token_ids keeps user ids + generation-config additions.
        assert set(sampling.stop_token_ids) == {13, 32000}
        assert len(sampling.stop_token_sequences) == 1
        assert output_config.stop_strings == ("STOP", )
        # Ordered association: token-id reasons first, then string reasons.
        reasons = output_config.stop_sequence_reasons
        assert reasons[-1][1] == "STOP"
        assert reasons[-1][0] == sampling.stop_token_sequences[0]
        assert all(pair[1] == pair[0][0] for pair in reasons[:-1])
        assert output_config.num_logprobs == 1
        assert engine_request.required_features == ()

    def test_guided_decoding_maps_to_spec_and_feature(self):
        params = prepared_params(guided_decoding=GuidedDecodingParams(json={"type": "object"}))
        engine_request, _ = convert(params)
        assert engine_request.guided_decoding.mode == "json_schema"
        assert "object" in engine_request.guided_decoding.payload
        assert engine_request.required_features == ("guided_decoding", )
        assert derive_required_features(engine_request) == ("guided_decoding", )

    def test_unprepared_params_rejected(self):
        with pytest.raises(ConversionError):
            convert(SamplingParams(max_tokens=8))

    def test_untokenized_stop_strings_rejected(self):
        params = SamplingParams(max_tokens=8, end_id=2, stop=["STOP"])
        with pytest.raises(ConversionError):
            convert(params)


REJECTION_CASES = [
    ("non_streaming", {}, dict(streaming=False)),
    ("echo", {}, dict(echo=True)),
    ("n_gt_1", dict(n=2, top_p=0.9), {}),
    ("n_gt_1", dict(best_of=2, top_p=0.9), {}),
    ("beam_search", dict(use_beam_search=True, best_of=1, n=1), {}),
    ("beam_search", dict(length_penalty=1.0), {}),
    ("top_logprobs", dict(logprobs=5), {}),
    ("prompt_top_logprobs", dict(prompt_logprobs=5), {}),
    ("logits_processor", dict(apply_batched_logits_processor=True), {}),
    ("embedding_bias", dict(embedding_bias=torch.zeros(4)), {}),
    ("bad_words", dict(bad=["x"]), {}),
    ("bad_words", dict(bad_token_ids=[3]), {}),
    ("ignore_eos", dict(ignore_eos=True), {}),
    ("min_p", dict(min_p=0.2), {}),
    ("top_p_extras", dict(top_p_min=0.5), {}),
    ("no_repeat_ngram", dict(no_repeat_ngram_size=2), {}),
    ("prompt_ignore_length", dict(prompt_ignore_length=1), {}),
    ("return_logits", dict(return_context_logits=True), {}),
    ("return_logits", dict(return_generation_logits=True), {}),
    ("return_logits", dict(additional_model_outputs=["hidden"]), {}),
    ("exclude_input_from_output", dict(exclude_input_from_output=False), {}),
    ("truncate_prompt_tokens", dict(truncate_prompt_tokens=4), {}),
    ("multimodal", {}, dict(multimodal_params=object())),
    ("lora", {}, dict(lora_request=object())),
    ("prompt_adapter", {}, dict(prompt_adapter_request=object())),
    ("disaggregated", {}, dict(disaggregated_params=object())),
    ("scheduling_params", {}, dict(scheduling_params=object())),
    ("conversation_params", {}, dict(conversation_params=object())),
    ("postproc_params", {}, dict(postproc_params=object())),
    ("trace_headers", {}, dict(trace_headers={"traceparent": "x"})),
    ("cache_salt", {}, dict(cache_salt="salt")),
    ("query_token_ids", {}, dict(query_token_ids=[1])),
    ("encoder_input", {}, dict(encoder_input_token_ids=[1])),
    ("kv_cache_retention", {}, dict(kv_cache_retention_config=object())),
    ("priority", {}, dict(priority=0.9)),
]


class TestRejections:

    @pytest.mark.parametrize("axis,param_kwargs,convert_kwargs", REJECTION_CASES)
    def test_rejected_pre_submit(self, axis, param_kwargs, convert_kwargs):
        params = prepared_params(**param_kwargs)
        streaming = convert_kwargs.pop("streaming", True)
        with pytest.raises(RequestIneligibleError) as excinfo:
            convert_request("req-1", [1, 2, 3], params, streaming=streaming,
                            **convert_kwargs)
        assert excinfo.value.axis == axis

    def test_logprobs_mode_rejected(self):
        non_raw = [m for m in LogprobMode if getattr(m, "value", m) not in ("raw", "RAW")]
        if not non_raw:
            pytest.skip("no non-RAW logprob mode in this build")
        params = prepared_params(logprobs=1)
        params.logprobs_mode = non_raw[0]
        with pytest.raises(RequestIneligibleError) as excinfo:
            convert(params)
        assert excinfo.value.axis == "logprobs_mode"

    def test_every_ineligible_axis_has_a_rejection_case(self):
        tested = {case[0] for case in REJECTION_CASES}
        tested |= {"logprobs_mode", "bart_forced_tokens", "thinking_token_budget",
                   "lookahead_config"}
        matrix_axes = {rule.axis for rule in ELIGIBILITY_MATRIX
                       if rule.classification == "ineligible"}
        assert matrix_axes <= tested, f"untested axes: {matrix_axes - tested}"

    def test_lookahead_rejected(self):
        params = prepared_params()
        params.lookahead_config = object()
        with pytest.raises(RequestIneligibleError) as excinfo:
            convert(params)
        assert excinfo.value.axis == "lookahead_config"


class TestOpenAiFieldInventoryCompleteness:
    """Machine-readable completeness over the live OpenAI request surface."""

    def test_every_openai_request_field_dispositioned(self):
        from tensorrt_llm.executor.engine_client.conversion import \
            OPENAI_REQUEST_FIELD_DISPOSITION
        from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                        CompletionRequest)
        surface = set(ChatCompletionRequest.model_fields) | set(
            CompletionRequest.model_fields)
        missing = surface - set(OPENAI_REQUEST_FIELD_DISPOSITION)
        assert not missing, f"OpenAI fields without a V0 disposition: {sorted(missing)}"
        stale = set(OPENAI_REQUEST_FIELD_DISPOSITION) - surface
        assert not stale, f"dispositions for unknown OpenAI fields: {sorted(stale)}"

    def test_axis_references_exist_in_matrix(self):
        from tensorrt_llm.executor.engine_client.conversion import \
            OPENAI_REQUEST_FIELD_DISPOSITION
        known = {"preprocessing", "supported", "frontend", "capability_gated"}
        matrix_axes = {rule.axis for rule in ELIGIBILITY_MATRIX}
        for field, disposition in OPENAI_REQUEST_FIELD_DISPOSITION.items():
            if disposition in known:
                continue
            assert disposition.startswith("axis:"), (field, disposition)
            axis = disposition[len("axis:"):]
            assert axis in matrix_axes, (
                f"{field} references unknown matrix axis {axis!r}")

    # The normative dispositions the contract's scope matrix promises:
    # completeness alone would let a wrong classification (e.g. `echo`
    # marked frontend-supported) pass silently.
    EXPECTED_DISPOSITIONS = {
        "echo": "axis:echo",
        "n": "axis:n_gt_1",
        "best_of": "axis:n_gt_1",
        "logit_bias": "axis:embedding_bias",
        "top_logprobs": "axis:top_logprobs",
        "stream": "supported",
        "stream_options": "frontend",
        "suffix": "frontend",
        "max_tokens": "supported",
        "temperature": "supported",
        "response_format": "capability_gated",
        "lora_request": "axis:lora",
        "disaggregated_params": "axis:disaggregated",
        "cache_salt": "axis:cache_salt",
        "messages": "preprocessing",
        "prompt": "preprocessing",
        "model": "preprocessing",
    }

    def test_expected_dispositions(self):
        from tensorrt_llm.executor.engine_client.conversion import \
            OPENAI_REQUEST_FIELD_DISPOSITION
        for field, expected in self.EXPECTED_DISPOSITIONS.items():
            assert OPENAI_REQUEST_FIELD_DISPOSITION.get(field) == expected, (
                f"{field}: expected disposition {expected!r}, got "
                f"{OPENAI_REQUEST_FIELD_DISPOSITION.get(field)!r}")


class TestEligibilityMatrixCompleteness:

    def test_every_sampling_params_field_classified(self):
        covered = set()
        for rule in ELIGIBILITY_MATRIX:
            covered.update(rule.sources)
        missing = []
        for field in dataclasses.fields(SamplingParams):
            if field.name in ("_stop_word_ids", "_bad_word_ids"):
                continue  # derived storage of stop/bad, covered via their axes
            if field.name not in covered:
                missing.append(field.name)
        assert not missing, f"SamplingParams fields not in ELIGIBILITY_MATRIX: {missing}"

    def test_classifications_are_known(self):
        for rule in ELIGIBILITY_MATRIX:
            assert rule.classification in ("supported", "normalized", "ineligible",
                                           "config_gate")


class TestCompositionalTranslation:

    def make_engine_request(self) -> EngineRequest:
        params = prepared_params(temperature=0.25, top_p=0.8, top_k=16, seed=99,
                                 repetition_penalty=1.05, presence_penalty=0.1,
                                 frequency_penalty=0.2, min_tokens=2,
                                 stop_token_ids=[13], stop=["END"], logprobs=1,
                                 prompt_logprobs=0)
        engine_request, _ = convert(params)
        return engine_request

    def test_translation_consumes_only_the_encoded_request(self):
        engine_request = self.make_engine_request()
        decoded = decode(encode(engine_request))
        direct = engine_request_to_generation_request(engine_request)
        via_wire = engine_request_to_generation_request(decoded)
        assert via_wire.prompt_token_ids == direct.prompt_token_ids
        assert via_wire.stop_token_sequences == direct.stop_token_sequences
        assert via_wire.sampling_params == direct.sampling_params

    def test_every_wire_field_reaches_worker_consumption(self):
        engine_request = self.make_engine_request()
        generation_request = engine_request_to_generation_request(engine_request)
        sp = generation_request.sampling_params
        sampling = engine_request.sampling

        # Fields the worker reads directly off SamplingParams at enqueue.
        assert sp.max_tokens == sampling.max_new_tokens
        assert sp.end_id == sampling.end_id
        assert sp.pad_id == sampling.pad_id
        assert sp.logprobs == sampling.num_logprobs
        assert sp.prompt_logprobs == sampling.num_prompt_logprobs

        # Fields consumed through _get_sampling_config into the runtime config.
        config = sp._get_sampling_config()
        assert config.temperature == pytest.approx(sampling.temperature)
        assert config.top_p == pytest.approx(sampling.top_p)
        assert config.top_k == sampling.top_k
        assert config.seed == sampling.seed
        assert config.repetition_penalty == pytest.approx(sampling.repetition_penalty)
        assert config.presence_penalty == pytest.approx(sampling.presence_penalty)
        assert config.frequency_penalty == pytest.approx(sampling.frequency_penalty)
        assert config.min_tokens == sampling.min_tokens
        assert config.beam_width == 1
        assert config.num_return_sequences == 1

        # Stop handling: ids through SamplingParams, sequences through the
        # carrier, merged by the worker's stop-word derivation.
        stop_words = _request_stop_words(generation_request)
        expected = [[token_id] for token_id in sampling.stop_token_ids]
        expected += [list(seq) for seq in sampling.stop_token_sequences]
        assert stop_words == expected

    def test_translation_refuses_guided_decoding(self):
        params = prepared_params(guided_decoding=GuidedDecodingParams(json_object=True))
        engine_request, _ = convert(params)
        with pytest.raises(ConversionError):
            engine_request_to_generation_request(engine_request)

    def test_no_setup_needed_downstream(self):
        engine_request = self.make_engine_request()
        generation_request = engine_request_to_generation_request(engine_request)
        # The synthetic params carry no stop strings, so nothing downstream
        # can trigger (or need) tokenizer-based re-setup.
        assert generation_request.sampling_params.stop is None
        assert generation_request.sampling_params._stop_word_ids is None
        assert generation_request.stop_token_sequences == [
            list(seq) for seq in engine_request.sampling.stop_token_sequences
        ]
