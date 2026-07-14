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
"""Construction-time validation tests for the engine contract wire types."""

import dataclasses
from types import MappingProxyType

import pytest

from tensorrt_llm.executor.engine_client.contract import (
    ENGINE_CONTRACT_VERSION, INT64_MAX, MAX_STOP_SEQUENCES,
    ContractConstructionError, EngineCapabilities, EngineRequest,
    EngineSamplingConfig, ErrorFrame, FrontendModelContext,
    FrontendOutputConfig, GuidedDecodingSpec, RequestComplete, Terminal,
    TokenDelta, TokenizerSpec, validate_no_callables)


def make_sampling(**overrides) -> EngineSamplingConfig:
    kwargs = dict(max_new_tokens=16)
    kwargs.update(overrides)
    return EngineSamplingConfig(**kwargs)


def make_request(**overrides) -> EngineRequest:
    kwargs = dict(request_id="req-1", prompt_token_ids=(1, 2, 3), sampling=make_sampling())
    kwargs.update(overrides)
    return EngineRequest(**kwargs)


class TestConstructionAccepts:

    def test_full_request(self):
        request = make_request(
            sampling=make_sampling(end_id=2, pad_id=0, stop_token_ids=[13],
                                   stop_token_sequences=[[7, 8]], temperature=0.5, seed=-1,
                                   num_logprobs=1, num_prompt_logprobs=1),
            guided_decoding=GuidedDecodingSpec(mode="json_object"),
            required_features=["guided_decoding"])
        assert request.protocol_version == ENGINE_CONTRACT_VERSION
        # Sequences normalize to tuples so the value graph is immutable.
        assert request.sampling.stop_token_ids == (13, )
        assert request.sampling.stop_token_sequences == ((7, 8), )
        assert request.required_features == ("guided_decoding", )

    def test_metrics_become_immutable_copies(self):
        source = {"cached_tokens": 3.0}
        delta = TokenDelta(request_id="r", sequence_id=0, new_token_ids=[1], metrics=source)
        assert isinstance(delta.metrics, MappingProxyType)
        with pytest.raises(TypeError):
            delta.metrics["cached_tokens"] = 9.0
        source["cached_tokens"] = 9.0
        assert delta.metrics["cached_tokens"] == 3.0

    def test_frames_are_frozen(self):
        terminal = Terminal(request_id="r", sequence_id=0, finish_reason="stop")
        with pytest.raises(dataclasses.FrozenInstanceError):
            terminal.finish_reason = "length"

    def test_ordered_stop_sequence_reasons(self):
        config = FrontendOutputConfig(stop_strings=["a", "b"],
                                      stop_sequence_reasons=[[[1, 2], "a"], [[1, 2], "b"]])
        # Order is preserved verbatim: first match wins downstream.
        assert config.stop_sequence_reasons == (((1, 2), "a"), ((1, 2), "b"))

    def test_model_context(self):
        context = FrontendModelContext(
            tokenizer=TokenizerSpec(uri="/models/m", files_manifest=[["tokenizer.json", "00"]]),
            capabilities=EngineCapabilities(features=["streaming"]),
            eos_id=2, max_context_length=4096)
        assert context.tokenizer.files_manifest == (("tokenizer.json", "00"), )

    def test_int_float_field_accepts_int(self):
        assert make_sampling(temperature=1).temperature == 1.0


class TestConstructionRejects:

    @pytest.mark.parametrize("kwargs", [
        dict(max_new_tokens=True),
        dict(max_new_tokens=0),
        dict(end_id=1.5),
        dict(stop_token_ids=[True]),
        dict(stop_token_sequences=[[]]),
        dict(stop_token_sequences=[[1]] * (MAX_STOP_SEQUENCES + 1)),
        dict(temperature=float("nan")),
        dict(top_p=float("inf")),
        dict(seed=INT64_MAX + 1),
        dict(num_logprobs=-1),
        dict(protocol_version=0),
    ])
    def test_sampling_config_rejects(self, kwargs):
        with pytest.raises(ContractConstructionError):
            make_sampling(**kwargs)

    @pytest.mark.parametrize("kwargs", [
        dict(request_id=""),
        dict(request_id=7),
        dict(prompt_token_ids=()),
        dict(prompt_token_ids=(1, "2")),
        dict(sampling=None),
        dict(guided_decoding="json"),
        dict(required_features=(1, )),
    ])
    def test_request_rejects(self, kwargs):
        with pytest.raises(ContractConstructionError):
            make_request(**kwargs)

    @pytest.mark.parametrize("kwargs", [
        dict(new_token_ids=()),
        dict(new_token_ids=(1, ), logprobs=(-0.5, -0.5)),
        dict(new_token_ids=(True, )),
        dict(sequence_id=-1),
        dict(event_seq=-1),
        dict(metrics={"k": "not-a-float"}),
        dict(metrics={1: 2.0}),
        dict(logprobs=(float("nan"), )),
    ])
    def test_token_delta_rejects(self, kwargs):
        base = dict(request_id="r", sequence_id=0, new_token_ids=(1, ))
        base.update(kwargs)
        with pytest.raises(ContractConstructionError):
            TokenDelta(**base)

    def test_finish_reason_must_be_known(self):
        with pytest.raises(ContractConstructionError):
            Terminal(request_id="r", sequence_id=0, finish_reason="timeout")

    def test_stop_reason_rejects_bool_and_float(self):
        for bad in (True, 1.5):
            with pytest.raises(ContractConstructionError):
                Terminal(request_id="r", sequence_id=0, finish_reason="stop", stop_reason=bad)

    def test_request_complete_status_must_be_known(self):
        with pytest.raises(ContractConstructionError):
            RequestComplete(request_id="r", status="done", prompt_tokens=1, completion_tokens=1)

    def test_request_complete_negative_counts(self):
        with pytest.raises(ContractConstructionError):
            RequestComplete(request_id="r", status="ok", prompt_tokens=-1, completion_tokens=0)

    def test_error_frame_requires_code(self):
        with pytest.raises(ContractConstructionError):
            ErrorFrame(request_id="r", error_code="")

    def test_guided_decoding_payload_required_unless_json_object(self):
        with pytest.raises(ContractConstructionError):
            GuidedDecodingSpec(mode="regex")
        assert GuidedDecodingSpec(mode="json_object").payload is None

    def test_guided_decoding_unknown_mode(self):
        with pytest.raises(ContractConstructionError):
            GuidedDecodingSpec(mode="yaml", payload="x")

    def test_output_config_rejects_unordered_or_malformed_pairs(self):
        with pytest.raises(ContractConstructionError):
            FrontendOutputConfig(stop_sequence_reasons=[[[1, 2]]])
        with pytest.raises(ContractConstructionError):
            FrontendOutputConfig(stop_sequence_reasons={(1, 2): "a"})
        with pytest.raises(ContractConstructionError):
            FrontendOutputConfig(stop_strings=[""])

    def test_bool_flags_reject_int(self):
        with pytest.raises(ContractConstructionError):
            FrontendOutputConfig(detokenize=1)

    def test_tokenizer_spec_manifest_pairs(self):
        with pytest.raises(ContractConstructionError):
            TokenizerSpec(uri="/m", files_manifest=[["only-path"]])


class TestNoCallables:

    def test_callable_rejected_anywhere(self):
        with pytest.raises(ContractConstructionError):
            validate_no_callables({"hook": lambda x: x})
        with pytest.raises(ContractConstructionError):
            validate_no_callables([1, 2, print])

    def test_bytes_rejected(self):
        with pytest.raises(ContractConstructionError):
            validate_no_callables(b"raw")

    def test_arbitrary_object_rejected(self):
        class Payload:
            pass

        with pytest.raises(ContractConstructionError):
            validate_no_callables({"obj": Payload()})

    def test_valid_graph_passes(self):
        validate_no_callables(make_request())
