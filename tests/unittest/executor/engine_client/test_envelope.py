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
"""Envelope normalization tests over the real PyTorch response shapes."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor.engine_client.envelope import (EnvelopeError,
                                                          normalize_response)
from tensorrt_llm.executor.postproc_worker import PostprocWorker
from tensorrt_llm.executor.result import Logprob, LogProbsResult, ResponseWrapper
from tensorrt_llm.executor.utils import ErrorResponse
from tensorrt_llm.metrics.enums import RequestEventTiming


def make_result(**overrides) -> SimpleNamespace:
    fields = dict(is_final=False, output_token_ids=[[5, 6]], finish_reasons=None,
                  log_probs=None, cum_log_probs=None, sequence_index=0)
    fields.update(overrides)
    return SimpleNamespace(**fields)


def make_response(client_id=7, **result_overrides) -> LlmResponse:
    return LlmResponse(request_id=1, result=make_result(**result_overrides),
                       client_id=client_id)


class TestTokenDeltaNormalization:

    def test_plain_delta(self):
        envelope = normalize_response(make_response())
        assert envelope.client_id == 7
        assert envelope.new_token_ids == (5, 6)
        assert not envelope.is_final
        assert envelope.finish_reason_name is None
        assert envelope.logprobs is None and envelope.prompt_logprobs is None

    def test_final_with_finish_reason(self):
        envelope = normalize_response(
            make_response(is_final=True,
                          finish_reasons=[tllm.FinishReason.END_ID]))
        assert envelope.is_final
        assert envelope.finish_reason_name == "END_ID"

    def test_no_token_final(self):
        envelope = normalize_response(
            make_response(is_final=True, output_token_ids=[[]],
                          finish_reasons=[tllm.FinishReason.CANCELLED]))
        assert envelope.new_token_ids == ()
        assert envelope.finish_reason_name == "CANCELLED"

    def test_snapshot_is_immutable_copy(self):
        source_tokens = [5, 6]
        response = make_response(output_token_ids=[source_tokens])
        envelope = normalize_response(response)
        source_tokens.append(999)
        assert envelope.new_token_ids == (5, 6)

    def test_cached_tokens_sourced(self):
        envelope = normalize_response(make_response(cached_tokens=3))
        assert envelope.cached_tokens == 3
        assert normalize_response(make_response()).cached_tokens is None

    def test_serialized_result_is_deserialized(self):
        inner = make_result()

        class SerializedResult:

            def __init__(self):
                self._result = b"opaque"

            def deserialize(self):
                self._result = inner
                for name, value in vars(inner).items():
                    setattr(self, name, value)

        response = LlmResponse(request_id=1, result=SerializedResult(), client_id=7)
        envelope = normalize_response(response)
        assert envelope.new_token_ids == (5, 6)


class TestLogprobNormalization:

    def test_float_list_passes_through(self):
        envelope = normalize_response(
            make_response(log_probs=[[-0.5, -1.5]]))
        assert envelope.logprobs == (-0.5, -1.5)

    def test_map_shape_exact_lookup(self):
        entries = [{5: Logprob(logprob=-0.5, rank=1)},
                   {6: Logprob(logprob=-1.5, rank=2), 9: Logprob(logprob=-0.1, rank=1)}]
        envelope = normalize_response(make_response(log_probs=[entries]))
        assert envelope.logprobs == (-0.5, -1.5)

    def test_map_missing_key_is_typed_error(self):
        entries = [{9: Logprob(logprob=-0.1, rank=1)}, {6: Logprob(logprob=-1.5, rank=1)}]
        with pytest.raises(EnvelopeError) as excinfo:
            normalize_response(make_response(log_probs=[entries]))
        assert excinfo.value.reason == "logprob_mismatch"

    def test_length_mismatch_is_typed_error(self):
        with pytest.raises(EnvelopeError) as excinfo:
            normalize_response(make_response(log_probs=[[-0.5]]))
        assert excinfo.value.reason == "logprob_mismatch"

    def test_prompt_logprobs_float_passthrough(self):
        wrapper = ResponseWrapper(
            make_response(),
            logprobs=LogProbsResult(prompt=[-2.0, -3.0], generation=None))
        envelope = normalize_response(wrapper)
        assert envelope.prompt_logprobs == (-2.0, -3.0)

    def test_prompt_logprobs_map_shape_needs_binding(self):
        prompt_entries = [{2: Logprob(-1.0, 1)}, {3: Logprob(-2.0, 1)}, {5: Logprob(-3.0, 1)}]
        wrapper = ResponseWrapper(
            make_response(),
            logprobs=LogProbsResult(prompt=prompt_entries, generation=None))
        with pytest.raises(EnvelopeError):
            normalize_response(wrapper)
        # With the bound prompt ids (offset by one, plus the first generated
        # token) the exact lookup succeeds.
        envelope = normalize_response(wrapper, prompt_token_ids=(1, 2, 3))
        assert envelope.prompt_logprobs == (-1.0, -2.0, -3.0)


class TestMetricsNormalization:

    def test_enum_keys_convert(self):
        wrapper = ResponseWrapper(
            make_response(),
            request_perf_metrics={RequestEventTiming.ARRIVAL_TIME: 1.5})
        envelope = normalize_response(wrapper)
        assert envelope.metrics == {RequestEventTiming.ARRIVAL_TIME.value: 1.5}

    def test_non_numeric_metric_is_typed_error(self):
        wrapper = ResponseWrapper(make_response(),
                                  request_perf_metrics={"bad": "value"})
        with pytest.raises(EnvelopeError) as excinfo:
            normalize_response(wrapper)
        assert excinfo.value.reason == "metrics_mismatch"


class TestErrorShapes:

    def test_error_response(self):
        envelope = normalize_response(ErrorResponse(client_id=3, error_msg="boom",
                                                    request_id=1))
        assert envelope.client_id == 3
        assert envelope.is_final
        assert envelope.error_msg == "boom"

    def test_llm_response_error(self):
        response = LlmResponse(request_id=1, error_msg="worker error", client_id=4)
        envelope = normalize_response(response)
        assert envelope.error_msg == "worker error"
        assert envelope.is_final


class TestRejectedShapes:

    def test_postproc_output_rejected(self):
        output = PostprocWorker.Output(client_id=1, res="text", is_final=True)
        with pytest.raises(EnvelopeError) as excinfo:
            normalize_response(output)
        assert excinfo.value.reason == "postproc_parallel_shape"

    def test_cpp_engine_shape_rejected(self):

        class FakeCppResponse:

            client_id = 1

            def has_error(self):
                return False

        with pytest.raises(EnvelopeError) as excinfo:
            normalize_response(FakeCppResponse())
        assert excinfo.value.reason == "cpp_engine_shape"

    def test_unknown_shape_rejected(self):
        with pytest.raises(EnvelopeError) as excinfo:
            normalize_response(object())
        assert excinfo.value.reason == "unknown_shape"
