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
"""Tests for the pre-tokenized stop-sequence carrier on GenerationRequest."""

import pickle

import pytest

from tensorrt_llm.executor.base_worker import _request_stop_words
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.sampling_params import SamplingParams


def make_request(**kwargs) -> GenerationRequest:
    defaults = dict(prompt_token_ids=[1, 2, 3],
                    sampling_params=SamplingParams(max_tokens=8),
                    streaming=True)
    defaults.update(kwargs)
    return GenerationRequest(**defaults)


class TestCarrierField:

    def test_default_is_none(self):
        assert make_request().stop_token_sequences is None

    def test_accepts_sequences(self):
        request = make_request(stop_token_sequences=[[7, 8], [9]])
        assert request.stop_token_sequences == [[7, 8], [9]]

    @pytest.mark.parametrize("bad", [
        [[]],
        [[1, "x"]],
        [[True]],
        ["ab"],
        [7],
    ])
    def test_rejects_malformed(self, bad):
        with pytest.raises((ValueError, TypeError)):
            make_request(stop_token_sequences=bad)

    def test_survives_pickle(self):
        request = make_request(stop_token_sequences=[[7, 8]])
        clone = pickle.loads(pickle.dumps(request))
        assert clone.stop_token_sequences == [[7, 8]]
        assert clone.prompt_token_ids == [1, 2, 3]


class TestRequestStopWords:

    def test_carrier_merges_with_sampling_stops(self):
        request = make_request(
            sampling_params=SamplingParams(max_tokens=8, stop_token_ids=[13]),
            stop_token_sequences=[[7, 8], [9]])
        assert _request_stop_words(request) == [[13], [7, 8], [9]]

    def test_carrier_alone(self):
        request = make_request(stop_token_sequences=[[7, 8]])
        assert _request_stop_words(request) == [[7, 8]]

    def test_no_stops(self):
        assert _request_stop_words(make_request()) == []

    def test_ignore_eos_suppresses_all(self):
        request = make_request(
            sampling_params=SamplingParams(max_tokens=8, stop_token_ids=[13],
                                           ignore_eos=True),
            stop_token_sequences=[[7, 8]])
        assert _request_stop_words(request) == []

    def test_legacy_requests_without_field(self):
        request = make_request(sampling_params=SamplingParams(max_tokens=8,
                                                              stop_token_ids=[13]))
        # Requests unpickled from an older writer may lack the attribute.
        del request.stop_token_sequences
        assert _request_stop_words(request) == [[13]]
