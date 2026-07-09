# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Golden-output oracle: exact old-path vs engine-client-pipeline equality."""

import asyncio
import copy

import pytest
from oracle_harness import (
    LLM_API_FIXTURES,
    OPENAI_FIXTURES,
    OracleFixture,
    StepSpec,
    assert_llm_api_equal,
    assert_openai_equal,
    run_new_path_llm_api,
    run_new_path_openai,
    run_old_path_llm_api,
    run_old_path_openai,
)


@pytest.mark.parametrize("fixture", OPENAI_FIXTURES, ids=lambda f: f.name)
def test_openai_parity_in_process(fixture: OracleFixture):
    """SSE chunk sequences / non-streaming bodies are exactly equal."""
    old_output = run_old_path_openai(fixture)
    new_output = asyncio.run(run_new_path_openai(fixture))
    assert_openai_equal(old_output, new_output, fixture.streaming)


@pytest.mark.parametrize("fixture", LLM_API_FIXTURES, ids=lambda f: f.name)
def test_llm_api_parity_in_process(fixture: OracleFixture):
    """RequestOutput/CompletionOutput field-level equality, incl. streaming steps."""
    old_snapshots = run_old_path_llm_api(fixture)
    new_snapshots = run_new_path_llm_api(fixture)
    assert_llm_api_equal(old_snapshots, new_snapshots)


class TestHarnessSelfChecks:
    """The oracle must catch seeded divergences — proof the comparison bites."""

    def test_seeded_stop_trim_divergence_is_caught(self):
        fixture = copy.deepcopy(
            next(f for f in OPENAI_FIXTURES if f.name == "chat_stream_stop_string_trimmed")
        )
        old_output = run_old_path_openai(fixture)
        # Seed an off-by-one stop trim on the new path's input: the stop
        # sequence arrives with one extra token so the trimmed text differs.
        fixture.steps[-1] = StepSpec([102, 105, 107], finish="STOP_WORDS", is_final=True)
        new_output = asyncio.run(run_new_path_openai(fixture))
        with pytest.raises(AssertionError):
            assert_openai_equal(old_output, new_output, streaming=True)

    def test_mutated_chunk_field_is_caught(self):
        fixture = next(f for f in OPENAI_FIXTURES if f.name == "chat_stream_basic_end_token")
        old_output = run_old_path_openai(fixture)
        new_output = asyncio.run(run_new_path_openai(fixture))
        mutated = list(new_output)
        mutated[-1] = mutated[-1].replace('"finish_reason":"stop"', '"finish_reason":"length"')
        with pytest.raises(AssertionError):
            assert_openai_equal(old_output, mutated, streaming=True)

    def test_mutated_llm_api_expectation_is_caught(self):
        fixture = next(f for f in LLM_API_FIXTURES if f.name == "llm_api_stream_basic")
        old_snapshots = run_old_path_llm_api(fixture)
        new_snapshots = run_new_path_llm_api(fixture)
        mutated = copy.deepcopy(new_snapshots)
        mutated[-1][0]["token_ids"].append(999)
        with pytest.raises(AssertionError):
            assert_llm_api_equal(old_snapshots, mutated)
