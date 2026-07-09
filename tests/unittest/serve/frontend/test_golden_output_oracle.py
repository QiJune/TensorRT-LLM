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
import json

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


def _socket_client_factory(executor):
    """Route the pipeline through the boundary socket, co-located."""
    from tensorrt_llm.engine_api.legacy_adapter import LegacyEngineClientAdapter
    from tensorrt_llm.engine_api.socket_transport import LocalProcessEngineClient

    return LocalProcessEngineClient(LegacyEngineClientAdapter(executor))


@pytest.mark.parametrize("fixture", OPENAI_FIXTURES, ids=lambda f: f.name)
def test_openai_parity_over_socket(fixture: OracleFixture):
    """The same fixtures must hold over the msgpack socket path."""
    old_output = run_old_path_openai(fixture)
    new_output = asyncio.run(run_new_path_openai(fixture, _socket_client_factory))
    assert_openai_equal(old_output, new_output, fixture.streaming)


@pytest.mark.parametrize("fixture", LLM_API_FIXTURES, ids=lambda f: f.name)
def test_llm_api_parity_over_socket(fixture: OracleFixture):
    old_snapshots = run_old_path_llm_api(fixture)
    new_snapshots = run_new_path_llm_api(fixture, _socket_client_factory)
    assert_llm_api_equal(old_snapshots, new_snapshots)


@pytest.mark.parametrize("fixture", OPENAI_FIXTURES, ids=lambda f: f.name)
def test_openai_parity_headless(fixture: OracleFixture):
    """The same fixtures must hold with the engine in a separate process.

    The Python LLM API never runs detached (it stays an in-process facade),
    so the headless variant covers the OpenAI fixtures only.
    """
    from oracle_harness import headless_client_factory

    old_output = run_old_path_openai(fixture)
    new_output = asyncio.run(run_new_path_openai(fixture, headless_client_factory(fixture)))
    assert_openai_equal(old_output, new_output, fixture.streaming)


class TestStreamingMultiSequenceParity:
    """Bug-for-bug parity pins for interleaved multi-sequence streaming.

    The historical path re-emits prior deltas and terminal chunks for
    untouched sequences on interleaved n>1 streaming (its diff cursors only
    advance for the sequence named by each response, and its formatter keeps
    finish state per call, not per request). The exact-equality gate (DEC-4)
    mandates reproducing that behavior bug-for-bug, so the pipeline does.

    These tests pin BOTH facts: the equality, and the duplicated shape in
    the OLD path's own output — so an upstream fix to the historical
    behavior fails here and forces a deliberate lockstep update of the
    pipeline (tracked as a queued side issue in the loop's goal tracker).
    """

    @staticmethod
    def _content_stream(chunks):
        decoded = []
        for chunk in chunks:
            payload = json.loads(chunk[len("data: ") :].strip())
            for choice in payload.get("choices") or []:
                delta = choice.get("delta") or {}
                decoded.append(
                    (
                        choice["index"],
                        delta.get("content") or choice.get("text"),
                        choice.get("finish_reason"),
                    )
                )
        return decoded

    def test_chat_n2_interleaved_duplication_is_shared_by_both_paths(self):
        fixture = next(f for f in OPENAI_FIXTURES if f.name == "chat_stream_n2")
        old_output = run_old_path_openai(fixture)
        new_output = asyncio.run(run_new_path_openai(fixture))
        assert_openai_equal(old_output, new_output, streaming=True)

        stream = self._content_stream(old_output)
        # The duplication signature of the historical path: choice 0's
        # "Hello" delta re-emitted when sequence 1's event arrives, and
        # choice 0's terminal re-emitted alongside sequence 1's terminal.
        content_events = [item for item in stream if item[1]]
        assert content_events.count((0, "Hello", None)) == 2, (
            "the historical path no longer duplicates untouched-sequence "
            "deltas on n>1 streaming — update the pipeline in lockstep and "
            "refresh this pin"
        )
        terminals = [item for item in stream if item[2] is not None]
        assert len([t for t in terminals if t[0] == 0]) == 2

    def test_completion_n2_interleaved_parity(self):
        fixture = next(f for f in OPENAI_FIXTURES if f.name == "completion_stream_n2")
        old_output = run_old_path_openai(fixture)
        new_output = asyncio.run(run_new_path_openai(fixture))
        assert_openai_equal(old_output, new_output, streaming=True)

        # Same bug-for-bug parity as chat: the historical completion stream
        # postprocessor emits a chunk for every choice on every step (empty
        # text for untouched sequences, re-emitted terminals), and the
        # pipeline reproduces it exactly. Pin the old path's shape so an
        # upstream fix forces a lockstep pipeline update.
        stream = self._content_stream(old_output)
        empty_untouched = [item for item in stream if item[1] == ""]
        assert empty_untouched, (
            "the historical completion stream no longer emits empty chunks "
            "for untouched sequences — update the pipeline in lockstep and "
            "refresh this pin"
        )
        terminals = [item for item in stream if item[2] is not None]
        assert len([t for t in terminals if t[0] == 0]) == 2


class TestToolParserStopLogprobsShape:
    """Semantic proof the AC-4 combination actually fires in the fixture.

    Old-vs-new equality alone could pass with an inert stop configuration;
    this decodes the OLD path's stream directly and asserts the parsed
    tool-call delta, the stop-derived terminal, and the logprob payloads
    are all present.
    """

    def test_old_path_stream_contains_tool_call_stop_and_logprobs(self):
        fixture = next(
            f for f in OPENAI_FIXTURES if f.name == "chat_stream_tool_parser_stop_logprobs"
        )
        chunks = [
            json.loads(chunk[len("data: ") :].strip()) for chunk in run_old_path_openai(fixture)
        ]

        tool_call_names = [
            call.get("function", {}).get("name")
            for payload in chunks
            for choice in payload.get("choices") or []
            for call in (choice.get("delta") or {}).get("tool_calls") or []
        ]
        assert "get_weather" in tool_call_names, (
            "the fixture no longer produces a parsed tool-call delta"
        )

        terminals = [
            choice
            for payload in chunks
            for choice in payload.get("choices") or []
            if choice.get("finish_reason") is not None
        ]
        assert terminals, "no terminal chunk emitted"
        # The stop string fired: the frontend scan set stop_reason and the
        # finish became stop-derived; with a parsed tool call present the
        # formatter rewrites 'stop' to 'tool_calls' (historical semantics),
        # so 'length' here would mean the stop never matched.
        assert terminals[-1]["finish_reason"] == "tool_calls", (
            "the configured stop string did not fire — the terminal must be "
            "stop-derived (rewritten to tool_calls), not length"
        )
        assert terminals[-1]["stop_reason"] == " world"

        logprob_chunks = [
            choice
            for payload in chunks
            for choice in payload.get("choices") or []
            if choice.get("logprobs")
        ]
        assert logprob_chunks, "no logprob payloads present in the stream"

    def test_new_path_stream_matches_the_same_shape(self):
        """The equality gate implies this, but assert it directly too."""
        fixture = next(
            f for f in OPENAI_FIXTURES if f.name == "chat_stream_tool_parser_stop_logprobs"
        )
        chunks = [
            json.loads(chunk[len("data: ") :].strip())
            for chunk in asyncio.run(run_new_path_openai(fixture))
        ]
        terminals = [
            choice
            for payload in chunks
            for choice in payload.get("choices") or []
            if choice.get("finish_reason") is not None
        ]
        assert terminals and terminals[-1]["finish_reason"] == "tool_calls"
        assert terminals[-1]["stop_reason"] == " world"


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
