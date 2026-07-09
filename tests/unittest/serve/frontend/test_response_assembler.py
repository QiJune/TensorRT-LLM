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
"""CPU-only unit tests for the frontend response assembler."""

import pytest

from tensorrt_llm.engine_api import (
    ContractViolationError,
    EngineError,
    EngineErrorCode,
    EngineEvent,
    EngineRequest,
    FrontendOutputConfig,
    ProtocolViolationError,
    RuntimeSamplingConfig,
    TerminalKind,
)
from tensorrt_llm.serve.frontend import FrontendResponseAssembler

VOCAB = {
    1: "Hello",
    2: " world",
    3: "!",
    4: " STOP",
    5: " again",
    6: " ST",
    7: "OP",
    8: "<eos>",
}


class FakeTokenizer:
    """Deterministic vocab-lookup tokenizer with incremental decoding."""

    def decode(self, token_ids, **kwargs):
        if isinstance(token_ids, int):
            return VOCAB[token_ids]
        return "".join(VOCAB[token_id] for token_id in token_ids)

    def decode_incrementally(
        self, token_ids, prev_text=None, states=None, *, flush=False, stream_interval=1, **kwargs
    ):
        prev_text = prev_text or ""
        return prev_text + "".join(VOCAB[token_id] for token_id in token_ids), (states or {})


def make_assembler(
    request_id="req-1",
    streaming=True,
    num_sequences=1,
    num_returns=1,
    detokenize=True,
    use_beam_search=False,
    **config_overrides,
) -> FrontendResponseAssembler:
    config = FrontendOutputConfig(detokenize=detokenize, **config_overrides)
    return FrontendResponseAssembler(
        request_id,
        config,
        num_sequences=num_sequences,
        num_returns=num_returns,
        use_beam_search=use_beam_search,
        streaming=streaming,
        tokenizer=FakeTokenizer() if detokenize else None,
    )


def event(request_id="req-1", sequence_index=0, event_index=0, token_ids=None, **kwargs):
    return EngineEvent(
        request_id=request_id,
        sequence_index=sequence_index,
        event_index=event_index,
        token_ids=token_ids if token_ids is not None else [],
        **kwargs,
    )


class TestStreamingAssembly:
    def test_incremental_detokenization_and_diffs(self):
        assembler = make_assembler()
        assembler.consume(event(event_index=0, token_ids=[1]))
        output = assembler.view.outputs[0]
        assert output.text == "Hello"
        assert output.text_diff == "Hello"
        assert output.token_ids_diff == [1]

        assembler.consume(event(event_index=1, token_ids=[2]))
        assert output.text == "Hello world"
        assert output.text_diff == " world"
        assert output.token_ids_diff == [2]

        assembler.consume(
            event(
                event_index=2,
                token_ids=[3],
                terminal_kind=TerminalKind.FINISHED,
                finish_reason="length",
            )
        )
        assert output.text == "Hello world!"
        assert output.finish_reason == "length"
        assert assembler.done

    def test_stateful_lifetime_spans_chunks(self):
        """Per-request incremental state must persist across streaming chunks."""
        assembler = make_assembler()
        assembler.consume(event(event_index=0, token_ids=[1]))
        states_after_first = assembler.view.outputs[0]._incremental_states
        assembler.consume(event(event_index=1, token_ids=[2]))
        assert assembler.view.outputs[0]._incremental_states is not None
        assert states_after_first is not None

    def test_logprobs_accumulate_with_diffs(self):
        assembler = make_assembler()
        assembler.consume(event(event_index=0, token_ids=[1], logprobs=[-0.1]))
        assembler.consume(
            event(
                event_index=1,
                token_ids=[2],
                logprobs=[-0.2],
                cumulative_logprob=-0.3,
                terminal_kind=TerminalKind.FINISHED,
                finish_reason="length",
            )
        )
        output = assembler.view.outputs[0]
        assert output.logprobs == [-0.1, -0.2]
        assert output.logprobs_diff == [-0.2]
        assert output.cumulative_logprob == -0.3

    def test_cumulative_beam_logprobs_replace_not_append(self):
        """Cumulative (beam) events replace logprobs, not append.

        The token list replaces the prefix, so logprobs must replace too —
        otherwise logprobs outgrows token_ids and the OpenAI logprob
        formatter's length assertion fails.
        """
        assembler = make_assembler(
            num_sequences=1, num_returns=1, use_beam_search=True, detokenize=False
        )
        assembler.consume(
            event(event_index=0, token_ids=[1, 2], cumulative=True, logprobs=[-0.1, -0.2])
        )
        assembler.consume(
            event(
                event_index=1,
                token_ids=[1, 2, 3],
                cumulative=True,
                logprobs=[-0.1, -0.2, -0.3],
                terminal_kind=TerminalKind.FINISHED,
                finish_reason="length",
            )
        )
        output = assembler.view.outputs[0]
        assert output.token_ids == [1, 2, 3]
        assert output.logprobs == [-0.1, -0.2, -0.3]
        assert len(output.logprobs) == len(output.token_ids)


class TestStopHandling:
    def test_stop_string_detected_and_trimmed(self):
        assembler = make_assembler(stop_strings=[" STOP"], stop_sequence_token_ids=[[4]])
        assembler.consume(event(event_index=0, token_ids=[1]))
        assembler.consume(event(event_index=1, token_ids=[6, 7]))
        output = assembler.view.outputs[0]
        # " ST" + "OP" forms the stop string across token boundaries.
        assert output.text == "Hello"
        assert output.finish_reason == "stop"
        assert output.stop_reason == " STOP"
        assert assembler.done

    def test_stop_string_included_when_requested(self):
        assembler = make_assembler(
            stop_strings=[" STOP"],
            stop_sequence_token_ids=[[4]],
            include_stop_str_in_output=True,
        )
        assembler.consume(event(event_index=0, token_ids=[1, 6, 7]))
        output = assembler.view.outputs[0]
        assert output.text == "Hello STOP"
        assert output.stop_reason == " STOP"

    def test_runtime_stop_sequence_trimmed_at_token_level(self):
        assembler = make_assembler(stop_strings=[" STOP"], stop_sequence_token_ids=[[4]])
        assembler.consume(
            event(
                event_index=0,
                token_ids=[1, 4],
                terminal_kind=TerminalKind.FINISHED,
                finish_reason="stop",
                stop_kind="stop_sequence",
            )
        )
        output = assembler.view.outputs[0]
        assert output.token_ids == [1]
        assert output.stop_reason == " STOP"
        assert output.text == "Hello"

    def test_runtime_stop_token_id_attribution(self):
        assembler = make_assembler(stop_token_ids=[8])
        assembler.consume(
            event(
                event_index=0,
                token_ids=[1, 8],
                terminal_kind=TerminalKind.FINISHED,
                finish_reason="stop",
                stop_kind="stop_sequence",
            )
        )
        output = assembler.view.outputs[0]
        assert output.token_ids == [1]
        assert output.stop_reason == 8

    def test_end_token_stop_keeps_tokens_and_no_stop_reason(self):
        assembler = make_assembler()
        assembler.consume(
            event(
                event_index=0,
                token_ids=[1, 2],
                terminal_kind=TerminalKind.FINISHED,
                finish_reason="stop",
                stop_kind="end_token",
            )
        )
        output = assembler.view.outputs[0]
        assert output.token_ids == [1, 2]
        assert output.stop_reason is None
        assert output.finish_reason == "stop"


class TestSequenceSelection:
    def test_top_n_selection_when_best_of_exceeds_n(self):
        assembler = make_assembler(num_sequences=2, num_returns=1)
        assembler.consume(
            event(
                sequence_index=0,
                event_index=0,
                token_ids=[1],
                cumulative_logprob=-2.0,
                terminal_kind=TerminalKind.FINISHED,
                finish_reason="length",
            )
        )
        assembler.consume(
            event(
                sequence_index=1,
                event_index=0,
                token_ids=[2],
                cumulative_logprob=-1.0,
                terminal_kind=TerminalKind.FINISHED,
                finish_reason="length",
            )
        )
        outputs = assembler.view.outputs
        assert len(outputs) == 1
        assert outputs[0].token_ids == [2]
        assert outputs[0].index == 0

    def test_all_sequences_terminal_marks_done(self):
        assembler = make_assembler(num_sequences=2, num_returns=2)
        assembler.consume(
            event(
                sequence_index=0,
                event_index=0,
                token_ids=[1],
                terminal_kind=TerminalKind.FINISHED,
                finish_reason="length",
            )
        )
        assert not assembler.done
        assembler.consume(
            event(
                sequence_index=1,
                event_index=0,
                token_ids=[2],
                terminal_kind=TerminalKind.FINISHED,
                finish_reason="length",
            )
        )
        assert assembler.done


class TestErrorsAndAbort:
    def test_aborted_terminal_maps_to_cancelled(self):
        assembler = make_assembler()
        assembler.consume(event(event_index=0, token_ids=[1]))
        assembler.consume(
            event(
                event_index=1,
                terminal_kind=TerminalKind.ABORTED,
                finish_reason="cancelled",
            )
        )
        assert assembler.view.outputs[0].finish_reason == "cancelled"
        assert assembler.done

    def test_error_terminal_records_typed_error(self):
        assembler = make_assembler()
        assembler.consume(
            event(
                event_index=0,
                terminal_kind=TerminalKind.ERROR,
                error=EngineError(code=EngineErrorCode.REQUEST_FAILED, message="boom"),
            )
        )
        assert assembler.done
        assert assembler.error.code is EngineErrorCode.REQUEST_FAILED

    def test_error_terminal_after_partial_output_is_position_independent(self):
        """ERROR terminals are position-independent, even after partial output.

        A socket-side engine failure arrives as an ERROR terminal with
        event_index 0 even after tokens streamed; it must preserve the typed
        error, not raise an out-of-order protocol violation.
        """
        assembler = make_assembler()
        assembler.consume(event(event_index=0, token_ids=[1]))
        assembler.consume(event(event_index=1, token_ids=[2]))
        # ERROR terminal with the position-independent event_index 0.
        assembler.consume(
            event(
                event_index=0,
                token_ids=[],
                terminal_kind=TerminalKind.ERROR,
                error=EngineError(code=EngineErrorCode.INTERNAL_ERROR, message="engine died"),
            )
        )
        assert assembler.done
        assert assembler.error.code is EngineErrorCode.INTERNAL_ERROR
        assert "engine died" in assembler.error.message


class TestInputContract:
    def test_events_cannot_carry_detokenized_text(self):
        with pytest.raises(TypeError):
            EngineEvent(request_id="req-1", event_index=0, text="pre-detokenized")

    def test_pre_trimmed_stop_sequence_rejected(self):
        assembler = make_assembler(stop_strings=[" STOP"], stop_sequence_token_ids=[[4]])
        with pytest.raises(ProtocolViolationError, match="pre-trimmed"):
            assembler.consume(
                event(
                    event_index=0,
                    token_ids=[1],  # stop tokens already removed upstream
                    terminal_kind=TerminalKind.FINISHED,
                    finish_reason="stop",
                    stop_kind="stop_sequence",
                )
            )

    def test_ordering_violations_rejected(self):
        assembler = make_assembler()
        assembler.consume(
            event(
                event_index=0,
                token_ids=[1],
                terminal_kind=TerminalKind.FINISHED,
                finish_reason="length",
            )
        )
        with pytest.raises(ProtocolViolationError, match="duplicate terminal"):
            assembler.consume(
                event(event_index=1, terminal_kind=TerminalKind.FINISHED, finish_reason="length")
            )

    def test_wrong_request_id_rejected(self):
        assembler = make_assembler()
        with pytest.raises(ContractViolationError, match="fed to assembler"):
            assembler.consume(event(request_id="other-request"))

    def test_non_event_input_rejected(self):
        assembler = make_assembler()
        with pytest.raises(ContractViolationError, match="must be EngineEvent"):
            assembler.consume({"token_ids": [1]})

    def test_formatter_callable_cannot_attach_to_request(self):
        """New-path requests reject callable-style postprocessing attachments."""

        def formatter(result, args):
            return result

        with pytest.raises(ContractViolationError, match="callable"):
            EngineRequest(
                request_id="req-1",
                prompt_token_ids=[1],
                sampling=RuntimeSamplingConfig(),
                disaggregated_metadata={"post_processor": formatter},
            )
        with pytest.raises(TypeError):
            EngineRequest(
                request_id="req-1",
                prompt_token_ids=[1],
                sampling=RuntimeSamplingConfig(),
                postproc_params=formatter,
            )

    def test_detokenize_requires_tokenizer(self):
        with pytest.raises(ContractViolationError, match="tokenizer"):
            FrontendResponseAssembler("req-1", FrontendOutputConfig(detokenize=True))

    def test_misaligned_stop_sequences_rejected(self):
        with pytest.raises(ContractViolationError, match="align"):
            make_assembler(stop_strings=[" STOP"], stop_sequence_token_ids=[[4], [5]])
