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
"""CPU-only unit tests for the engine boundary data contracts."""

import dataclasses

import pytest
import torch

from tensorrt_llm.engine_api import (
    ContractViolationError,
    EngineError,
    EngineErrorCode,
    EngineEvent,
    EngineRequest,
    EventOrderingChecker,
    FrontendOutputConfig,
    ProtocolViolationError,
    PythonExtension,
    RuntimeSamplingConfig,
    TensorAuxiliaryPayload,
    TerminalKind,
    TokenLogprob,
    validate_plain_data,
)


def make_request(**overrides) -> EngineRequest:
    defaults = dict(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling=RuntimeSamplingConfig(max_tokens=16, temperature=0.7, stop_token_ids=[42]),
        streaming=True,
    )
    defaults.update(overrides)
    return EngineRequest(**defaults)


class TestContractConstruction:
    """Construction and invariant tests for the text path."""

    def test_request_text_path(self):
        request = make_request(
            trace_context={"traceparent": "00-abc-def-01"},
            disaggregated_metadata={"request_type": "context_and_generation"},
        )
        assert request.request_id == "req-1"
        assert request.crosses_neutral_wire
        assert request.sampling.stop_token_ids == [42]

    def test_frontend_output_config(self):
        config = FrontendOutputConfig(
            stop_strings=["\n\nUser:"],
            include_stop_str_in_output=True,
            num_return_sequences=2,
        )
        assert config.detokenize
        assert config.stop_strings == ["\n\nUser:"]

    def test_event_stream_first_mid_terminal(self):
        first = EngineEvent(
            request_id="req-1",
            event_index=0,
            token_ids=[7],
            prompt_token_ids=[1, 2, 3],
            logprobs=[{7: TokenLogprob(logprob=-0.1, rank=1)}],
        )
        mid = EngineEvent(request_id="req-1", event_index=1, token_ids=[8])
        terminal = EngineEvent(
            request_id="req-1",
            event_index=2,
            token_ids=[9],
            terminal_kind=TerminalKind.FINISHED,
            finish_reason="stop",
            stop_reason=42,
        )
        assert not first.is_terminal
        assert not mid.is_terminal
        assert terminal.is_terminal
        assert terminal.finish_reason == "stop"

    def test_cumulative_token_semantics_flag(self):
        beam_event = EngineEvent(
            request_id="req-1",
            sequence_index=1,
            event_index=0,
            token_ids=[7, 8, 9],
            cumulative=True,
        )
        assert beam_event.cumulative

    def test_typed_error_event(self):
        event = EngineEvent(
            request_id="req-1",
            event_index=0,
            terminal_kind=TerminalKind.ERROR,
            error=EngineError(
                code=EngineErrorCode.REQUEST_FAILED,
                message="sampler rejected the request",
                request_id="req-1",
            ),
        )
        assert event.error.code is EngineErrorCode.REQUEST_FAILED

    def test_python_extension_side_channel_accepts_callables(self):
        request = make_request(
            python_extension=PythonExtension(
                logits_processor=lambda ids, logits: logits,
                extra_fields={"py_custom": object()},
            )
        )
        assert not request.crosses_neutral_wire

    def test_tensor_side_channel_accepts_tensors(self):
        event = EngineEvent(
            request_id="req-1",
            event_index=0,
            token_ids=[7],
            tensor_payload=TensorAuxiliaryPayload(tensors={"generation_logits": torch.zeros(2)}),
        )
        assert "generation_logits" in event.tensor_payload.tensors


class TestPlainDataIntrospection:
    """Walk all plain-data fields and confirm no callables/tensors leak outside side channels."""

    def _assert_plain_outside_side_channels(self, obj) -> None:
        for f in dataclasses.fields(obj):
            if f.name in type(obj).SIDE_CHANNEL_FIELDS:
                continue
            value = getattr(obj, f.name)
            if isinstance(value, RuntimeSamplingConfig):
                self._assert_plain_outside_side_channels(value)
            else:
                validate_plain_data(value, f"{type(obj).__name__}.{f.name}")

    def test_request_fields_are_plain_data(self):
        request = make_request(
            trace_context={"traceparent": "00-abc-def-01"},
            disaggregated_metadata={"first_gen_tokens": [11], "opaque_state": b"\x00\x01"},
            python_extension=PythonExtension(logits_processor=lambda ids, logits: logits),
        )
        self._assert_plain_outside_side_channels(request)

    def test_event_fields_are_plain_data(self):
        event = EngineEvent(
            request_id="req-1",
            event_index=0,
            token_ids=[7],
            logprobs=[{7: TokenLogprob(logprob=-0.5)}],
            prompt_token_ids=[1, 2, 3],
            prompt_logprobs=[-0.2, -0.3, -0.4],
            metrics={"ttft_ms": 12.5},
            tensor_payload=TensorAuxiliaryPayload(tensors={"logits": torch.zeros(1)}),
        )
        self._assert_plain_outside_side_channels(event)

    def test_declared_side_channels_are_exactly_the_documented_ones(self):
        assert EngineRequest.SIDE_CHANNEL_FIELDS == {"python_extension", "tensor_payload"}
        assert EngineEvent.SIDE_CHANNEL_FIELDS == {"tensor_payload"}
        assert RuntimeSamplingConfig.SIDE_CHANNEL_FIELDS == frozenset()
        assert FrontendOutputConfig.SIDE_CHANNEL_FIELDS == frozenset()


class TestValidationRejections:
    """Non-plain data in plain-data fields must fail validation."""

    def test_callable_in_request_plain_field_rejected(self):
        with pytest.raises(ContractViolationError, match="callable"):
            make_request(disaggregated_metadata={"handler": lambda: None})

    def test_tensor_in_request_plain_field_rejected(self):
        with pytest.raises(ContractViolationError, match="Tensor"):
            make_request(prompt_token_ids=torch.tensor([1, 2, 3]))

    def test_callable_in_event_plain_field_rejected(self):
        with pytest.raises(ContractViolationError, match="callable"):
            EngineEvent(
                request_id="req-1",
                event_index=0,
                disaggregated_metadata={"post_processor": lambda x: x},
            )

    def test_tensor_in_event_plain_field_rejected(self):
        with pytest.raises(ContractViolationError, match="Tensor"):
            EngineEvent(request_id="req-1", event_index=0, token_ids=torch.tensor([7]))

    def test_arbitrary_object_rejected(self):
        class Opaque:
            pass

        with pytest.raises(ContractViolationError, match="Opaque"):
            make_request(disaggregated_metadata={"state": Opaque()})

    def test_callable_in_sampling_config_rejected(self):
        with pytest.raises(ContractViolationError, match="callable"):
            RuntimeSamplingConfig(stop_token_ids=[lambda: 42])

    def test_error_terminal_requires_typed_error(self):
        with pytest.raises(ContractViolationError, match="typed EngineError"):
            EngineEvent(request_id="req-1", event_index=0, terminal_kind=TerminalKind.ERROR)

    def test_error_payload_only_on_error_terminal(self):
        with pytest.raises(ContractViolationError, match="terminal_kind=ERROR"):
            EngineEvent(
                request_id="req-1",
                event_index=0,
                error=EngineError(code=EngineErrorCode.INTERNAL_ERROR, message="oops"),
            )

    def test_finish_reason_requires_terminal(self):
        with pytest.raises(ContractViolationError, match="terminal"):
            EngineEvent(request_id="req-1", event_index=0, finish_reason="stop")

    def test_empty_request_id_rejected(self):
        with pytest.raises(ContractViolationError, match="request_id"):
            make_request(request_id="")

    def test_priority_out_of_range_rejected(self):
        with pytest.raises(ContractViolationError, match="priority"):
            make_request(priority=1.5)


class TestEventOrderingChecker:
    """Per-sequence ordering and terminal-event invariants."""

    @staticmethod
    def _event(request_id="req-1", sequence_index=0, event_index=0, **kwargs):
        return EngineEvent(
            request_id=request_id,
            sequence_index=sequence_index,
            event_index=event_index,
            token_ids=kwargs.pop("token_ids", [7]),
            **kwargs,
        )

    def test_valid_interleaved_streams_accepted(self):
        checker = EventOrderingChecker()
        checker.observe(self._event("a", 0, 0, prompt_token_ids=[1]))
        checker.observe(self._event("b", 0, 0))
        checker.observe(self._event("a", 1, 0))
        checker.observe(self._event("a", 0, 1))
        checker.observe(self._event("b", 0, 1, terminal_kind=TerminalKind.FINISHED))
        checker.observe(self._event("a", 0, 2, terminal_kind=TerminalKind.FINISHED))
        checker.observe(self._event("a", 1, 1, terminal_kind=TerminalKind.FINISHED))
        assert checker.sequence_finished("a", 0)
        assert checker.sequence_finished("a", 1)
        assert checker.sequence_finished("b", 0)

    def test_second_terminal_event_rejected(self):
        checker = EventOrderingChecker()
        checker.observe(self._event(event_index=0, terminal_kind=TerminalKind.FINISHED))
        with pytest.raises(ProtocolViolationError, match="duplicate terminal"):
            checker.observe(self._event(event_index=1, terminal_kind=TerminalKind.FINISHED))

    def test_event_after_terminal_rejected(self):
        checker = EventOrderingChecker()
        checker.observe(self._event(event_index=0, terminal_kind=TerminalKind.ABORTED))
        with pytest.raises(ProtocolViolationError, match="after terminal"):
            checker.observe(self._event(event_index=1))

    def test_out_of_order_event_index_rejected(self):
        checker = EventOrderingChecker()
        checker.observe(self._event(event_index=0))
        with pytest.raises(ProtocolViolationError, match="out-of-order"):
            checker.observe(self._event(event_index=2))

    def test_prompt_metadata_on_later_event_rejected(self):
        checker = EventOrderingChecker()
        checker.observe(self._event(event_index=0))
        with pytest.raises(ProtocolViolationError, match="prompt metadata"):
            checker.observe(self._event(event_index=1, prompt_token_ids=[1, 2]))

    def test_terminal_per_sequence_is_independent(self):
        checker = EventOrderingChecker()
        checker.observe(
            self._event(sequence_index=0, event_index=0, terminal_kind=TerminalKind.FINISHED)
        )
        # Other sequences of the same request may continue.
        checker.observe(self._event(sequence_index=1, event_index=0))
        checker.observe(
            self._event(sequence_index=1, event_index=1, terminal_kind=TerminalKind.FINISHED)
        )

    def test_forget_clears_request_state(self):
        checker = EventOrderingChecker()
        checker.observe(self._event(event_index=0, terminal_kind=TerminalKind.FINISHED))
        checker.forget("req-1")
        assert not checker.sequence_finished("req-1", 0)
        checker.observe(self._event(event_index=0))
