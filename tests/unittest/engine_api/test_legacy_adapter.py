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
"""CPU-only unit tests for LegacyEngineClientAdapter with fake runtime responses."""

import dataclasses
from typing import Optional

import pytest

from tensorrt_llm.engine_api import (
    EngineClientError,
    EngineErrorCode,
    EngineEvent,
    EngineRequest,
    EventOrderingChecker,
    RuntimeSamplingConfig,
    TerminalKind,
)
from tensorrt_llm.engine_api.legacy_adapter import LegacyEngineClientAdapter
from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.result import GenerationResult, LogProbsResult, ResponseWrapper
from tensorrt_llm.executor.utils import ErrorResponse


class FakeFinishReason:
    """Mimics the runtime finish-reason enum by name."""

    def __init__(self, name: str) -> None:
        self.name = name


class FakeResult:
    """Duck-typed runtime response payload (LlmResult / bindings Result)."""

    def __init__(
        self,
        output_token_ids,
        finish_reasons=None,
        sequence_index: int = 0,
        is_final: bool = False,
        cum_log_probs=None,
        log_probs=None,
        context_phase_params=None,
    ) -> None:
        self.output_token_ids = output_token_ids
        self.finish_reasons = finish_reasons or [FakeFinishReason("NOT_FINISHED")]
        self.sequence_index = sequence_index
        self.is_final = is_final
        self.cum_log_probs = cum_log_probs
        self.log_probs = log_probs
        self.context_phase_params = context_phase_params
        self.decoding_iter = 1


class FakeResponse:
    """Duck-typed runtime response (passes is_llm_response)."""

    def __init__(self, client_id: int, result: FakeResult, error_msg: Optional[str] = None):
        self.client_id = client_id
        self._result = result
        self.error_msg = error_msg

    def has_error(self) -> bool:
        return self.error_msg is not None

    @property
    def result(self) -> FakeResult:
        return self._result


class FakeContextPhaseParams:
    def __init__(self) -> None:
        self.first_gen_tokens = [17]
        self.req_id = 991
        self.opaque_state = b"\x01\x02"
        self.draft_tokens = None
        self.ctx_dp_rank = 0
        self.disagg_info_endpoint = "tcp://ctx:5555"


class FakeExecutor(GenerationExecutor):
    """In-memory executor: captures submissions, lets tests inject raw responses."""

    def __init__(self) -> None:
        super().__init__(num_postprocess_workers=0)
        self.submitted: list[GenerationRequest] = []
        self.aborted_ids: list[int] = []
        self.results: dict[int, GenerationResult] = {}
        self.was_shutdown = False

    def submit(self, request: GenerationRequest) -> GenerationResult:
        request.set_id(self._get_next_client_id())
        self.submitted.append(request)
        result = GenerationResult(request, executor=self)
        self.results[request.id] = result
        return result

    def abort_request(self, request_id: int) -> None:
        self.aborted_ids.append(request_id)

    def shutdown(self) -> None:
        self.was_shutdown = True

    # Test helper: inject a raw runtime response for a submitted request.
    def push(self, request_id: int, response) -> None:
        self.results[request_id].queue.put(response)


@pytest.fixture
def executor() -> FakeExecutor:
    return FakeExecutor()


@pytest.fixture
def adapter(executor) -> LegacyEngineClientAdapter:
    return LegacyEngineClientAdapter(executor)


def make_request(request_id="req-1", streaming=True, **sampling_overrides) -> EngineRequest:
    sampling = RuntimeSamplingConfig(
        max_tokens=8, end_id=2, stop_token_ids=[42], **sampling_overrides
    )
    return EngineRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        sampling=sampling,
        streaming=streaming,
    )


def submitted_id(executor: FakeExecutor) -> int:
    return executor.submitted[-1].id


class TestSubmission:
    def test_runtime_detokenization_disabled(self, adapter, executor):
        adapter.submit(make_request())
        params = executor.submitted[-1].sampling_params
        assert params.detokenize is False

    def test_no_callables_on_runtime_request(self, adapter, executor):
        adapter.submit(make_request())
        request = executor.submitted[-1]
        assert request.postproc_params is None
        callables = [
            name
            for name in vars(request)
            if callable(getattr(request, name)) and not name.startswith("__")
        ]
        assert callables == []
        assert request.sampling_params.logits_processor is None

    def test_stop_sequences_cross_pre_tokenized(self, adapter, executor):
        adapter.submit(make_request(stop_sequence_token_ids=[[7, 8], [9]]))
        params = executor.submitted[-1].sampling_params
        assert params._get_stop_words() == [[42], [7, 8], [9]]

    def test_duplicate_request_id_rejected(self, adapter):
        adapter.submit(make_request())
        with pytest.raises(EngineClientError) as excinfo:
            adapter.submit(make_request())
        assert excinfo.value.error.code is EngineErrorCode.INVALID_REQUEST

    def test_postproc_worker_executor_rejected(self):
        class PostprocExecutor(FakeExecutor):
            def __init__(self):
                GenerationExecutor.__init__(self, num_postprocess_workers=2)

        with pytest.raises(EngineClientError) as excinfo:
            LegacyEngineClientAdapter(PostprocExecutor())
        assert excinfo.value.error.code is EngineErrorCode.UNSUPPORTED_CAPABILITY


class TestEventNormalization:
    def test_streaming_stream_normalizes_to_ordered_events(self, adapter, executor):
        handle = adapter.submit(make_request())
        rid = submitted_id(executor)
        executor.push(rid, FakeResponse(rid, FakeResult([[5]])))
        executor.push(rid, FakeResponse(rid, FakeResult([[6]])))
        executor.push(
            rid,
            FakeResponse(
                rid,
                FakeResult([[42]], finish_reasons=[FakeFinishReason("STOP_WORDS")], is_final=True),
            ),
        )

        events = list(handle.events())
        checker = EventOrderingChecker()
        for event in events:
            checker.observe(event)

        assert [event.event_index for event in events] == [0, 1, 2]
        assert [event.token_ids for event in events] == [[5], [6], [42]]
        assert events[0].prompt_token_ids == [1, 2, 3]
        assert events[1].prompt_token_ids is None
        assert events[-1].terminal_kind is TerminalKind.FINISHED
        assert events[-1].finish_reason == "stop"
        assert events[-1].stop_reason == 42

    def test_non_streaming_single_final_event(self, adapter, executor):
        handle = adapter.submit(make_request(streaming=False))
        rid = submitted_id(executor)
        executor.push(
            rid,
            FakeResponse(
                rid,
                FakeResult([[5, 6, 2]], finish_reasons=[FakeFinishReason("END_ID")], is_final=True),
            ),
        )
        events = list(handle.events())
        assert len(events) == 1
        assert events[0].token_ids == [5, 6, 2]
        assert events[0].finish_reason == "stop"
        assert events[0].stop_reason == 2

    def test_length_finish_reason(self, adapter, executor):
        handle = adapter.submit(make_request())
        rid = submitted_id(executor)
        executor.push(
            rid,
            FakeResponse(
                rid, FakeResult([[5]], finish_reasons=[FakeFinishReason("LENGTH")], is_final=True)
            ),
        )
        assert list(handle.events())[-1].finish_reason == "length"

    def test_multiple_sequences_each_get_terminal(self, adapter, executor):
        handle = adapter.submit(make_request(n=2, best_of=2, temperature=0.8))
        rid = submitted_id(executor)
        executor.push(rid, FakeResponse(rid, FakeResult([[5]], sequence_index=0)))
        executor.push(rid, FakeResponse(rid, FakeResult([[7]], sequence_index=1)))
        executor.push(
            rid,
            FakeResponse(
                rid,
                FakeResult([[6]], sequence_index=0, finish_reasons=[FakeFinishReason("LENGTH")]),
            ),
        )
        executor.push(
            rid,
            FakeResponse(
                rid,
                FakeResult(
                    [[8]],
                    sequence_index=1,
                    finish_reasons=[FakeFinishReason("LENGTH")],
                    is_final=True,
                ),
            ),
        )
        events = list(handle.events())
        checker = EventOrderingChecker()
        for event in events:
            checker.observe(event)
        assert checker.sequence_finished("req-1", 0)
        assert checker.sequence_finished("req-1", 1)
        terminals = [event for event in events if event.is_terminal]
        assert {event.sequence_index for event in terminals} == {0, 1}

    def test_beam_search_events_are_cumulative(self, adapter, executor):
        handle = adapter.submit(make_request(use_beam_search=True, n=2, best_of=2))
        rid = submitted_id(executor)
        executor.push(
            rid,
            FakeResponse(
                rid,
                FakeResult(
                    [[5, 6], [5, 7]],
                    finish_reasons=[FakeFinishReason("LENGTH"), FakeFinishReason("LENGTH")],
                    is_final=True,
                ),
            ),
        )
        events = list(handle.events())
        assert all(event.cumulative for event in events)
        assert [event.sequence_index for event in events] == [0, 1]
        assert [event.token_ids for event in events] == [[5, 6], [5, 7]]

    def test_logprobs_from_response_tensors(self, adapter, executor):
        handle = adapter.submit(make_request(logprobs=0, logprobs_simple_format=True))
        rid = submitted_id(executor)
        executor.push(
            rid,
            FakeResponse(
                rid,
                FakeResult(
                    [[5]],
                    log_probs=[[-0.25]],
                    cum_log_probs=[-0.25],
                    finish_reasons=[FakeFinishReason("LENGTH")],
                    is_final=True,
                ),
            ),
        )
        event = list(handle.events())[0]
        assert event.logprobs == [-0.25]
        assert event.cumulative_logprob == -0.25

    def test_cumulative_logprobs_source_sliced_to_delta(self, adapter, executor):
        """Cumulative log_probs are sliced to the per-event delta.

        A streamwise-cumulative source must be sliced so each delta event
        carries only the newly generated logprob entries — else later
        events carry more logprobs than tokens.
        """
        handle = adapter.submit(make_request(logprobs=0, logprobs_simple_format=True))
        rid = submitted_id(executor)
        # log_probs grows cumulatively while each event's token_ids is a delta.
        executor.push(rid, FakeResponse(rid, FakeResult([[5]], log_probs=[[-0.1]])))
        executor.push(rid, FakeResponse(rid, FakeResult([[6]], log_probs=[[-0.1, -0.2]])))
        executor.push(
            rid,
            FakeResponse(
                rid,
                FakeResult(
                    [[7]],
                    log_probs=[[-0.1, -0.2, -0.3]],
                    finish_reasons=[FakeFinishReason("LENGTH")],
                    is_final=True,
                ),
            ),
        )
        events = list(handle.events())
        for e in events:
            assert len(e.logprobs) == len(e.token_ids), (
                f"event {e.event_index} logprobs {e.logprobs} misaligned with tokens {e.token_ids}"
            )
        assert [e.logprobs for e in events] == [[-0.1], [-0.2], [-0.3]]

    def test_prompt_logprobs_from_response_wrapper_first_event_only(self, adapter, executor):
        handle = adapter.submit(make_request(prompt_logprobs=0))
        rid = submitted_id(executor)
        wrapped = ResponseWrapper(
            FakeResponse(rid, FakeResult([[5]])),
            logprobs=LogProbsResult(prompt=[-0.5, -0.6, -0.7], generation=None),
        )
        executor.push(rid, wrapped)
        executor.push(
            rid,
            FakeResponse(
                rid, FakeResult([[6]], finish_reasons=[FakeFinishReason("LENGTH")], is_final=True)
            ),
        )
        events = list(handle.events())
        assert events[0].prompt_logprobs == [-0.5, -0.6, -0.7]
        assert events[1].prompt_logprobs is None

    def test_cached_prompt_logprobs_on_every_response_first_event_only(self, adapter, executor):
        """Cached prompt logprobs attach only to a sequence's first event.

        The PyTorch worker returns cached prompt logprobs on every response;
        the adapter must attach them only to the first event so the ordering
        checker does not reject event_index > 0.
        """
        handle = adapter.submit(make_request(prompt_logprobs=0))
        rid = submitted_id(executor)
        prompt_lp = LogProbsResult(prompt=[-0.5, -0.6, -0.7], generation=None)
        # Cached prompt logprobs ride along on every response, including the
        # second and the terminal.
        executor.push(
            rid, ResponseWrapper(FakeResponse(rid, FakeResult([[5]])), logprobs=prompt_lp)
        )
        executor.push(
            rid, ResponseWrapper(FakeResponse(rid, FakeResult([[6]])), logprobs=prompt_lp)
        )
        executor.push(
            rid,
            ResponseWrapper(
                FakeResponse(
                    rid,
                    FakeResult([[7]], finish_reasons=[FakeFinishReason("LENGTH")], is_final=True),
                ),
                logprobs=prompt_lp,
            ),
        )
        events = list(handle.events())
        # The whole stream must satisfy the ordering contract (this is what
        # the socket client enforces).
        checker = EventOrderingChecker()
        for event in events:
            checker.observe(event)
        assert events[0].prompt_logprobs == [-0.5, -0.6, -0.7]
        assert events[0].prompt_token_ids == [1, 2, 3]
        assert all(e.prompt_logprobs is None for e in events[1:])
        assert all(e.prompt_token_ids is None for e in events[1:])

    def test_disaggregated_metadata_opaque_passthrough(self, adapter, executor):
        handle = adapter.submit(make_request(streaming=False))
        rid = submitted_id(executor)
        executor.push(
            rid,
            FakeResponse(
                rid,
                FakeResult(
                    [[17]],
                    finish_reasons=[FakeFinishReason("NOT_FINISHED")],
                    is_final=True,
                    context_phase_params=FakeContextPhaseParams(),
                ),
            ),
        )
        event = list(handle.events())[0]
        assert event.disaggregated_metadata == {
            "request_type": "context_only",
            "first_gen_tokens": [17],
            "ctx_request_id": 991,
            "opaque_state": b"\x01\x02",
            "ctx_dp_rank": 0,
            "ctx_info_endpoint": "tcp://ctx:5555",
        }
        assert event.finish_reason == "not_finished"


class TestAbortAndErrors:
    def test_mid_stream_abort_terminates_with_terminal_event(self, adapter, executor):
        handle = adapter.submit(make_request())
        rid = submitted_id(executor)
        executor.push(rid, FakeResponse(rid, FakeResult([[5]])))
        handle.abort()
        assert executor.aborted_ids == [rid]
        executor.push(
            rid,
            FakeResponse(
                rid,
                FakeResult([[]], finish_reasons=[FakeFinishReason("CANCELLED")], is_final=True),
            ),
        )
        events = list(handle.events())
        assert events[-1].terminal_kind is TerminalKind.ABORTED
        assert events[-1].finish_reason == "cancelled"

    def test_abort_unknown_request_raises_typed_error(self, adapter):
        with pytest.raises(EngineClientError) as excinfo:
            adapter.abort("never-submitted")
        assert excinfo.value.error.code is EngineErrorCode.UNKNOWN_REQUEST

    def test_error_response_becomes_typed_error_event(self, adapter, executor):
        handle = adapter.submit(make_request())
        rid = submitted_id(executor)
        executor.push(rid, ErrorResponse(rid, "worker exploded", rid))
        events = list(handle.events())
        assert len(events) == 1
        assert events[0].terminal_kind is TerminalKind.ERROR
        assert events[0].error.code is EngineErrorCode.REQUEST_FAILED
        assert "worker exploded" in events[0].error.message

    def test_response_with_error_flag_becomes_typed_error_event(self, adapter, executor):
        handle = adapter.submit(make_request())
        rid = submitted_id(executor)
        executor.push(rid, FakeResponse(rid, FakeResult([[5]]), error_msg="request failed"))
        events = list(handle.events())
        assert events[0].terminal_kind is TerminalKind.ERROR
        assert "request failed" in events[0].error.message


class TestRawOutputConformance:
    """The adapter must deliver raw token-level events and leak no legacy objects."""

    def test_contract_has_no_text_field(self):
        assert "text" not in {f.name for f in dataclasses.fields(EngineEvent)}

    def test_stop_tokens_not_trimmed(self, adapter, executor):
        """Raw events keep the full stop sequence; trimming is frontend work."""
        handle = adapter.submit(make_request(stop_sequence_token_ids=[[7, 8]]))
        rid = submitted_id(executor)
        executor.push(
            rid,
            FakeResponse(
                rid,
                FakeResult(
                    [[5, 7, 8]], finish_reasons=[FakeFinishReason("STOP_WORDS")], is_final=True
                ),
            ),
        )
        event = list(handle.events())[0]
        assert event.token_ids == [5, 7, 8]

    def test_only_engine_events_escape(self, adapter, executor):
        handle = adapter.submit(make_request())
        rid = submitted_id(executor)
        executor.push(rid, FakeResponse(rid, FakeResult([[5]])))
        executor.push(
            rid,
            FakeResponse(
                rid, FakeResult([[6]], finish_reasons=[FakeFinishReason("LENGTH")], is_final=True)
            ),
        )
        assert not isinstance(handle, GenerationResult)
        for event in handle.events():
            assert isinstance(event, EngineEvent)

    def test_unknown_response_type_raises_typed_error(self, adapter, executor):
        handle = adapter.submit(make_request())
        rid = submitted_id(executor)
        executor.push(rid, object())
        with pytest.raises(EngineClientError) as excinfo:
            list(handle.events())
        assert excinfo.value.error.code is EngineErrorCode.INTERNAL_ERROR


class TestControlPlane:
    def test_health_and_capabilities(self, adapter, executor):
        assert adapter.check_health() is True
        capabilities = adapter.get_capabilities()
        assert capabilities["control"]["health"] is True

    def test_shutdown_delegates(self, adapter, executor):
        adapter.shutdown()
        assert executor.was_shutdown
