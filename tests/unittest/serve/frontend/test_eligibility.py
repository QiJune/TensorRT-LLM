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
"""Eligibility predicate and fallback/reject routing tests for the pipeline."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm.engine_api import (
    EngineClient,
    EngineClientError,
    EngineErrorCode,
    EngineEvent,
    EngineRequest,
    RequestHandle,
    TerminalKind,
)
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.frontend.eligibility import (
    PipelineDeploymentMode,
    check_deployment,
    check_request,
)
from tensorrt_llm.serve.frontend.llm_api_pipeline import LlmApiEnginePipeline
from tensorrt_llm.serve.frontend.openai_pipeline import OpenAIServingPipeline
from tensorrt_llm.serve.frontend.request_processor import FrontendProcessor
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, CompletionRequest


def deployment_args(**overrides) -> SimpleNamespace:
    defaults = dict(backend="pytorch", orchestrator_type=None, num_postprocess_workers=0)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestDeploymentPredicate:
    def test_default_pytorch_deployment_is_eligible(self):
        assert check_deployment(deployment_args())

    def test_tensorrt_backend_ineligible(self):
        decision = check_deployment(deployment_args(backend="tensorrt"))
        assert not decision and "pytorch" in decision.reason

    def test_ray_orchestrator_ineligible(self):
        decision = check_deployment(deployment_args(orchestrator_type="ray"))
        assert not decision and "orchestrator" in decision.reason

    def test_rpc_orchestrator_ineligible(self):
        assert not check_deployment(deployment_args(orchestrator_type="rpc"))

    def test_postprocess_workers_ineligible(self):
        decision = check_deployment(deployment_args(num_postprocess_workers=2))
        assert not decision and "num_postprocess_workers=2" in decision.reason

    def test_post_processor_hook_disables_pipeline(self):
        """A configured hook disables the pipeline entirely.

        The hook runs at the in-process detok chokepoint; the pipeline
        must never bypass it.
        """
        decision = check_deployment(deployment_args(post_processor_hook="my_pkg.my_hook"))
        assert not decision and "post_processor_hook" in decision.reason


class TestRequestPredicate:
    def test_plain_text_chat_request_eligible(self):
        assert check_request(SamplingParams(end_id=2), endpoint="chat")

    def test_harmony_and_responses_endpoints_ineligible(self):
        assert not check_request(SamplingParams(end_id=2), endpoint="harmony")
        assert not check_request(SamplingParams(end_id=2), endpoint="responses")

    def test_multimodal_ineligible(self):
        assert not check_request(SamplingParams(end_id=2), endpoint="chat", has_multimodal=True)

    def test_guided_decoding_ineligible(self):
        from tensorrt_llm.sampling_params import GuidedDecodingParams

        params = SamplingParams(end_id=2, guided_decoding=GuidedDecodingParams(json_object=True))
        decision = check_request(params, endpoint="chat")
        assert not decision and "guided decoding" in decision.reason

    def test_logits_processor_ineligible(self):
        params = SamplingParams(end_id=2)
        params.logits_processor = lambda ids, logits, *a: logits
        assert not check_request(params, endpoint="chat")

    def test_thinking_token_budget_ineligible(self):
        params = SamplingParams(end_id=2, thinking_token_budget=128)
        assert not check_request(params, endpoint="chat")

    def test_embedding_bias_ineligible(self):
        params = SamplingParams(end_id=2)
        params.embedding_bias = torch.zeros(8)
        assert not check_request(params, endpoint="chat")

    def test_logits_returns_ineligible(self):
        assert not check_request(
            SamplingParams(end_id=2, return_context_logits=True), endpoint="chat"
        )
        assert not check_request(
            SamplingParams(end_id=2, return_generation_logits=True), endpoint="completions"
        )

    def test_lora_and_prompt_adapter_ineligible(self):
        params = SamplingParams(end_id=2)
        assert not check_request(params, endpoint="chat", lora_request=object())
        assert not check_request(params, endpoint="chat", prompt_adapter_request=object())

    def test_disaggregated_params_ineligible(self):
        assert not check_request(
            SamplingParams(end_id=2), endpoint="chat", disaggregated_params=object()
        )

    def test_multi_prompt_ineligible(self):
        assert not check_request(SamplingParams(end_id=2), endpoint="completions", num_prompts=3)


VOCAB_TEXT = {5: "Hello", 6: " world"}


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    chat_template = "fake-template"

    def encode(self, text, add_special_tokens=True, **kwargs):
        return [hash(w) % 100 + 10 for w in text.split()]

    def get_chat_template(self, chat_template=None, tools=None):
        return chat_template or self.chat_template

    def apply_chat_template(self, conversation=None, **kwargs):
        return " ".join(m["content"] for m in conversation)

    def decode(self, token_ids, **kwargs):
        if isinstance(token_ids, int):
            return VOCAB_TEXT.get(token_ids, f"<{token_ids}>")
        return "".join(VOCAB_TEXT.get(t, f"<{t}>") for t in token_ids)

    def decode_incrementally(
        self, token_ids, prev_text=None, states=None, *, flush=False, stream_interval=1, **kwargs
    ):
        return (prev_text or "") + self.decode(token_ids), (states or {})


class FakeHandle(RequestHandle):
    def __init__(self, request: EngineRequest):
        self._request = request

    @property
    def request_id(self) -> str:
        return self._request.request_id

    def _events(self):
        return [
            EngineEvent(
                request_id=self._request.request_id,
                event_index=0,
                token_ids=[5, 6],
                prompt_token_ids=list(self._request.prompt_token_ids),
                terminal_kind=TerminalKind.FINISHED,
                finish_reason="length",
            )
        ]

    def events(self):
        yield from self._events()

    async def aevents(self):
        for event in self._events():
            yield event

    def abort(self) -> None:
        pass


class FakeEngineClient(EngineClient):
    def __init__(self):
        self.submitted: list[EngineRequest] = []

    def submit(self, request: EngineRequest) -> RequestHandle:
        self.submitted.append(request)
        return FakeHandle(request)

    def abort(self, request_id: str) -> None:
        pass

    def get_capabilities(self):
        return {}

    def check_health(self) -> bool:
        return True

    def get_stats(self, timeout: float):
        return []

    def get_kv_events(self, timeout: float):
        return []

    def shutdown(self) -> None:
        pass


@pytest.fixture(autouse=True)
def reset_instrumentation():
    FrontendProcessor.reset_invocation_count()
    yield
    FrontendProcessor.reset_invocation_count()


@pytest.fixture
def client():
    return FakeEngineClient()


class _ModelConfig:
    model_type = "llama"


def make_pipeline(client, mode=PipelineDeploymentMode.COLOCATED) -> OpenAIServingPipeline:
    return OpenAIServingPipeline(
        client,
        FrontendProcessor(FakeTokenizer(), model_config=_ModelConfig()),
        model_label="test-model",
        mode=mode,
    )


class TestOpenAIRouting:
    def test_eligible_completion_served_by_pipeline(self, client):
        pipeline = make_pipeline(client)
        request = CompletionRequest(model="test-model", prompt="hi there", max_tokens=4)
        response = asyncio.run(pipeline.try_completion(request))
        assert response is not None
        assert len(client.submitted) == 1
        assert client.submitted[0].crosses_neutral_wire
        assert response.choices[0].text == "Hello world"
        assert response.usage.completion_tokens == 2

    def test_chat_cache_salt_reaches_engine_request(self, client):
        pipeline = make_pipeline(client)
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4,
            cache_salt="tenant-B",
        )
        response = asyncio.run(pipeline.try_chat(request))
        assert response is not None
        assert client.submitted[0].cache_salt == "tenant-B"

    def test_prompt_ignore_length_reaches_engine_request(self, client):
        pipeline = make_pipeline(client)
        request = CompletionRequest(
            model="test-model", prompt="hi there", max_tokens=4, prompt_ignore_length=2
        )
        response = asyncio.run(pipeline.try_completion(request))
        assert response is not None
        assert client.submitted[0].sampling.prompt_ignore_length == 2

    def test_pipeline_valueerror_becomes_typed_error_response(self, client):
        """New-path validation failures become typed error responses.

        E.g. greedy n>1 must be converted by the endpoint, not a 500.
        """
        from tensorrt_llm.serve.openai_server import OpenAIServer

        pipeline = make_pipeline(client)
        # Greedy n>1 raises ValueError inside try_chat's to_sampling_params.
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            n=2,
            temperature=0.0,
        )
        with pytest.raises(ValueError):
            asyncio.run(pipeline.try_chat(request))

        # The endpoint wrapper converts it to a BadRequest error response.
        fake_self = SimpleNamespace(
            _engine_pipeline=pipeline,
            create_error_response=OpenAIServer.create_error_response,
        )
        response = asyncio.run(OpenAIServer.openai_chat(fake_self, request, None))
        assert response.status_code == 400
        assert client.submitted == []

    def test_pipeline_engine_client_error_becomes_typed_error_response(self, client):
        from tensorrt_llm.engine_api import EngineError
        from tensorrt_llm.serve.openai_server import OpenAIServer

        class RaisingClient(FakeEngineClient):
            def submit(self, request):
                raise EngineClientError(
                    EngineError(code=EngineErrorCode.REQUEST_FAILED, message="submit boom")
                )

        pipeline = make_pipeline(RaisingClient())
        request = CompletionRequest(model="test-model", prompt="hi there", max_tokens=4)
        with pytest.raises(EngineClientError):
            asyncio.run(pipeline.try_completion(request))

        fake_self = SimpleNamespace(
            _engine_pipeline=pipeline,
            create_error_response=OpenAIServer.create_error_response,
        )
        response = asyncio.run(OpenAIServer.openai_completion(fake_self, request, None))
        assert response.status_code == 400

    def test_engine_terminal_error_becomes_typed_error_response(self, client):
        """An engine ERROR terminal becomes a structured error response.

        A non-streaming request whose engine stream ends in an ERROR
        terminal must surface a structured error, not a 500.
        """
        from tensorrt_llm.engine_api import EngineError
        from tensorrt_llm.serve.openai_server import OpenAIServer

        class ErrorTerminalHandle(FakeHandle):
            def _events(self):
                return [
                    EngineEvent(
                        request_id=self._request.request_id,
                        event_index=0,
                        terminal_kind=TerminalKind.ERROR,
                        error=EngineError(
                            code=EngineErrorCode.REQUEST_FAILED, message="runtime request failed"
                        ),
                    )
                ]

        class ErrorTerminalClient(FakeEngineClient):
            def submit(self, request):
                self.submitted.append(request)
                return ErrorTerminalHandle(request)

        pipeline = make_pipeline(ErrorTerminalClient())
        request = CompletionRequest(model="test-model", prompt="hi there", max_tokens=4)
        # _collect surfaces the typed engine error.
        with pytest.raises(EngineClientError):
            asyncio.run(pipeline.try_completion(request))

        fake_self = SimpleNamespace(
            _engine_pipeline=pipeline,
            create_error_response=OpenAIServer.create_error_response,
        )
        response = asyncio.run(OpenAIServer.openai_completion(fake_self, request, None))
        assert response.status_code == 400

    def test_ineligible_request_falls_back_colocated(self, client):
        pipeline = make_pipeline(client)
        request = CompletionRequest(
            model="test-model", prompt="hi", max_tokens=4, logit_bias={"5": 2.0}
        )
        response = asyncio.run(pipeline.try_completion(request))
        assert response is None
        assert client.submitted == []

    def test_ineligible_request_rejected_detached_with_no_partial_submission(self, client):
        pipeline = make_pipeline(client, mode=PipelineDeploymentMode.DETACHED)
        request = CompletionRequest(
            model="test-model", prompt="hi", max_tokens=4, logit_bias={"5": 2.0}
        )
        response = asyncio.run(pipeline.try_completion(request))
        assert response is not None
        assert response.status_code == 400
        assert EngineErrorCode.UNSUPPORTED_CAPABILITY.value in response.body.decode()
        # Nothing partial crossed the boundary.
        assert client.submitted == []

    def test_multi_prompt_completion_falls_back(self, client):
        pipeline = make_pipeline(client)
        request = CompletionRequest(model="test-model", prompt=["a", "b"], max_tokens=4)
        assert asyncio.run(pipeline.try_completion(request)) is None
        assert client.submitted == []

    def test_request_chat_template_rejected_by_server_policy(self, client):
        """Server policy applies before any routing decision."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            chat_template="{{ custom }}",
        )
        pipeline = make_pipeline(client)  # allow_request_chat_template=False
        assert asyncio.run(pipeline.try_chat(request)) is None
        assert client.submitted == []

        detached = make_pipeline(client, mode=PipelineDeploymentMode.DETACHED)
        response = asyncio.run(detached.try_chat(request))
        assert response.status_code == 400
        assert client.submitted == []

    def test_strict_tools_synthesize_guided_decoding_and_fall_back(self, client):
        """Strict tools become guided decoding pre-eligibility.

        The request must never cross the seam this loop.
        """
        from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory

        class _Info:
            trigger = "<tool>"
            begin = "<tool>"
            end = "</tool>"

        class StrictCapableParser:
            needs_raw_special_tokens = True

            def supports_structural_tag(self):
                return True

            def structure_info(self):
                return lambda name: _Info()

        ToolParserFactory.parsers["_oracle_strict_parser"] = StrictCapableParser
        try:
            request = ChatCompletionRequest(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "strict": True,
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )
            pipeline = OpenAIServingPipeline(
                client,
                FrontendProcessor(FakeTokenizer()),
                model_label="test-model",
                tool_parser="_oracle_strict_parser",
            )
            assert asyncio.run(pipeline.try_chat(request)) is None
            assert client.submitted == []

            detached = OpenAIServingPipeline(
                client,
                FrontendProcessor(FakeTokenizer()),
                model_label="test-model",
                mode=PipelineDeploymentMode.DETACHED,
                tool_parser="_oracle_strict_parser",
            )
            response = asyncio.run(detached.try_chat(request))
            assert response.status_code == 400
            assert client.submitted == []
        finally:
            ToolParserFactory.parsers.pop("_oracle_strict_parser", None)

    def test_raw_special_tokens_parser_mutation_applied_before_split(self, client):
        """Non-strict tools with a raw-special-tokens parser stay eligible.

        The historical skip_special_tokens=False mutation reaches the
        frontend output config.
        """
        from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory

        class RawTokensParser:
            needs_raw_special_tokens = True

            def supports_structural_tag(self):
                return False

            def parse_streaming_increment(self, text, tools):
                from types import SimpleNamespace

                return SimpleNamespace(normal_text=text, calls=[])

            def detect_and_parse(self, text, tools):
                from types import SimpleNamespace

                return SimpleNamespace(normal_text=text, calls=[])

        ToolParserFactory.parsers["_oracle_raw_parser"] = RawTokensParser
        observed = {}

        class RecordingProcessor(FrontendProcessor):
            def process_chat(self, conversation, sampling_params, **kwargs):
                observed["skip_special_tokens"] = sampling_params.skip_special_tokens
                return super().process_chat(conversation, sampling_params, **kwargs)

        try:
            request = ChatCompletionRequest(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "parameters": {}},
                    }
                ],
            )

            class ModelConfig:
                model_type = "llama"

            pipeline = OpenAIServingPipeline(
                client,
                RecordingProcessor(FakeTokenizer(), model_config=ModelConfig()),
                model_label="test-model",
                tool_parser="_oracle_raw_parser",
            )
            response = asyncio.run(pipeline.try_chat(request))
            assert response is not None
            assert observed["skip_special_tokens"] is False
            assert len(client.submitted) == 1
        finally:
            ToolParserFactory.parsers.pop("_oracle_raw_parser", None)

    def test_multimodal_chat_falls_back(self, client):
        pipeline = make_pipeline(client)
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                    ],
                }
            ],
        )
        assert asyncio.run(pipeline.try_chat(request)) is None
        assert client.submitted == []


class TestLlmApiRouting:
    def test_eligible_text_request_served(self, client):
        pipeline = LlmApiEnginePipeline(client, FrontendProcessor(FakeTokenizer()))
        result = pipeline.try_generate_async("hello world", SamplingParams(end_id=2))
        assert result is not None
        output = result.result()
        assert output.finished
        assert output.outputs[0].token_ids == [5, 6]
        assert output.outputs[0].text == "Hello world"
        assert output.outputs[0].finish_reason == "length"

    def test_priority_and_cache_salt_reach_engine_request(self, client):
        pipeline = LlmApiEnginePipeline(client, FrontendProcessor(FakeTokenizer()))
        result = pipeline.try_generate_async(
            "hello world",
            SamplingParams(end_id=2),
            priority=0.9,
            cache_salt="tenant-C",
        )
        assert result is not None
        assert client.submitted[0].priority == 0.9
        assert client.submitted[0].cache_salt == "tenant-C"

    def test_prompt_ignore_length_reaches_engine_request(self, client):
        pipeline = LlmApiEnginePipeline(client, FrontendProcessor(FakeTokenizer()))
        result = pipeline.try_generate_async(
            "hello world", SamplingParams(end_id=2, prompt_ignore_length=3)
        )
        assert result is not None
        assert client.submitted[0].sampling.prompt_ignore_length == 3

    def test_return_perf_metrics_from_normalized_params_reaches_engine(self, client):
        """Normalized sampling-param defaults reach the engine request.

        LLM.generate_async applies LLM-level defaults (e.g.
        return_perf_metrics) before the pipeline handles the request; the
        pipeline must carry the normalized value through.
        """
        pipeline = LlmApiEnginePipeline(client, FrontendProcessor(FakeTokenizer()))
        # Simulates the post-_prepare_sampling_params state where the LLM
        # applied its return_perf_metrics default.
        result = pipeline.try_generate_async(
            "hello world", SamplingParams(end_id=2, return_perf_metrics=True)
        )
        assert result is not None
        assert client.submitted[0].sampling.return_perf_metrics is True

    def test_unmapped_kwargs_fall_back(self, client):
        pipeline = LlmApiEnginePipeline(client, FrontendProcessor(FakeTokenizer()))
        result = pipeline.try_generate_async(
            "hello", SamplingParams(end_id=2), kv_cache_retention_config=object()
        )
        assert result is None
        assert client.submitted == []

    def test_guided_decoding_falls_back(self, client):
        from tensorrt_llm.sampling_params import GuidedDecodingParams

        pipeline = LlmApiEnginePipeline(client, FrontendProcessor(FakeTokenizer()))
        params = SamplingParams(end_id=2, guided_decoding=GuidedDecodingParams(json_object=True))
        assert pipeline.try_generate_async("hello", params) is None
        assert client.submitted == []


class TestFlagOffBehavior:
    def test_flag_defaults_off(self):
        field = TorchLlmArgs.model_fields["enable_engine_client_pipeline"]
        assert field.default is False

    def test_flag_off_constructs_no_pipeline_objects(self):
        from tensorrt_llm.serve.openai_server import OpenAIServer

        fake_server = SimpleNamespace(
            generator=SimpleNamespace(args=SimpleNamespace(enable_engine_client_pipeline=False))
        )
        assert OpenAIServer._maybe_create_engine_pipeline(fake_server) is None
        assert FrontendProcessor.invocation_count == 0

    def test_flag_on_but_ineligible_deployment_constructs_no_pipeline(self):
        from tensorrt_llm.serve.openai_server import OpenAIServer

        fake_server = SimpleNamespace(
            generator=SimpleNamespace(
                args=SimpleNamespace(
                    enable_engine_client_pipeline=True,
                    backend="pytorch",
                    orchestrator_type="ray",
                    num_postprocess_workers=0,
                )
            )
        )
        assert OpenAIServer._maybe_create_engine_pipeline(fake_server) is None
        assert FrontendProcessor.invocation_count == 0
