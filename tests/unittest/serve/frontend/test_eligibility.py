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

    def encode(self, text, add_special_tokens=True, **kwargs):
        return [hash(w) % 100 + 10 for w in text.split()]

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


def make_pipeline(client, mode=PipelineDeploymentMode.COLOCATED) -> OpenAIServingPipeline:
    return OpenAIServingPipeline(
        client,
        FrontendProcessor(FakeTokenizer()),
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
