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
"""Golden-output oracle harness: old path vs engine-client pipeline.

Given a fixed synthetic stream of runtime responses, the same request must
produce exactly equal output on the historical in-process path and on the
engine-client pipeline: text, decoded SSE chunk structure, usage,
finish_reason, logprobs. The oracle compares the output *transformation*
given fixed token-event streams — generation itself is not exercised.

Volatile identity metadata (response ``id``, ``created`` timestamp) is
pinned/normalized on both paths before comparison; everything else must be
byte-equal after JSON decoding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

from tensorrt_llm.engine_api.legacy_adapter import LegacyEngineClientAdapter
from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.result import GenerationResult, Logprob
from tensorrt_llm.inputs.utils import apply_chat_template
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines
from tensorrt_llm.serve.frontend.eligibility import PipelineDeploymentMode
from tensorrt_llm.serve.frontend.llm_api_pipeline import LlmApiEnginePipeline
from tensorrt_llm.serve.frontend.openai_pipeline import OpenAIServingPipeline
from tensorrt_llm.serve.frontend.request_processor import FrontendProcessor
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, CompletionRequest
from tensorrt_llm.serve.postprocess_handlers import (
    ChatPostprocArgs,
    CompletionPostprocArgs,
    chat_response_post_processor,
    chat_stream_post_processor,
    completion_response_post_processor,
    completion_stream_post_processor,
)

PINNED_STREAM_ID = "oracle-stream"
PINNED_CREATED = 1111111111


# --- deterministic tokenizer -------------------------------------------------

VOCAB = {
    101: "Hello",
    102: " world",
    103: "!",
    104: "<eos>",
    105: " foo",
    106: " bar",
    107: "<STOP>",
    108: " baz",
    109: " User:",
    110: " ST",
    111: "OP",
}
REVERSE_VOCAB = {text: token_id for token_id, text in VOCAB.items()}


class OracleTokenizer:
    """Deterministic tokenizer shared by both paths."""

    eos_token_id = 104
    pad_token_id = 100
    chat_template = "oracle-template"

    def encode(self, text, add_special_tokens=True, truncation=False, max_length=None):
        if text in REVERSE_VOCAB:
            return [REVERSE_VOCAB[text]]
        token_ids = [hash(word) % 1000 + 200 for word in text.split()]
        if add_special_tokens:
            token_ids = [199] + token_ids
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return token_ids

    def decode(self, token_ids, **kwargs):
        if isinstance(token_ids, int):
            return VOCAB.get(token_ids, f"<{token_ids}>")
        return "".join(VOCAB.get(t, f"<{t}>") for t in token_ids)

    def decode_incrementally(
        self, token_ids, prev_text=None, states=None, *, flush=False, stream_interval=1, **kwargs
    ):
        return (prev_text or "") + self.decode(token_ids), (states or {})

    def get_chat_template(self, chat_template=None, tools=None):
        return chat_template or self.chat_template

    def apply_chat_template(self, conversation=None, tools=None, documents=None, **kwargs):
        rendered = "".join(f"[{m['role']}]{m['content']}" for m in conversation)
        if kwargs.get("add_generation_prompt"):
            rendered += "[assistant]"
        return rendered


class OracleModelConfig:
    model_type = "llama"


# --- fixture model ---------------------------------------------------------


@dataclass
class StepSpec:
    """One synthetic runtime response step."""

    token_ids: list[int]
    sequence_index: int = 0
    finish: str = "NOT_FINISHED"
    is_final: bool = False
    logprobs: Optional[list] = None  # entries: float or {token_id: Logprob}
    cum_logprob: Optional[float] = None


@dataclass
class OracleFixture:
    """A complete oracle case: request shape plus a fixed response stream."""

    name: str
    endpoint: str  # "chat" | "completions"
    streaming: bool
    request_kwargs: dict
    steps: list[StepSpec]
    model: str = "oracle-model"

    def __post_init__(self):
        self.request_kwargs.setdefault("model", self.model)
        self.request_kwargs.setdefault("stream", self.streaming)


def _finish_reason(name: str):
    from tensorrt_llm.bindings import executor as tllm

    return getattr(tllm.FinishReason, name)


class _FakeResult:
    def __init__(self, spec: StepSpec) -> None:
        self.output_token_ids = [list(spec.token_ids)]
        self.finish_reasons = [_finish_reason(spec.finish)]
        self.sequence_index = spec.sequence_index
        self.is_final = spec.is_final
        self.cum_log_probs = [spec.cum_logprob] if spec.cum_logprob is not None else None
        self.log_probs = [list(spec.logprobs)] if spec.logprobs is not None else None
        self.context_phase_params = None
        self.decoding_iter = 1
        self.cached_tokens = 0
        self.avg_decoded_tokens_per_iter = None
        self.generation_logits = None
        self.context_logits = None
        self.request_perf_metrics = None


class _FakeResponse:
    def __init__(self, client_id: int, spec: StepSpec) -> None:
        self.client_id = client_id
        self.result = _FakeResult(spec)

    def has_error(self) -> bool:
        return False


class _OracleExecutor(GenerationExecutor):
    """Auto-responding executor: every submission receives the fixture stream."""

    def __init__(self, steps: list[StepSpec]) -> None:
        super().__init__(num_postprocess_workers=0)
        self._steps = steps
        self.aborted_ids: list[int] = []

    def submit(self, request: GenerationRequest) -> GenerationResult:
        request.set_id(self._get_next_client_id())
        result = GenerationResult(request, executor=self)
        for spec in self._steps:
            result.queue.put(_FakeResponse(request.id, spec))
        return result

    def abort_request(self, request_id: int) -> None:
        self.aborted_ids.append(request_id)

    def shutdown(self) -> None:
        pass


# --- request construction ----------------------------------------------------


def build_request_model(fixture: OracleFixture):
    if fixture.endpoint == "chat":
        return ChatCompletionRequest(**fixture.request_kwargs)
    return CompletionRequest(**fixture.request_kwargs)


def _chat_prompt_and_conversation(request, tokenizer):
    conversation, _mm, mm_placeholder_counts = parse_chat_messages_coroutines(
        request.messages, OracleModelConfig(), None
    )
    tool_dicts = None if request.tools is None else [t.model_dump() for t in request.tools]
    prompt = apply_chat_template(
        model_type="llama",
        tokenizer=tokenizer,
        processor=None,
        conversation=conversation,
        add_generation_prompt=request.add_generation_prompt,
        mm_placeholder_counts=mm_placeholder_counts,
        tools=tool_dicts,
        documents=request.documents,
        chat_template=request.chat_template,
        chat_template_kwargs=request.chat_template_kwargs or {},
    )
    return prompt, conversation


# --- old path -----------------------------------------------------------------


def run_old_path_openai(fixture: OracleFixture):
    """Drive the historical result machinery + serve-side postprocessors."""
    tokenizer = OracleTokenizer()
    request = build_request_model(fixture)
    if fixture.endpoint == "chat":
        sampling_params = request.to_sampling_params(backend="pytorch")
        prompt, conversation = _chat_prompt_and_conversation(request, tokenizer)
        args = ChatPostprocArgs.from_request(request)
        role = "assistant" if request.add_generation_prompt else request.messages[-1]["role"]
        if (
            conversation
            and conversation[-1].get("content")
            and conversation[-1].get("role") == role
        ):
            args.last_message_content = conversation[-1]["content"]
        stream_post_processor = chat_stream_post_processor
        response_post_processor = chat_response_post_processor
    else:
        sampling_params = request.to_sampling_params(backend="pytorch")
        prompt = request.prompt
        args = CompletionPostprocArgs.from_request(request)
        args.prompt = prompt
        stream_post_processor = completion_stream_post_processor
        response_post_processor = completion_response_post_processor

    if sampling_params.end_id is None:
        sampling_params._setup(tokenizer, None, None)
    sampling_params._stream_interval = 1
    prompt_token_ids = tokenizer.encode(
        prompt, add_special_tokens=sampling_params.add_special_tokens
    )

    generation_request = GenerationRequest(
        list(prompt_token_ids), sampling_params=sampling_params, streaming=fixture.streaming
    )
    executor = _OracleExecutor(fixture.steps)
    result = executor.submit(generation_request)
    request_output = RequestOutput._from_generation_result(result, prompt, tokenizer)

    args.tokenizer = tokenizer
    args.num_prompt_tokens = len(prompt_token_ids)
    args.stream_response_id = PINNED_STREAM_ID
    args.stream_created = PINNED_CREATED

    if fixture.streaming:
        chunks: list[str] = []
        while not request_output.finished:
            next(request_output)
            chunks.extend(stream_post_processor(request_output, args))
        return chunks
    request_output.result()
    return response_post_processor(request_output, args)


def run_old_path_llm_api(fixture: OracleFixture):
    """Historical LLM API path: per-step output snapshots plus the final state."""
    tokenizer = OracleTokenizer()
    sampling_params = _llm_api_sampling_params(fixture)
    if sampling_params.end_id is None:
        sampling_params._setup(tokenizer, None, None)
    sampling_params._stream_interval = 1
    prompt = fixture.request_kwargs["prompt"]
    prompt_token_ids = tokenizer.encode(
        prompt, add_special_tokens=sampling_params.add_special_tokens
    )
    generation_request = GenerationRequest(
        list(prompt_token_ids), sampling_params=sampling_params, streaming=fixture.streaming
    )
    executor = _OracleExecutor(fixture.steps)
    result = executor.submit(generation_request)
    request_output = RequestOutput._from_generation_result(result, prompt, tokenizer)
    return _collect_llm_api_snapshots(request_output, fixture.streaming)


# --- new path -----------------------------------------------------------------


def _new_pipeline_components(fixture: OracleFixture, client_factory: Optional[Callable] = None):
    tokenizer = OracleTokenizer()
    executor = _OracleExecutor(fixture.steps)
    if client_factory is None:
        client = LegacyEngineClientAdapter(executor)
    else:
        client = client_factory(executor)
    processor = FrontendProcessor(
        tokenizer, model_config=OracleModelConfig(), default_stream_interval=1
    )
    return tokenizer, client, processor


async def run_new_path_openai(fixture: OracleFixture, client_factory: Optional[Callable] = None):
    """Drive the engine-client pipeline end to end (production code path)."""
    _tokenizer, client, processor = _new_pipeline_components(fixture, client_factory)
    try:
        pipeline = OpenAIServingPipeline(
            client,
            processor,
            model_label=fixture.model,
            mode=PipelineDeploymentMode.COLOCATED,
        )
        request = build_request_model(fixture)
        if fixture.endpoint == "chat":
            response = await pipeline.try_chat(request)
        else:
            response = await pipeline.try_completion(request)
        assert response is not None, f"fixture {fixture.name} unexpectedly ineligible"
        if fixture.streaming:
            chunks = [chunk async for chunk in response.body_iterator]
            assert chunks and chunks[-1] == "data: [DONE]\n\n"
            return chunks[:-1]
        return response
    finally:
        client.shutdown()


def run_new_path_llm_api(fixture: OracleFixture, client_factory: Optional[Callable] = None):
    _tokenizer, client, processor = _new_pipeline_components(fixture, client_factory)
    try:
        pipeline = LlmApiEnginePipeline(client, processor)
        sampling_params = _llm_api_sampling_params(fixture)
        request_output = pipeline.try_generate_async(
            fixture.request_kwargs["prompt"], sampling_params, streaming=fixture.streaming
        )
        assert request_output is not None, f"fixture {fixture.name} unexpectedly ineligible"
        return _collect_llm_api_snapshots(request_output, fixture.streaming)
    finally:
        client.shutdown()


def _llm_api_sampling_params(fixture: OracleFixture) -> SamplingParams:
    kwargs = {
        k: v for k, v in fixture.request_kwargs.items() if k not in ("prompt", "model", "stream")
    }
    return SamplingParams(**kwargs)


# --- normalization + comparison ---------------------------------------------


def _normalized_sse(chunks: list[str]) -> list[dict]:
    normalized = []
    for chunk in chunks:
        assert chunk.startswith("data: "), f"malformed SSE chunk: {chunk!r}"
        payload = json.loads(chunk[len("data: ") :].strip())
        payload["id"] = "ID"
        payload["created"] = 0
        normalized.append(payload)
    return normalized


def _normalized_body(response) -> dict:
    payload = response.model_dump()
    payload["id"] = "ID"
    payload["created"] = 0
    return payload


def assert_openai_equal(old_output, new_output, streaming: bool) -> None:
    """Exact equality of SSE chunk sequences or non-streaming bodies."""
    if streaming:
        assert _normalized_sse(old_output) == _normalized_sse(new_output)
    else:
        assert _normalized_body(old_output) == _normalized_body(new_output)


def _plain_logprob_entries(entries) -> Optional[list]:
    if entries is None:
        return None
    plain = []
    for entry in entries:
        if isinstance(entry, dict):
            plain.append(
                {int(t): round(float(getattr(lp, "logprob", lp)), 9) for t, lp in entry.items()}
            )
        else:
            plain.append(round(float(entry), 9))
    return plain


def _output_snapshot(output) -> dict:
    return {
        "index": output.index,
        "text": output.text,
        "token_ids": list(output.token_ids),
        "text_diff": output.text_diff,
        "token_ids_diff": list(output.token_ids_diff),
        "logprobs": _plain_logprob_entries(output.logprobs),
        "cumulative_logprob": output.cumulative_logprob,
        "finish_reason": output.finish_reason,
        "stop_reason": output.stop_reason,
        "length": output.length,
    }


def _collect_llm_api_snapshots(request_output, streaming: bool) -> list[list[dict]]:
    snapshots = []
    if streaming:
        for _step in request_output:
            snapshots.append([_output_snapshot(o) for o in _step.outputs])
    else:
        request_output.result()
        snapshots.append([_output_snapshot(o) for o in request_output.outputs])
    return snapshots


def assert_llm_api_equal(old_snapshots, new_snapshots) -> None:
    assert old_snapshots == new_snapshots


# --- fixture corpus -----------------------------------------------------------

CHAT_MESSAGES = [{"role": "user", "content": "say hello"}]

OPENAI_FIXTURES: list[OracleFixture] = [
    OracleFixture(
        name="chat_stream_basic_end_token",
        endpoint="chat",
        streaming=True,
        request_kwargs={"messages": CHAT_MESSAGES, "max_tokens": 8},
        steps=[
            StepSpec([101]),
            StepSpec([102]),
            StepSpec([103], finish="END_ID", is_final=True),
        ],
    ),
    OracleFixture(
        name="chat_nonstream_basic",
        endpoint="chat",
        streaming=False,
        request_kwargs={"messages": CHAT_MESSAGES, "max_tokens": 8},
        steps=[StepSpec([101, 102, 103], finish="END_ID", is_final=True)],
    ),
    OracleFixture(
        name="chat_stream_stop_string_trimmed",
        endpoint="chat",
        streaming=True,
        request_kwargs={"messages": CHAT_MESSAGES, "max_tokens": 8, "stop": ["<STOP>"]},
        steps=[
            StepSpec([101]),
            StepSpec([102, 107], finish="STOP_WORDS", is_final=True),
        ],
    ),
    OracleFixture(
        name="chat_stream_stop_string_included",
        endpoint="chat",
        streaming=True,
        request_kwargs={
            "messages": CHAT_MESSAGES,
            "max_tokens": 8,
            "stop": ["<STOP>"],
            "include_stop_str_in_output": True,
        },
        steps=[
            StepSpec([101]),
            StepSpec([102, 107], finish="STOP_WORDS", is_final=True),
        ],
    ),
    OracleFixture(
        name="chat_stream_cross_token_stop_string",
        endpoint="chat",
        streaming=True,
        request_kwargs={"messages": CHAT_MESSAGES, "max_tokens": 8, "stop": ["<STOP>"]},
        steps=[
            StepSpec([101]),
            StepSpec([110, 111]),
            StepSpec([102], finish="LENGTH", is_final=True),
        ],
    ),
    OracleFixture(
        name="chat_stream_stop_token_ids",
        endpoint="chat",
        streaming=True,
        request_kwargs={"messages": CHAT_MESSAGES, "max_tokens": 8, "stop_token_ids": [105]},
        steps=[
            StepSpec([101]),
            StepSpec([102, 105], finish="STOP_WORDS", is_final=True),
        ],
    ),
    OracleFixture(
        name="chat_stream_n2",
        endpoint="chat",
        streaming=True,
        request_kwargs={"messages": CHAT_MESSAGES, "max_tokens": 8, "n": 2, "temperature": 0.8},
        steps=[
            StepSpec([101], sequence_index=0),
            StepSpec([105], sequence_index=1),
            StepSpec([102], sequence_index=0, finish="LENGTH"),
            StepSpec([106], sequence_index=1, finish="LENGTH", is_final=True),
        ],
    ),
    OracleFixture(
        name="chat_nonstream_n2",
        endpoint="chat",
        streaming=False,
        request_kwargs={"messages": CHAT_MESSAGES, "max_tokens": 8, "n": 2, "temperature": 0.8},
        steps=[
            StepSpec([101, 102], sequence_index=0, finish="LENGTH", cum_logprob=-1.0),
            StepSpec(
                [105, 106], sequence_index=1, finish="LENGTH", is_final=True, cum_logprob=-2.0
            ),
        ],
    ),
    OracleFixture(
        name="chat_stream_logprobs",
        endpoint="chat",
        streaming=True,
        request_kwargs={
            "messages": CHAT_MESSAGES,
            "max_tokens": 8,
            "logprobs": True,
            "top_logprobs": 1,
        },
        steps=[
            StepSpec([101], logprobs=[{101: Logprob(-0.25, rank=1)}]),
            StepSpec(
                [102],
                logprobs=[{102: Logprob(-0.5, rank=1)}],
                finish="LENGTH",
                is_final=True,
            ),
        ],
    ),
    OracleFixture(
        name="chat_stream_usage_continuous",
        endpoint="chat",
        streaming=True,
        request_kwargs={
            "messages": CHAT_MESSAGES,
            "max_tokens": 8,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        },
        steps=[
            StepSpec([101]),
            StepSpec([102], finish="END_ID", is_final=True),
        ],
    ),
    OracleFixture(
        name="completion_stream_basic",
        endpoint="completions",
        streaming=True,
        request_kwargs={"prompt": "hello there", "max_tokens": 8},
        steps=[
            StepSpec([101]),
            StepSpec([102], finish="END_ID", is_final=True),
        ],
    ),
    OracleFixture(
        name="completion_stream_stop_string",
        endpoint="completions",
        streaming=True,
        request_kwargs={"prompt": "hello there", "max_tokens": 8, "stop": ["<STOP>"]},
        steps=[
            StepSpec([101]),
            StepSpec([107], finish="STOP_WORDS", is_final=True),
        ],
    ),
    OracleFixture(
        name="completion_nonstream_echo",
        endpoint="completions",
        streaming=False,
        request_kwargs={"prompt": "hello there", "max_tokens": 8, "echo": True},
        steps=[StepSpec([101, 102], finish="LENGTH", is_final=True)],
    ),
    OracleFixture(
        name="completion_stream_usage",
        endpoint="completions",
        streaming=True,
        request_kwargs={
            "prompt": "hello there",
            "max_tokens": 8,
            "stream_options": {"include_usage": True},
        },
        steps=[
            StepSpec([101]),
            StepSpec([102], finish="END_ID", is_final=True),
        ],
    ),
    OracleFixture(
        name="completion_stream_logprobs",
        endpoint="completions",
        streaming=True,
        request_kwargs={"prompt": "hello there", "max_tokens": 8, "logprobs": 1},
        steps=[
            StepSpec([101], logprobs=[{101: Logprob(-0.25, rank=1)}]),
            StepSpec(
                [102], logprobs=[{102: Logprob(-0.5, rank=1)}], finish="LENGTH", is_final=True
            ),
        ],
    ),
]

LLM_API_FIXTURES: list[OracleFixture] = [
    OracleFixture(
        name="llm_api_stream_basic",
        endpoint="llm_api",
        streaming=True,
        request_kwargs={"prompt": "hello there", "max_tokens": 8},
        steps=[
            StepSpec([101]),
            StepSpec([102], finish="END_ID", is_final=True),
        ],
    ),
    OracleFixture(
        name="llm_api_stream_stop_string",
        endpoint="llm_api",
        streaming=True,
        request_kwargs={"prompt": "hello there", "max_tokens": 8, "stop": ["<STOP>"]},
        steps=[
            StepSpec([101]),
            StepSpec([102, 107], finish="STOP_WORDS", is_final=True),
        ],
    ),
    OracleFixture(
        name="llm_api_nonstream_stop_token_ids",
        endpoint="llm_api",
        streaming=False,
        request_kwargs={"prompt": "hello there", "max_tokens": 8, "stop_token_ids": [105]},
        steps=[StepSpec([101, 105], finish="STOP_WORDS", is_final=True)],
    ),
    OracleFixture(
        name="llm_api_stream_logprobs",
        endpoint="llm_api",
        streaming=True,
        request_kwargs={"prompt": "hello there", "max_tokens": 8, "logprobs": 1},
        steps=[
            StepSpec([101], logprobs=[{101: Logprob(-0.25, rank=1)}], cum_logprob=-0.25),
            StepSpec(
                [102],
                logprobs=[{102: Logprob(-0.5, rank=1)}],
                cum_logprob=-0.75,
                finish="LENGTH",
                is_final=True,
            ),
        ],
    ),
    OracleFixture(
        name="llm_api_nonstream_basic",
        endpoint="llm_api",
        streaming=False,
        request_kwargs={"prompt": "hello there", "max_tokens": 8},
        steps=[StepSpec([101, 102, 103], finish="END_ID", is_final=True)],
    ),
]
