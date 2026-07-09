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
"""Frontend-internal OpenAI chat/completions formatters for the assembled path.

These functions own the OpenAI response and SSE shaping for requests served
through the engine-boundary pipeline. They are selected per endpoint by the
frontend and are never attached to requests — no formatter callable crosses
to the engine. Their output must match the historical serve-side
postprocessing byte for byte; the golden-output conformance fixtures gate
that equivalence.

Formatter-parameter objects carry the per-request stateful pieces (reasoning
and tool parser instances, first-iteration flag, stream metadata) whose
lifetime spans all streaming chunks of one request.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Union

from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory, ReasoningParserResult
from tensorrt_llm.serve.tool_call_id import make_tool_call_id

if os.environ.get("TLLM_LIGHTWEIGHT_IMPORT", "0") == "1":
    # Detached frontends reject disaggregated-serving requests with typed
    # capability errors, so ctx_usage is always None here and the rewrite
    # helpers reduce to identities; the real implementations live in the
    # engine-side disagg utilities, which pull the runtime import graph.
    def rewrite_usage_info_from_ctx(usage, ctx_usage):
        return usage

    def rewrite_usage_response_from_ctx(response, ctx_usage):
        return response
else:
    from tensorrt_llm.llmapi.disagg_utils import (
        rewrite_usage_info_from_ctx,
        rewrite_usage_response_from_ctx,
    )
from tensorrt_llm.serve.frontend.response_assembler import AssembledRequestView
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionToolsParam,
    ChatMessage,
    CompletionLogProbs,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    PromptTokensDetails,
    StreamOptions,
    ToolCall,
    UsageInfo,
    to_disaggregated_params,
)
from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory

__all__ = [
    "ChatFormatterParams",
    "CompletionFormatterParams",
    "format_chat_response",
    "format_chat_stream_chunks",
    "format_completion_response",
    "format_completion_stream_chunks",
]


@dataclass(kw_only=True)
class ChatFormatterParams:
    """Per-request chat formatting parameters and stateful parser instances."""

    role: str
    model: str
    first_iteration: bool = True
    num_prompt_tokens: Optional[int] = None
    tokenizer: Any = None
    echo: bool = False
    num_choices: int = 1
    tools: Optional[List[ChatCompletionToolsParam]] = None
    tool_choice: Optional[Union[Literal["none"], ChatCompletionNamedToolChoiceParam]] = "none"
    return_logprobs: bool = False
    top_logprobs: bool = False
    stream_options: Optional[StreamOptions] = None
    last_message_content: Optional[str] = None
    reasoning_parser: Optional[str] = None
    tool_parser: Optional[str] = None
    reasoning_parser_dict: dict[int, Any] = field(default_factory=dict)
    tool_parser_dict: dict[int, Any] = field(default_factory=dict)
    has_tool_call: dict[int, bool] = field(default_factory=dict)
    tool_call_id_type: str = "random"
    chat_template_kwargs: Optional[dict[str, Any]] = None
    ctx_usage: Optional[UsageInfo] = None
    stream_response_id: Optional[str] = None
    stream_created: Optional[int] = None

    @classmethod
    def from_request(
        cls,
        request: ChatCompletionRequest,
        *,
        model: Optional[str] = None,
    ) -> "ChatFormatterParams":
        return cls(
            echo=request.echo,
            role="assistant" if request.add_generation_prompt else request.messages[-1]["role"],
            model=model or request.model,
            num_choices=request.n if request.n else 1,
            tools=request.tools,
            tool_choice=request.tool_choice,
            stream_options=request.stream_options,
            return_logprobs=bool(request.logprobs),
            top_logprobs=bool(request.top_logprobs),
            chat_template_kwargs=request.chat_template_kwargs,
            ctx_usage=None
            if request.disaggregated_params is None
            else request.disaggregated_params.ctx_usage,
        )


@dataclass(kw_only=True)
class CompletionFormatterParams:
    """Per-request completion formatting parameters."""

    model: str
    first_iteration: bool = True
    num_prompt_tokens: Optional[int] = None
    tokenizer: Any = None
    echo: bool = False
    num_choices: int = 1
    prompt_idx: int = 0
    detokenize: bool = True
    prompt: Optional[str] = None
    return_logprobs: bool = False
    stream_options: Optional[StreamOptions] = None
    ctx_usage: Optional[UsageInfo] = None
    stream_response_id: Optional[str] = None
    stream_created: Optional[int] = None

    @classmethod
    def from_request(
        cls,
        request: CompletionRequest,
        *,
        model: Optional[str] = None,
    ) -> "CompletionFormatterParams":
        return cls(
            echo=request.echo,
            model=model or request.model,
            num_choices=request.n if request.n else 1,
            stream_options=request.stream_options,
            detokenize=request.detokenize,
            return_logprobs=bool(request.logprobs),
            ctx_usage=None
            if request.disaggregated_params is None
            else request.disaggregated_params.ctx_usage,
        )


def _ctx_usage_for_view(params: Any, view: AssembledRequestView) -> Optional[UsageInfo]:
    ctx_usage = params.ctx_usage
    if ctx_usage is not None:
        return ctx_usage
    for output in view.outputs:
        disaggregated_params = getattr(output, "disaggregated_params", None)
        if disaggregated_params is None:
            continue
        candidate = disaggregated_params.ctx_usage
        if candidate is None:
            continue
        if isinstance(candidate, UsageInfo):
            return candidate
        return UsageInfo.model_validate(candidate)
    return None


def _ensure_stream_metadata(
    params: Any, view: AssembledRequestView, prefix: str
) -> Tuple[str, int]:
    if params.stream_response_id is None:
        params.stream_response_id = f"{prefix}-{view.id}"
    if params.stream_created is None:
        params.stream_created = int(time.time())
    return params.stream_response_id, params.stream_created


def create_chat_logprobs(
    token_ids: List[int],
    tokenizer: Any,
    logprobs: list,
    top_logprobs: bool,
) -> ChatCompletionLogProbs:
    assert len(token_ids) == len(logprobs), "token_ids and logprobs have different lengths"
    content: List[ChatCompletionLogProbsContent] = []
    for token_id, logprob in zip(token_ids, logprobs):
        token = tokenizer.decode(token_id)
        chat_logprob = ChatCompletionLogProbsContent(
            token=token,
            bytes=list(token.encode("utf-8", errors="replace")),
        )
        if isinstance(logprob, dict):
            if token_id in logprob:
                chat_logprob.logprob = max(logprob[token_id].logprob, -9999.0)
                if top_logprobs:
                    chat_logprob.top_logprobs = [
                        ChatCompletionLogProbsContent(
                            token=(tk := tokenizer.decode(tid)),
                            logprob=max(lp.logprob, -9999.0),
                            bytes=list(tk.encode("utf-8", errors="replace")),
                        )
                        for tid, lp in logprob.items()
                    ]
        else:
            chat_logprob.logprob = max(logprob, -9999.0)
        content.append(chat_logprob)
    return ChatCompletionLogProbs(content=content)


def create_completion_logprobs(
    token_ids: List[int],
    tokenizer: Any,
    logprobs: list,
    initial_offset: int = 0,
) -> CompletionLogProbs:
    assert len(token_ids) == len(logprobs), "token_ids and logprobs have different lengths"
    text_offset: List[int] = []
    token_logprobs: List[float] = []
    top_logprobs_list: List[dict] = []
    tokens: List[str] = []
    for token_id, logprob in zip(token_ids, logprobs):
        if isinstance(logprob, dict):
            token_logprobs.append(max(logprob[token_id].logprob, -9999.0))
            top_logprobs_list.append(
                {tokenizer.decode(tid): max(lp.logprob, -9999.0) for tid, lp in logprob.items()}
            )
        else:
            token_logprobs.append(max(logprob, -9999.0))

        token = tokenizer.decode(token_id)
        if len(text_offset) == 0:
            text_offset.append(initial_offset)
        else:
            text_offset.append(text_offset[-1] + len(token))
        tokens.append(token)
    return CompletionLogProbs(
        text_offset=text_offset,
        token_logprobs=token_logprobs,
        tokens=tokens,
        top_logprobs=top_logprobs_list,
    )


def _apply_reasoning_parser(
    params: ChatFormatterParams,
    output_index: int,
    text: str,
    streaming: bool,
    finished: bool = False,
) -> Tuple[str, str]:
    reasoning_parser = None
    if params.reasoning_parser is not None:
        if output_index not in params.reasoning_parser_dict:
            params.reasoning_parser_dict[output_index] = (
                ReasoningParserFactory.create_reasoning_parser(
                    params.reasoning_parser, params.chat_template_kwargs
                )
            )
        reasoning_parser = params.reasoning_parser_dict[output_index]

    if reasoning_parser is not None:
        if not streaming:
            result = reasoning_parser.parse(text)
        else:
            result = reasoning_parser.parse_delta(text)
            if finished:
                finish_result = reasoning_parser.finish()
                result = ReasoningParserResult(
                    content=result.content + finish_result.content,
                    reasoning_content=result.reasoning_content + finish_result.reasoning_content,
                )
        content, reasoning_content = result.content, result.reasoning_content
    else:
        content, reasoning_content = text, ""

    return content, reasoning_content


def _apply_tool_parser(
    params: ChatFormatterParams,
    output_index: int,
    text: str,
    streaming: bool,
) -> Tuple[str, list]:
    tool_parser = None
    tools = params.tools
    if params.tool_parser is not None and tools is not None:
        if output_index not in params.tool_parser_dict:
            params.tool_parser_dict[output_index] = ToolParserFactory.create_tool_parser(
                params.tool_parser
            )
        tool_parser = params.tool_parser_dict[output_index]

    if tool_parser is not None and tools is not None:
        if not streaming:
            result = tool_parser.detect_and_parse(text, tools)
        else:
            result = tool_parser.parse_streaming_increment(text, tools)
        normal_text, calls = result.normal_text, result.calls
        if result.calls:
            params.has_tool_call[output_index] = True
    else:
        normal_text, calls = text, []

    return normal_text, calls


def format_chat_stream_chunks(view: AssembledRequestView, params: ChatFormatterParams) -> List[str]:
    """SSE chunks for one assembly step of a streaming chat request."""

    def yield_first_chat(
        num_tokens: int, idx: int, role: str | None = None, content: str | None = None
    ):
        choice_data = ChatCompletionResponseStreamChoice(
            index=idx, delta=DeltaMessage(role=role, content=content), finish_reason=None
        )
        chunk = ChatCompletionStreamResponse(
            choices=[choice_data],
            model=params.model,
            id=stream_response_id,
            created=stream_created,
        )
        if include_continuous_usage:
            chunk.usage = UsageInfo(
                prompt_tokens=num_tokens,
                total_tokens=num_tokens,
                completion_tokens=0,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=view.cached_tokens),
            )
            rewrite_usage_info_from_ctx(chunk.usage, ctx_usage)
        data = chunk.model_dump_json(exclude_none=True)
        return data

    res: List[str] = []
    # DEC-4 bug-for-bug parity: `finish_reason_sent` is intentionally a
    # per-call local, exactly like the historical postprocessor
    # (serve/postprocess_handlers.py chat_stream_post_processor,
    # `finish_reason_sent = [False] * args.num_choices`). For interleaved
    # n>1 streaming the runtime sends one sequence per response
    # (executor/result.py _handle_response -> _handle_sequence updates only
    # that sequence's diff cursor), so an already-finished choice re-emits
    # its terminal chunk and unchanged choices re-emit their last delta.
    # The oracle pins this (TestStreamingMultiSequenceParity, chat_stream_n2).
    # Persisting this state across calls would diverge flag-on from the
    # byte-identical flag-off output; the de-duplication is a lockstep fix to
    # BOTH this formatter and the historical postprocessor in a follow-up loop.
    finish_reason_sent = [False] * params.num_choices
    prompt_tokens = params.num_prompt_tokens
    ctx_usage = _ctx_usage_for_view(params, view)
    stream_response_id, stream_created = _ensure_stream_metadata(params, view, "chatcmpl")
    if stream_option := params.stream_options:
        include_usage = stream_option.include_usage
        include_continuous_usage = include_usage and stream_option.continuous_usage_stats
    else:
        include_usage = False
        include_continuous_usage = False
    if params.first_iteration:
        for i in range(params.num_choices):
            res.append(f"data: {yield_first_chat(prompt_tokens, i, role=params.role)} \n\n")
            if params.echo and params.last_message_content:
                res.append(
                    f"data: {yield_first_chat(prompt_tokens, i, content=params.last_message_content)} \n\n"
                )
        params.first_iteration = False

    for output in view.outputs:
        i = output.index

        if finish_reason_sent[i]:
            continue

        has_token_delta = bool(output.token_ids_diff)
        delta_text = output.text_diff
        delta_text, reasoning_delta_text = _apply_reasoning_parser(
            params, i, delta_text, True, finished=(output.finish_reason is not None)
        )

        if params.tool_choice and type(params.tool_choice) is ChatCompletionNamedToolChoiceParam:
            delta_message = DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        function=DeltaFunctionCall(
                            name=params.tool_choice.function.name, arguments=delta_text
                        ),
                        index=i,
                    ),
                ],
            )
        else:
            delta_text, calls = _apply_tool_parser(params, i, delta_text, True)
            tool_calls = []
            for call_item in calls:
                # Tool call ID should be generated only once per tool call
                if call_item.name:
                    # First chunk: include ID and function name
                    tool_call_id = make_tool_call_id(
                        id_type=params.tool_call_id_type,
                        func_name=call_item.name,
                        idx=call_item.tool_index,
                    )
                    function_name = call_item.name
                else:
                    # Subsequent chunks: null ID and name for argument deltas
                    tool_call_id = None
                    function_name = None

                tool_calls.append(
                    DeltaToolCall(
                        id=tool_call_id,
                        index=call_item.tool_index,
                        function=DeltaFunctionCall(
                            name=function_name,
                            arguments=call_item.parameters,
                        ),
                    )
                )
            # Keep token-bearing chunks visible even when detokenization has no
            # text to flush yet.
            if (
                tool_calls
                or delta_text
                or reasoning_delta_text
                or output.finish_reason
                or has_token_delta
            ):
                delta_message = DeltaMessage(
                    content=delta_text,
                    reasoning_content=reasoning_delta_text,
                    tool_calls=tool_calls if tool_calls else None,
                )
            else:
                continue

        choice = ChatCompletionResponseStreamChoice(
            index=i,
            delta=delta_message,
            avg_decoded_tokens_per_iter=view.avg_decoded_tokens_per_iter,
            stop_reason=output.stop_reason,
        )
        if params.return_logprobs:
            logprobs = output.logprobs_diff
            token_ids = output.token_ids_diff
            choice.logprobs = create_chat_logprobs(
                token_ids, params.tokenizer, logprobs, params.top_logprobs
            )
        if output.finish_reason is not None:
            if output.finish_reason == "stop" and params.has_tool_call.get(i, False):
                choice.finish_reason = "tool_calls"
            else:
                choice.finish_reason = output.finish_reason
            choice.stop_reason = output.stop_reason
            finish_reason_sent[i] = True
        chunk = ChatCompletionStreamResponse(
            choices=[choice],
            model=params.model,
            id=stream_response_id,
            created=stream_created,
        )
        if include_continuous_usage:
            chunk.usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=output.length,
                total_tokens=output.length + prompt_tokens,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=view.cached_tokens),
            )
            rewrite_usage_info_from_ctx(chunk.usage, ctx_usage)
        data = chunk.model_dump_json(exclude_none=True)
        res.append(f"data: {data}\n\n")

    if include_usage and view._done:
        completion_tokens = sum(output.length for output in view.outputs)
        final_usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=view.cached_tokens),
        )
        rewrite_usage_info_from_ctx(final_usage, ctx_usage)

        final_usage_chunk = ChatCompletionStreamResponse(
            choices=[],
            model=params.model,
            usage=final_usage,
            id=stream_response_id,
            created=stream_created,
        )
        final_usage_data = final_usage_chunk.model_dump_json()
        res.append(f"data: {final_usage_data}\n\n")
    return res


def format_chat_response(
    view: AssembledRequestView, params: ChatFormatterParams
) -> ChatCompletionResponse:
    """Non-streaming chat completion response for a finished assembly."""
    choices: List[ChatCompletionResponseChoice] = []
    role = params.role
    for output in view.outputs:
        text, reasoning_text = _apply_reasoning_parser(params, output.index, output.text, False)

        if params.tool_choice and isinstance(
            params.tool_choice, ChatCompletionNamedToolChoiceParam
        ):
            message = ChatMessage(
                role=role,
                content="",
                tool_calls=[
                    ToolCall(
                        function=FunctionCall(name=params.tool_choice.function.name, arguments=text)
                    )
                ],
            )
        else:
            if text is None:
                text = ""
            text, calls = _apply_tool_parser(params, output.index, text, False)
            tool_calls = [
                ToolCall(function=FunctionCall(name=call.name or "", arguments=call.parameters))
                for call in calls
            ]
            message = ChatMessage(
                role=role, content=text, reasoning_content=reasoning_text, tool_calls=tool_calls
            )
        disaggregated_params = to_disaggregated_params(output.disaggregated_params)
        choice = ChatCompletionResponseChoice(
            index=output.index,
            message=message,
            stop_reason=output.stop_reason,
            disaggregated_params=disaggregated_params,
            avg_decoded_tokens_per_iter=view.avg_decoded_tokens_per_iter,
        )
        if output.finish_reason == "stop" and params.has_tool_call.get(output.index, False):
            choice.finish_reason = "tool_calls"
        else:
            choice.finish_reason = output.finish_reason

        if params.return_logprobs:
            choice.logprobs = create_chat_logprobs(
                output.token_ids, params.tokenizer, output.logprobs, params.top_logprobs
            )
        choices.append(choice)

    if params.echo and params.last_message_content:
        for choice in choices:
            full_message = params.last_message_content + choice.message.content
            choice.message.content = full_message

    num_prompt_tokens = params.num_prompt_tokens
    num_generated_tokens = sum(len(output.token_ids) for output in view.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=view.cached_tokens),
    )
    ctx_usage = _ctx_usage_for_view(params, view)
    response = ChatCompletionResponse(
        model=params.model,
        choices=choices,
        usage=usage,
    )
    rewrite_usage_response_from_ctx(response, ctx_usage)
    return response


def format_completion_stream_chunks(
    view: AssembledRequestView, params: CompletionFormatterParams
) -> List[str]:
    """SSE chunks for one assembly step of a streaming completion request."""
    res: List[str] = []
    prompt_tokens = params.num_prompt_tokens
    ctx_usage = _ctx_usage_for_view(params, view)
    stream_response_id, stream_created = _ensure_stream_metadata(params, view, "cmpl")
    if stream_option := params.stream_options:
        include_usage = stream_option.include_usage
        include_continuous_usage = include_usage and stream_option.continuous_usage_stats
    else:
        include_usage = False
        include_continuous_usage = False

    # DEC-4 bug-for-bug parity: like the historical
    # completion_stream_post_processor, this iterates every choice on every
    # engine event and emits its current diff/terminal without persisting
    # per-choice sent-state. For interleaved n>1 streaming (one sequence per
    # response, executor/result.py:_handle_sequence) unchanged choices
    # re-emit their last delta and finished choices re-emit their terminal.
    # completion_stream_n2 pins this exact shape; skipping unchanged choices
    # would diverge from the byte-identical flag-off output. De-duplication is
    # a lockstep fix to both formatters and the historical postprocessors in a
    # follow-up loop.
    for output in view.outputs:
        delta_text = output.text_diff
        if params.echo and params.first_iteration:
            delta_text = params.prompt + delta_text
        choice = CompletionResponseStreamChoice(
            index=params.prompt_idx * params.num_choices + output.index,
            text=delta_text if params.detokenize else "",
            token_ids=None if params.detokenize else output.token_ids_diff,
            finish_reason=output.finish_reason,
            stop_reason=output.stop_reason,
            avg_decoded_tokens_per_iter=view.avg_decoded_tokens_per_iter,
        )
        if params.return_logprobs:
            logprobs = output.logprobs_diff
            token_ids = output.token_ids_diff
            choice.logprobs = create_completion_logprobs(
                token_ids, params.tokenizer, logprobs, output._last_text_len
            )

        chunk = CompletionStreamResponse(
            model=params.model,
            choices=[choice],
            id=stream_response_id,
            created=stream_created,
        )
        if include_continuous_usage:
            chunk.usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=output.length,
                total_tokens=output.length + prompt_tokens,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=view.cached_tokens),
            )
            rewrite_usage_info_from_ctx(chunk.usage, ctx_usage)
        data = chunk.model_dump_json(exclude_unset=False)
        res.append(f"data: {data}\n\n")

    if include_usage and view._done:
        completion_tokens = sum(output.length for output in view.outputs)
        final_usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=view.cached_tokens),
        )
        rewrite_usage_info_from_ctx(final_usage, ctx_usage)

        final_usage_chunk = CompletionStreamResponse(
            choices=[],
            model=params.model,
            usage=final_usage,
            id=stream_response_id,
            created=stream_created,
        )
        final_usage_data = final_usage_chunk.model_dump_json()
        res.append(f"data: {final_usage_data}\n\n")
    params.first_iteration = False
    return res


def format_completion_response(
    view: AssembledRequestView, params: CompletionFormatterParams
) -> CompletionResponse:
    """Non-streaming completion response for a finished assembly."""
    prompt_tokens = params.num_prompt_tokens
    completion_tokens = 0
    choices = []
    for output in view.outputs:
        text = output.text
        if params.echo:
            text = params.prompt + text
        disaggregated_params = to_disaggregated_params(output.disaggregated_params)
        choice = CompletionResponseChoice(
            text=text if params.detokenize else "",
            token_ids=None if params.detokenize else output.token_ids,
            index=params.prompt_idx * params.num_choices + output.index,
            disaggregated_params=disaggregated_params,
            context_logits=None if view.context_logits is None else view.context_logits.tolist(),
            stop_reason=output.stop_reason,
            finish_reason=output.finish_reason,
            avg_decoded_tokens_per_iter=view.avg_decoded_tokens_per_iter,
        )
        if params.return_logprobs:
            choice.logprobs = create_completion_logprobs(
                output.token_ids, params.tokenizer, output.logprobs
            )

        completion_tokens += output.length
        choices.append(choice)

    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=completion_tokens + prompt_tokens,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=view.cached_tokens),
    )
    response = CompletionResponse(choices=choices, model=params.model, usage=usage)
    ctx_usage = _ctx_usage_for_view(params, view)
    rewrite_usage_response_from_ctx(response, ctx_usage)
    return response
