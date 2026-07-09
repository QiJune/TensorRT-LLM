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
"""Shared frontend input processing: chat templating, tokenization, params split.

``FrontendProcessor`` is the single input pipeline used by both serving
facades on the engine-boundary path — the OpenAI HTTP server and the Python
LLM API. It owns:

- chat-template application (chat endpoint),
- prompt tokenization with the same semantics as the historical input
  processor (special-token handling, truncation, tiktoken fallback),
- the decomposition of the sampling-params bag into the engine-facing
  :class:`~tensorrt_llm.engine_api.contracts.RuntimeSamplingConfig` (plain
  data that crosses the boundary) and the frontend-only
  :class:`~tensorrt_llm.engine_api.contracts.FrontendOutputConfig`.

Pre-tokenized prompts pass straight through (the ``PreprocessedInputs``
fast-path behavior is preserved: token ids are never re-tokenized).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

from tensorrt_llm.engine_api.contracts import FrontendOutputConfig, RuntimeSamplingConfig

__all__ = ["FrontendProcessor", "ProcessedInput"]

# Mirrors the special-token allowance of the historical default input
# processor's tiktoken fallback path.
_TIKTOKEN_SPECIAL_TOKENS = {
    "<|startoftext|>",
    "<|endoftext|>",
    "<|reserved_200000|>",
    "<|reserved_200001|>",
    "<|return|>",
    "<|constrain|>",
    "<|reserved_200004|>",
    "<|channel|>",
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|reserved_200009|>",
    "<|reserved_200010|>",
    "<|reserved_200011|>",
    "<|call|>",
    "<|reserved_200013|>",
}


@dataclass(slots=True)
class ProcessedInput:
    """Result of frontend input processing for one request."""

    prompt_token_ids: list[int]
    prompt: Optional[str]
    sampling: RuntimeSamplingConfig
    output_config: FrontendOutputConfig


class FrontendProcessor:
    """Tokenize + chat-template pipeline shared by both serving facades.

    Args:
        tokenizer: The frontend tokenizer (encode/decode; chat templates for
            the chat endpoint).
        model_config: HF model config used for chat-template model-type
            resolution; only needed for the chat endpoint.
        processor: Optional HF processor forwarded to chat templating.
        generation_config: HF generation config; its extra ``eos_token_id``
            values extend the runtime stop tokens exactly like the historical
            sampling-params setup.
        default_chat_template: Server-level chat template override.
        default_stream_interval: Streaming detokenization interval used when
            the request does not pin one.
        lightweight_templates: Apply chat templates through the import-light
            text-only path (detached frontends, which reject multimodal)
            instead of the full multimodal-aware pipeline.
    """

    # Test instrumentation: counts pipeline invocations process-wide so
    # flag-off runs can assert the new path was never exercised.
    invocation_count: int = 0

    def __init__(
        self,
        tokenizer: Any,
        *,
        model_config: Any = None,
        processor: Any = None,
        generation_config: Any = None,
        default_chat_template: Optional[str] = None,
        default_stream_interval: int = 1,
        lightweight_templates: bool = False,
    ) -> None:
        self._tokenizer = tokenizer
        self._model_config = model_config
        self._processor = processor
        self._generation_config = generation_config
        self._default_chat_template = default_chat_template
        self._default_stream_interval = default_stream_interval
        self._lightweight_templates = lightweight_templates

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    # --- input processing --------------------------------------------------

    def process_text(self, prompt: Union[str, list[int]], sampling_params: Any) -> ProcessedInput:
        """Process a plain-text (or pre-tokenized) prompt.

        Pre-tokenized prompts pass through unchanged; text prompts are
        tokenized with the same semantics as the historical input processor.
        """
        type(self).invocation_count += 1
        if isinstance(prompt, str):
            prompt_token_ids = self.tokenize(prompt, sampling_params)
            prompt_text: Optional[str] = prompt
        else:
            prompt_token_ids = list(prompt)
            prompt_text = None
        sampling, output_config = self.split_sampling_params(sampling_params)
        return ProcessedInput(
            prompt_token_ids=prompt_token_ids,
            prompt=prompt_text,
            sampling=sampling,
            output_config=output_config,
        )

    def process_chat(
        self,
        conversation: list[dict],
        sampling_params: Any,
        *,
        add_generation_prompt: bool = True,
        mm_placeholder_counts: Optional[list] = None,
        tools: Optional[list[dict]] = None,
        documents: Optional[list[dict]] = None,
        chat_template: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
    ) -> ProcessedInput:
        """Apply the chat template to a parsed conversation, then tokenize."""
        type(self).invocation_count += 1
        if self._lightweight_templates:
            from tensorrt_llm.tokenizer.chat_template import apply_text_chat_template

            prompt = apply_text_chat_template(
                tokenizer=self._tokenizer,
                conversation=conversation,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
                documents=documents,
                chat_template=chat_template or self._default_chat_template,
                chat_template_kwargs=chat_template_kwargs or {},
            )
            prompt_token_ids = self.tokenize(prompt, sampling_params)
            sampling, output_config = self.split_sampling_params(sampling_params)
            return ProcessedInput(
                prompt_token_ids=prompt_token_ids,
                prompt=prompt,
                sampling=sampling,
                output_config=output_config,
            )
        # Deferred: pulls the historical template machinery only on the chat
        # endpoint; the text path stays independent of it.
        from tensorrt_llm.inputs.utils import apply_chat_template
        from tensorrt_llm.serve.chat_utils import resolve_top_level_model_type

        prompt = apply_chat_template(
            model_type=resolve_top_level_model_type(self._model_config),
            tokenizer=self._tokenizer,
            processor=self._processor,
            conversation=conversation,
            add_generation_prompt=add_generation_prompt,
            mm_placeholder_counts=mm_placeholder_counts or [],
            tools=tools,
            documents=documents,
            chat_template=chat_template or self._default_chat_template,
            chat_template_kwargs=chat_template_kwargs or {},
        )
        prompt_token_ids = self.tokenize(prompt, sampling_params)
        sampling, output_config = self.split_sampling_params(sampling_params)
        return ProcessedInput(
            prompt_token_ids=prompt_token_ids,
            prompt=prompt,
            sampling=sampling,
            output_config=output_config,
        )

    def tokenize(self, prompt: str, sampling_params: Any) -> list[int]:
        """Tokenize text with the historical default-input-processor semantics."""
        if self._tokenizer is None:
            raise ValueError("tokenizer is required to tokenize string prompt")
        kwargs = {}
        if sampling_params.truncate_prompt_tokens is not None:
            kwargs = dict(truncation=True, max_length=sampling_params.truncate_prompt_tokens)
        try:
            token_ids = self._tokenizer.encode(
                prompt, add_special_tokens=sampling_params.add_special_tokens, **kwargs
            )
        except Exception:
            # Tiktoken path
            token_ids = self._tokenizer.encode(prompt, allowed_special=_TIKTOKEN_SPECIAL_TOKENS)
        return token_ids

    # --- params-bag decomposition -------------------------------------------

    def split_sampling_params(
        self, params: Any
    ) -> tuple[RuntimeSamplingConfig, FrontendOutputConfig]:
        """Split a SamplingParams bag into runtime-crossing and frontend-only halves.

        Applies the same defaults the historical setup applied: ``end_id`` and
        ``pad_id`` fall back to the tokenizer, and extra generation-config
        ``eos_token_id`` values extend the runtime stop tokens. Stop and bad
        strings are tokenized frontend-side; only their token sequences cross.
        """
        end_id = params.end_id
        pad_id = params.pad_id
        if end_id is None and self._tokenizer is not None:
            end_id = self._tokenizer.eos_token_id
            pad_id = self._tokenizer.pad_token_id
            if pad_id is None:
                pad_id = end_id

        stop_token_ids = list(params.stop_token_ids) if params.stop_token_ids else []
        generation_config = self._generation_config
        if params.end_id is None and generation_config is not None:
            eos_token_id = getattr(generation_config, "eos_token_id", None)
            if eos_token_id is not None:
                eos_token_ids = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
                for stop_token in eos_token_ids:
                    if stop_token != end_id and stop_token not in stop_token_ids:
                        stop_token_ids.append(stop_token)

        stop_strings = None
        stop_sequence_token_ids = None
        if params.stop is not None:
            stop_strings = [params.stop] if isinstance(params.stop, str) else list(params.stop)
            stop_sequence_token_ids = [self._encode_word(s) for s in stop_strings]

        bad_sequence_token_ids = None
        if params.bad is not None:
            bad_strings = [params.bad] if isinstance(params.bad, str) else list(params.bad)
            bad_sequence_token_ids = [self._encode_word(s) for s in bad_strings]

        sampling = RuntimeSamplingConfig(
            max_tokens=params.max_tokens,
            n=params.n,
            best_of=params.best_of,
            use_beam_search=params.use_beam_search,
            end_id=end_id,
            pad_id=pad_id,
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            top_p_min=params.top_p_min,
            top_p_reset_ids=params.top_p_reset_ids,
            top_p_decay=params.top_p_decay,
            min_p=params.min_p,
            seed=params.seed,
            min_tokens=params.min_tokens,
            presence_penalty=params.presence_penalty,
            frequency_penalty=params.frequency_penalty,
            repetition_penalty=params.repetition_penalty,
            length_penalty=params.length_penalty,
            early_stopping=params.early_stopping,
            no_repeat_ngram_size=params.no_repeat_ngram_size,
            prompt_ignore_length=params.prompt_ignore_length,
            beam_search_diversity_rate=params.beam_search_diversity_rate,
            beam_width_array=params.beam_width_array,
            stop_token_ids=stop_token_ids or None,
            stop_sequence_token_ids=stop_sequence_token_ids,
            bad_token_ids=list(params.bad_token_ids) if params.bad_token_ids else None,
            bad_sequence_token_ids=bad_sequence_token_ids,
            ignore_eos=params.ignore_eos,
            logprobs=params.logprobs,
            prompt_logprobs=params.prompt_logprobs,
            logprobs_simple_format=params.logprobs_simple_format,
            prompt_logprobs_simple_format=params.prompt_logprobs_simple_format,
            return_perf_metrics=params.return_perf_metrics,
        )
        output_config = FrontendOutputConfig(
            detokenize=params.detokenize,
            skip_special_tokens=params.skip_special_tokens,
            spaces_between_special_tokens=params.spaces_between_special_tokens,
            stop_strings=stop_strings,
            stop_sequence_token_ids=stop_sequence_token_ids,
            stop_token_ids=stop_token_ids or None,
            include_stop_str_in_output=params.include_stop_str_in_output,
            num_return_sequences=params.n,
            stream_interval=getattr(params, "_stream_interval", None)
            or self._default_stream_interval,
        )
        return sampling, output_config

    def _encode_word(self, text: str) -> list[int]:
        """Tokenize a stop/bad word exactly like the historical params setup."""
        try:
            return self._tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            # For tiktokenizer, encode does not have add_special_tokens.
            return self._tokenizer.encode(text)

    @classmethod
    def reset_invocation_count(cls) -> None:
        cls.invocation_count = 0
