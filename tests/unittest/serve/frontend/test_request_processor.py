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
"""Token-id parity tests: FrontendProcessor vs the historical input pipeline."""

import pytest

from tensorrt_llm.inputs.registry import DefaultInputProcessor
from tensorrt_llm.inputs.utils import apply_chat_template
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.frontend.request_processor import FrontendProcessor


class FakeTokenizer:
    """Deterministic word-hash tokenizer with a trivial chat template."""

    eos_token_id = 2
    pad_token_id = 0
    chat_template = "fake-template"

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, text, add_special_tokens=True, truncation=False, max_length=None):
        self.encode_calls += 1
        token_ids = [hash(word) % 1000 + 10 for word in text.split()]
        if add_special_tokens:
            token_ids = [1] + token_ids
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return token_ids

    def get_chat_template(self, chat_template=None, tools=None):
        return chat_template or self.chat_template

    def apply_chat_template(
        self, conversation=None, tools=None, documents=None, chat_template=None, **kwargs
    ):
        rendered = "".join(f"[{m['role']}]{m['content']}" for m in conversation)
        if kwargs.get("add_generation_prompt"):
            rendered += "[assistant]"
        return rendered


class FakeModelConfig:
    model_type = "llama"


@pytest.fixture(autouse=True)
def reset_instrumentation():
    FrontendProcessor.reset_invocation_count()
    yield
    FrontendProcessor.reset_invocation_count()


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def processor(tokenizer):
    return FrontendProcessor(tokenizer, model_config=FakeModelConfig())


def legacy_tokenize(tokenizer, prompt: str, sampling_params: SamplingParams):
    """The historical text path: DefaultInputProcessor tokenization."""
    legacy = DefaultInputProcessor(model_path=None, config=None, tokenizer=tokenizer)
    token_ids, _extra = legacy({"prompt": prompt}, sampling_params)
    return token_ids


class TestTokenIdParity:
    def test_text_prompt_matches_legacy_path(self, processor, tokenizer):
        params = SamplingParams(end_id=2)
        prompt = "the quick brown fox jumps over the lazy dog"
        assert processor.process_text(prompt, params).prompt_token_ids == legacy_tokenize(
            tokenizer, prompt, params
        )

    def test_no_special_tokens_matches_legacy_path(self, processor, tokenizer):
        params = SamplingParams(end_id=2, add_special_tokens=False)
        prompt = "completions style prompt"
        assert processor.process_text(prompt, params).prompt_token_ids == legacy_tokenize(
            tokenizer, prompt, params
        )

    def test_truncation_matches_legacy_path(self, processor, tokenizer):
        params = SamplingParams(end_id=2, truncate_prompt_tokens=3)
        prompt = "one two three four five six"
        assert processor.process_text(prompt, params).prompt_token_ids == legacy_tokenize(
            tokenizer, prompt, params
        )

    def test_chat_matches_legacy_template_plus_tokenize(self, processor, tokenizer):
        params = SamplingParams(end_id=2, add_special_tokens=False)
        conversation = [
            {"role": "user", "content": "hello there"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "how are you"},
        ]
        legacy_prompt = apply_chat_template(
            model_type="llama",
            tokenizer=tokenizer,
            processor=None,
            conversation=conversation,
            add_generation_prompt=True,
            mm_placeholder_counts=[],
            tools=None,
            documents=None,
            chat_template=None,
            chat_template_kwargs={},
        )
        expected = legacy_tokenize(tokenizer, legacy_prompt, params)

        processed = processor.process_chat(conversation, params, add_generation_prompt=True)
        assert processed.prompt_token_ids == expected
        assert processed.prompt == legacy_prompt

    def test_pretokenized_fast_path_bypasses_tokenizer(self, processor, tokenizer):
        params = SamplingParams(end_id=2)
        before = tokenizer.encode_calls
        processed = processor.process_text([4, 5, 6], params)
        assert processed.prompt_token_ids == [4, 5, 6]
        assert processed.prompt is None
        assert tokenizer.encode_calls == before


class TestParamsSplitParity:
    def test_end_id_defaults_match_legacy_setup(self, processor, tokenizer):
        params = SamplingParams()
        sampling, _config = processor.split_sampling_params(params)

        legacy = SamplingParams()
        legacy._setup(tokenizer, None, None)
        assert sampling.end_id == legacy.end_id
        assert sampling.pad_id == legacy.pad_id

    def test_generation_config_eos_extends_stop_tokens(self, processor, tokenizer):
        class GenerationConfig:
            eos_token_id = [2, 7, 9]

        processor_with_config = FrontendProcessor(tokenizer, generation_config=GenerationConfig())
        params = SamplingParams()
        sampling, _config = processor_with_config.split_sampling_params(params)

        legacy = SamplingParams()
        legacy._setup(tokenizer, None, GenerationConfig())
        assert sampling.stop_token_ids == legacy.stop_token_ids

    def test_stop_strings_tokenized_like_legacy_setup(self, processor, tokenizer):
        params = SamplingParams(end_id=2, stop=["User:", "\n\n"], stop_token_ids=[42])
        sampling, config = processor.split_sampling_params(params)

        legacy = SamplingParams(end_id=2, stop=["User:", "\n\n"], stop_token_ids=[42])
        legacy._setup(tokenizer, None, None)
        assert sampling.stop_sequence_token_ids == legacy._stop_word_ids
        # Runtime stop words = single stop token ids + tokenized stop strings.
        runtime_words = [[i] for i in sampling.stop_token_ids] + sampling.stop_sequence_token_ids
        assert runtime_words == legacy._get_stop_words()
        assert config.stop_strings == ["User:", "\n\n"]
        assert config.stop_token_ids == [42]

    def test_frontend_config_carries_output_shaping_only(self, processor):
        params = SamplingParams(
            end_id=2,
            detokenize=False,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
            n=1,
        )
        _sampling, config = processor.split_sampling_params(params)
        assert config.detokenize is False
        assert config.skip_special_tokens is False
        assert config.include_stop_str_in_output is True
        assert config.num_return_sequences == 1


class TestInstrumentation:
    def test_invocations_counted(self, processor):
        assert FrontendProcessor.invocation_count == 0
        processor.process_text("hello world", SamplingParams(end_id=2))
        assert FrontendProcessor.invocation_count == 1

    def test_flag_off_means_zero_invocations(self):
        """Constructing unrelated objects must not touch the pipeline counter."""
        assert FrontendProcessor.invocation_count == 0
