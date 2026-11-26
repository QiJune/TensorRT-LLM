# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A streamed request that stops on a multi-token stop word must end with the non-streamed text.

When the sampler matches a stop word by token id, ``_handle_sequence`` trims
the stop ids from ``token_ids``. In a streamed request some of those ids may
have arrived in earlier responses and already been decoded incrementally, so
the final ``.text`` has to be decoded again from the kept ids: otherwise it
keeps the start of the stop string (``"<answer>42</answer"``), or ends in
U+FFFD when the decoder still holds the lead bytes of a character that only
the stop word completes (a Thai digit such as ``"๒"``).

The executor's requests (``LlmRequest``) play the sampler's steps, and their
responses reach the result as a frontend gets them: rank 0 publishes them and
the frontend decodes the frame (``test_common.executor.fakes.Rank0``).
CPU-only; no engine needed.
"""

import pytest
from test_common.executor.fakes import Rank0
from tokenizers import Tokenizer
from tokenizers.decoders import ByteFallback, Fuse, Sequence
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.llmapi.engine_client.result import DetokenizedGenerationResultBase
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.tokenizer import TransformersTokenizer

pytestmark = pytest.mark.cpu_only

# "๑" and "๒" are E0 B9 91 and E0 B9 92 in UTF-8: one byte token each, so the
# decoder holds <0xE0><0xB9> until the last byte completes the character.
VOCAB = {
    "<unk>": 0,
    "<0xE0>": 1,
    "<0xB9>": 2,
    "<0x91>": 3,
    "<0x92>": 4,
    "ข้อ": 5,
    " ": 6,
    "<answer>": 7,
    "42": 8,
    "</": 9,
    "answer": 10,
    ">": 11,
}
THAI_DIGIT_TWO = [1, 2, 4]
# "ข้อ ๑ ข้อ ๒"
THAI_GENERATED = [5, 6, 1, 2, 3, 6, 5, 6, *THAI_DIGIT_TWO]
ANSWER_CLOSE = [9, 10, 11]
# "<answer>42</answer>"
ANSWER_GENERATED = [7, 8, *ANSWER_CLOSE]
PROMPT = [5, 6]
CLIENT_ID = 1


@pytest.fixture(scope="module")
def tokenizer() -> TransformersTokenizer:
    backend = Tokenizer(WordLevel(VOCAB, unk_token="<unk>"))
    backend.decoder = Sequence([ByteFallback(), Fuse()])
    return TransformersTokenizer(PreTrainedTokenizerFast(tokenizer_object=backend))


@pytest.fixture(params=["HF", "TRTLLM"], autouse=True)
def detokenization_backend(request, monkeypatch: pytest.MonkeyPatch) -> str:
    from tensorrt_llm.tokenizer import tokenizer as tokenizer_module

    monkeypatch.setattr(tokenizer_module, "TLLM_INCREMENTAL_DETOKENIZATION_BACKEND", request.param)
    return request.param


def _sample(llm_request, token_ids, finish_reason=None):
    """One sampler step of ``llm_request``: its new tokens, and its end with ``finish_reason``."""
    for token_id in token_ids:
        llm_request.add_new_token(token_id, 0)
    if finish_reason is not None:
        llm_request.state = LlmRequestState.GENERATION_COMPLETE
        llm_request.set_finished_reason(finish_reason, 0)


def _sampling_params(stop, stop_ids, **kwargs):
    sampling_params = SamplingParams(stop=stop, **kwargs)
    # Stands in for SamplingParams._setup tokenizing the stop string.
    sampling_params._stop_word_ids = [stop_ids]
    sampling_params._stream_interval = 1
    return sampling_params


def _serve(tokenizer, sampling_params, streaming=True):
    """A detokenizing result for a request, the rank 0 serving it, and the executor's requests."""
    request = GenerationRequest(list(PROMPT), sampling_params, streaming=streaming)
    request.set_id(CLIENT_ID)
    result = DetokenizedGenerationResultBase(
        CLIENT_ID, sampling_params, tokenizer=tokenizer, streaming=streaming
    )
    rank0 = Rank0()
    return result, rank0, rank0.submit(request, result)


def _run(tokenizer, sampling_params, chunks, finish_reason, streaming=True):
    """Sample one chunk per step; the last one finishes the request with finish_reason."""
    result, rank0, (sequence,) = _serve(tokenizer, sampling_params, streaming=streaming)
    deltas = []
    for index, chunk in enumerate(chunks):
        is_final = index == len(chunks) - 1
        _sample(sequence, chunk, finish_reason if is_final else None)
        rank0.publish(sequence)
        deltas.append(result.outputs[0].text_diff)
    return result.outputs[0], "".join(deltas)


def _chunked(token_ids, size):
    return [token_ids[i : i + size] for i in range(0, len(token_ids), size)]


# The last field is the joined text_diff values. For "</answer>" they still end in
# "</answer": earlier responses already streamed the decoded stop-word prefix, and
# deltas cannot take text back (#21318). Change it here once streaming holds back a
# possible stop-word prefix.
CASES = [
    pytest.param("๒", THAI_DIGIT_TWO, THAI_GENERATED, "ข้อ ๑ ข้อ ", "ข้อ ๑ ข้อ ", id="thai_digit"),
    pytest.param(
        "</answer>",
        ANSWER_CLOSE,
        ANSWER_GENERATED,
        "<answer>42",
        "<answer>42</answer",
        id="answer_tag",
    ),
]


@pytest.mark.parametrize("chunk_size", [1, 2])
@pytest.mark.parametrize("stop, stop_ids, generated, expected, expected_joined", CASES)
def test_streamed_text_drops_stop_word_ids_streamed_earlier(
    tokenizer, stop, stop_ids, generated, expected, expected_joined, chunk_size
):
    sampling_params = _sampling_params(stop, stop_ids)
    output, joined = _run(
        tokenizer, sampling_params, _chunked(generated, chunk_size), tllm.FinishReason.STOP_WORDS
    )
    reference, _ = _run(
        tokenizer,
        _sampling_params(stop, stop_ids),
        [generated],
        tllm.FinishReason.STOP_WORDS,
        streaming=False,
    )

    assert reference.text == expected
    assert output.text == expected
    assert output.token_ids == reference.token_ids
    assert output.finish_reason == "stop"
    assert output.stop_reason == stop
    assert joined == expected_joined


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_streamed_deltas_do_not_end_in_replacement_character(tokenizer, chunk_size):
    """The held lead bytes of the stop character must not be flushed into the last delta."""
    _, joined = _run(
        tokenizer,
        _sampling_params("๒", THAI_DIGIT_TWO),
        _chunked(THAI_GENERATED, chunk_size),
        tllm.FinishReason.STOP_WORDS,
    )
    assert "\N{REPLACEMENT CHARACTER}" not in joined
    assert joined == "ข้อ ๑ ข้อ "


def test_stop_word_inside_final_response_keeps_streamed_text(tokenizer):
    """Stop ids that all arrive in the final response never reached the decoder."""
    chunks = [THAI_GENERATED[:-3], THAI_DIGIT_TWO]
    output, joined = _run(
        tokenizer, _sampling_params("๒", THAI_DIGIT_TWO), chunks, tllm.FinishReason.STOP_WORDS
    )
    assert output.text == joined == "ข้อ ๑ ข้อ "


def test_include_stop_str_in_output_keeps_stop_word(tokenizer):
    output, joined = _run(
        tokenizer,
        _sampling_params("๒", THAI_DIGIT_TWO, include_stop_str_in_output=True),
        _chunked(THAI_GENERATED, 1),
        tllm.FinishReason.STOP_WORDS,
    )
    assert output.token_ids == THAI_GENERATED
    assert output.text == joined == "ข้อ ๑ ข้อ ๒"


def test_length_finish_flushes_held_bytes(tokenizer):
    """Without a stop-word trim the final flush still emits what a full decode gives."""
    generated = THAI_GENERATED[:-1]
    output, _ = _run(
        tokenizer,
        _sampling_params("๒", THAI_DIGIT_TWO),
        _chunked(generated, 1),
        tllm.FinishReason.LENGTH,
    )
    assert output.finish_reason == "length"
    assert output.text == tokenizer.decode(generated)


def test_trimmed_candidate_is_decoded_once_when_siblings_stream_on(tokenizer, monkeypatch):
    """With n > 1, a candidate that stopped on a trimmed stop word is not decoded again.

    Candidate 0 streams to its stop word first; candidate 1 then streams on and
    each of its responses walks every output. Candidate 0 must keep the
    non-streamed text and an empty text_diff without another full decode.
    """
    sampling_params = _sampling_params(
        "๒", THAI_DIGIT_TWO, n=2, best_of=2, temperature=0.8, top_p=0.9
    )
    result, rank0, sequences = _serve(tokenizer, sampling_params)
    full_decodes = []
    decode = tokenizer.decode

    def counting_decode(token_ids, *args, **kwargs):
        full_decodes.append(list(token_ids))
        return decode(token_ids, *args, **kwargs)

    monkeypatch.setattr(tokenizer, "decode", counting_decode)

    sibling = [5, 6, 7, 8]
    steps = [
        (0, [token_id], tllm.FinishReason.STOP_WORDS if index == len(THAI_GENERATED) - 1 else None)
        for index, token_id in enumerate(THAI_GENERATED)
    ] + [
        (1, [token_id], tllm.FinishReason.LENGTH if index == len(sibling) - 1 else None)
        for index, token_id in enumerate(sibling)
    ]
    first, second = result._outputs
    first_diffs_after_stop = []
    for sequence_index, chunk, finish_reason in steps:
        sequence = sequences[sequence_index]
        _sample(sequence, chunk, finish_reason)
        rank0.publish(sequence)
        if sequence_index == 1:
            first_diffs_after_stop.append(first.text_diff)

    assert first.text == "ข้อ ๑ ข้อ "
    assert first.finish_reason == "stop"
    assert first_diffs_after_stop == [""] * len(sibling)
    assert full_decodes == [THAI_GENERATED[:-3]]
    assert second.text == "ข้อ <answer>42"
    assert second.finish_reason == "length"
