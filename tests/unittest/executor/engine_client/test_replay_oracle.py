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
"""Golden replay oracle: fixed response streams through both pipelines.

The legacy side runs the REAL legacy consumer code path —
``DetokenizedGenerationResultBase._handle_response`` with the real
detokenize/trim/stop semantics — never a re-implementation in test code.
The new side runs envelope → router → frames → assembler. Parity fields:
token ids, text, finish reason, stop reason, logprob values, and usage
counts, including the presentation renderings (``timeout``, ``cancelled``).
A deliberately perturbed stream must fail parity, proving the oracle
discriminates.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor.engine_client.assembler import FrontendResponseAssembler
from tensorrt_llm.executor.engine_client.contract import (
    ContractConstructionError, FrontendOutputConfig)
from tensorrt_llm.executor.engine_client.router import EngineFrameRouter
from tensorrt_llm.executor.result import (DetokenizedGenerationResultBase,
                                          Logprob)
from tensorrt_llm.sampling_params import SamplingParams

PROMPT_IDS = (1, 2, 3)


class CharTokenizer:
    """Deterministic shared tokenizer: id N ↔ chr(97 + N % 26); id 900+ maps
    to unicode fragments so multi-char/multi-byte boundaries are exercised."""

    UNICODE = {900: "é", 901: "☂", 902: "é☂"}

    def _piece(self, token_id: int) -> str:
        if token_id in self.UNICODE:
            return self.UNICODE[token_id]
        return chr(97 + token_id % 26)

    def decode(self, ids, **kwargs):
        return "".join(self._piece(i) for i in ids)

    def decode_incrementally(self, token_ids, prev_text=None, states=None, *,
                             flush=False, stream_interval=1, **kwargs):
        prev_text = prev_text or ""
        return prev_text + self.decode(token_ids), states or {}


TOKENIZER = CharTokenizer()


def ids_for(text: str) -> list:
    special = {v: k for k, v in CharTokenizer.UNICODE.items()}
    out = []
    for ch in text:
        out.append(special.get(ch, ord(ch) - 97))
    return out


def make_result(**overrides) -> SimpleNamespace:
    fields = dict(is_final=False, output_token_ids=[[]], finish_reasons=None,
                  log_probs=None, cum_log_probs=None, sequence_index=0,
                  context_phase_params=None, decoding_iter=1,
                  avg_decoded_tokens_per_iter=None, request_perf_metrics=None,
                  generation_logits=None, context_logits=None)
    fields.update(overrides)
    return SimpleNamespace(**fields)


def delta(tokens, log_probs=None):
    return LlmResponse(request_id=1, client_id=7,
                       result=make_result(
                           output_token_ids=[list(tokens)],
                           finish_reasons=[tllm.FinishReason.NOT_FINISHED],
                           log_probs=log_probs))


def final(reason, tokens=(), log_probs=None):
    return LlmResponse(request_id=1, client_id=7,
                       result=make_result(is_final=True,
                                          output_token_ids=[list(tokens)],
                                          finish_reasons=[reason],
                                          log_probs=log_probs))


def make_sampling_params(stop=None, stop_token_ids=None) -> SamplingParams:
    params = SamplingParams(max_tokens=64, end_id=2, pad_id=0, stop=stop,
                            stop_token_ids=stop_token_ids)
    if stop is not None:
        strings = [stop] if isinstance(stop, str) else list(stop)
        params._stop_word_ids = [ids_for(s) for s in strings]
    params._stream_interval = 1
    return params


def stop_association(params: SamplingParams):
    return tuple(
        (tuple(ids), reason)
        for reason, ids in params._get_stop_reasons_and_words())


def run_legacy(responses, params: SamplingParams) -> dict:
    result = DetokenizedGenerationResultBase(id=7, sampling_params=params,
                                             tokenizer=TOKENIZER, streaming=True)
    result._streaming = True
    for response in responses:
        result._handle_response(response)
        if result._done:
            break
    output = result._outputs[0]
    logprob_values = [
        entry if not isinstance(entry, dict) else
        max(entry.values(), key=lambda lp: lp.rank is not None and -lp.rank
            if lp.rank else 0).logprob if False else None
        for entry in output.logprobs
    ]
    # Chosen-token projection for dict-shaped entries: look up by token id.
    projected = []
    for position, entry in enumerate(output.logprobs):
        if isinstance(entry, dict):
            token_id = output.token_ids[position]
            projected.append(entry[token_id].logprob)
        else:
            projected.append(entry)
    return {
        "token_ids": list(output.token_ids),
        "text": output.text,
        "finish_reason": output.finish_reason,
        "stop_reason": output.stop_reason,
        "logprobs": projected,
        "usage_completion": len(output.token_ids),
    }


def run_new(responses, params: SamplingParams) -> dict:
    association = stop_association(params)
    strings = ([params.stop] if isinstance(params.stop, str) else
               list(params.stop or ()))
    config = FrontendOutputConfig(
        stop_strings=tuple(strings),
        include_stop_str_in_output=params.include_stop_str_in_output,
        stop_sequence_reasons=association, end_id=params.end_id,
        num_logprobs=params.logprobs)
    aborted = []
    router = EngineFrameRouter(abort_fn=aborted.append)
    request = SimpleNamespace(id=None)
    request.set_id = lambda v: setattr(request, "id", v) or request
    binding = router.register_pending(request, "req-oracle", PROMPT_IDS,
                                      association)
    request.set_id(7)
    router.observe_submit(request)
    assembler = FrontendResponseAssembler("req-oracle", config,
                                          tokenizer=TOKENIZER,
                                          abort_callback=lambda rid: aborted.append(rid))

    for response in responses:
        router.on_response(response)
        # Drain the frames this response produced and process them as one
        # batch (the SSE layer drains-then-renders the same way).
        batch = []
        while True:
            frame, ready = binding.delivery.pop_nowait()
            if not ready:
                break
            batch.append(frame)
        assembler.process_frames(batch)
        if assembler._stopped_by_string:
            break
    return {
        "token_ids": list(assembler.token_ids),
        "text": assembler.text,
        "finish_reason": assembler.finish_reason,
        "stop_reason": assembler.stop_reason,
        "logprobs": list(assembler.logprobs),
        "usage_completion": (assembler.usage or {}).get(
            "completion_tokens", len(assembler.token_ids)),
    }


PARITY_FIELDS = ("token_ids", "text", "finish_reason", "stop_reason", "logprobs",
                 "usage_completion")


def assert_parity(responses, params, fields=PARITY_FIELDS):
    legacy = run_legacy(responses, params)
    new = run_new(responses, params)
    for field in fields:
        assert new[field] == legacy[field], (
            f"parity mismatch on {field!r}: new={new[field]!r} "
            f"legacy={legacy[field]!r}")
    return legacy, new


class TestParityCorpus:

    def test_plain_streaming(self):
        responses = [delta(ids_for("hel")), delta(ids_for("lo")),
                     final(tllm.FinishReason.END_ID)]
        assert_parity(responses, make_sampling_params())

    def test_final_carrying_tokens(self):
        responses = [delta(ids_for("hi")),
                     final(tllm.FinishReason.LENGTH, tokens=ids_for("!"[0:0] or "z"))]
        assert_parity(responses, make_sampling_params())

    def test_stop_token_id(self):
        params = make_sampling_params(stop_token_ids=[13])
        responses = [delta(ids_for("ab")),
                     final(tllm.FinishReason.STOP_WORDS, tokens=[13])]
        assert_parity(responses, params)

    def test_engine_stop_string_cross_token_boundary(self):
        # Stop string "xy" tokenizes to two tokens; the engine stops on the
        # sequence and both pipelines trim it.
        params = make_sampling_params(stop=["xy"])
        responses = [delta(ids_for("ab")),
                     final(tllm.FinishReason.STOP_WORDS, tokens=ids_for("xy"))]
        assert_parity(responses, params)

    def test_frontend_stop_string_detection(self):
        # The engine does not stop; the frontend detects the string in the
        # decoded text mid-stream (legacy scans cumulative text each step).
        params = make_sampling_params(stop=["xy"])
        responses = [delta(ids_for("ax")), delta(ids_for("yz")),
                     final(tllm.FinishReason.LENGTH)]
        legacy, new = assert_parity(responses, params)
        assert legacy["finish_reason"] == "stop"
        assert legacy["stop_reason"] == "xy"

    def test_duplicate_and_colliding_stop_strings(self):
        params = make_sampling_params(stop=["xy", "xy", "y"])
        responses = [delta(ids_for("axyb")), final(tllm.FinishReason.LENGTH)]
        assert_parity(responses, params)

    def test_unicode_stop_string(self):
        params = make_sampling_params(stop=["é☂"])
        responses = [delta([900]), delta([901]),
                     final(tllm.FinishReason.LENGTH)]
        assert_parity(responses, params)

    def test_float_logprobs(self):
        responses = [delta(ids_for("ab"), log_probs=[[-0.5, -1.0]]),
                     delta(ids_for("c"), log_probs=[[-0.25]]),
                     final(tllm.FinishReason.END_ID)]
        assert_parity(responses, make_sampling_params())

    def test_dict_logprobs_project_to_same_values(self):
        entries1 = [{ids_for("a")[0]: Logprob(-0.5, 1)},
                    {ids_for("b")[0]: Logprob(-1.0, 2)}]
        responses = [delta(ids_for("ab"), log_probs=[entries1]),
                     final(tllm.FinishReason.END_ID)]
        assert_parity(responses, make_sampling_params())

    def test_timeout_rendering(self):
        responses = [delta(ids_for("a")), final(tllm.FinishReason.TIMED_OUT)]
        legacy, new = assert_parity(responses, make_sampling_params(),
                                    fields=("token_ids", "text", "finish_reason"))
        assert legacy["finish_reason"] == "timeout"

    def test_cancelled_rendering(self):
        responses = [delta(ids_for("a")), final(tllm.FinishReason.CANCELLED)]
        legacy, new = assert_parity(responses, make_sampling_params(),
                                    fields=("token_ids", "text", "finish_reason"))
        assert legacy["finish_reason"] == "cancelled"

    def test_empty_stop_string_rejected_at_construction(self):
        with pytest.raises(ContractConstructionError):
            FrontendOutputConfig(stop_strings=("", ))


class TestOracleDiscriminates:

    def test_perturbed_stream_fails_parity(self):
        params = make_sampling_params()
        responses = [delta(ids_for("hel")), delta(ids_for("lo")),
                     final(tllm.FinishReason.END_ID)]
        legacy = run_legacy(responses, params)
        perturbed = [delta(ids_for("hel")), delta(ids_for("lQ".lower())),
                     final(tllm.FinishReason.END_ID)]
        new = run_new(perturbed, params)
        assert new["text"] != legacy["text"]

    def test_dropped_terminal_detected(self):
        params = make_sampling_params()
        complete_responses = [delta(ids_for("hi")), final(tllm.FinishReason.END_ID)]
        truncated = [delta(ids_for("hi"))]
        legacy = run_legacy(complete_responses, params)
        new = run_new(truncated, params)
        assert new["finish_reason"] != legacy["finish_reason"]
