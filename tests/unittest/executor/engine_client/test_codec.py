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
"""Codec round-trip, golden-fixture stability, and strict rejection tests."""

import os

import msgpack
import pytest

from tensorrt_llm.executor.engine_client.codec import (MAX_MESSAGE_BYTES,
                                                       DecodeError,
                                                       EncodeError, decode,
                                                       encode)
from tensorrt_llm.executor.engine_client.contract import (
    ENGINE_CONTRACT_VERSION, EngineCapabilities, EngineHealth, EngineRequest,
    EngineSamplingConfig, ErrorFrame, FrontendModelContext,
    FrontendOutputConfig, GuidedDecodingSpec, IterationStatsBatch,
    KvCacheEventsBatch, RequestComplete, Terminal, TokenDelta, TokenizerSpec)


def full_engine_request() -> EngineRequest:
    return EngineRequest(
        request_id="req-Ω-你好",
        prompt_token_ids=(1, 2, 3, 2**62, 0),
        sampling=EngineSamplingConfig(
            max_new_tokens=128, end_id=2, pad_id=0,
            stop_token_ids=(13, 14), stop_token_sequences=((7, 8), (9, )),
            min_tokens=1, temperature=0.5, top_p=0.9, top_k=40, seed=-12345,
            repetition_penalty=1.1, presence_penalty=0.0, frequency_penalty=-0.5,
            num_logprobs=1, num_prompt_logprobs=1),
        guided_decoding=GuidedDecodingSpec(mode="json_schema", payload='{"type":"object"}'),
        required_features=("guided_decoding", ))


def full_token_delta() -> TokenDelta:
    return TokenDelta(request_id="r1", sequence_id=0, new_token_ids=(5, 6, 7),
                      logprobs=(-0.25, -1.5, -0.001), prompt_logprobs=(-2.0, ),
                      metrics={"cached_tokens": 3.0, "arrival_time": 12.5}, event_seq=4)


def full_model_context() -> FrontendModelContext:
    return FrontendModelContext(
        tokenizer=TokenizerSpec(uri="/models/llama",
                                files_manifest=(("tokenizer.json", "ab" * 32), ),
                                revision="main", fast=True, trust_remote_code=False,
                                added_tokens_json="[]"),
        capabilities=EngineCapabilities(features=("streaming", "logprobs")),
        chat_template_source="{{ messages }}", chat_template_version="v1",
        eos_id=2, pad_id=0, max_context_length=8192)


ROUND_TRIP_CASES = [
    full_engine_request(),
    EngineRequest(request_id="minimal", prompt_token_ids=(1, ),
                  sampling=EngineSamplingConfig(max_new_tokens=1)),
    full_token_delta(),
    TokenDelta(request_id="m", sequence_id=1, new_token_ids=(42, )),
    Terminal(request_id="r1", sequence_id=0, finish_reason="stop", stop_reason="</s>",
             event_seq=9),
    Terminal(request_id="r1", sequence_id=0, finish_reason="length"),
    Terminal(request_id="r1", sequence_id=0, finish_reason="error", stop_reason="timeout"),
    RequestComplete(request_id="r1", status="ok", prompt_tokens=5, completion_tokens=3,
                    cached_tokens=3, metrics={"e2e_time": 1.25}, event_seq=10),
    RequestComplete(request_id="r1", status="failed", prompt_tokens=0, completion_tokens=0),
    ErrorFrame(request_id="r2", error_code="worker_died",
               message="crashed before first response ☠"),
    EngineCapabilities(features=("streaming", )),
    EngineSamplingConfig(max_new_tokens=4),
    GuidedDecodingSpec(mode="json_object"),
    FrontendOutputConfig(stop_strings=("stop", ),
                         stop_sequence_reasons=(((1, 2), "stop"), ((3, ), 13)), end_id=2,
                         num_logprobs=1),
    TokenizerSpec(uri="/m"),
    full_model_context(),
    EngineHealth(healthy=True, detail="ok"),
    IterationStatsBatch(entries=('{"iter": 1}', )),
    KvCacheEventsBatch(entries=()),
]

# Checked-in binary golden fixtures live in golden/<name>.msgpack; regenerate
# with golden/generate_golden.py. A diff in any .msgpack file is a wire-format
# change of the same protocol version and must be treated as such.
GOLDEN_OBJECTS = [
    ("engine_request_full", full_engine_request),
    ("token_delta_full", full_token_delta),
    ("terminal_stop",
     lambda: Terminal(request_id="r1", sequence_id=0, finish_reason="stop", stop_reason="</s>",
                      event_seq=9)),
    ("request_complete",
     lambda: RequestComplete(request_id="r1", status="ok", prompt_tokens=5, completion_tokens=3,
                             cached_tokens=3, metrics={"e2e_time": 1.25}, event_seq=10)),
    ("error_frame",
     lambda: ErrorFrame(request_id="r2", error_code="worker_died",
                        message="crashed before first response \u2620")),
    ("model_context", full_model_context),
    ("minimal_delta", lambda: TokenDelta(request_id="m", sequence_id=1, new_token_ids=(42, ))),
]

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")



class TestRoundTrip:

    @pytest.mark.parametrize("obj", ROUND_TRIP_CASES, ids=lambda o: type(o).__name__)
    def test_round_trip(self, obj):
        data = encode(obj)
        assert decode(data) == obj
        assert encode(decode(data)) == data

    def test_canonical_metric_key_order(self):
        a = TokenDelta(request_id="r", sequence_id=0, new_token_ids=(1, ),
                       metrics={"a": 1.0, "b": 2.0})
        b = TokenDelta(request_id="r", sequence_id=0, new_token_ids=(1, ),
                       metrics={"b": 2.0, "a": 1.0})
        assert encode(a) == encode(b)


class TestGoldenFixtures:

    @pytest.mark.parametrize("name,factory", GOLDEN_OBJECTS, ids=lambda c: c
                             if isinstance(c, str) else "")
    def test_golden(self, name, factory):
        expected = factory()
        with open(os.path.join(GOLDEN_DIR, f"{name}.msgpack"), "rb") as f:
            data = f.read()
        assert decode(data) == expected, f"golden fixture {name} decode mismatch"
        assert encode(expected) == data, f"golden fixture {name} encode not byte-stable"


def _envelope(kind, fields) -> bytes:
    return msgpack.packb({"k": kind, "d": fields}, use_bin_type=True)


def _terminal_fields(**overrides):
    fields = {"request_id": "r", "sequence_id": 0, "finish_reason": "stop"}
    fields.update(overrides)
    return fields


class TestDecodeRejects:

    def test_unknown_kind(self):
        with pytest.raises(DecodeError) as excinfo:
            decode(_envelope("mystery_frame", {}))
        assert excinfo.value.reason == "unknown_kind"

    def test_unknown_field(self):
        with pytest.raises(DecodeError) as excinfo:
            decode(_envelope("terminal", _terminal_fields(surprise=1)))
        assert excinfo.value.reason == "unknown_field"

    def test_missing_required_field(self):
        with pytest.raises(DecodeError) as excinfo:
            decode(_envelope("terminal", {"request_id": "r"}))
        assert excinfo.value.reason == "missing_field"

    def test_wrong_item_type(self):
        with pytest.raises(DecodeError) as excinfo:
            decode(_envelope("terminal", _terminal_fields(sequence_id="zero")))
        assert excinfo.value.reason == "invalid_content"

    def test_bool_is_not_int(self):
        with pytest.raises(DecodeError) as excinfo:
            decode(_envelope("terminal", _terminal_fields(sequence_id=True)))
        assert excinfo.value.reason == "invalid_content"

    def test_newer_protocol_version(self):
        with pytest.raises(DecodeError) as excinfo:
            decode(_envelope("terminal",
                             _terminal_fields(protocol_version=ENGINE_CONTRACT_VERSION + 1)))
        assert excinfo.value.reason == "version_unsupported"

    def test_nested_version_check(self):
        fields = {
            "request_id": "r",
            "prompt_token_ids": [1],
            "sampling": {"max_new_tokens": 1,
                         "protocol_version": ENGINE_CONTRACT_VERSION + 1},
        }
        with pytest.raises(DecodeError) as excinfo:
            decode(_envelope("engine_request", fields))
        assert excinfo.value.reason == "version_unsupported"

    def test_nan_in_payload(self):
        with pytest.raises(DecodeError) as excinfo:
            decode(_envelope("token_delta",
                             {"request_id": "r", "sequence_id": 0, "new_token_ids": [1],
                              "logprobs": [float("nan")]}))
        assert excinfo.value.reason == "invalid_content"

    def test_extension_type_rejected(self):
        data = msgpack.packb({"k": "terminal", "d": msgpack.ExtType(4, b"x")})
        with pytest.raises(DecodeError):
            decode(data)

    def test_bytes_payload_rejected(self):
        data = msgpack.packb({"k": "terminal", "d": _terminal_fields(stop_reason=b"raw")},
                             use_bin_type=True)
        with pytest.raises(DecodeError):
            decode(data)

    def test_malformed_bytes(self):
        with pytest.raises(DecodeError):
            decode(b"\xc1\x00\x00")

    def test_wrong_envelope_shape(self):
        with pytest.raises(DecodeError) as excinfo:
            decode(msgpack.packb({"kind": "terminal"}))
        assert excinfo.value.reason == "malformed"

    def test_oversized_message(self):
        with pytest.raises(DecodeError) as excinfo:
            decode(b"\x00" * (MAX_MESSAGE_BYTES + 1))
        assert excinfo.value.reason == "limit_exceeded"

    def test_nesting_depth_limit(self):
        nested = 1
        for _ in range(40):
            nested = [nested]
        with pytest.raises(DecodeError):
            decode(msgpack.packb({"k": "terminal", "d": _terminal_fields(stop_reason=nested)}))

    def test_non_bytes_input(self):
        with pytest.raises(DecodeError):
            decode("not-bytes")


class TestEncodeRejects:

    def test_unregistered_type(self):
        with pytest.raises(EncodeError) as excinfo:
            encode({"not": "a wire type"})
        assert excinfo.value.reason == "unknown_type"

    def test_smuggled_nan_rejected(self):
        delta = TokenDelta(request_id="r", sequence_id=0, new_token_ids=(1, ), logprobs=(-0.5, ))
        object.__setattr__(delta, "logprobs", (float("nan"), ))
        with pytest.raises(EncodeError) as excinfo:
            encode(delta)
        assert excinfo.value.reason == "not_finite"

    def test_smuggled_object_rejected(self):
        delta = TokenDelta(request_id="r", sequence_id=0, new_token_ids=(1, ))
        object.__setattr__(delta, "metrics", {"cb": print})
        with pytest.raises(EncodeError) as excinfo:
            encode(delta)
        assert excinfo.value.reason == "non_primitive"

    def test_smuggled_bytes_rejected(self):
        frame = ErrorFrame(request_id="r", error_code="x")
        object.__setattr__(frame, "message", b"raw")
        with pytest.raises(EncodeError) as excinfo:
            encode(frame)
        assert excinfo.value.reason == "non_primitive"
