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
"""Normalization of raw worker responses into a uniform envelope.

The proxy's result-dispatch loop delivers several shapes: the PyTorch
path's ``LlmResponse``, ``ResponseWrapper`` (post-computed logprobs and
perf metrics attached by the worker), and ``ErrorResponse``. This module
normalizes the PyTorch shapes into ``RuntimeResponseEnvelope`` — immutable
snapshots of exactly the fields the contract needs — and rejects, with
typed errors, the shapes V0 does not support (the C++-engine
``tllm.Response`` and the postprocess-parallel ``PostprocWorker.Output``).

Normalization rules (contract DEC-1/DEC-8 and divergence note 5):
- Both runtime logprob shapes are accepted: chosen-token float lists pass
  through; token-id→``Logprob`` maps resolve by exact token-id lookup, and
  any missing key, misalignment, or length mismatch is a typed error —
  never a silent drop or pad.
- Perf-metric keys convert enum→string via ``.value``.
- Cached-token accounting is read from the same runtime field the legacy
  usage path reads.
- A no-token response produces an envelope with no new tokens; the router
  decides what (if any) frame that becomes.
"""

import dataclasses
import math
from typing import Optional

from ..._torch.pyexecutor.llm_request import LlmResponse
from ..postproc_worker import PostprocWorker
from ..result import ResponseWrapper
from ..utils import ErrorResponse, is_llm_response
from .contract import ContractError

__all__ = [
    "EnvelopeError",
    "RuntimeResponseEnvelope",
    "normalize_response",
]


class EnvelopeError(ContractError):
    """A worker response cannot be normalized. Carries a stable ``reason``."""

    def __init__(self, reason: str, message: str):
        super().__init__(f"{reason}: {message}")
        self.reason = reason


@dataclasses.dataclass(frozen=True)
class RuntimeResponseEnvelope:
    """Immutable snapshot of one worker response, normalized for the router.

    Not a wire type: this is the rank-0-internal handoff between the proxy
    tap and the frame router. ``new_token_ids`` is this response's delta
    (possibly empty for a no-token final); ``finish_reason_name`` is the raw
    engine reason name (e.g. ``"END_ID"``) or ``None`` when the sequence is
    not finished.
    """

    client_id: int
    is_final: bool
    error_msg: Optional[str] = None
    sequence_index: int = 0
    new_token_ids: tuple = ()
    finish_reason_name: Optional[str] = None
    logprobs: Optional[tuple] = None
    prompt_logprobs: Optional[tuple] = None
    # Map-shaped prompt logprobs that arrived on a no-token response cannot
    # be normalized yet (the first generated token supplies the last lookup
    # key); they are carried raw for the router to hold and normalize on the
    # next token-carrying delta via ``resolve_held_prompt_logprobs``.
    raw_prompt_logprob_entries: Optional[tuple] = None
    metrics: Optional[dict] = None
    cached_tokens: Optional[int] = None


def _snapshot_float(value, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EnvelopeError("logprob_mismatch", f"{context}: expected a number, got "
                            f"{type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise EnvelopeError("logprob_mismatch", f"{context}: non-finite value")
    return value


def _logprob_entry(entry, expected_token_id: int, context: str) -> float:
    """Normalize one logprob entry: float passes, mapping needs exact lookup."""
    if isinstance(entry, dict):
        if expected_token_id not in entry:
            raise EnvelopeError(
                "logprob_mismatch",
                f"{context}: token id {expected_token_id} missing from logprob map "
                f"(keys={sorted(entry.keys())[:8]!r})")
        item = entry[expected_token_id]
        value = getattr(item, "logprob", item)
        return _snapshot_float(value, context)
    return _snapshot_float(entry, context)


def _normalize_generation_logprobs(log_probs, new_token_ids, client_id) -> Optional[tuple]:
    if log_probs is None:
        return None
    entries = list(log_probs)
    if len(entries) != len(new_token_ids):
        raise EnvelopeError(
            "logprob_mismatch",
            f"client_id={client_id}: {len(entries)} logprob entries for "
            f"{len(new_token_ids)} new tokens")
    return tuple(
        _logprob_entry(entry, token_id, f"client_id={client_id} logprobs[{i}]")
        for i, (entry, token_id) in enumerate(zip(entries, new_token_ids)))


def _normalize_prompt_logprobs(prompt_entries, prompt_token_ids, new_token_ids,
                               client_id) -> Optional[tuple]:
    if prompt_entries is None:
        return None
    entries = list(prompt_entries)
    if all(not isinstance(entry, dict) for entry in entries):
        return tuple(
            _snapshot_float(entry, f"client_id={client_id} prompt_logprobs[{i}]")
            for i, entry in enumerate(entries))
    # Map-shaped prompt logprobs need the exact per-position token ids: the
    # prompt shifted by one plus the first generated token (mirroring the
    # worker-side computation offset).
    if prompt_token_ids is None:
        raise EnvelopeError(
            "logprob_mismatch",
            f"client_id={client_id}: map-shaped prompt logprobs need the bound "
            "prompt token ids")
    if not new_token_ids:
        # Legitimately possible: the worker re-attaches cached prompt
        # logprobs to later (even terminal-only) responses. The caller holds
        # the raw entries or drops them if they were already delivered.
        return None
    expected = list(prompt_token_ids[1:]) + [new_token_ids[0]]
    if len(entries) != len(expected):
        raise EnvelopeError(
            "logprob_mismatch",
            f"client_id={client_id}: {len(entries)} prompt-logprob entries for "
            f"{len(expected)} expected positions")
    return tuple(
        _logprob_entry(entry, token_id, f"client_id={client_id} prompt_logprobs[{i}]")
        for i, (entry, token_id) in enumerate(zip(entries, expected)))


def _normalize_metrics(raw_metrics, client_id) -> Optional[dict]:
    if raw_metrics is None:
        return None
    if not isinstance(raw_metrics, dict):
        raise EnvelopeError(
            "metrics_mismatch",
            f"client_id={client_id}: expected a metrics dict, got "
            f"{type(raw_metrics).__name__}")
    metrics = {}
    for key, value in raw_metrics.items():
        name = getattr(key, "value", key)
        if not isinstance(name, str):
            name = str(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EnvelopeError(
                "metrics_mismatch",
                f"client_id={client_id}: metric {name!r} has non-numeric value "
                f"{type(value).__name__}")
        value = float(value)
        if not math.isfinite(value):
            raise EnvelopeError(
                "metrics_mismatch",
                f"client_id={client_id}: metric {name!r} is not finite")
        metrics[name] = value
    return metrics


def normalize_response(raw, prompt_token_ids=None) -> RuntimeResponseEnvelope:
    """Normalize one raw dispatch-loop item into an envelope.

    Args:
        raw: One item as delivered to the proxy result-dispatch loop (after
            batched lists are unrolled by the caller).
        prompt_token_ids: The bound request's prompt token ids, required only
            to resolve map-shaped prompt logprobs.

    Raises:
        EnvelopeError: for unsupported shapes (C++-engine ``tllm.Response``,
            ``PostprocWorker.Output``, unknown objects) and for logprob or
            metric content that cannot be normalized losslessly.
    """
    wrapper_metrics = None
    logprobs_result = None
    if isinstance(raw, ResponseWrapper):
        wrapper_metrics = raw.request_perf_metrics
        logprobs_result = raw.logprobs
        raw = raw._response

    if isinstance(raw, ErrorResponse):
        return RuntimeResponseEnvelope(client_id=raw.client_id, is_final=True,
                                       error_msg=str(raw.error_msg))
    if isinstance(raw, PostprocWorker.Output):
        raise EnvelopeError(
            "postproc_parallel_shape",
            "PostprocWorker.Output requires num_postprocess_workers=0 in V0")
    if not isinstance(raw, LlmResponse):
        if is_llm_response(raw):
            raise EnvelopeError(
                "cpp_engine_shape",
                f"{type(raw).__name__} (C++-engine response) is not supported in V0")
        raise EnvelopeError("unknown_shape",
                            f"{type(raw).__name__} is not a recognized response shape")

    client_id = raw.client_id
    if raw.has_error():
        return RuntimeResponseEnvelope(client_id=client_id, is_final=True,
                                       error_msg=str(raw.error_msg))

    result = raw.result
    if hasattr(result, "_result") and isinstance(getattr(result, "_result"), bytes):
        result.deserialize()

    sequence_index = getattr(result, "sequence_index", 0) or 0
    output_token_ids = result.output_token_ids
    source_tokens = output_token_ids[0] if output_token_ids else []
    new_token_ids = tuple(int(token) for token in source_tokens)

    finish_reasons = result.finish_reasons
    finish_reason_name = None
    if finish_reasons:
        reason = finish_reasons[0]
        finish_reason_name = getattr(reason, "name", str(reason))

    log_probs = result.log_probs
    logprobs = _normalize_generation_logprobs(
        log_probs[0] if log_probs else None, new_token_ids, client_id)
    raw_prompt_entries = (logprobs_result.prompt
                          if logprobs_result is not None else None)
    prompt_logprobs = _normalize_prompt_logprobs(raw_prompt_entries,
                                                 prompt_token_ids,
                                                 new_token_ids, client_id)
    raw_prompt_logprob_entries = None
    if raw_prompt_entries is not None and prompt_logprobs is None:
        # Map-shaped entries on a no-token response: not normalizable yet.
        raw_prompt_logprob_entries = tuple(raw_prompt_entries)

    cached_tokens = getattr(result, "cached_tokens", None)
    if cached_tokens is not None and (isinstance(cached_tokens, bool)
                                      or not isinstance(cached_tokens, int)):
        cached_tokens = None

    return RuntimeResponseEnvelope(
        client_id=client_id,
        is_final=bool(result.is_final),
        sequence_index=int(sequence_index),
        new_token_ids=new_token_ids,
        finish_reason_name=finish_reason_name,
        logprobs=logprobs,
        prompt_logprobs=prompt_logprobs,
        raw_prompt_logprob_entries=raw_prompt_logprob_entries,
        metrics=_normalize_metrics(wrapper_metrics, client_id),
        cached_tokens=cached_tokens,
    )


def resolve_held_prompt_logprobs(raw_entries, prompt_token_ids, first_token_id,
                                 client_id) -> tuple:
    """Normalize held map-shaped prompt logprobs once the first token is known."""
    return _normalize_prompt_logprobs(list(raw_entries), prompt_token_ids,
                                      (first_token_id, ), client_id)
