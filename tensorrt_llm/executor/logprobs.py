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
"""Log probabilities: the types the sampler produces and results report.

Also the prompt log probabilities the engine computes from context logits.
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional, TypeAlias

import torch
import torch.nn.functional as F

__all__ = [
    "Logprob",
    "TokenLogprobs",
    "SimpleTokenLogprobs",
    "LogProbsResult",
    "reduce_prompt_logprobs",
    "prompt_logprobs_from_reduced",
    "compute_prompt_logprobs",
]


@dataclass(slots=True)
class Logprob:
    """Holds logprob and vocab rank for a token."""

    logprob: float
    rank: Optional[int] = None


# List of token_id_to_Logprob dict for prompt or generation texts.
TokenLogprobs: TypeAlias = list[dict[int, Logprob]]


# Compact format: one logprob per token (the sampled token's logprob).
# Returned when `SamplingParams.{logprobs,prompt_logprobs}_simple_format` is set
# and the corresponding `logprobs` / `prompt_logprobs` is 0. Avoids the
# per-token `dict[int, Logprob]` allocation overhead of `TokenLogprobs`.
SimpleTokenLogprobs: TypeAlias = list[float]


class LogProbsResult(NamedTuple):
    """Optional prompt log probabilities computed post runtime."""

    prompt: Optional[TokenLogprobs | SimpleTokenLogprobs] = None


def _reduce_prompt_logprobs_chunk(
    logits: torch.Tensor, token_ids: Optional[list[int]], top_k: int, normalized: bool
) -> torch.Tensor:
    logprobs = logits.to(dtype=torch.float32)
    if not normalized:
        logprobs = F.log_softmax(logprobs, dim=-1)
    columns = []
    if token_ids is not None:
        targets = torch.tensor(token_ids, dtype=torch.int64, device=logprobs.device).unsqueeze(1)
        target_logprobs = logprobs.gather(1, targets)
        ranks = (logprobs > target_logprobs).sum(dim=1, keepdim=True) + 1
        columns += [target_logprobs, ranks]
    if top_k > 0:
        top_values, top_ids = torch.topk(logprobs, k=top_k, dim=-1)
        columns += [top_values, top_ids]
    if not columns:
        return torch.empty((logprobs.shape[0], 0), dtype=torch.float64)
    # FP64 holds FP32 logprobs and integer ranks/token ids exactly.
    return torch.cat([column.to(torch.float64) for column in columns], dim=1).cpu()


def reduce_prompt_logprobs(
    logits: torch.Tensor, token_ids: Optional[list[int]], top_k: int, *, normalized: bool = False
) -> torch.Tensor:
    """Reduce context logits to the values reported as prompt logprobs.

    The reduction runs on the logits' device and copies one small host tensor
    per bounded chunk, so no per-token host synchronization is needed.

    Args:
        logits: ``[rows, vocab]`` (or ``[1, rows, vocab]``) logits of any
            floating dtype; row ``t`` predicts ``token_ids[t]``.
        token_ids: Target token ids, one per used row, or None to report only
            the top-k tokens.
        top_k: Number of most likely tokens to report per row.
        normalized: ``logits`` already hold log probabilities, so they are
            reported as is instead of being log-softmaxed again.

    Returns:
        ``[rows, columns]`` FP64 host tensor. With ``token_ids``, the columns
        start with the target logprob and its 1-based vocabulary rank; ``top_k``
        logprobs and then ``top_k`` token ids follow.
    """
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    if token_ids is not None:
        logits = logits[: len(token_ids)]
    # Bound softmax scratch independently of prompt length. On the GPU, a
    # slice can fall back to CPU without retaining a full-vocabulary GPU result.
    rows_per_chunk = max(1, (16 * 1024 * 1024) // (4 * logits.shape[-1]))
    chunks = []
    for start in range(0, logits.shape[0], rows_per_chunk):
        chunk = logits[start : start + rows_per_chunk]
        tokens = None if token_ids is None else token_ids[start : start + chunk.shape[0]]
        try:
            reduced = _reduce_prompt_logprobs_chunk(chunk, tokens, top_k, normalized)
        except torch.OutOfMemoryError:
            if not chunk.is_cuda:
                raise
            reduced = None
        if reduced is None:
            reduced = _reduce_prompt_logprobs_chunk(chunk.cpu(), tokens, top_k, normalized)
        chunks.append(reduced)
    if not chunks:
        columns = (0 if token_ids is None else 2) + 2 * top_k
        return torch.empty((0, columns), dtype=torch.float64)
    return torch.cat(chunks)


def prompt_logprobs_from_reduced(
    reduced: torch.Tensor, token_ids: Optional[list[int]], top_k: int, simple: bool
) -> TokenLogprobs | SimpleTokenLogprobs:
    """Build the user-facing prompt logprobs from ``reduce_prompt_logprobs`` rows.

    Args:
        reduced: ``[rows, columns]`` FP64 host tensor from
            ``reduce_prompt_logprobs`` with the same ``token_ids`` presence
            and ``top_k``.
        token_ids: The target token ids the rows were reduced against, or None.
        top_k: Number of top tokens reduced per row.
        simple: Return a flat list of target logprobs when ``top_k`` is 0.
    """
    rows = reduced.tolist()
    if top_k == 0:
        if token_ids is None:
            return []
        if simple:
            return [row[0] for row in rows]
        return [
            {token_id: Logprob(logprob=row[0], rank=int(row[1]))}
            for token_id, row in zip(token_ids, rows)
        ]
    first = 0 if token_ids is None else 2
    results: TokenLogprobs = []
    for index, row in enumerate(rows):
        values = row[first : first + top_k]
        ids = row[first + top_k : first + 2 * top_k]
        token_dict = {
            int(token_id): Logprob(logprob=value, rank=rank + 1)
            for rank, (value, token_id) in enumerate(zip(values, ids))
        }
        if token_ids is not None and token_ids[index] not in token_dict:
            token_dict[token_ids[index]] = Logprob(logprob=row[0], rank=int(row[1]))
        results.append(token_dict)
    return results


def compute_prompt_logprobs(
    k_prompt_logprobs: int,
    context_logits: Optional[torch.Tensor],
    prompt_token_ids: Optional[list[int]] = None,
    simple_prompt_logprobs: bool = False,
) -> LogProbsResult:
    """Compute prompt token logprobs from context logits.

    When ``simple_prompt_logprobs`` is True and ``k_prompt_logprobs`` is 0,
    return a flat list of sampled-token logprobs instead of per-token dicts.
    """
    if k_prompt_logprobs is None or context_logits is None:
        return LogProbsResult(prompt=None)
    reduced = reduce_prompt_logprobs(context_logits, prompt_token_ids, k_prompt_logprobs)
    tokens = None if prompt_token_ids is None else prompt_token_ids[: reduced.shape[0]]
    return LogProbsResult(
        prompt=prompt_logprobs_from_reduced(
            reduced, tokens, k_prompt_logprobs, simple_prompt_logprobs
        )
    )
