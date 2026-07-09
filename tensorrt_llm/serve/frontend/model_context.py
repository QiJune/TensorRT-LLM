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
"""Data-only model context for frontends detached from the engine.

Everything a detached frontend knows about the model comes from the engine
handshake: this read-only structure carries it. The frontend builds its
tokenizer and chat templates from this context alone — it never constructs
a model, an executor, or a proxy, and never reaches into engine-side
objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(slots=True, frozen=True)
class FrontendModelContext:
    """Read-only model context built from the engine handshake.

    Args:
        model: Model name/label echoed in responses.
        tokenizer_dir: Where the frontend loads its tokenizer from (a local
            path or HF id the frontend can resolve).
        max_seq_len: Engine sequence-length limit, if advertised.
        reasoning_parser: Reasoning-parser name configured engine-side.
        return_perf_metrics: Whether the engine collects per-request
            performance metrics.
        stream_interval: Streaming detokenization interval.
        model_type: HF ``model_type`` of the served model (e.g. ``gpt_oss``),
            so the frontend can detect chat paths it cannot serve (Harmony).
        trust_remote_code: Whether the engine loaded the tokenizer with
            ``trust_remote_code``; the frontend must match to load the same
            tokenizer and produce identical token ids.
        tokenizer_mode: Engine tokenizer mode (``auto``/``slow``); controls
            fast-vs-slow tokenizer selection.
        capabilities: The engine's advertised capability set.
    """

    model: str
    tokenizer_dir: Optional[str] = None
    max_seq_len: Optional[int] = None
    reasoning_parser: Optional[str] = None
    return_perf_metrics: bool = False
    stream_interval: int = 1
    model_type: Optional[str] = None
    trust_remote_code: bool = False
    tokenizer_mode: str = "auto"
    capabilities: Optional[dict] = None

    @classmethod
    def from_handshake(
        cls, model_context: dict[str, Any], capabilities: Optional[dict] = None
    ) -> "FrontendModelContext":
        return cls(
            model=model_context.get("model") or "unknown",
            tokenizer_dir=model_context.get("tokenizer_dir"),
            max_seq_len=model_context.get("max_seq_len"),
            reasoning_parser=model_context.get("reasoning_parser"),
            return_perf_metrics=bool(model_context.get("return_perf_metrics", False)),
            stream_interval=int(model_context.get("stream_interval") or 1),
            model_type=model_context.get("model_type"),
            trust_remote_code=bool(model_context.get("trust_remote_code", False)),
            tokenizer_mode=model_context.get("tokenizer_mode") or "auto",
            capabilities=capabilities,
        )

    @property
    def is_harmony_model(self) -> bool:
        """Whether the served model uses the Harmony chat format (gpt_oss)."""
        return self.model_type == "gpt_oss"

    def build_tokenizer(self) -> Any:
        """Load the frontend tokenizer from the context alone (no runtime).

        Mirrors the engine's tokenizer init (``trust_remote_code`` and
        fast/slow mode) so the frontend produces identical token ids.
        """
        if self.tokenizer_dir is None:
            raise ValueError(
                "the engine handshake advertised no tokenizer source; the detached "
                "frontend cannot detokenize"
            )
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            self.tokenizer_dir,
            trust_remote_code=self.trust_remote_code,
            use_fast=self.tokenizer_mode != "slow",
        )
