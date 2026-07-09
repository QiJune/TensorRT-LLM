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
"""Eligibility predicate for the engine-client pipeline.

A request is served by the engine-client pipeline only when **all** of the
following hold; everything else is served by the in-process path (silent
fallback with a debug log when co-located, typed capability error when the
frontend runs detached from the engine):

Deployment conditions (evaluated once per server/LLM instance):

- PyTorch backend.
- Default MPI/IPC orchestration (in-process worker or IPC proxy); Ray and
  RPC orchestrators are not wired to the pipeline.
- ``num_postprocess_workers == 0`` (postprocess worker processes carry
  formatter callables to the engine side, which the pipeline forbids).

Per-request conditions:

- Endpoint is chat or completions (or the LLM API text path). Harmony and
  the Responses API keep their historical handlers.
- No multimodal inputs.
- No guided decoding.
- No Python logits processor of any kind (including implicitly attached
  ones, e.g. the thinking-budget processor).
- No LoRA / prompt-adapter request.
- No disaggregated-serving parameters (disagg role servers keep the
  historical path).
- No tensor-carrying sampling fields (``embedding_bias`` /
  ``logit_bias``) and no context/generation-logits returns.
- Completions: a single prompt per request.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "EligibilityResult",
    "PipelineDeploymentMode",
    "check_deployment",
    "check_request",
    "check_sampling_params",
]


class PipelineDeploymentMode(str, enum.Enum):
    """How the frontend runs relative to the engine."""

    COLOCATED = "colocated"
    DETACHED = "detached"


@dataclass(slots=True, frozen=True)
class EligibilityResult:
    """Outcome of an eligibility check; falsy when ineligible."""

    eligible: bool
    reason: Optional[str] = None

    def __bool__(self) -> bool:
        return self.eligible


_ELIGIBLE = EligibilityResult(True)


def _ineligible(reason: str) -> EligibilityResult:
    return EligibilityResult(False, reason)


def check_deployment(llm_args: Any) -> EligibilityResult:
    """Evaluate the deployment-level half of the predicate against LLM args."""
    backend = getattr(llm_args, "backend", None)
    if backend != "pytorch":
        return _ineligible(f"backend {backend!r} is not the pytorch backend")
    orchestrator_type = getattr(llm_args, "orchestrator_type", None)
    if orchestrator_type is not None:
        return _ineligible(
            f"orchestrator_type {orchestrator_type!r} is not the default MPI/IPC orchestration"
        )
    num_postprocess_workers = getattr(llm_args, "num_postprocess_workers", 0) or 0
    if num_postprocess_workers > 0:
        return _ineligible(
            f"num_postprocess_workers={num_postprocess_workers} requires the in-process path"
        )
    return _ELIGIBLE


def check_sampling_params(sampling_params: Any) -> EligibilityResult:
    """Evaluate sampling-params conditions shared by every endpoint."""
    if getattr(sampling_params, "guided_decoding", None) is not None:
        return _ineligible("guided decoding is served by the in-process path")
    if getattr(sampling_params, "logits_processor", None) is not None:
        return _ineligible("Python logits processors are served by the in-process path")
    if getattr(sampling_params, "apply_batched_logits_processor", False):
        return _ineligible("batched logits processors are served by the in-process path")
    if getattr(sampling_params, "thinking_token_budget", None) is not None:
        return _ineligible(
            "thinking_token_budget attaches a logits processor; served by the in-process path"
        )
    if getattr(sampling_params, "embedding_bias", None) is not None:
        return _ineligible("embedding bias (logit_bias) tensors cannot cross the engine boundary")
    if getattr(sampling_params, "return_context_logits", False) or getattr(
        sampling_params, "return_generation_logits", False
    ):
        return _ineligible("logits returns are served by the in-process path")
    if getattr(sampling_params, "return_encoder_output", False):
        return _ineligible("encoder outputs are served by the in-process path")
    if getattr(sampling_params, "additional_model_outputs", None):
        return _ineligible("additional model outputs are served by the in-process path")
    if getattr(sampling_params, "lookahead_config", None) is not None:
        return _ineligible("lookahead decoding config is served by the in-process path")
    return _ELIGIBLE


def check_request(
    sampling_params: Any,
    *,
    endpoint: str,
    has_multimodal: bool = False,
    lora_request: Any = None,
    prompt_adapter_request: Any = None,
    disaggregated_params: Any = None,
    num_prompts: int = 1,
) -> EligibilityResult:
    """Evaluate the per-request half of the predicate.

    Args:
        sampling_params: The request's SamplingParams (post request parsing).
        endpoint: One of ``"chat"``, ``"completions"``, ``"llm_api_text"``.
            Harmony and the Responses API never reach this check; their
            handlers are separate routes that stay on the historical path.
        has_multimodal: Whether the request carries any multimodal content.
        lora_request: LoRA adapter request, if any.
        prompt_adapter_request: Prompt-adapter request, if any.
        disaggregated_params: Disaggregated-serving parameters, if any.
        num_prompts: Number of prompts in the request (completions may batch).
    """
    if endpoint not in ("chat", "completions", "llm_api_text"):
        return _ineligible(f"endpoint {endpoint!r} is served by the in-process path")
    if has_multimodal:
        return _ineligible("multimodal requests are served by the in-process path")
    if lora_request is not None:
        return _ineligible("LoRA requests are served by the in-process path")
    if prompt_adapter_request is not None:
        return _ineligible("prompt-adapter requests are served by the in-process path")
    if disaggregated_params is not None:
        return _ineligible("disaggregated-serving requests are served by the in-process path")
    if num_prompts != 1:
        return _ineligible("multi-prompt requests are served by the in-process path")
    return check_sampling_params(sampling_params)
