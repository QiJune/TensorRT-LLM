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
"""Compatibility shim for ``tensorrt_llm.executor.result``.

The results moved to ``tensorrt_llm.llmapi.engine_client.result``, the log-probability types to
``tensorrt_llm.executor.logprobs`` and the response wrapper to ``tensorrt_llm.executor.utils``.
Each name read here warns once with a ``FutureWarning``; the module is removed in 1.6.0.

DO NOT ADD ANYTHING TO THIS FILE.
"""

import importlib
import warnings

_LAZY_NAMES = {
    "CompletionOutput": "tensorrt_llm.llmapi.engine_client.result",
    "DetokenizedGenerationResultBase": "tensorrt_llm.llmapi.engine_client.result",
    "GenerationResult": "tensorrt_llm.llmapi.engine_client.result",
    "GenerationResultBase": "tensorrt_llm.llmapi.engine_client.result",
    "IterationResult": "tensorrt_llm.llmapi.engine_client.result",
    "PostprocWorker": "tensorrt_llm.llmapi.engine_client.result",
    "LogProbsResult": "tensorrt_llm.executor.logprobs",
    "Logprob": "tensorrt_llm.executor.logprobs",
    "SimpleTokenLogprobs": "tensorrt_llm.executor.logprobs",
    "TokenLogprobs": "tensorrt_llm.executor.logprobs",
    "compute_prompt_logprobs": "tensorrt_llm.executor.logprobs",
    "prompt_logprobs_from_reduced": "tensorrt_llm.executor.logprobs",
    "reduce_prompt_logprobs": "tensorrt_llm.executor.logprobs",
}


def __getattr__(name):
    if name in _LAZY_NAMES:
        warnings.warn(
            f"tensorrt_llm.executor.result.{name} has moved to {_LAZY_NAMES[name]} "
            "and will be removed in 1.6.0.",
            FutureWarning,
            stacklevel=2,
        )
        value = getattr(importlib.import_module(_LAZY_NAMES[name]), name)
        globals()[name] = value  # warn once: later reads skip __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
