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
"""Identity, FutureWarning and old-path-import checks of the executor's kept old paths.

The old paths kept are the ones code outside the repository imports: the names the
``tensorrt_llm.executor`` package exported, ``executor.result`` (Dynamo, see
``tests/unittest/dynamo/test_imports.py``) and ``executor.postprocessor_hook`` (shown in the
documentation).
"""

from pathlib import Path

import pytest
from test_common.serve.shims import ShimRow, check_identity, check_warnings, find_old_path_imports

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[3]

_CLIENT = "tensorrt_llm.llmapi.engine_client"
_RESULT_NAMES = (
    "CompletionOutput",
    "DetokenizedGenerationResultBase",
    "GenerationResult",
    "GenerationResultBase",
    "IterationResult",
)
_LOGPROB_NAMES = (
    "LogProbsResult",
    "Logprob",
    "SimpleTokenLogprobs",
    "TokenLogprobs",
    "compute_prompt_logprobs",
    "prompt_logprobs_from_reduced",
    "reduce_prompt_logprobs",
)
_HOOK_NAMES = (
    "PostProcessorHook",
    "PostProcessorHookAction",
    "PostProcessorHookChunk",
    "PostProcessorHookVerdict",
    "apply_post_processor_hook",
    "emit",
    "load_post_processor_hook",
    "suppress",
    "terminate",
)

# (old module, new module, names)
_MOVED = (
    ("tensorrt_llm.executor", f"{_CLIENT}.executor", ("CppExecutorError", "GenerationExecutor")),
    ("tensorrt_llm.executor", f"{_CLIENT}.postproc_worker", ("PostprocWorker",)),
    (
        "tensorrt_llm.executor",
        f"{_CLIENT}.proxy",
        ("GenerationExecutorFrontendProxy", "GenerationExecutorProxy"),
    ),
    ("tensorrt_llm.executor", f"{_CLIENT}.result", _RESULT_NAMES),
    (
        "tensorrt_llm.executor.result",
        f"{_CLIENT}.result",
        _RESULT_NAMES + ("PostprocWorker", "ResponseWrapper", "get_metrics_dict"),
    ),
    ("tensorrt_llm.executor.result", "tensorrt_llm.executor.logprobs", _LOGPROB_NAMES),
    ("tensorrt_llm.executor.postprocessor_hook", f"{_CLIENT}.postprocessor_hook", _HOOK_NAMES),
)

SHIM_ROWS = tuple(
    ShimRow(old, name, new, retire_in="1.6.0") for old, new, names in _MOVED for name in names
)


def test_shim_identity():
    assert check_identity(SHIM_ROWS) == []


def test_shim_warnings():
    assert check_warnings(SHIM_ROWS) == []


def test_no_old_path_imports():
    assert find_old_path_imports(SHIM_ROWS, REPO_ROOT) == []
