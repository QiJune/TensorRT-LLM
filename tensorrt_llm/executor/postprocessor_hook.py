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
"""Compatibility shim for ``tensorrt_llm.executor.postprocessor_hook``.

Will be removed in 1.6.0; import from ``tensorrt_llm.llmapi.engine_client.postprocessor_hook``.

DO NOT ADD ANYTHING TO THIS FILE.
"""

import warnings

from tensorrt_llm.llmapi.engine_client.postprocessor_hook import (
    PostProcessorHook,
    PostProcessorHookAction,
    PostProcessorHookChunk,
    PostProcessorHookVerdict,
    apply_post_processor_hook,
    emit,
    load_post_processor_hook,
    suppress,
    terminate,
)

warnings.warn(
    "tensorrt_llm.executor.postprocessor_hook has moved to "
    "tensorrt_llm.llmapi.engine_client.postprocessor_hook and will be removed in 1.6.0.",
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    "PostProcessorHook",
    "PostProcessorHookAction",
    "PostProcessorHookChunk",
    "PostProcessorHookVerdict",
    "apply_post_processor_hook",
    "emit",
    "load_post_processor_hook",
    "suppress",
    "terminate",
]
