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
"""Experimental typed engine-client contract (gated by TLLM_EXPERIMENTAL_ENGINE_CLIENT).

Internal package: not part of the public LLM API surface. See
``ENGINE_CONTRACT.md`` for the contract specification, scope matrix, and
divergence notes.
"""

from .codec import (CodecError, DecodeError, EncodeError, decode,  # noqa: F401
                    encode)
from .contract import (ENGINE_CONTRACT_VERSION, ContractConstructionError,  # noqa: F401
                       ContractError, EngineCapabilities, EngineHealth,
                       EngineRequest, EngineSamplingConfig, ErrorFrame,
                       FrontendModelContext, FrontendOutputConfig,
                       GuidedDecodingSpec, IterationStatsBatch,
                       KvCacheEventsBatch, OutputFrame, RequestComplete,
                       Terminal, TokenDelta, TokenizerSpec,
                       validate_no_callables)

__all__ = [
    "ENGINE_CONTRACT_VERSION",
    "ContractError",
    "ContractConstructionError",
    "CodecError",
    "EncodeError",
    "DecodeError",
    "encode",
    "decode",
    "EngineCapabilities",
    "EngineRequest",
    "EngineSamplingConfig",
    "GuidedDecodingSpec",
    "TokenDelta",
    "Terminal",
    "RequestComplete",
    "ErrorFrame",
    "OutputFrame",
    "FrontendOutputConfig",
    "TokenizerSpec",
    "FrontendModelContext",
    "EngineHealth",
    "IterationStatsBatch",
    "KvCacheEventsBatch",
    "validate_no_callables",
]
