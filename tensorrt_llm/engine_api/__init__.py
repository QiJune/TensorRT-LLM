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
"""Narrow, language-neutral interface between the serving frontend and the engine.

Modules in this package must stay import-light: no ``torch``, no
``tensorrt_llm.bindings``, no model machinery at module import time. The
detached serving frontend imports this package without a GPU stack present.
Engine-side implementations that need the heavy runtime live in modules that
are only imported on the engine side (e.g. the legacy executor adapter).
"""

from tensorrt_llm.engine_api.contracts import (
    ContractViolationError,
    EngineClient,
    EngineClientError,
    EngineError,
    EngineErrorCode,
    EngineEvent,
    EngineRequest,
    EventOrderingChecker,
    FinishReason,
    FrontendOutputConfig,
    GenerationClient,
    ProtocolViolationError,
    PythonExtension,
    RequestHandle,
    RuntimeControl,
    RuntimeSamplingConfig,
    TensorAuxiliaryPayload,
    TerminalKind,
    TokenLogprob,
    validate_plain_data,
)

__all__ = [
    "ContractViolationError",
    "EngineClient",
    "EngineClientError",
    "EngineError",
    "EngineErrorCode",
    "EngineEvent",
    "EngineRequest",
    "EventOrderingChecker",
    "FinishReason",
    "FrontendOutputConfig",
    "GenerationClient",
    "ProtocolViolationError",
    "PythonExtension",
    "RequestHandle",
    "RuntimeControl",
    "RuntimeSamplingConfig",
    "TensorAuxiliaryPayload",
    "TerminalKind",
    "TokenLogprob",
    "validate_plain_data",
]
