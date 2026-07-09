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
"""Frontend tier of the engine-boundary serving path.

Owns everything between raw token-level engine events and client-facing
responses: response assembly (detokenization, stop handling, diff tracking)
and per-endpoint OpenAI formatting. Formatters are frontend-internal
functions selected per endpoint — they are never request payload and never
cross to the engine.
"""

from tensorrt_llm.serve.frontend.request_processor import FrontendProcessor, ProcessedInput
from tensorrt_llm.serve.frontend.response_assembler import (
    AssembledRequestView,
    AssembledSequenceOutput,
    FrontendResponseAssembler,
)

__all__ = [
    "AssembledRequestView",
    "AssembledSequenceOutput",
    "FrontendProcessor",
    "FrontendResponseAssembler",
    "ProcessedInput",
]
