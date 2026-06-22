# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Surviving TensorRT-free surface of the ``tensorrt_llm.models`` package.

The legacy TensorRT model implementations and engine-build helpers were removed
along with the TensorRT backend. Only the backend-agnostic config and
quantization-config symbols that the LLM API and benchmarking utilities still
import are retained here. ``MODEL_MAP`` is kept (empty) so the ``automodel``
discovery helpers remain importable; they raise a clear ``NotImplementedError``
when asked to build a (now-removed) TensorRT model.
"""

from ..quantization.quant_config import LayerQuantConfig, QuantConfig
from .modeling_utils import PretrainedConfig, QuantAlgo, SpeculativeDecodingMode

# The legacy TensorRT architecture registry is intentionally empty: the PyTorch
# backend resolves model classes via ``tensorrt_llm._torch.models`` instead.
MODEL_MAP = {}

__all__ = [
    'PretrainedConfig',
    'SpeculativeDecodingMode',
    'QuantConfig',
    'LayerQuantConfig',
    'QuantAlgo',
    'MODEL_MAP',
]
