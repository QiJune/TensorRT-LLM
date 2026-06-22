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
"""Serializable quantization configuration classes shared across backends.

These classes live in the (TensorRT-free) ``quantization`` package so that the
PyTorch backend, the VisualGen API, and the LLM API can depend on them without
importing the legacy TensorRT model implementations.
"""

import fnmatch
import re
from functools import cached_property
from typing import Dict, List, Optional

from pydantic import Field, PrivateAttr

from .._utils import QuantModeWrapper
from ..llmapi.utils import StrictBaseModel
from .mode import (KV_CACHE_QUANT_ALGO_LIST, QUANT_ALGO_LIST,
                   W8A8_SQ_PLUGIN_LIST, QuantAlgo, QuantMode)


class QuantConfig(StrictBaseModel):
    """Serializable quantization configuration class, part of the PretrainedConfig."""

    quant_algo: Optional[QuantAlgo] = Field(
        default=None,
        description="Quantization algorithm.",
        json_schema_extra={"telemetry": True})
    kv_cache_quant_algo: Optional[QuantAlgo] = Field(
        default=None, description="KV cache quantization algorithm.")
    group_size: Optional[int] = Field(
        default=128, description="Group size for group-wise quantization.")
    smoothquant_val: float = Field(
        default=0.5,
        description="Smoothing parameter alpha used in smooth quant.")
    clamp_val: Optional[List[float]] = Field(
        default=None,
        description="Clamp values used in FP8 rowwise quantization.")
    use_meta_recipe: bool = Field(
        default=False,
        description="Whether to use Meta's recipe for FP8 rowwise quantization."
    )
    has_zero_point: bool = Field(
        default=False,
        description="Whether to use zero point for quantization.")
    pre_quant_scale: bool = Field(
        default=False,
        description="Whether to use pre-quant scale for quantization.")
    exclude_modules: Optional[List[str]] = Field(
        default=None,
        description="Module name patterns that are skipped in quantization.")
    mamba_ssm_cache_dtype: Optional[str] = Field(
        default=None, description="Data type for mamba SSM cache.")
    mamba_ssm_stochastic_rounding: bool = Field(
        default=False,
        description=
        "Enable stochastic rounding for Mamba SSM state updates. Requires fp16 cache."
    )
    mamba_ssm_philox_rounds: int = Field(
        default=10,
        ge=1,
        description=
        "Number of Philox rounds for stochastic rounding PRNG. Higher values give better randomness."
    )

    @cached_property
    def quant_mode(self) -> QuantModeWrapper:
        quant_mode_list = [
            QuantMode.from_quant_algo(
                self.quant_algo,
                self.kv_cache_quant_algo,
            )
        ]
        return QuantModeWrapper(quant_mode_list)

    @cached_property
    def layer_quant_mode(self) -> QuantMode:
        return QuantMode.from_quant_algo(
            self.quant_algo,
            self.kv_cache_quant_algo,
        )

    @property
    def _use_plugin_sq(self):
        return self.quant_algo in W8A8_SQ_PLUGIN_LIST

    @property
    def _requires_calibration(self):
        return self.quant_algo in (set(QUANT_ALGO_LIST) - {
            QuantAlgo.W8A16, QuantAlgo.W4A16,
            QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN
        }) or self.kv_cache_quant_algo in KV_CACHE_QUANT_ALGO_LIST

    @property
    def _requires_modelopt_quantization(self):
        if self.quant_algo in [
                QuantAlgo.NVFP4, QuantAlgo.FP8, QuantAlgo.W4A16_AWQ,
                QuantAlgo.W4A8_AWQ, QuantAlgo.W8A8_SQ_PER_CHANNEL,
                QuantAlgo.MIXED_PRECISION
        ]:
            return True
        elif self.quant_algo is None and self.kv_cache_quant_algo == QuantAlgo.FP8:
            return True
        else:
            return False

    def _get_quant_cfg(self, module_name=None):
        if (module_name is not None
                and self.is_module_excluded_from_quantization(module_name)):
            return LayerQuantConfig(quant_algo=None, quantized_layers={})
        return self

    def _get_modelopt_qformat(self):
        algo_to_modelopt_map = {
            QuantAlgo.W8A16: "int8_wo",
            QuantAlgo.W4A16: "int4_wo",
            QuantAlgo.NVFP4: "nvfp4",
            QuantAlgo.FP8: "fp8",
            QuantAlgo.W4A16_AWQ: "int4_awq",
            QuantAlgo.W4A8_AWQ: "w4a8_awq",
            QuantAlgo.W8A8_SQ_PER_CHANNEL: "int8_sq",
        }
        assert self.quant_algo != QuantAlgo.MIXED_PRECISION, f"We don't support mixed precision in QuantConfig"
        if self.quant_algo is not None:
            assert self.quant_algo in algo_to_modelopt_map, f"We don't use Modelopt for quantization algorithm {self.quant_algo}, you probably shall not call this"
            return algo_to_modelopt_map[self.quant_algo]
        else:
            return 'full_prec'

    def _get_modelopt_kv_cache_dtype(self):
        algo_to_modelopt_map = {
            QuantAlgo.FP8: 'fp8',
            QuantAlgo.INT8: 'int8',
        }
        if self.kv_cache_quant_algo is not None:
            assert self.kv_cache_quant_algo in algo_to_modelopt_map, f"We don't use Modelopt for quantization algorithm {self.kv_cache_quant_algo}, you probably shall not call this"
            return algo_to_modelopt_map[self.kv_cache_quant_algo]
        else:
            return None

    def is_module_excluded_from_quantization(self, name: str) -> bool:
        """Check if the module is excluded from quantization.

        A module is excluded if its own name or any ancestor (split on
        ``.``) matches an entry in ``exclude_modules`` via ``fnmatch`` or
        a ``re:`` prefixed regex. The ancestor walk means listing a parent
        module (without a glob suffix) implicitly excludes all of its
        children.

        Args:
            name (str): The name of the module.

        Returns:
            bool: True if the module is excluded from quantization, False otherwise.
        """
        if self.exclude_modules is None:
            return False
        candidate = name
        while True:
            for exclude_module in self.exclude_modules:
                if exclude_module.startswith("re:"):
                    if re.fullmatch(exclude_module[3:], candidate):
                        return True
                elif fnmatch.fnmatchcase(candidate, exclude_module):
                    return True
            if '.' not in candidate:
                return False
            candidate = candidate.rsplit('.', 1)[0]

    # NOTE: this is kept for backward compatibility with external libraries (e.g., modelopt).
    # For new code, prefer directly using QuantConfig(**config) instead.
    @classmethod
    def from_dict(cls, config: dict) -> 'QuantConfig':
        """Create a QuantConfig instance from a dict.

        Args:
            config (dict): The dict used to create QuantConfig.

        Returns:
            tensorrt_llm.models.modeling_utils.QuantConfig: The QuantConfig created from dict.
        """
        obj = cls(**config)
        return obj


class LayerQuantConfig(StrictBaseModel):
    """Configuration for layer-wise/mixed-precision quantization."""

    quant_algo: Optional[QuantAlgo] = Field(
        default=None,
        description="Quantization algorithm (typically MIXED_PRECISION).")
    kv_cache_quant_algo: Optional[QuantAlgo] = Field(
        default=None, description="KV cache quantization algorithm.")
    quantized_layers: Dict[str, QuantConfig] = Field(
        default_factory=dict,
        description="Per-layer quantization configurations.")

    # Computed cache, not serialized
    _auto_quant_mode: Dict[str, QuantMode] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context) -> None:
        """Compute auto_quant_mode after initialization."""
        self._auto_quant_mode = {}
        if self.quantized_layers:
            for name, layer_config in self.quantized_layers.items():
                self._auto_quant_mode[name] = QuantMode.from_quant_algo(
                    layer_config.quant_algo,
                    self.kv_cache_quant_algo,
                )

    @property
    def auto_quant_mode(self) -> Dict[str, QuantMode]:
        return self._auto_quant_mode

    @property
    def quant_mode(self) -> QuantModeWrapper:
        quant_mode_list = list(set(self._auto_quant_mode.values()))
        return QuantModeWrapper(quant_mode_list)

    def layer_quant_mode(self, layer_name) -> QuantMode:
        for name, quant_mode in self._auto_quant_mode.items():
            if fnmatch.fnmatch(layer_name, name):
                return quant_mode
        return QuantMode(0)

    @property
    def auto_quant_list(self) -> List[QuantAlgo]:
        if not self.quantized_layers:
            return []
        return list(set(lc.quant_algo for lc in self.quantized_layers.values()))

    def _get_quant_cfg(self, module_name) -> QuantConfig:
        for name, quant_cfg in self.quantized_layers.items():
            if fnmatch.fnmatch(module_name, name):
                return quant_cfg
        return QuantConfig()

    def _get_modelopt_qformat(self):
        algo_to_modelopt_map = {
            QuantAlgo.NVFP4: "nvfp4",
            QuantAlgo.FP8: "fp8",
            QuantAlgo.W4A16_AWQ: "int4_awq",
            QuantAlgo.W4A8_AWQ: "w4a8_awq",
            QuantAlgo.W8A8_SQ_PER_CHANNEL: "int8_sq",
        }
        assert self.quant_algo == QuantAlgo.MIXED_PRECISION, \
            "We only support mixed precision quantization in LayerQuantConfig"
        autoq_format = ','.join(
            [algo_to_modelopt_map[item] for item in self.auto_quant_list])
        return autoq_format

    # NOTE: this is kept for backward compatibility with external libraries (e.g., modelopt).
    # For new code, prefer directly using LayerQuantConfig(**config) instead.
    @classmethod
    def from_dict(cls, config: dict) -> 'LayerQuantConfig':
        return cls(**config)
