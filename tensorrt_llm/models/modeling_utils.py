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
"""Surviving TensorRT-free surface of ``tensorrt_llm.models``.

After the removal of the legacy TensorRT backend, the only symbols retained
consumers (the LLM API and benchmarking utilities) need from this module are the
backend-agnostic configuration types ``PretrainedConfig`` and
``SpeculativeDecodingMode`` plus the quantization-config re-exports. The legacy
TensorRT model base classes (``PretrainedModel``, ``DecoderModelForCausalLM``,
the per-architecture implementations, and the engine-build helpers) have been
removed along with the TensorRT backend.
"""

import argparse
import copy
import dataclasses
import json
import os
from enum import IntFlag, auto
from typing import TYPE_CHECKING, Generator, Optional, Union

from ..bindings.executor import RuntimeDefaults
from ..functional_enums import PositionEmbeddingType
from ..logger import logger
from ..mapping import Mapping
from ..quantization.mode import QuantAlgo
# QuantConfig and LayerQuantConfig live in the (TensorRT-free)
# ``tensorrt_llm.quantization`` package; re-exported here for backward
# compatibility with existing import sites.
from ..quantization.quant_config import LayerQuantConfig, QuantConfig

__all__ = [
    'PretrainedConfig',
    'SpeculativeDecodingMode',
    'QuantConfig',
    'LayerQuantConfig',
    'QuantAlgo',
]


@dataclasses.dataclass(kw_only=True, frozen=True)
class Gemma2ConfigGroup:
    query_pre_attn_scalar: int
    final_logit_softcapping: Optional[float]
    attn_logit_softcapping: Optional[float]

    @classmethod
    def keys(cls):
        return {f.name for f in dataclasses.fields(cls)}


@dataclasses.dataclass(kw_only=True, frozen=True)
class Gemma3ConfigGroup:
    query_pre_attn_scalar: float
    final_logit_softcapping: Optional[float]
    _sliding_window_pattern: int
    rope_local_base_freq: int
    sliding_window: int

    @classmethod
    def keys(cls):
        return {f.name for f in dataclasses.fields(cls)}


if TYPE_CHECKING:
    from typing import Type, TypeVar

    from typing_extensions import Self

    ConfigGroups = Union[Gemma2ConfigGroup, Gemma3ConfigGroup]
    """Groupings of config where, if one of said properties exists, we assume all of the properties exist (even if they are `None`)"""
    CG = TypeVar("CG", bound=ConfigGroups)

    RuntimeDefaultsIn = Optional[Union[RuntimeDefaults, dict]]


class SpeculativeDecodingMode(IntFlag):
    # [WARNING] KEEP BELOW DEFINITION IN SYNC WITH cpp/tensorrt_llm/runtime/speculativeDecodingMode.h
    NONE = auto()
    DRAFT_TOKENS_EXTERNAL = auto()
    MEDUSA = auto()
    LOOKAHEAD_DECODING = auto()
    EXPLICIT_DRAFT_TOKENS = auto()
    EAGLE = auto()
    NGRAM = auto()
    USER_PROVIDED = auto()
    SAVE_HIDDEN_STATES = auto()
    AUTO = auto()

    @staticmethod
    def from_arguments(args: argparse.Namespace):
        if args.speculative_decoding_mode is None:
            return SpeculativeDecodingMode.NONE
        elif args.speculative_decoding_mode == "draft_tokens_external":
            return SpeculativeDecodingMode.DRAFT_TOKENS_EXTERNAL
        elif args.speculative_decoding_mode == "medusa":
            return SpeculativeDecodingMode.MEDUSA
        elif args.speculative_decoding_mode == "lookahead_decoding":
            return SpeculativeDecodingMode.LOOKAHEAD_DECODING
        elif args.speculative_decoding_mode == "explicit_draft_tokens":
            return SpeculativeDecodingMode.EXPLICIT_DRAFT_TOKENS
        elif args.speculative_decoding_mode == "eagle":
            return SpeculativeDecodingMode.EAGLE
        elif args.speculative_decoding_mode == "ngram":
            return SpeculativeDecodingMode.NGRAM
        elif args.speculative_decoding_mode == "user_provided":
            return SpeculativeDecodingMode.USER_PROVIDED
        elif args.speculative_decoding_mode == "auto":
            return SpeculativeDecodingMode.AUTO
        elif args.speculative_decoding_mode == "save_hidden_states":
            return SpeculativeDecodingMode.SAVE_HIDDEN_STATES
        else:
            assert False, "Unknown speculative_decoding_mode " + args.speculative_decoding_mode


class PretrainedConfig:

    def __init__(self,
                 *,
                 architecture: str,
                 dtype: str,
                 hidden_size: int,
                 num_hidden_layers: int,
                 num_attention_heads: int,
                 vocab_size: Optional[int] = None,
                 hidden_act: str = 'gelu',
                 logits_dtype: str = 'float32',
                 norm_epsilon: float = 1e-5,
                 position_embedding_type: Union[
                     PositionEmbeddingType,
                     str] = PositionEmbeddingType.learned_absolute,
                 max_position_embeddings: Optional[int] = None,
                 rotary_embedding_dim: Optional[int] = None,
                 num_key_value_heads: Optional[int] = None,
                 intermediate_size: Optional[int] = None,
                 mapping: Optional[Union[Mapping, dict]] = None,
                 quantization: Optional[Union[QuantConfig, dict]] = None,
                 use_parallel_embedding: bool = False,
                 embedding_sharding_dim: int = 0,
                 head_size: Optional[int] = None,
                 qk_layernorm: bool = False,
                 runtime_defaults: "RuntimeDefaultsIn" = None,
                 **kwargs):
        self.architecture = architecture
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act

        self.logits_dtype = logits_dtype
        self.norm_epsilon = norm_epsilon

        self.runtime_defaults = self.create_runtime_defaults(runtime_defaults)

        if isinstance(position_embedding_type, str):
            position_embedding_type = PositionEmbeddingType.from_string(
                position_embedding_type)
        assert isinstance(position_embedding_type, PositionEmbeddingType)
        self.position_embedding_type = position_embedding_type

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        if intermediate_size is None:
            intermediate_size = hidden_size * 4
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings

        if mapping is None:
            mapping = Mapping()
        elif isinstance(mapping, dict):
            mapping = Mapping.from_dict(mapping)
        assert isinstance(mapping, Mapping)
        self.mapping = mapping

        if quantization is None:
            quantization = QuantConfig()
        elif isinstance(quantization, dict):
            quantization = QuantConfig(**quantization)
        assert isinstance(quantization, (QuantConfig, LayerQuantConfig))
        self.quantization = quantization

        self.use_parallel_embedding = use_parallel_embedding
        self.embedding_sharding_dim = embedding_sharding_dim

        if head_size is None:
            head_size = hidden_size // num_attention_heads
        self.head_size = head_size
        self.qk_layernorm = qk_layernorm

        if rotary_embedding_dim is None:
            rotary_embedding_percentage = kwargs.get('rotary_pct', 1.0)
            rotary_embedding_dim = kwargs.get(
                'rotary_dim', int(head_size * rotary_embedding_percentage))
        self.rotary_embedding_dim = rotary_embedding_dim

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
                logger.warning(
                    f"Implicitly setting {self.__class__.__name__}.{key} = {value}"
                )
            except AttributeError as err:
                raise err

    @staticmethod
    def create_runtime_defaults(
            defaults: "RuntimeDefaultsIn" = None) -> Optional[RuntimeDefaults]:
        if isinstance(defaults, dict):
            return RuntimeDefaults(**defaults)
        return defaults

    @property
    def kv_dtype(self):
        # TODO: need to align the kv dtype
        # now assume the kv cache is for all layers
        if self.quant_mode.has_int8_kv_cache():
            return 'int8'
        elif self.quant_mode.has_fp8_kv_cache():
            return 'fp8'
        elif self.quant_mode.has_fp4_kv_cache():
            return 'fp4'
        else:
            return self.dtype

    def set_if_not_exist(self, key, value):
        if not hasattr(self, key):
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, config: dict):
        # Maybe we need AutoConfig for this
        from . import MODEL_MAP
        model_cls = MODEL_MAP[config['architecture']]
        config_cls = getattr(model_cls, 'config_class', cls)
        return config_cls(**config)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)

        output['position_embedding_type'] = str(self.position_embedding_type)
        output['mapping'] = self.mapping.to_dict()
        output['mapping'].pop('rank')
        output['quantization'] = self.quantization.model_dump()

        return output

    @classmethod
    def from_json_file(cls, config_file: str):
        with open(config_file) as f:
            config = json.load(f)
        obj = cls.from_dict(config)
        if obj.quantization.quant_algo == QuantAlgo.MIXED_PRECISION:
            try:
                layer_config_path = str(config_file).replace(
                    'config.json', 'quant_cfg.json')
                obj.to_layer_quant_config(layer_config_path)
            except Exception as e:
                raise RuntimeError(
                    f"Encounter error '{e}' for read quantization config '{layer_config_path}'"
                )
        return obj

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str):
        return cls.from_json_file(os.path.join(ckpt_dir, 'config.json'))

    def to_json_file(self, config_file: str):
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def to_layer_quant_config(self, config_file: str):
        with open(config_file) as f:
            config = json.load(f)

        if self.architecture == "MixtralForCausalLM":
            for layer_name in list(config["quantized_layers"].keys()):
                quant_cfg = config["quantized_layers"][layer_name]
                if "mlp.fc" in layer_name or "mlp.proj" in layer_name:
                    moe_name, _ = layer_name.rsplit('.', 1)
                    if moe_name not in config["quantized_layers"]:
                        config["quantized_layers"][moe_name] = quant_cfg
                    else:
                        assert quant_cfg == config["quantized_layers"][
                            moe_name], "MoE module needs to have the same quantization format for non-router sub-modules"

        self.quantization = LayerQuantConfig.model_validate(config)

    @property
    def quant_mode(self):
        return self.quantization.quant_mode

    @property
    def quant_algo(self):
        return self.quantization.quant_algo

    def _get_quant_cfg(self, module_name: str):
        return self.quantization._get_quant_cfg(module_name)

    def set_rank(self, rank: int):
        self.mapping.rank = rank

    def get_config_group(self, group_cls: "Type[CG]") -> "CG":
        cfg = {k: v for k, v in self.to_dict().items() if k in group_cls.keys()}
        return group_cls(**cfg)

    def has_config_group(self, group_cls: "Type[CG]") -> "bool":
        return all(hasattr(self, key) for key in group_cls.keys())

    def for_each_rank(self) -> "Generator[Self, None, None]":
        for rank in range(self.mapping.world_size):
            config_copy = copy.deepcopy(self)
            config_copy.set_rank(rank)
            yield config_copy
