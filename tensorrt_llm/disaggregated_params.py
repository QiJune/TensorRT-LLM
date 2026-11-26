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

import os
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from tensorrt_llm.bindings import executor as tllme

if TYPE_CHECKING:
    from tensorrt_llm.inputs.multimodal import MultimodalInputLayout


class DisaggScheduleStyle(IntEnum):
    CONTEXT_FIRST = 0
    GENERATION_FIRST = 1


@dataclass(slots=True, kw_only=True)
class DisaggregatedParams:
    """Disaggregated serving parameters.

    Args:
        request_type (str): The type of request ("context_only" | "generation_only" | "context_and_generation")
        first_gen_tokens (List[int]): The first tokens of the generation request
        ctx_request_id (int): The context request id
        opaque_state(bytes): Any additional state needing to be exchanged between context and gen instances
        draft_tokens (List[int]): The draft tokens of the generation request
        disagg_request_id (int): The disaggregated request id, if set, both context and generation requests will use it
         as underlying request id.
        first_gen_log_probs (List): The logprobs for first_gen_tokens, produced during prefill.
         Each entry is a list (one per beam) of either ``TokenLogprobs`` (``list[dict[int, Logprob]]``,
         default format) or ``SimpleTokenLogprobs`` (``list[float]``, simple format).
        first_gen_logits (List): The generation logits for first_gen_tokens, produced during prefill.
         Each entry is a torch.Tensor of shape [num_tokens, vocab_size] (one per beam/sequence).
        ctx_usage (Dict[str, Any]): The context usage payload to preserve exact
         usage accounting on the generation server.
        remote_tail_start (int): Prompt-token boundary transferred by a remote-tail
         context worker. The generation worker replays the suffix starting here.
        orig_elapsed_s (float): Seconds since the request arrived at the disagg orchestrator,
         measured when the generation request was dispatched. Used for GEN admission ordering.
        multimodal_embedding_handles (List[Dict[str, Any]]): The resulting multimodal embedding handles from ViT.
        multimodal_hashes (List[List[int]]): The multimodal hashes of each multimodal item in the request.
        multimodal_positions (List[int]): Prompt-token offset of each multimodal item.
        multimodal_lengths (List[int]): Prompt-token count of each multimodal item.
        multimodal_input_layout (MultimodalInputLayout): Exact encoder-side
         prompt and multimodal layout to preserve during E/P handoff.
    """

    request_type: Optional[str] = None
    # P-D Disaggregated Params
    first_gen_tokens: Optional[List[int]] = None
    first_gen_log_probs: Optional[List] = None
    first_gen_logits: Optional[List] = None
    ctx_request_id: Optional[int] = None
    opaque_state: Optional[bytes] = None
    draft_tokens: Optional[List[int]] = None
    # If disagg_request_id is set, both context and generation requests will use it as underlying request id.
    disagg_request_id: Optional[int] = None
    ctx_dp_rank: Optional[int] = None
    ctx_info_endpoint: Optional[str] = None
    schedule_style: Optional[DisaggScheduleStyle] = None
    ctx_usage: Optional[Dict[str, Any]] = None
    remote_tail_start: Optional[int] = None
    multimodal_positions: Optional[List[int]] = None
    multimodal_lengths: Optional[List[int]] = None
    orig_elapsed_s: Optional[float] = None

    # E-P Disaggregated Params
    multimodal_embedding_handles: Optional[List[Dict[str, Any]]] = (
        None  # multimodal embedding handles should be a list of cudaIPC handles for each mm_embedding
    )
    multimodal_hashes: Optional[List[List[int]]] = (
        None  # user provided mm hashes should be a list of 8 integers
    )
    mrope_position_ids_handle: Optional[Dict[str, Any]] = None
    mrope_position_deltas_handle: Optional[Dict[str, Any]] = None
    multimodal_input_layout: Optional["MultimodalInputLayout"] = None

    def get_context_phase_params(self) -> tllme.ContextPhaseParams:
        # Prefer disagg_request_id over ctx_request_id
        request_id = (
            self.disagg_request_id if self.disagg_request_id is not None else self.ctx_request_id
        )
        # `first_gen_tokens` is now required by bindings and cannot be None.
        first_gen_tokens = self.first_gen_tokens if self.first_gen_tokens is not None else []
        return tllme.ContextPhaseParams(
            first_gen_tokens,
            request_id,
            self.opaque_state,
            self.draft_tokens,
            self.ctx_dp_rank,
            self.ctx_info_endpoint,
        )

    def get_request_type(self) -> tllme.RequestType:
        if self.request_type == "context_only":
            return tllme.RequestType.REQUEST_TYPE_CONTEXT_ONLY
        elif self.request_type == "generation_only":
            return tllme.RequestType.REQUEST_TYPE_GENERATION_ONLY
        elif self.request_type == "context_and_generation":
            return tllme.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION
        else:
            raise ValueError(
                f"Unknown request type: {self.request_type}. Must be context_only, generation_only or "
                "context_and_generation"
            )

    def __post_init__(self):
        """Restore serialized layout and validate request and multimodal metadata."""
        if self.multimodal_input_layout is not None:
            # Defer the import because inputs and executor import these parameters.
            from tensorrt_llm.inputs.multimodal import MultimodalInputLayout

            if isinstance(self.multimodal_input_layout, dict):
                self.multimodal_input_layout = MultimodalInputLayout(**self.multimodal_input_layout)
            elif not isinstance(self.multimodal_input_layout, MultimodalInputLayout):
                raise TypeError("multimodal_input_layout must be MultimodalInputLayout")
        if (self.multimodal_positions is None) != (self.multimodal_lengths is None):
            raise ValueError(
                "multimodal_positions and multimodal_lengths must be provided together"
            )
        if self.multimodal_positions is not None:
            if len(self.multimodal_positions) != len(self.multimodal_lengths):
                raise ValueError(
                    "multimodal_positions and multimodal_lengths must have the same length"
                )
            if any(position < 0 for position in self.multimodal_positions):
                raise ValueError("multimodal_positions must be non-negative")
            if any(length <= 0 for length in self.multimodal_lengths):
                raise ValueError("multimodal_lengths must be positive")
        if self.request_type is not None:
            self.request_type = self.request_type.lower()
            if self.request_type not in [
                "context_only",
                "generation_only",
                "context_and_generation",
            ]:
                raise ValueError(
                    f"Unknown request type: {self.request_type}. Must be context_only, generation_only or "
                    "context_and_generation"
                )
        if self.multimodal_embedding_handles is not None:
            if self.multimodal_hashes is not None:
                # if mm hashes are provided, kvcache reuse can be enabled
                assert len(self.multimodal_embedding_handles) == len(self.multimodal_hashes), (
                    "multimodal_embedding_handles and multimodal_hashes must have the same length"
                )
                for mm_hash in self.multimodal_hashes:
                    assert isinstance(mm_hash, list), "mm_hash must be a list"
                    assert len(mm_hash) == 8, "mm_hash must be a list of 8 integers"
                    assert all(isinstance(x, int) for x in mm_hash), "mm_hash must contain integers"
            else:
                # if user did not provide mm embedding handles, kvcache reuse will be disabled
                assert len(self.multimodal_embedding_handles) > 0, (
                    "multimodal_embedding_handles must be provided"
                )
                vals = np.random.randint(
                    np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=8, dtype=np.int32
                ).tolist()
                self.multimodal_hashes = [vals] * len(self.multimodal_embedding_handles)


# Snowflake global disagg request id, 64-bit / positive int64 (MSB reserved 0):
#   [ 0 (1) | timestamp_ms (39) | node_id (8) | process_id (6) | counter (10) ]
# The (node_id, process_id) pair identifies a fleet worker process, so co-located
# workers never emit the same id in the same millisecond. See docs/source/
# advanced/disaggregated-service.md for the full disagg-request-id design.
DISAGG_TIMESTAMP_BITS = 39


DISAGG_NODE_ID_BITS = 8


DISAGG_PROCESS_ID_BITS = 6


DISAGG_COUNTER_BITS = 10


# Local ids [0, MIN_GLOBAL_ID) and global disagg ids [MIN_GLOBAL_ID, 2^63) are
# disjoint by construction so they never collide. Power of two (masked in
# get_local_request_id).
MIN_GLOBAL_ID = 1 << 40


# Consider GIL being removed in the future, use a lock to protect the counter
_global_disagg_request_id_lock = threading.Lock()


_global_disagg_request_id_counter = 0


def get_global_disagg_request_id(node_id: int, process_id: int = 0) -> int:
    """A snowflake global disagg request id (does not guarantee monotonicity).

    Layout: 0(1) | timestamp_ms(39) | node_id(8) | process_id(6) | counter(10).
    node_id identifies the node, process_id the fleet worker process on it -- the
    pair makes the id unique across co-located workers without any coordination.
    """
    global _global_disagg_request_id_lock
    global _global_disagg_request_id_counter

    NODE_ID_SPACE = 1 << DISAGG_NODE_ID_BITS
    PROCESS_ID_SPACE = 1 << DISAGG_PROCESS_ID_BITS
    COUNTER_MASK = (1 << DISAGG_COUNTER_BITS) - 1
    TIMESTAMP_MASK = (1 << DISAGG_TIMESTAMP_BITS) - 1
    MAX_INT64 = (1 << 63) - 1

    if node_id not in range(0, NODE_ID_SPACE):
        raise ValueError(f"node_id must be in range [0, {NODE_ID_SPACE})")
    if process_id not in range(0, PROCESS_ID_SPACE):
        raise ValueError(f"process_id must be in range [0, {PROCESS_ID_SPACE})")

    timestamp_ms = int(time.monotonic() * 1000) & TIMESTAMP_MASK
    with _global_disagg_request_id_lock:
        counter = _global_disagg_request_id_counter & COUNTER_MASK
        _global_disagg_request_id_counter += 1

    global_id = (
        (timestamp_ms << (DISAGG_NODE_ID_BITS + DISAGG_PROCESS_ID_BITS + DISAGG_COUNTER_BITS))
        | (node_id << (DISAGG_PROCESS_ID_BITS + DISAGG_COUNTER_BITS))
        | (process_id << DISAGG_COUNTER_BITS)
        | counter
    )
    # Rotate into [MIN_GLOBAL_ID, MAX_INT64); [0, MIN_GLOBAL_ID) is local-id space.
    global_id_int64 = global_id % (MAX_INT64 - MIN_GLOBAL_ID) + MIN_GLOBAL_ID
    return global_id_int64


def get_local_request_id(last_id: int) -> int:
    """Increment the last_id by 1 and mod by MIN_GLOBAL_ID."""
    return (last_id + 1) & (MIN_GLOBAL_ID - 1)


def disagg_process_id_space() -> int:
    """Number of distinct process_id slots in the snowflake id (2^bits)."""
    return 1 << DISAGG_PROCESS_ID_BITS


def worker_local_process_id() -> int:
    """Return this fleet worker's process index.

    The fleet launcher sets ``TRTLLM_DISAGG_WORKER_PROCESS_ID`` to a distinct
    value per process. A standalone disaggregated server defaults to 0.
    """
    process_id = int(os.environ.get("TRTLLM_DISAGG_WORKER_PROCESS_ID", "0"))
    process_id_space = disagg_process_id_space()
    if not 0 <= process_id < process_id_space:
        raise ValueError(
            "TRTLLM_DISAGG_WORKER_PROCESS_ID must be between 0 and "
            f"{process_id_space - 1}, got {process_id}"
        )
    return process_id
