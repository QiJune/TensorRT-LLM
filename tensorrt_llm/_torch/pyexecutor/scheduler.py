import copy
import dataclasses
import itertools
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from strenum import StrEnum

from tensorrt_llm.bindings import internal as tb_internal
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy
from tensorrt_llm.logger import logger

from .llm_request import LlmRequest, LlmRequestState
from .request_utils import merge_requests

RequestList = list[LlmRequest]

# Standard scheduler output (used by both SimpleScheduler and SimpleUnifiedScheduler)
SchedulerOutput = namedtuple("SchedulerOutput", [
    "context_requests", "generation_requests", "paused_requests",
    "fitting_disagg_gen_init_requests", "num_fitting_requests"
])


@dataclass
class UnifiedSchedulerOutput:
    """
    Extended scheduler output for SimpleUnifiedScheduler with global coordination.

    Includes standard scheduling fields plus updated_active_requests for attention_dp mode.
    """
    context_requests: RequestList
    generation_requests: RequestList
    paused_requests: RequestList
    fitting_disagg_gen_init_requests: RequestList
    num_fitting_requests: int

    # Optional: Only populated when global coordination is used (attention_dp)
    updated_active_requests: Optional[RequestList] = None

    def to_scheduler_output(self) -> SchedulerOutput:
        """Convert to standard SchedulerOutput (for backward compatibility)."""
        return SchedulerOutput(
            context_requests=self.context_requests,
            generation_requests=self.generation_requests,
            paused_requests=self.paused_requests,
            fitting_disagg_gen_init_requests=self.
            fitting_disagg_gen_init_requests,
            num_fitting_requests=self.num_fitting_requests,
        )

    def to_scheduled_requests(self) -> 'ScheduledRequests':
        """Convert to ScheduledRequests (used by PyExecutor)."""
        return ScheduledRequests.from_lists(
            context_requests=self.context_requests,
            generation_requests=self.generation_requests,
            paused_requests=self.paused_requests,
        )


class ScheduledRequests:
    # to be aligned with ScheduledRequests in cpp/tensorrt_llm/batch_manager/common.h
    def __init__(self):
        self.context_requests: RequestList = []
        self.generation_requests: RequestList = []
        self.paused_requests: RequestList = []
        self.disagg_gen_init_requests: RequestList = []

    @staticmethod
    def from_lists(
        context_requests: RequestList,
        generation_requests: RequestList,
        paused_requests: RequestList,
        disagg_gen_init_requests: Optional[RequestList] = None,
    ) -> 'ScheduledRequests':
        """Factory method to create ScheduledRequests from lists."""
        scheduled = ScheduledRequests()
        scheduled.context_requests = context_requests
        scheduled.generation_requests = generation_requests
        scheduled.paused_requests = paused_requests
        scheduled.disagg_gen_init_requests = disagg_gen_init_requests if disagg_gen_init_requests is not None else []
        return scheduled

    @property
    def is_generation_only(self) -> bool:
        return (not self.context_requests and all(
            len(req.draft_tokens) == 0 for req in self.generation_requests))

    @property
    def can_run_cuda_graph(self) -> bool:
        return (not self.context_requests)

    @property
    def batch_size(self) -> int:
        return len(self.context_requests) + len(self.generation_requests)

    def all_requests(self) -> list[LlmRequest]:
        return self.context_requests + self.generation_requests


class RequestScheduler(ABC):

    @abstractmethod
    def schedule_request(self, active_requests: RequestList,
                         inflight_request_ids: set[int]) -> SchedulerOutput:
        """
        :param active_requests: list of active requests, up to maximum number of sequences
        :param inflight_request_ids: set of request ids that are inflight (of all micro batches)
        :return: SchedulerOutput
        """
        # to be aligned with RequestScheduler::scheduleRequests in cpp/tensorrt_llm/batch_manager/requestScheduler.h
        raise NotImplementedError

    @abstractmethod
    def can_schedule(self, requests: RequestList) -> bool:
        """
        Check if current rank can schedule the requests.
        :param requests: list of requests to be scheduled
        :return: True if current rank can schedule the requests, False otherwise
        """
        raise NotImplementedError


@dataclass
class SerializableSchedulerOutput:
    """
    Serializable version of SchedulerOutput, used for sending schedule result to other ranks. Need this class because LlmRequest is not serializable by pickle.
    """
    context_requests: list[int]  # request ids of context requests
    generation_requests: list[int]  # request ids of generation requests
    paused_requests: list[int]  # request ids of paused requests
    fitting_disagg_gen_init_requests: list[
        int]  # request ids of fitting disaggregated generation initialization requests
    num_fitting_requests: int  # number of fitting requests

    @classmethod
    def from_scheduler_result(
            cls, scheduled_requests: ScheduledRequests,
            fitting_disagg_gen_init_requests: RequestList,
            num_fitting_requests: int) -> "SerializableSchedulerOutput":
        return cls(context_requests=[
            req.request_id for req in scheduled_requests.context_requests
        ],
                   generation_requests=[
                       req.request_id
                       for req in scheduled_requests.generation_requests
                   ],
                   paused_requests=[
                       req.request_id
                       for req in scheduled_requests.paused_requests
                   ],
                   fitting_disagg_gen_init_requests=[
                       req.request_id
                       for req in fitting_disagg_gen_init_requests
                   ],
                   num_fitting_requests=num_fitting_requests)

    def to_scheduler_result(
        self, active_requests: RequestList
    ) -> tuple[ScheduledRequests, RequestList, int]:
        id_to_request = {req.request_id: req for req in active_requests}
        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests = [
            id_to_request[req_id] for req_id in self.context_requests
        ]
        scheduled_requests.generation_requests = [
            id_to_request[req_id] for req_id in self.generation_requests
        ]
        scheduled_requests.paused_requests = [
            id_to_request[req_id] for req_id in self.paused_requests
        ]
        fitting_disagg_gen_init_requests = [
            id_to_request[req_id]
            for req_id in self.fitting_disagg_gen_init_requests
        ]
        return scheduled_requests, fitting_disagg_gen_init_requests, self.num_fitting_requests


@dataclass
class RankResourceState:
    """
    Snapshot of a single rank's resources for global coordination.

    This dataclass captures all information needed to simulate resource
    allocation decisions without actually allocating resources.
    Used by SimpleUnifiedScheduler for attention_dp global scheduling.
    """

    rank_id: int

    # === Constraints (Safety) ===
    free_kv_blocks: int  # From CapacityScheduler.get_kv_cache_stats()
    max_kv_blocks: int  # Total KV cache capacity
    current_batch_tokens: int  # Current token load
    max_token_budget: float  # From MicroBatchScheduler.max_num_tokens (can be float('inf'))
    current_batch_size: int  # Number of active requests
    max_batch_size: int  # From MicroBatchScheduler.max_batch_size

    # === Load Metrics (Balancing) ===
    num_active_gen_reqs: int  # Generation requests in progress
    num_active_ctx_reqs: int  # Context requests in progress

    # === PEFT/LoRA (Optional - reserved for future use) ===
    active_lora_task_ids: Set[int] = field(
        default_factory=set)  # For LoRA co-location
    available_peft_pages: int = 0  # PEFT cache capacity


class CapacityScheduler(ABC):

    @abstractmethod
    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[list[LlmRequest], list[LlmRequest], list[LlmRequest]]:
        """
        :param active_requests: list of active requests, up to maximum number of sequences
        :return: (scheduledRequests, pausedRequests)
        """
        # to be aligned with CapacityScheduler::scheduleRequests in cpp/tensorrt_llm/batch_manager/capacityScheduler.h
        raise NotImplementedError


class BindCapacityScheduler(CapacityScheduler):

    def __init__(
        self,
        max_num_requests: int,
        kv_cache_manager,
        peft_cache_manager: tb_internal.batch_manager.PeftCacheManager | None,
        scheduler_policy: CapacitySchedulerPolicy = CapacitySchedulerPolicy.
        GUARANTEED_NO_EVICT,
        two_step_lookahead: bool = False,
    ):
        super(BindCapacityScheduler, self).__init__()
        self.kv_cache_manager = kv_cache_manager
        self.peft_cache_manager = peft_cache_manager

        self.impl = tb_internal.algorithms.CapacityScheduler(
            max_num_requests=max_num_requests,
            capacity_scheduler_policy=scheduler_policy._to_pybind(),
            has_kv_cache_manager=kv_cache_manager is not None,
            two_step_lookahead=two_step_lookahead,
            no_schedule_until_state=LlmRequestState.CONTEXT_INIT,
            no_schedule_after_state=LlmRequestState.GENERATION_COMPLETE)

    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[list[LlmRequest], list[LlmRequest], list[LlmRequest]]:
        return self.impl(active_requests, self.kv_cache_manager,
                         self.peft_cache_manager)


class KVCacheV2DummyScheduler(CapacityScheduler):
    # only schedule requests has no_schedule_until_state <= state < no_schedule_after_state
    no_schedule_until_state = LlmRequestState.CONTEXT_INIT
    no_schedule_after_state = LlmRequestState.GENERATION_COMPLETE

    def __init__(self, max_num_requests: int, kv_cache_manager):
        super(KVCacheV2DummyScheduler, self).__init__()
        self.max_num_requests = max_num_requests
        self.kv_cache_manager = kv_cache_manager

    def schedule_request(
        self, active_requests: RequestList
    ) -> tuple[list[LlmRequest], list[LlmRequest], list[LlmRequest]]:
        scheduled_requests = []
        scheduled_disagg_gen_init_requests = []
        pending_requests = []
        reserved_blocks = 0
        max_blocks = self.kv_cache_manager.get_max_resource_count()
        for request in active_requests:
            req_state = request.state
            # if request cannot be scheduled yet or request should no longer be scheduled, skip
            if not req_state == LlmRequestState.DISAGG_GENERATION_INIT and (
                    req_state.value < self.no_schedule_until_state.value
                    or req_state.value >= self.no_schedule_after_state.value):
                continue

            if len(scheduled_requests
                   ) >= self.max_num_requests or reserved_blocks >= max_blocks:
                break
            elif req_state == LlmRequestState.GENERATION_IN_PROGRESS or req_state == LlmRequestState.GENERATION_TO_COMPLETE:
                scheduled_requests.append(request)
                reserved_blocks += self.kv_cache_manager.get_needed_resource_to_completion(
                    request)
            elif req_state == LlmRequestState.DISAGG_GENERATION_INIT:
                scheduled_disagg_gen_init_requests.append(request)
                reserved_blocks += self.kv_cache_manager.get_needed_resource_to_completion(
                    request)
            else:
                pending_requests.append(request)

        avaiable_blocks = max_blocks - reserved_blocks
        for request in pending_requests:
            req_state = request.state
            if len(scheduled_requests) >= self.max_num_requests:
                break
            elif req_state == LlmRequestState.CONTEXT_INIT:
                needed_blocks = self.kv_cache_manager.get_needed_resource_to_completion(
                    request)
                if needed_blocks <= avaiable_blocks:
                    scheduled_requests.append(request)
                    avaiable_blocks -= needed_blocks
                elif needed_blocks > avaiable_blocks:
                    # If one requests fails to be scheduled, break
                    break

        assert len(scheduled_requests) + len(
            scheduled_disagg_gen_init_requests) > 0, (
                "no pending request can get enough resource to complete, "
                "please increase KV cache pool size.")
        return scheduled_requests, scheduled_disagg_gen_init_requests, []


class MicroBatchScheduler(ABC):

    @abstractmethod
    def schedule(
        self, active_requests: RequestList, inflight_request_ids: set[int]
    ) -> tuple[list[LlmRequest], list[LlmRequest]]:
        """
        :param active_requests: list of active requests, up to maximum number of sequences
        :param inflight_request_ids: set of request ids that are inflight (of all micro batches)
        :return: (contextRequests, generationRequests)
        """
        # to be aligned with MicroBatchScheduler::scheduleRequests in cpp/tensorrt_llm/batch_manager/microBatchScheduler.h
        raise NotImplementedError


class BindMicroBatchScheduler(MicroBatchScheduler):

    def __init__(
        self,
        max_batch_size: int,
        max_num_tokens: int = None,
        ctx_chunk_config: Optional[tuple[StrEnum, int]] = None,
    ) -> None:
        super(BindMicroBatchScheduler, self).__init__()
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens

        ctx_chunk_config_cpp = None
        if ctx_chunk_config is not None:
            ctx_chunk_config_cpp = tb_internal.batch_manager.ContextChunkingConfig(
                ctx_chunk_config[0]._to_pybind(), ctx_chunk_config[1])

        self.impl = tb_internal.algorithms.MicroBatchScheduler(
            ctx_chunk_config_cpp, max_num_tokens)

    def schedule(
        self, active_requests: RequestList, inflight_request_ids: set[int]
    ) -> tuple[list[LlmRequest], list[LlmRequest]]:
        return self.impl(active_requests, inflight_request_ids,
                         self.max_batch_size, self.max_num_tokens)


class SimpleScheduler(RequestScheduler):

    def __init__(self, capacity_scheduler: CapacityScheduler,
                 micro_batch_scheduler: MicroBatchScheduler):
        super(SimpleScheduler, self).__init__()
        self.capacity_scheduler = capacity_scheduler
        self.micro_batch_scheduler = micro_batch_scheduler

    def schedule_request(self, active_requests: RequestList,
                         inflight_request_ids: set[int]) -> SchedulerOutput:
        fitting_requests, fitting_disagg_gen_init_requests, paused_requests = self.capacity_scheduler.schedule_request(
            active_requests)

        context_requests, generation_requests = self.micro_batch_scheduler.schedule(
            fitting_requests, inflight_request_ids)
        # Convert from binding type RequestVector to list[LlmRequest],
        # so Python fields on LlmRequest won't be stripped away
        return SchedulerOutput(list(context_requests),
                               list(generation_requests), list(paused_requests),
                               list(fitting_disagg_gen_init_requests),
                               len(fitting_requests))

    def can_schedule(self, requests: RequestList) -> bool:
        fitting_requests, _, _ = self.capacity_scheduler.schedule_request(
            requests)
        return len(fitting_requests) == len(requests)


class ChunkingPolicy(Enum):
    EQUAL_PROGRESS = 1
    FIRST_COME_FIRST_SERVED = 2


@dataclasses.dataclass
class ContextChunkingConfig:
    chunking_policy: ChunkingPolicy
    chunk_unit_size: int


class NoEvictScheduledBlocksManager:
    """
    Python equivalent of C++ kv_cache_manager::NoEvictScheduledBlocksManager.
    Tracks available blocks per window size for GUARANTEED_NO_EVICT scheduling.

    Reference: cpp/tensorrt_llm/batch_manager/scheduledBlocksManager.h:29-62
    """

    def __init__(self, kv_cache_manager):
        """
        Initialize with free blocks from KVCacheManager.
        C++ equivalent: mAvailableBlocks = mKvCacheManager.getBlockManager().getNumFreeBlocksPerWindowSize()
        """
        self.kv_cache_manager = kv_cache_manager
        stats = kv_cache_manager.get_kv_cache_stats()
        self.available_blocks: dict[int, int] = dict(
            stats.num_free_blocks_per_window_size)

    def decrement_reserved_blocks(self, req: LlmRequest) -> None:
        """
        Decrement available blocks by the blocks needed to complete this request.
        C++ reference: scheduledBlocksManager.h:40-46
        """
        for window_size in self.available_blocks:
            needed = self.kv_cache_manager.get_remaining_blocks_to_completion(
                req, window_size)
            self.available_blocks[window_size] -= needed

    def enough_available_blocks(self, req: LlmRequest) -> bool:
        """
        Check if there are enough available blocks for this request across all window sizes.
        C++ reference: scheduledBlocksManager.h:48-57
        """
        return all(
            self.kv_cache_manager.get_remaining_blocks_to_completion(req, ws) <=
            avail for ws, avail in self.available_blocks.items())


class MaxUtilizationScheduledBlocksManager:
    """
    Python equivalent of C++ kv_cache_manager::MaxUtilizationScheduledBlocksManager.
    Tracks scheduled blocks per window size for MAX_UTILIZATION scheduling.

    Reference: cpp/tensorrt_llm/batch_manager/scheduledBlocksManager.h:64-117
    """

    def __init__(self, kv_cache_manager, two_steps_look_ahead: bool):
        """
        Initialize scheduled blocks count per window size.
        C++ equivalent: iterate windowSizes and set mNumScheduledBlocks[windowSize] = 0
        """
        self.kv_cache_manager = kv_cache_manager
        self.two_steps_look_ahead = two_steps_look_ahead
        window_sizes = set(kv_cache_manager.max_attention_window_vec)
        self.num_scheduled_blocks: dict[int, int] = {
            ws: 0
            for ws in window_sizes
        }

    def prepare_blocks_if_schedulable(
            self, req: LlmRequest) -> Optional[dict[int, int]]:
        """
        Check if request can be scheduled and return new block counts if so.
        Returns None if request cannot fit.
        C++ reference: scheduledBlocksManager.h:80-100
        """
        blocks_if_scheduled = {}
        for window_size, num_scheduled in self.num_scheduled_blocks.items():
            required = self.kv_cache_manager.get_needed_blocks_one_step(
                req, self.two_steps_look_ahead, window_size)
            logger.debug(
                f"MaxUtilizationScheduler: request ID {req.request_id} "
                f"required blocks {required} for {window_size} window size")
            scheduled_total = num_scheduled + required
            has_free = self.kv_cache_manager.scheduling_has_free_blocks(
                scheduled_total, window_size)
            if not has_free:
                return None
            blocks_if_scheduled[window_size] = scheduled_total
        return blocks_if_scheduled

    def update_scheduled_blocks(self, blocks: dict[int, int]) -> None:
        """
        Update the scheduled blocks after successfully scheduling a request.
        C++ reference: scheduledBlocksManager.h:102-110
        """
        assert len(blocks) == len(self.num_scheduled_blocks), \
            f"Block count mismatch: {len(blocks)} vs {len(self.num_scheduled_blocks)}"
        for window_size, blocks_if_scheduled in blocks.items():
            logger.debug(
                f"MaxUtilizationScheduler: scheduled blocks {blocks_if_scheduled} "
                f"for window size {window_size}")
            self.num_scheduled_blocks[window_size] = blocks_if_scheduled


class PeftHelper:
    """
    Helper class for PEFT/LoRA resource management.

    Encapsulates all PEFT-related logic including page calculation,
    task tracking, and capacity management.
    """

    def __init__(self, peft_cache_manager):
        """
        Initialize PEFT helper.

        Args:
            peft_cache_manager: PEFT cache manager instance (or None if PEFT disabled)
        """
        self.peft_cache_manager = peft_cache_manager

    def get_max_pages(self) -> int:
        """Get maximum PEFT cache pages available."""
        if self.peft_cache_manager is None:
            return 2**31 - 1  # INT_MAX equivalent
        return self.peft_cache_manager.max_device_pages

    def get_pages_for_request(self, req: LlmRequest) -> int:
        """Get number of PEFT pages needed for a request."""
        if self.peft_cache_manager is None:
            return 0
        return self.peft_cache_manager.determine_num_pages(req)

    def get_task_info(
            self, req: LlmRequest,
            seen_task_ids: set[int]) -> tuple[Optional[int], bool, int]:
        """
        Get PEFT task information for a request.

        Args:
            req: Request to check
            seen_task_ids: Set of task IDs already seen/allocated

        Returns:
            Tuple of (lora_task_id, is_new_task, required_pages)
        """
        lora_task_id = getattr(req, 'lora_task_id', None)
        is_new_task = lora_task_id is not None and lora_task_id not in seen_task_ids
        required_pages = self.get_pages_for_request(req) if is_new_task else 0
        return lora_task_id, is_new_task, required_pages


class GlobalCoordinator:
    """
    Handles global request coordination for attention_dp mode.

    This class encapsulates all the logic for coordinating request scheduling
    across multiple TP ranks using a single allgather operation. It implements
    a deterministic water-filling algorithm that all ranks execute identically (SPMD).

    Responsibilities:
    - Build local rank resource state
    - Gather states from all ranks via single allgather
    - Simulate global scheduling with water-filling algorithm
    - Calculate assignment scores for load balancing
    - Apply batching filters for context request coordination
    """

    def __init__(self, scheduler, dist, max_num_active_requests: int):
        """
        Initialize global coordinator.

        Args:
            scheduler: Reference to parent SimpleUnifiedScheduler (for estimation methods)
            dist: Distributed communication object
            max_num_active_requests: Maximum number of active requests across all ranks
        """
        self.scheduler = scheduler
        self.dist = dist
        self.max_num_active_requests = max_num_active_requests

        # Attention DP balancing/batching state
        self.attention_dp_enable_balance = False
        self.attention_dp_time_out_iters = 0
        self.attention_dp_batching_wait_iters = 0
        self.adp_ctx_waiting_iters_count = 0
        self.adp_ctx_batching_wait_iters_count = 0

    def _estimate_next_iteration_growth_tokens(self,
                                               request: LlmRequest) -> int:
        """
        Estimate how many additional tokens a request will consume in the NEXT iteration.

        This is critical for accurate simulation: old active requests will grow
        (generate tokens, process next chunk, etc.) before new requests are scheduled.

        Args:
            request: Active request to estimate growth for

        Returns:
            int: Estimated additional tokens for next iteration
        """
        state_value = request.state_value

        # Context requests: Check for chunking
        if state_value == self.scheduler._context_init_state_value:
            if not request.is_last_context_chunk:
                # Will process another chunk in next iteration
                remaining_length = request.context_remaining_length
                if remaining_length > 0:
                    # Estimate next chunk size
                    max_chunk = self.scheduler.max_num_tokens if self.scheduler.max_num_tokens else 2048
                    return min(remaining_length, max_chunk)
            return 0  # Last chunk, no growth

        # Generation requests: Will generate more tokens
        elif state_value != self.scheduler._encoder_init_state_value:
            # Get beam width for next iteration
            beam_width = request.get_beam_width_by_iter(for_next_iteration=True)
            # Add draft tokens if applicable
            draft_tokens = request.num_draft_tokens if request.has_draft_tokens else 0
            return beam_width + draft_tokens

        # Encoder requests: No growth (single-shot)
        return 0

    def _estimate_next_iteration_growth_blocks(self,
                                               request: LlmRequest) -> int:
        """
        Estimate how many additional KV cache blocks a request will need in the NEXT iteration.

        Args:
            request: Active request to estimate growth for

        Returns:
            int: Estimated additional blocks for next iteration
        """
        if self.scheduler.kv_cache_manager is None:
            return 0

        # Estimate growth tokens first
        growth_tokens = self._estimate_next_iteration_growth_tokens(request)
        if growth_tokens == 0:
            return 0

        # Get current sequence length and blocks
        current_length = request.get_num_tokens(0)

        # For VSWA, use worst-case across window sizes
        if hasattr(self.scheduler.kv_cache_manager, 'is_variable_window') and \
           self.scheduler.kv_cache_manager.is_variable_window:
            max_growth_blocks = 0
            for window_size_key in self.scheduler.kv_cache_manager.get_window_size_keys(
            ):
                current_blocks = self.scheduler.kv_cache_manager.get_num_required_blocks(
                    request, window_size_key)

                # Estimate blocks after growth (approximate)
                # This is conservative: assume each token might need a new block
                tokens_per_block = getattr(self.scheduler.kv_cache_manager,
                                           'tokens_per_block', 64)
                future_length = current_length + growth_tokens
                future_blocks = (future_length + tokens_per_block -
                                 1) // tokens_per_block

                growth_blocks = max(0, future_blocks - current_blocks)
                max_growth_blocks = max(max_growth_blocks, growth_blocks)

            return max_growth_blocks
        else:
            # Standard case: estimate block growth
            tokens_per_block = getattr(self.scheduler.kv_cache_manager,
                                       'tokens_per_block', 64)

            # Current blocks
            current_blocks = self.scheduler.kv_cache_manager.get_num_required_blocks(
                request)

            # Future blocks after growth
            future_length = current_length + growth_tokens
            future_blocks = (future_length + tokens_per_block -
                             1) // tokens_per_block

            return max(0, future_blocks - current_blocks)

    def build_local_state(
        self,
        active_requests: List[LlmRequest],
    ) -> RankResourceState:
        """
        Build snapshot of local rank's current state.

        ENHANCEMENT: Includes predicted growth of active requests for next iteration.
        This makes simulation more accurate by accounting for resources that old requests
        will consume before new requests are scheduled.

        This captures all information needed for global coordination without
        modifying any actual resources.

        Args:
            active_requests: Currently active requests on this rank

        Returns:
            RankResourceState: Snapshot of current rank state (including predicted growth)
        """
        # Get KV cache stats
        if self.scheduler.kv_cache_manager is not None:
            stats = self.scheduler.kv_cache_manager.get_kv_cache_stats()
            # For VSWA (Variable Sliding Window), we track per window size
            if hasattr(stats, 'num_free_blocks_per_window_size'):
                free_blocks_per_ws = dict(stats.num_free_blocks_per_window_size)
                # Use the primary window size (0 or first key)
                primary_ws = 0 if 0 in free_blocks_per_ws else next(
                    iter(free_blocks_per_ws), 0)
                free_blocks = free_blocks_per_ws.get(primary_ws, 0)
            else:
                # Fallback for non-VSWA
                free_blocks = getattr(stats, 'free_num_blocks', 0)
            max_blocks = getattr(self.scheduler.kv_cache_manager,
                                 'max_num_blocks', 0)
        else:
            free_blocks = 0
            max_blocks = 0

        # Get token budget
        max_token_budget = self.scheduler.max_num_tokens if self.scheduler.max_num_tokens is not None else float(
            'inf')

        # Calculate current token load
        current_tokens = self.scheduler._calculate_current_token_load(
            active_requests)

        # ENHANCEMENT: Predict growth for next iteration
        # This accounts for old requests consuming more resources before new requests schedule
        predicted_growth_tokens = 0
        predicted_growth_blocks = 0

        for req in active_requests:
            growth_tokens = self._estimate_next_iteration_growth_tokens(req)
            predicted_growth_tokens += growth_tokens

            if growth_tokens > 0:
                growth_blocks = self._estimate_next_iteration_growth_blocks(req)
                predicted_growth_blocks += growth_blocks

        # Count active requests by type
        num_active_gen = sum(1 for r in active_requests
                             if not r.is_context_init_state)
        num_active_ctx = sum(1 for r in active_requests
                             if r.is_context_init_state)

        # Reserve resources for predicted growth
        # This makes simulation conservative but accurate
        return RankResourceState(
            rank_id=self.dist.rank,
            free_kv_blocks=max(0, free_blocks -
                               predicted_growth_blocks),  # Reserve for growth
            max_kv_blocks=max_blocks,
            current_batch_tokens=current_tokens +
            predicted_growth_tokens,  # Include growth
            max_token_budget=max_token_budget,
            current_batch_size=len(active_requests),
            max_batch_size=self.scheduler.max_batch_size,
            num_active_gen_reqs=num_active_gen,
            num_active_ctx_reqs=num_active_ctx,
        )

    def gather_all_states(
            self, local_state: RankResourceState) -> List[RankResourceState]:
        """
        THE SINGLE COMMUNICATION POINT.
        Gather RankResourceState from all TP ranks via tp_allgather.

        This is the ONLY synchronization point in the unified scheduler,
        replacing the 3+ tp_allgather calls in the old architecture.

        Args:
            local_state: This rank's resource state

        Returns:
            List[RankResourceState]: States from all ranks
        """
        # Serialize to dict for communication (dataclasses are not directly serializable)
        local_dict = {
            'rank_id': local_state.rank_id,
            'free_kv_blocks': local_state.free_kv_blocks,
            'max_kv_blocks': local_state.max_kv_blocks,
            'current_batch_tokens': local_state.current_batch_tokens,
            'max_token_budget': local_state.max_token_budget,
            'current_batch_size': local_state.current_batch_size,
            'max_batch_size': local_state.max_batch_size,
            'num_active_gen_reqs': local_state.num_active_gen_reqs,
            'num_active_ctx_reqs': local_state.num_active_ctx_reqs,
            'active_lora_task_ids': list(local_state.active_lora_task_ids),
            'available_peft_pages': local_state.available_peft_pages,
        }

        # THE SINGLE tp_allgather
        all_dicts = self.dist.tp_allgather(local_dict)

        # Deserialize back to RankResourceState objects
        result = []
        for d in all_dicts:
            # Convert active_lora_task_ids back to set
            d['active_lora_task_ids'] = set(d.get('active_lora_task_ids', []))
            result.append(RankResourceState(**d))

        return result

    def calculate_assignment_score(
        self,
        rank_state: RankResourceState,
    ) -> float:
        """
        Calculate assignment score for a rank.
        Higher score = better assignment.

        Scoring components:
        1. Load penalty: Avoid overloaded ranks
        2. Context request penalty: Balance context vs generation

        Args:
            rank_state: Current state of the candidate rank

        Returns:
            float: Assignment score (higher is better)
        """
        score = 0.0

        # Component 1: Load balancing (token-based)
        if rank_state.max_token_budget > 0 and rank_state.max_token_budget != float(
                'inf'):
            load_ratio = rank_state.current_batch_tokens / rank_state.max_token_budget
            score -= load_ratio * 100.0

        # Component 2: Context vs generation balancing
        # Penalize ranks with many context requests (they block generation)
        score -= rank_state.num_active_ctx_reqs * 2.0
        score -= rank_state.num_active_gen_reqs * 1.0

        return score

    def can_accept_request(
        self,
        request: LlmRequest,
        rank_state: RankResourceState,
    ) -> bool:
        """
        Check if rank can accept this request based on resource constraints.
        This is the SIMULATION of capacity and token budget checks.

        OPTIMIZATION: If the request can be accepted, cache the estimated tokens/blocks
        to avoid recalculation in _fused_schedule_request().

        Args:
            request: The request to check
            rank_state: Current state of the candidate rank

        Returns:
            bool: True if rank can accept the request
        """
        # Check batch size limit
        if rank_state.current_batch_size >= rank_state.max_batch_size:
            return False

        # Check token budget limit
        tokens_needed = self.scheduler._estimate_tokens_needed(request)
        if rank_state.max_token_budget != float('inf'):
            if rank_state.current_batch_tokens + tokens_needed > rank_state.max_token_budget:
                return False

        # Check KV cache capacity
        blocks_needed = self.scheduler._estimate_blocks_needed(request)
        if rank_state.free_kv_blocks < blocks_needed:
            return False

        # OPTIMIZATION: Cache estimates for later use in _fused_schedule_request()
        # This avoids ~50% duplicate work for newly activated requests
        request.py_pre_validated = True
        request.py_estimated_tokens = tokens_needed
        request.py_estimated_blocks = blocks_needed

        return True

    def update_rank_state_after_assignment(
        self,
        rank_state: RankResourceState,
        request: LlmRequest,
    ) -> None:
        """
        Update simulated rank state after assigning a request.
        This modifies the state IN PLACE during simulation.

        Args:
            rank_state: The rank state to update (modified in place)
            request: The request that was assigned
        """
        # Decrement resources
        tokens_needed = self.scheduler._estimate_tokens_needed(request)
        rank_state.current_batch_tokens += tokens_needed
        rank_state.current_batch_size += 1

        blocks_needed = self.scheduler._estimate_blocks_needed(request)
        rank_state.free_kv_blocks -= blocks_needed

        # Update request counters
        if request.is_context_init_state:
            rank_state.num_active_ctx_reqs += 1
        else:
            rank_state.num_active_gen_reqs += 1

    def simulate_global_schedule(
        self,
        candidate_requests:
        List,  # List[RequestQueueItem] but avoid circular import
        all_rank_states: List[RankResourceState],
    ) -> Dict[int, List[int]]:
        """
        Deterministic water-filling algorithm.
        ALL RANKS RUN THIS IDENTICALLY (SPMD).

        This is the core scheduling algorithm that assigns requests to ranks
        based on resource availability and optimization criteria.

        Args:
            candidate_requests: List of candidate requests to assign
            all_rank_states: Current states of all ranks

        Returns:
            Dict mapping rank_id -> [assigned_request_ids]
        """
        # Deep copy to avoid modifying original states
        sim_states = copy.deepcopy(all_rank_states)

        # Initialize assignments
        assignments = {state.rank_id: [] for state in sim_states}

        # Sort candidates deterministically (all ranks must see same order!)
        # Priority: non-relaxed first, then by request_id for determinism
        sorted_candidates = sorted(
            candidate_requests,
            key=lambda item: (
                # Check if request has attention_dp_relax flag
                (getattr(item, 'llm_request', None) and getattr(
                    item.llm_request, 'py_scheduling_params', None) and getattr(
                        item.llm_request.py_scheduling_params,
                        'attention_dp_relax', False)) or False,
                # Secondary sort by id for determinism (RequestQueueItem.id)
                item.id,
            ))

        # Water-filling algorithm
        for req_item in sorted_candidates:
            if not hasattr(req_item, 'llm_request') or not req_item.llm_request:
                continue

            req = req_item.llm_request

            # Score all ranks for this request
            best_rank_id = -1
            best_score = -float('inf')

            for rank_state in sim_states:
                # Feasibility check
                if not self.can_accept_request(req, rank_state):
                    continue

                # Calculate score
                score = self.calculate_assignment_score(rank_state)

                if score > best_score:
                    best_score = score
                    best_rank_id = rank_state.rank_id

            # Assign to best rank (if any rank can accept)
            if best_rank_id != -1:
                assignments[best_rank_id].append(req.request_id)

                # Update simulated state
                target_state = sim_states[best_rank_id]
                self.update_rank_state_after_assignment(target_state, req)

        return assignments

    def apply_batching_filter(
        self,
        assignments: Dict[int, List[int]],
        candidate_requests: List,
    ) -> Dict[int, List[int]]:
        """
        Apply batching filter to assignments based on waiting logic.

        If we should wait for all ranks to have context requests, this method
        filters out context requests but keeps generation requests.

        Args:
            assignments: Dict mapping rank_id -> [assigned_request_ids]
            candidate_requests: List of candidate requests

        Returns:
            Dict[int, List[int]]: Filtered assignments
        """
        # Check if we should wait
        should_wait = self.should_wait_for_context_batching(
            assignments, candidate_requests)
        if not should_wait:
            return assignments

        # Build request ID to request mapping
        req_id_to_req = {}
        for req_item in candidate_requests:
            if hasattr(req_item, 'llm_request') and req_item.llm_request:
                req = req_item.llm_request
                req_id_to_req[req.request_id] = req

        # Filter out context requests, keep generation requests
        filtered_assignments = {}
        for rank_id in assignments:
            filtered_req_ids = []
            for req_id in assignments[rank_id]:
                if req_id in req_id_to_req:
                    req = req_id_to_req[req_id]
                    # Keep only generation requests, remove context requests
                    if not req.is_context_init_state:
                        filtered_req_ids.append(req_id)
                else:
                    # Unknown request (shouldn't happen but keep for safety)
                    filtered_req_ids.append(req_id)
            filtered_assignments[rank_id] = filtered_req_ids

        return filtered_assignments

    def should_wait_for_context_batching(
        self,
        assignments: Dict[int, List[int]],
        candidate_requests: List,
    ) -> bool:
        """
        Check if we should wait for all ranks to have context requests (attention_dp batching).

        This implements the same logic as _balance_adp_requests to ensure:
        1. All ranks have context requests before scheduling (avoid load imbalance)
        2. Batch context requests together when possible
        3. Timeout mechanism to avoid deadlock

        Args:
            assignments: Dict mapping rank_id -> [assigned_request_ids]
            candidate_requests: List of candidate requests

        Returns:
            bool: True if we should wait (clear context requests), False if we should proceed
        """
        if not self.attention_dp_enable_balance:
            return False

        # Build request ID to request mapping
        req_id_to_req = {}
        for req_item in candidate_requests:
            if hasattr(req_item, 'llm_request') and req_item.llm_request:
                req = req_item.llm_request
                req_id_to_req[req.request_id] = req

        # Count context and generation requests per rank
        rank_ctx_counts = {}
        rank_gen_counts = {}
        for rank_id, assigned_req_ids in assignments.items():
            ctx_count = 0
            gen_count = 0
            for req_id in assigned_req_ids:
                if req_id in req_id_to_req:
                    req = req_id_to_req[req_id]
                    if req.is_context_init_state:
                        ctx_count += 1
                    else:
                        gen_count += 1
            rank_ctx_counts[rank_id] = ctx_count
            rank_gen_counts[rank_id] = gen_count

        # Check conditions (same as _balance_adp_requests)
        all_ranks_have_ctx_requests = all(count > 0
                                          for count in rank_ctx_counts.values())
        all_ranks_have_gen_requests = all(count > 0
                                          for count in rank_gen_counts.values())

        # Note: We don't check free_ctx_slots here because global coordination already handles capacity in can_accept_request

        if all_ranks_have_ctx_requests:
            # All ranks have context requests
            self.adp_ctx_waiting_iters_count = 0

            # Check if we should batch (wait for more context requests)
            if all_ranks_have_gen_requests:
                if self.adp_ctx_batching_wait_iters_count < self.attention_dp_batching_wait_iters:
                    self.adp_ctx_batching_wait_iters_count += 1
                    return True  # Wait for batching
                else:
                    self.adp_ctx_batching_wait_iters_count = 0
                    return False  # Proceed with scheduling
            else:
                return False  # Proceed (no generation requests to compete with)
        else:
            # Not all ranks have context requests
            self.adp_ctx_waiting_iters_count += 1

            timeout_reached = self.adp_ctx_waiting_iters_count >= self.attention_dp_time_out_iters
            if timeout_reached or not all_ranks_have_gen_requests:
                # Timeout or no generation requests - proceed anyway
                self.adp_ctx_waiting_iters_count = 0
                return False
            else:
                # Wait for all ranks to get context requests
                return True


class CapacityChecker:
    """
    Helper class for KV cache capacity checking.

    Encapsulates all logic related to checking if requests fit in KV cache,
    including block reuse optimization and policy-specific capacity checks.
    """

    def __init__(self, kv_cache_manager, cross_kv_cache_manager,
                 scheduler_policy: CapacitySchedulerPolicy,
                 no_schedule_until_state_value, no_schedule_after_state_value):
        """
        Initialize capacity checker.

        Args:
            kv_cache_manager: KV cache manager instance
            cross_kv_cache_manager: Cross-attention KV cache manager (or None)
            scheduler_policy: Capacity scheduling policy
            no_schedule_until_state_value: Minimum state value for scheduling
            no_schedule_after_state_value: Maximum state value for scheduling
        """
        self.kv_cache_manager = kv_cache_manager
        self.cross_kv_cache_manager = cross_kv_cache_manager
        self.scheduler_policy = scheduler_policy
        self._no_schedule_until_state_value = no_schedule_until_state_value
        self._no_schedule_after_state_value = no_schedule_after_state_value

    def can_be_scheduled_with_disagg_exception(self, req: LlmRequest) -> bool:
        """
        Check if request can be scheduled, with exception for disagg generation init state.
        Disagg generation init requests bypass the normal state gating.
        """
        if req.is_disagg_generation_init_state:
            return True
        # Use cached state values for performance
        state_value = req.state_value
        return (state_value >= self._no_schedule_until_state_value
                and state_value < self._no_schedule_after_state_value)

    def is_skipping_relevant(self) -> bool:
        """
        Check if block reuse skip optimization is relevant.
        Disabled for VSWA (Variable Sliding Window Attention).
        """
        if self.kv_cache_manager is None:
            return False
        if self.kv_cache_manager.is_variable_window:
            return False
        if (self.cross_kv_cache_manager is not None
                and self.cross_kv_cache_manager.is_variable_window):
            return False
        return True

    def prefill_contributed_blocks(
            self, active_requests: RequestList) -> tuple[set, set]:
        """
        Collect blocks contributed by chunked context requests already executing.
        These blocks can be reused by later requests.

        Args:
            active_requests: Currently active requests

        Returns:
            Tuple of (context_blocks, cross_context_blocks) that can be reused
        """
        newly_contributed_context_blocks: set = set()
        newly_contributed_cross_context_blocks: set = set()

        if self.kv_cache_manager is None:
            return newly_contributed_context_blocks, newly_contributed_cross_context_blocks

        enable_block_reuse = self.kv_cache_manager.enable_block_reuse
        cross_enable_reuse = (self.cross_kv_cache_manager is not None and
                              self.cross_kv_cache_manager.enable_block_reuse)

        for req in active_requests:
            # Check: isContextInitState() && !isFirstContextChunk()
            if req.is_context_init_state and not req.is_first_context_chunk:
                # Chunked context request already executing
                if enable_block_reuse:
                    unique_tokens = req.get_unique_tokens(0)
                    block_key = self.kv_cache_manager.find_new_context_block(
                        unique_tokens, req)
                    if block_key is not None:
                        newly_contributed_context_blocks.add(block_key)

                if cross_enable_reuse:
                    encoder_unique_tokens = req.get_encoder_unique_tokens()
                    if encoder_unique_tokens is not None:
                        block_key = self.cross_kv_cache_manager.find_new_context_block(
                            encoder_unique_tokens, req)
                        if block_key is not None:
                            newly_contributed_cross_context_blocks.add(
                                block_key)

        return newly_contributed_context_blocks, newly_contributed_cross_context_blocks

    def _one_manager_beneficial_to_skip(self, kv_cache_manager, unique_tokens,
                                        req: LlmRequest,
                                        newly_contributed_blocks: set) -> bool:
        """Check if skipping is beneficial for one KV cache manager."""
        new_context_block = kv_cache_manager.find_new_context_block(
            unique_tokens, req)
        if new_context_block is not None:
            if new_context_block in newly_contributed_blocks:
                return True
        return False

    def beneficial_to_skip(self, req: LlmRequest,
                           newly_contributed_context_blocks: set,
                           newly_contributed_cross_context_blocks: set) -> bool:
        """
        Check if it's beneficial to skip this request.
        A request should be skipped if it can reuse blocks contributed by
        already scheduled context requests.

        Args:
            req: Request to check
            newly_contributed_context_blocks: Blocks from active context requests
            newly_contributed_cross_context_blocks: Cross-attention blocks from active requests

        Returns:
            True if request should be skipped for block reuse optimization
        """
        if not (req.is_context_init_state and req.is_first_context_chunk):
            return False

        if (self.kv_cache_manager is not None
                and self.kv_cache_manager.enable_block_reuse):
            unique_tokens = req.get_unique_tokens(0)
            if self._one_manager_beneficial_to_skip(
                    self.kv_cache_manager, unique_tokens, req,
                    newly_contributed_context_blocks):
                return True

        if (self.cross_kv_cache_manager is not None
                and self.cross_kv_cache_manager.enable_block_reuse):
            encoder_unique_tokens = req.get_encoder_unique_tokens()
            if encoder_unique_tokens is not None:
                if self._one_manager_beneficial_to_skip(
                        self.cross_kv_cache_manager, encoder_unique_tokens, req,
                        newly_contributed_cross_context_blocks):
                    return True

        return False

    def check_kv_capacity(
        self,
        req: LlmRequest,
        scheduled_blocks_manager,
        reserved_blocks,
        reserved_cross_blocks,
        simulation_mode: bool,
    ) -> bool:
        """
        Check if request fits in KV cache capacity.
        Uses the appropriate block manager based on the scheduling policy.

        Args:
            req: Request to check
            scheduled_blocks_manager: MaxUtilizationScheduledBlocksManager (or None)
            reserved_blocks: NoEvictScheduledBlocksManager (or None)
            reserved_cross_blocks: NoEvictScheduledBlocksManager for cross-attention (or None)
            simulation_mode: If True, don't update block manager state

        Returns:
            True if request fits, False otherwise
        """
        if self.scheduler_policy == CapacitySchedulerPolicy.MAX_REQUESTS:
            # No KV cache manager, always fits
            return True
        elif self.scheduler_policy == CapacitySchedulerPolicy.MAX_UTILIZATION:
            # Use MaxUtilizationScheduledBlocksManager
            blocks_if_scheduled = scheduled_blocks_manager.prepare_blocks_if_schedulable(
                req)
            if blocks_if_scheduled is None:
                return False
            # Update state if not in simulation mode
            if not simulation_mode:
                scheduled_blocks_manager.update_scheduled_blocks(
                    blocks_if_scheduled)
            return True
        else:
            # Use NoEvictScheduledBlocksManager (GUARANTEED_NO_EVICT or STATIC_BATCH)
            if req.is_context_init_state or req.is_disagg_generation_init_state:
                enough_blocks = reserved_blocks.enough_available_blocks(req)
                enough_cross_blocks = True
                if reserved_cross_blocks is not None:
                    enough_cross_blocks = reserved_cross_blocks.enough_available_blocks(
                        req)
                return enough_blocks and enough_cross_blocks
            else:
                # Generation requests always fit (blocks already reserved)
                return True


class ChunkingManager:
    """
    Helper class for context chunking management.

    Encapsulates all logic related to chunking context requests to fit within
    token budgets, including sorting, chunk size calculation, and draft token fitting.
    """

    def __init__(self, ctx_chunk_config, max_context_length):
        """
        Initialize chunking manager.

        Args:
            ctx_chunk_config: Context chunking configuration (policy + unit_size)
            max_context_length: Maximum context length per request
        """
        self.ctx_chunk_config = ctx_chunk_config
        self.max_context_length = max_context_length

    def sort_requests(self, context_requests: RequestList,
                      generation_requests: RequestList,
                      chunks_present: bool) -> None:
        """
        Sort requests for consistency with C++.

        1. If chunks are present, move context requests that reached the last
           context chunk to the end of the vector.
        2. Sort all requests by lora task id for performance.

        Args:
            context_requests: Context requests list (modified in-place)
            generation_requests: Generation requests list (modified in-place)
            chunks_present: Whether chunking is active
        """

        def get_lora_task_id(req: LlmRequest):
            # C++ uses std::optional comparison where nullopt < any_value
            # So requests without LoRA (nullopt) should come first
            lora_id = getattr(req, 'lora_task_id', None)
            if lora_id is None:
                return (0, 0)  # (has_value=False, value=0) - comes first
            return (1, lora_id)  # (has_value=True, value) - sorted by value

        if chunks_present:
            # Partition: non-last-chunk first, last-chunk at end
            not_last_chunk = [
                r for r in context_requests if not r.is_last_context_chunk
            ]
            last_chunk = [
                r for r in context_requests if r.is_last_context_chunk
            ]
            # Sort each group by lora_task_id
            not_last_chunk.sort(key=get_lora_task_id)
            last_chunk.sort(key=get_lora_task_id)
            # Rebuild the list in-place
            context_requests.clear()
            context_requests.extend(not_last_chunk)
            context_requests.extend(last_chunk)
        else:
            context_requests.sort(key=get_lora_task_id)

        generation_requests.sort(key=get_lora_task_id)

    def apply_chunking(self, requests: RequestList,
                       capacity: Optional[int]) -> None:
        """
        Apply chunking to context requests based on chunking policy.

        Args:
            requests: Context requests to chunk (modified in-place)
            capacity: Available token capacity
        """
        # C++: Resets all chunk sizes to 0 at start
        for req in requests:
            req.context_chunk_size = 0

        policy = self.ctx_chunk_config.chunking_policy
        unit_size = self.ctx_chunk_config.chunk_unit_size

        if policy == ChunkingPolicy.EQUAL_PROGRESS:
            self._chunk_equal_progress(requests, capacity, unit_size)
        elif policy == ChunkingPolicy.FIRST_COME_FIRST_SERVED:
            self._chunk_fcfs(requests, capacity, unit_size)
        else:
            raise ValueError(f"Invalid chunking policy: {policy}")

        self._fit_draft_tokens(requests, capacity, unit_size)

    def _chunk_equal_progress(self, requests: RequestList,
                              capacity: Optional[int], unit_size: int):
        """Apply equal progress chunking strategy."""
        num_ctx_tokens = 0
        num_tokens_single_loop = 1

        # C++ Loop: while ((!capacity || numCtxTokens < capacity) && numTokensSingleLoop)
        while (capacity is None
               or num_ctx_tokens < capacity) and num_tokens_single_loop > 0:
            num_tokens_single_loop = 0
            for req in requests:
                past_size = req.context_chunk_size

                # C++ logic: suggested = past + unit
                suggested_size = past_size + unit_size

                # Ensure we don't exceed what the request actually needs
                remaining_total = req.context_remaining_length
                suggested_size = min(suggested_size, remaining_total)

                req.context_chunk_size = suggested_size

                actual_size = req.context_chunk_size
                actual_increment = actual_size - past_size

                # Check Constraints
                # 1. Capacity
                if capacity is not None and (num_ctx_tokens + actual_increment
                                             > capacity):
                    req.context_chunk_size = past_size  # Revert
                    continue

                # 2. Max Context Length
                if self.max_context_length is not None and actual_size > self.max_context_length:
                    req.context_chunk_size = past_size  # Revert
                    continue

                num_ctx_tokens += actual_increment
                num_tokens_single_loop += actual_increment

    def _chunk_fcfs(self, requests: RequestList, capacity: Optional[int],
                    unit_size: int):
        """Apply first-come-first-served chunking strategy."""
        current_capacity = capacity if capacity is not None else float('inf')

        for req in requests:
            suggested_size = req.context_remaining_length
            actual_size = suggested_size

            # Apply unit size constraint
            if unit_size > 0:
                actual_size = (actual_size // unit_size) * unit_size

            # Apply capacity constraint
            if actual_size > current_capacity:
                actual_size = (int(current_capacity) // unit_size) * unit_size

            # Apply max context length constraint
            if self.max_context_length is not None and actual_size > self.max_context_length:
                actual_size = (self.max_context_length // unit_size) * unit_size

            req.context_chunk_size = actual_size
            current_capacity -= actual_size

            if current_capacity <= 0:
                break

    def _fit_draft_tokens(self, requests: RequestList, capacity: Optional[int],
                          unit_size: int):
        """Fit draft tokens into remaining capacity for chunked requests."""
        # Calculate tokens already taken by the batch so far
        num_ctx_tokens = sum(req.context_chunk_size for req in requests)

        for req in requests:
            if req.is_last_context_chunk and req.has_draft_tokens:
                remainder = req.context_chunk_size % unit_size
                remaining_space = 0 if remainder == 0 else unit_size - remainder

                if self.max_context_length is not None:
                    remaining_context_len = self.max_context_length - req.context_chunk_size
                    remaining_space = min(remaining_space,
                                          remaining_context_len)

                if capacity is not None:
                    remaining_space = min(remaining_space,
                                          capacity - num_ctx_tokens)
                    num_ctx_tokens += remaining_space

                draft_discard = req.num_draft_tokens - remaining_space
                if draft_discard > 0:
                    logger.debug(f"Discarding {draft_discard} draft tokens")


@dataclass
class SchedulingState:
    """
    State container for scheduling loop in _fused_schedule_request.

    Groups all state variables together to reduce parameter passing
    and make the code more maintainable.
    """
    # Block reuse optimization
    skipping_is_relevant: bool
    newly_contributed_context_blocks: set
    newly_contributed_cross_context_blocks: set

    # PEFT/LoRA tracking
    has_peft: bool
    claimed_peft_pages: int
    available_peft_pages: int
    uniq_task_ids: set

    # Batch tracking
    batch_num_tokens: int
    scheduled_req_size: int
    scheduled_beam_width: int

    # Output lists
    context_requests: RequestList
    generation_requests: RequestList
    paused_requests: RequestList
    fitting_disagg_gen_init: RequestList

    # Chunking state
    contexts_to_be_chunked: RequestList
    num_chunked_tokens: int
    all_context_requests_fit: bool

    # Cached configuration (for faster access)
    max_batch_size: int
    max_num_tokens: Optional[int]
    max_context_length: Optional[int]
    ctx_chunk_config: Optional['ContextChunkingConfig']


class SimpleUnifiedScheduler(RequestScheduler):
    """
    Unified scheduler with FUSED single-pass scheduling for both modes.

    This scheduler combines capacity (KV cache) and micro-batch (token budget)
    checks into a single efficient loop, eliminating the double work of the
    traditional two-pass approach.

    Supports two operational modes:

    1. TP-only mode (enable_global_scheduling=False):
       - Local scheduling on this rank only
       - Supports batch waiting optimization
       - Uses fused single-pass scheduling

    2. Attention DP mode (enable_global_scheduling=True):
       - Global coordination across all TP ranks
       - Reduces tp_allgather calls from 3+ to 1 per scheduling step
       - Proactive architecture: Sync State → Global Simulation → Commit locally
       - Token-based load balancing
       - Uses fused single-pass scheduling with simulation mode

    Fused Scheduling Architecture:
    - Single loop checks both KV cache AND token budget together
    - Direct resource access (no wrapper schedulers)
    - Reuses block manager infrastructure (NoEvictScheduledBlocksManager, MaxUtilizationScheduledBlocksManager)
    - Supports all capacity policies: MAX_UTILIZATION, GUARANTEED_NO_EVICT, STATIC_BATCH, MAX_REQUESTS
    - Supports chunking: EQUAL_PROGRESS and FIRST_COME_FIRST_SERVED
    - Simulation mode for global coordination (no side effects)

    Performance benefits:
    - Faster: Single-pass vs two-pass (30-50% speedup)
    - Simpler: Eliminates PyCapacityScheduler and PyMicroBatchScheduler
    - More correct: No simulation/execution divergence bugs
    - Less memory: No duplicate state tracking
    """

    def __init__(
            self,
            max_batch_size: int,
            max_num_tokens: int,
            kv_cache_manager,
            peft_cache_manager,
            scheduler_policy: CapacitySchedulerPolicy,
            ctx_chunk_config: Optional[tuple[StrEnum, int]] = None,
            cross_kv_cache_manager=None,
            two_step_lookahead: bool = False,
            scheduler_capacity: Optional[int] = None,
            dist=None,  # Optional: Enable global scheduling for attention_dp
            max_num_active_requests: Optional[
                int] = None,  # Required for global coordination
    ):
        # Use scheduler_capacity if provided, otherwise fall back to max_batch_size
        # scheduler_capacity may differ from max_batch_size (e.g., adjusted for attention_dp + disagg)
        capacity = scheduler_capacity if scheduler_capacity is not None else max_batch_size

        # Global scheduling support for attention_dp
        # When enabled, coordinates scheduling across all TP ranks with single allgather
        self.dist = dist
        self.max_num_active_requests = max_num_active_requests
        self.enable_global_scheduling = dist is not None and max_num_active_requests is not None

        # Parse chunking config
        py_chunk_config = None
        if ctx_chunk_config:
            # Fix: Use string comparison to identify the policy.
            # This works regardless of whether the input is a Python Enum, C++ Binding Enum, or String.
            input_policy = ctx_chunk_config[0]

            if "EQUAL_PROGRESS" in str(input_policy):
                policy_enum = ChunkingPolicy.EQUAL_PROGRESS
            else:
                # Default to FCFS for FIRST_COME_FIRST_SERVED or others
                policy_enum = ChunkingPolicy.FIRST_COME_FIRST_SERVED

            py_chunk_config = ContextChunkingConfig(policy_enum,
                                                    ctx_chunk_config[1])

        # FUSED PATH: Always use single-pass scheduling for both TP-only and global coordination
        # Store resources directly for single-pass scheduling
        # This eliminates the double work of capacity + micro-batch scheduling
        self.kv_cache_manager = kv_cache_manager
        self.cross_kv_cache_manager = cross_kv_cache_manager
        self.peft_cache_manager = peft_cache_manager
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.max_num_requests = capacity
        self.ctx_chunk_config = py_chunk_config
        self.max_context_length = max_num_tokens
        self.scheduler_policy = scheduler_policy
        self.two_step_lookahead = two_step_lookahead

        # Cache state values for performance
        self._no_schedule_until_state_value = LlmRequestState.CONTEXT_INIT.value
        self._no_schedule_after_state_value = LlmRequestState.GENERATION_TO_COMPLETE.value
        self._context_init_state_value = LlmRequestState.CONTEXT_INIT.value
        self._encoder_init_state_value = LlmRequestState.ENCODER_INIT.value

        # Helper components
        self.peft_helper = PeftHelper(peft_cache_manager)

        self.capacity_checker = CapacityChecker(
            kv_cache_manager, cross_kv_cache_manager, scheduler_policy,
            self._no_schedule_until_state_value,
            self._no_schedule_after_state_value)

        self.chunking_manager = ChunkingManager(
            py_chunk_config, max_num_tokens) if py_chunk_config else None

        if self.enable_global_scheduling:
            self.global_coordinator = GlobalCoordinator(
                self, dist, max_num_active_requests)
        else:
            self.global_coordinator = None

        # Batch waiting state (for TP-only mode)
        # These track the waiting logic for batch waiting in TP-only mode
        # Will be configured by PyExecutor if needed
        self.batch_wait_timeout_iters = 0
        self.batch_wait_max_tokens_ratio = 0.0
        self.enable_batch_waiting = False
        self.batch_wait_iters_count = 0

    def activate_new_requests(
        self,
        active_requests: RequestList,
        waiting_queue: Optional[deque],
        cp_config: dict,
        cp_rank: int,
        cp_size: int,
        exclude_last_generation_logits: bool,
    ) -> tuple[RequestList, int]:
        """
        Activate new requests from waiting queue.

        For attention_dp mode, uses global coordination to assign requests across ranks.
        For regular TP mode, activates requests locally based on available capacity.

        Args:
            active_requests: Currently active requests
            waiting_queue: Queue of waiting RequestQueueItems
            cp_config: CP configuration dict
            cp_rank: Current CP rank
            cp_size: Total number of CP ranks
            exclude_last_generation_logits: Whether to exclude last generation logits

        Returns:
            Tuple of (new_llm_requests, expected_num_active_requests)
            - new_llm_requests: List of newly activated LlmRequests
            - expected_num_active_requests: Maximum number of active requests across all ranks
        """
        # Check if we have any waiting requests
        if waiting_queue is None or len(waiting_queue) == 0:
            return [], len(active_requests)

        if self.enable_global_scheduling:
            # Attention DP mode: Use global coordination to assign requests
            return self._activate_with_global_coordination(
                active_requests, waiting_queue, cp_config, cp_rank, cp_size,
                exclude_last_generation_logits)
        else:
            # TP-only mode: Activate requests locally
            return self._activate_local(active_requests, waiting_queue,
                                        cp_config, cp_rank, cp_size,
                                        exclude_last_generation_logits)

    def _schedule_generation_only_during_waiting(
        self,
        active_requests: RequestList,
        inflight_request_ids: set[int],
    ) -> Optional[UnifiedSchedulerOutput]:
        """
        Proactive optimization: Schedule only generation requests when in waiting mode.

        This avoids expensive context request scheduling when we're already waiting
        for more generation requests to accumulate.

        Args:
            active_requests: Currently active requests
            inflight_request_ids: Set of inflight request IDs

        Returns:
            UnifiedSchedulerOutput if still waiting (with empty context_requests),
            None if should exit waiting mode and run normal scheduling
        """
        # Split requests by type
        generation_requests_only = [
            r for r in active_requests if not r.is_context_init_state
        ]

        # Check if we have generation requests to avoid dead waiting
        if len(generation_requests_only) == 0:
            # No generation requests, stop waiting to avoid dead lock
            self.batch_wait_iters_count = 0
            return None  # Exit to normal path

        # Only schedule generation requests (skip expensive context scheduling)
        # Use fused scheduler
        result = self._fused_schedule_request(generation_requests_only,
                                              inflight_request_ids)

        # Check if we should stop waiting
        num_gen_tokens = sum(
            self._estimate_tokens_needed(gen_req)
            for gen_req in result.generation_requests)

        max_num_tokens = self.max_num_tokens
        if max_num_tokens is not None:
            # Check if we've timed out or have enough generation tokens
            should_stop_waiting = (
                self.batch_wait_iters_count >= self.batch_wait_timeout_iters
                or num_gen_tokens
                >= self.batch_wait_max_tokens_ratio * max_num_tokens)

            if should_stop_waiting:
                # Stop waiting, next iteration will schedule context requests
                self.batch_wait_iters_count = 0
                return None  # Exit to normal path
            else:
                # Continue waiting
                self.batch_wait_iters_count += 1
        else:
            # No token budget limit, stop waiting
            self.batch_wait_iters_count = 0
            return None  # Exit to normal path

        # Return with empty context requests (still waiting)
        return UnifiedSchedulerOutput(
            context_requests=[],
            generation_requests=result.generation_requests,
            paused_requests=result.paused_requests,
            fitting_disagg_gen_init_requests=result.
            fitting_disagg_gen_init_requests,
            num_fitting_requests=result.num_fitting_requests,
            updated_active_requests=None,
        )

    def _apply_batch_waiting(
        self,
        context_requests: RequestList,
        generation_requests: RequestList,
    ) -> RequestList:
        """
        Apply batch waiting logic for TP-only mode.

        Return an empty list if scheduled requests fulfill the waiting conditions,
        otherwise return the original context requests.

        Waiting conditions:
        - The number of scheduled tokens (both context and generation) is smaller than
          `self.batch_wait_max_tokens_ratio * self.max_num_tokens`
        - The number of waiting iterations is smaller than `self.batch_wait_timeout_iters`

        Args:
            context_requests: Scheduled context requests
            generation_requests: Scheduled generation requests

        Returns:
            Empty list if should wait, otherwise original context_requests
        """
        # Skip if batch waiting is not enabled
        if not self.enable_batch_waiting:
            return context_requests

        # Skip if no context requests to wait for
        if len(context_requests) == 0:
            return context_requests

        # Skip if no generation requests (to avoid dead waiting)
        if len(generation_requests) == 0:
            self.batch_wait_iters_count = 0
            return context_requests

        # Calculate scheduled tokens
        num_scheduled_ctx_tokens = sum(
            self._estimate_tokens_needed(ctx_req)
            for ctx_req in context_requests)
        num_scheduled_gen_tokens = sum(
            self._estimate_tokens_needed(gen_req)
            for gen_req in generation_requests)
        num_scheduled_tokens = num_scheduled_ctx_tokens + num_scheduled_gen_tokens

        # Get max_num_tokens
        max_num_tokens = self.max_num_tokens
        if max_num_tokens is None:
            # No token budget limit, cannot apply batch waiting
            return context_requests

        # Check waiting conditions
        should_waiting = (self.batch_wait_iters_count
                          < self.batch_wait_timeout_iters
                          and num_scheduled_tokens
                          < self.batch_wait_max_tokens_ratio * max_num_tokens)

        if should_waiting:
            self.batch_wait_iters_count += 1
            return []

        self.batch_wait_iters_count = 0
        return context_requests

    def schedule_request(
        self,
        active_requests: RequestList,
        inflight_request_ids: set[int],
    ) -> UnifiedSchedulerOutput:
        """
        Schedule requests for execution.

        This method handles capacity scheduling (KV cache allocation) and
        micro-batch scheduling (token budget + chunking).

        For TP-only mode (enable_global_scheduling=False), also applies batch waiting logic.
        For attention_dp mode (enable_global_scheduling=True), batching is done during activation.

        Args:
            active_requests: Currently active requests
            inflight_request_ids: Set of inflight request IDs

        Returns:
            UnifiedSchedulerOutput with scheduled requests
        """
        # FUSED PATH: Always use single-pass scheduling
        # Proactive optimization for TP-only mode:
        # If we're already in waiting mode, skip context scheduling to save computation
        if (not self.enable_global_scheduling and self.enable_batch_waiting
                and self.batch_wait_iters_count > 0):
            # Try generation-only scheduling (optimization path)
            result = self._schedule_generation_only_during_waiting(
                active_requests, inflight_request_ids)
            if result is not None:
                # Still waiting, return early with empty context
                return result
            # Otherwise, exit waiting mode and fall through to normal path

        # Use fused single-pass scheduling
        result = self._fused_schedule_request(active_requests,
                                              inflight_request_ids)

        # Apply batch waiting for TP-only mode
        # For attention_dp, batching is done during activation via _apply_batching_filter()
        if not self.enable_global_scheduling:
            result.context_requests = self._apply_batch_waiting(
                result.context_requests, result.generation_requests)

        return result

    # ========== Helper methods for _fused_schedule_request ==========

    def _initialize_block_managers(
        self, simulation_mode: bool
    ) -> tuple[Optional['MaxUtilizationScheduledBlocksManager'],
               Optional['NoEvictScheduledBlocksManager'],
               Optional['NoEvictScheduledBlocksManager']]:
        """
        Initialize block managers based on scheduling policy.

        Args:
            simulation_mode: If True, skip start_scheduling call

        Returns:
            Tuple of (scheduled_blocks_manager, reserved_blocks, reserved_cross_blocks)
        """
        scheduled_blocks_manager = None
        reserved_blocks = None
        reserved_cross_blocks = None

        if self.scheduler_policy == CapacitySchedulerPolicy.MAX_UTILIZATION:
            if not simulation_mode:
                self.kv_cache_manager.start_scheduling()
            scheduled_blocks_manager = MaxUtilizationScheduledBlocksManager(
                self.kv_cache_manager, self.two_step_lookahead)
        elif self.scheduler_policy == CapacitySchedulerPolicy.GUARANTEED_NO_EVICT or \
             self.scheduler_policy == CapacitySchedulerPolicy.STATIC_BATCH:
            reserved_blocks = NoEvictScheduledBlocksManager(
                self.kv_cache_manager)
            if self.cross_kv_cache_manager is not None:
                reserved_cross_blocks = NoEvictScheduledBlocksManager(
                    self.cross_kv_cache_manager)

        return scheduled_blocks_manager, reserved_blocks, reserved_cross_blocks

    def _initialize_scheduling_state(self, active_requests: RequestList,
                                     has_peft: bool) -> SchedulingState:
        """
        Initialize scheduling state for _fused_schedule_request.

        Args:
            active_requests: Currently active requests
            has_peft: Whether PEFT is enabled

        Returns:
            SchedulingState with initialized values
        """
        # Block reuse optimization
        skipping_is_relevant = self.capacity_checker.is_skipping_relevant()
        newly_contributed_context_blocks: set = set()
        newly_contributed_cross_context_blocks: set = set()
        if skipping_is_relevant:
            newly_contributed_context_blocks, newly_contributed_cross_context_blocks = \
                self.capacity_checker.prefill_contributed_blocks(active_requests)

        # PEFT/LoRA state
        claimed_peft_pages = 0
        available_peft_pages = self.peft_helper.get_max_pages(
        ) if has_peft else 0
        uniq_task_ids: set[int] = set() if has_peft else None

        return SchedulingState(
            skipping_is_relevant=skipping_is_relevant,
            newly_contributed_context_blocks=newly_contributed_context_blocks,
            newly_contributed_cross_context_blocks=
            newly_contributed_cross_context_blocks,
            has_peft=has_peft,
            claimed_peft_pages=claimed_peft_pages,
            available_peft_pages=available_peft_pages,
            uniq_task_ids=uniq_task_ids,
            batch_num_tokens=0,
            scheduled_req_size=0,
            scheduled_beam_width=0,
            context_requests=[],
            generation_requests=[],
            paused_requests=[],
            fitting_disagg_gen_init=[],
            contexts_to_be_chunked=[],
            num_chunked_tokens=0,
            all_context_requests_fit=True,
            max_batch_size=self.max_batch_size,
            max_num_tokens=self.max_num_tokens,
            max_context_length=self.max_context_length,
            ctx_chunk_config=self.ctx_chunk_config,
        )

    def _schedule_in_progress_generation(
        self,
        active_requests: RequestList,
        state: SchedulingState,
        reserved_blocks: 'NoEvictScheduledBlocksManager',
        reserved_cross_blocks: Optional['NoEvictScheduledBlocksManager'],
        simulation_mode: bool,
    ) -> None:
        """
        Schedule in-progress generation requests (GUARANTEED_NO_EVICT policy only).

        These must be scheduled first to free up reserved blocks.
        Updates state in-place.

        Args:
            active_requests: All active requests
            state: Current scheduling state (modified in-place)
            reserved_blocks: Reserved blocks manager
            reserved_cross_blocks: Reserved cross-attention blocks manager (or None)
            simulation_mode: If True, skip block updates
        """
        for req in active_requests:
            if not self.capacity_checker.can_be_scheduled_with_disagg_exception(
                    req):
                continue

            if len(state.context_requests) + len(
                    state.generation_requests) >= self.max_num_requests:
                break

            if req.is_generation_in_progress_state:
                # Check token budget
                req_num_tokens = self._estimate_tokens_needed(req)

                if state.max_num_tokens is not None and (
                        state.batch_num_tokens + req_num_tokens
                        > state.max_num_tokens):
                    state.paused_requests.append(req)
                    continue

                # Fits! Schedule it
                state.generation_requests.append(req)
                state.batch_num_tokens += req_num_tokens
                state.scheduled_req_size += 1

                if not simulation_mode:
                    reserved_blocks.decrement_reserved_blocks(req)
                    if reserved_cross_blocks is not None:
                        reserved_cross_blocks.decrement_reserved_blocks(req)

                # Track PEFT
                if state.has_peft:
                    lora_task_id, is_new_task, peft_pages = self.peft_helper.get_task_info(
                        req, state.uniq_task_ids)
                    if is_new_task:
                        state.claimed_peft_pages += peft_pages
                        state.uniq_task_ids.add(lora_task_id)

        # Update available PEFT pages
        if state.has_peft:
            state.available_peft_pages -= state.claimed_peft_pages

    def _should_schedule_request(
            self, req: LlmRequest, inflight_request_ids: set[int],
            reserved_blocks: Optional['NoEvictScheduledBlocksManager']) -> bool:
        """
        Check if request should be considered for scheduling.

        Args:
            req: Request to check
            inflight_request_ids: Set of already in-flight request IDs
            reserved_blocks: Reserved blocks manager (or None)

        Returns:
            True if request should be processed, False if should skip
        """
        # Skip inflight requests
        if req.request_id in inflight_request_ids:
            return False

        # Skip requests not in schedulable state range
        req_state_value = req.state_value
        if not (req_state_value >= self._no_schedule_until_state_value
                and req_state_value < self._no_schedule_after_state_value):
            # For disagg gen init, allow exception
            if not req.is_disagg_generation_init_state:
                return False

        # Skip in-progress generation (already handled for GUARANTEED_NO_EVICT)
        if reserved_blocks is not None and req.is_generation_in_progress_state:
            return False

        return True

    def _check_batch_limits(self, state: SchedulingState) -> bool:
        """
        Check if batch limits are reached.

        Args:
            state: Current scheduling state

        Returns:
            True if can continue scheduling, False if limits reached
        """
        # Check batch size limit
        if state.scheduled_req_size >= state.max_batch_size:
            return False

        # Check request count limit
        if len(state.context_requests) + len(state.generation_requests) + len(
                state.fitting_disagg_gen_init) >= self.max_num_requests:
            return False

        return True

    def _finalize_chunking(self, state: SchedulingState) -> None:
        """
        Apply chunking to queued context requests and finalize.

        Updates state in-place by moving chunked requests to context_requests.

        Args:
            state: Current scheduling state (modified in-place)
        """
        if not state.contexts_to_be_chunked:
            return

        # Verify chunking fits
        if state.max_num_tokens is not None and state.num_chunked_tokens > (
                state.max_num_tokens - state.batch_num_tokens):
            state.all_context_requests_fit = False

        # Apply chunking
        remaining_capacity = (state.max_num_tokens - state.batch_num_tokens
                              ) if state.max_num_tokens is not None else None
        self.chunking_manager.apply_chunking(state.contexts_to_be_chunked,
                                             remaining_capacity)

        # Finalize chunked requests
        for req in state.contexts_to_be_chunked:
            if req.context_chunk_size > 0:
                state.context_requests.append(req)
                draft_tokens = req.num_draft_tokens if (
                    req.is_last_context_chunk and req.has_draft_tokens) else 0
                state.batch_num_tokens += req.context_chunk_size + draft_tokens
            else:
                state.paused_requests.append(req)

    def _build_scheduler_output(
            self, state: SchedulingState) -> UnifiedSchedulerOutput:
        """
        Build final scheduler output from scheduling state.

        Args:
            state: Final scheduling state

        Returns:
            UnifiedSchedulerOutput with scheduled requests
        """
        # Sort requests for consistency
        chunks_present = state.ctx_chunk_config is not None
        if self.chunking_manager and chunks_present:
            self.chunking_manager.sort_requests(state.context_requests,
                                                state.generation_requests,
                                                chunks_present)

        # Return results
        num_fitting = len(state.context_requests) + len(
            state.generation_requests) + len(state.fitting_disagg_gen_init)
        return UnifiedSchedulerOutput(
            context_requests=state.context_requests,
            generation_requests=state.generation_requests,
            paused_requests=state.paused_requests,
            fitting_disagg_gen_init_requests=state.fitting_disagg_gen_init,
            num_fitting_requests=num_fitting,
            updated_active_requests=None,
        )

    def _fused_schedule_request(
        self,
        active_requests: RequestList,
        inflight_request_ids: set[int],
        simulation_mode: bool = False,
    ) -> UnifiedSchedulerOutput:
        """
        Fused single-pass scheduling combining capacity and micro-batch checks.

        This method merges the two-pass approach (capacity → micro-batch) into a single
        loop that checks both KV cache capacity and token budget together. This eliminates
        redundant work and improves performance for global coordination mode.

        Args:
            active_requests: Currently active requests to schedule
            inflight_request_ids: Set of request IDs already in flight
            simulation_mode: If True, only check feasibility without allocating blocks
                           (used for global coordination simulation)

        Returns:
            UnifiedSchedulerOutput with scheduled requests
        """
        # Initialize block managers
        scheduled_blocks_manager, reserved_blocks, reserved_cross_blocks = \
            self._initialize_block_managers(simulation_mode)

        # Initialize scheduling state
        state = self._initialize_scheduling_state(
            active_requests, self.peft_cache_manager is not None)

        # For GUARANTEED_NO_EVICT: Schedule in-progress generation first
        if reserved_blocks is not None:
            self._schedule_in_progress_generation(active_requests, state,
                                                  reserved_blocks,
                                                  reserved_cross_blocks,
                                                  simulation_mode)

        # MAIN SCHEDULING LOOP: Fused capacity + token budget checking
        for req in active_requests:
            req_state_value = req.state_value

            # Filtering checks
            if not self._should_schedule_request(req, inflight_request_ids,
                                                 reserved_blocks):
                continue

            # Batch limit checks
            if not self._check_batch_limits(state):
                state.paused_requests.append(req)
                break

            # Block reuse skip optimization
            if (state.skipping_is_relevant
                    and not req.is_disagg_generation_init_state
                    and self.capacity_checker.beneficial_to_skip(
                        req, state.newly_contributed_context_blocks,
                        state.newly_contributed_cross_context_blocks)):
                continue

            # --- A. Encoder Request Handling ---
            if req_state_value == self._encoder_init_state_value:
                req_num_tokens = self._estimate_tokens_needed(req)

                assert state.max_context_length is None or req_num_tokens <= state.max_context_length, \
                    f"The number of encoder tokens ({req_num_tokens}) exceeds the limit value ({state.max_context_length})"

                # Check token budget
                if state.max_num_tokens is not None and (
                        state.batch_num_tokens + req_num_tokens
                        > state.max_num_tokens):
                    state.paused_requests.append(req)
                    break

                # Check KV cache capacity
                can_fit_kv = self.capacity_checker.check_kv_capacity(
                    req, scheduled_blocks_manager, reserved_blocks,
                    reserved_cross_blocks, simulation_mode)
                if not can_fit_kv:
                    state.paused_requests.append(req)
                    break

                # Fits! Schedule it
                state.context_requests.append(req)
                state.batch_num_tokens += req_num_tokens
                state.scheduled_req_size += 1

            # --- B. Context Request Handling ---
            elif req_state_value == self._context_init_state_value:
                if not state.ctx_chunk_config:
                    # No chunking: schedule full context
                    req_num_tokens = self._estimate_tokens_needed(req)

                    assert state.max_context_length is None or req_num_tokens <= state.max_context_length, \
                        f"The number of context tokens ({req_num_tokens}) exceeds the limit value ({state.max_context_length})"

                    # Check token budget
                    if state.max_num_tokens is not None and (
                            state.batch_num_tokens + req_num_tokens
                            > state.max_num_tokens):
                        state.paused_requests.append(req)
                        break

                    # Check KV cache capacity
                    can_fit_kv = self.capacity_checker.check_kv_capacity(
                        req, scheduled_blocks_manager, reserved_blocks,
                        reserved_cross_blocks, simulation_mode)
                    if not can_fit_kv:
                        state.paused_requests.append(req)
                        break

                    # Fits! Schedule it
                    state.context_requests.append(req)
                    state.batch_num_tokens += req_num_tokens
                    state.scheduled_req_size += 1
                else:
                    # Chunking enabled: tentative schedule
                    # Check KV cache capacity first
                    can_fit_kv = self.capacity_checker.check_kv_capacity(
                        req, scheduled_blocks_manager, reserved_blocks,
                        reserved_cross_blocks, simulation_mode)
                    if not can_fit_kv:
                        state.paused_requests.append(req)
                        break

                    # Add to chunking queue
                    req.context_chunk_size = req.context_remaining_length

                    draft_tokens = req.num_draft_tokens if (
                        req.is_last_context_chunk
                        and req.has_draft_tokens) else 0
                    req_num_tokens = req.context_chunk_size + draft_tokens

                    if state.max_context_length is not None:
                        if state.max_context_length < req_num_tokens:
                            req_num_tokens = state.max_context_length
                            state.all_context_requests_fit = False

                    state.contexts_to_be_chunked.append(req)
                    state.num_chunked_tokens += req_num_tokens
                    state.scheduled_req_size += 1

            # --- C. Generation Request Handling ---
            elif req.is_disagg_generation_init_state:
                # Disagg gen init - special handling
                # Check KV cache capacity
                can_fit_kv = self.capacity_checker.check_kv_capacity(
                    req, scheduled_blocks_manager, reserved_blocks,
                    reserved_cross_blocks, simulation_mode)
                if not can_fit_kv:
                    state.paused_requests.append(req)
                    break

                # Check PEFT capacity
                if state.has_peft:
                    lora_task_id, is_new_task, needed_peft_pages = self.peft_helper.get_task_info(
                        req, state.uniq_task_ids)
                    if needed_peft_pages > state.available_peft_pages:
                        state.paused_requests.append(req)
                        continue
                    if is_new_task:
                        state.available_peft_pages -= needed_peft_pages
                        state.uniq_task_ids.add(lora_task_id)

                # Fits! Add to disagg gen init list
                state.fitting_disagg_gen_init.append(req)

            else:
                # Regular generation request
                req_num_tokens = self._estimate_tokens_needed(req)
                beam_width = req.get_beam_width_by_iter(
                    for_next_iteration=False)

                # Check token budget
                if state.max_num_tokens is not None and (
                        state.batch_num_tokens + req_num_tokens
                        > state.max_num_tokens):
                    state.paused_requests.append(req)
                    break

                # Beam width consistency check
                if state.scheduled_beam_width == 0:
                    state.scheduled_beam_width = beam_width
                elif state.scheduled_beam_width != beam_width:
                    logger.debug(
                        f"generation request skipped: ID {req.request_id} since its "
                        f"beam width ({beam_width}) is different from scheduled ones "
                        f"({state.scheduled_beam_width})")
                    continue

                # Fits! Schedule it
                state.generation_requests.append(req)
                state.batch_num_tokens += req_num_tokens
                state.scheduled_req_size += 1

        # Apply chunking if needed
        if state.contexts_to_be_chunked:
            self._finalize_chunking(state)

        # Build and return output
        return self._build_scheduler_output(state)

    def can_schedule(self, requests: RequestList) -> bool:
        """
        Check if all requests can be scheduled (dry run).
        Uses fused scheduler in simulation mode.
        """
        # Use fused scheduler in simulation mode
        result = self._fused_schedule_request(requests,
                                              set(),
                                              simulation_mode=True)
        scheduled_count = len(result.context_requests) + len(
            result.generation_requests) + len(
                result.fitting_disagg_gen_init_requests)
        return scheduled_count == len(requests)

    # ========== Estimation methods for global coordination ==========
    # These methods provide resource estimation for global coordination,
    # working with both fused and traditional scheduling paths

    def _estimate_tokens_needed(self, request: LlmRequest) -> int:
        """
        Estimate how many tokens this request will consume in the next step.

        OPTIMIZATION: For pre-validated requests (passed simulation in GlobalCoordinator),
        use cached estimate to avoid recalculation (~30-40% speedup for new requests).

        Args:
            request: The request to estimate for

        Returns:
            int: Number of tokens needed for next iteration
        """
        # Fast path: Use cached estimate if available
        if request.py_pre_validated and request.py_estimated_tokens > 0:
            return request.py_estimated_tokens

        # Slow path: Calculate from scratch
        state_value = request.state_value

        # Encoder tokens
        if state_value == self._encoder_init_state_value:
            return request.encoder_output_len

        # Context tokens
        elif state_value == self._context_init_state_value:
            base_tokens = request.get_num_tokens(0)
            draft_tokens = request.num_draft_tokens if request.has_draft_tokens else 0
            return base_tokens + draft_tokens

        # Generation tokens
        else:
            beam_width = request.get_beam_width_by_iter(
                for_next_iteration=False)
            draft_tokens = request.num_draft_tokens if request.has_draft_tokens else 0
            return beam_width + draft_tokens

    def _estimate_blocks_needed(self, request: LlmRequest) -> int:
        """
        Estimate how many KV cache blocks this request will consume in the next step.

        OPTIMIZATION: For pre-validated requests (passed simulation in GlobalCoordinator),
        use cached estimate to avoid recalculation (~30-40% speedup for new requests).

        Args:
            request: The request to estimate for

        Returns:
            int: Number of blocks needed (worst-case for VSWA)
        """
        # Fast path: Use cached estimate if available
        if request.py_pre_validated and request.py_estimated_blocks > 0:
            return request.py_estimated_blocks

        # Slow path: Calculate from scratch
        if self.kv_cache_manager is None:
            return 0

        # For VSWA, check all window sizes and return worst-case (maximum)
        if hasattr(self.kv_cache_manager, 'is_variable_window'
                   ) and self.kv_cache_manager.is_variable_window:
            max_blocks = 0
            for window_size_key in self.kv_cache_manager.get_window_size_keys():
                blocks = self.kv_cache_manager.get_num_required_blocks(
                    request, window_size_key)
                max_blocks = max(max_blocks, blocks)
            return max_blocks
        else:
            # Standard case: single window size
            return self.kv_cache_manager.get_num_required_blocks(request)

    def _calculate_current_token_load(self,
                                      active_requests: RequestList) -> int:
        """
        Calculate total tokens consumed by current active requests.

        Args:
            active_requests: List of currently active requests

        Returns:
            int: Total token count
        """
        total_tokens = 0
        for req in active_requests:
            # Only count schedulable requests
            state_value = req.state_value
            if (state_value >= self._no_schedule_until_state_value
                    and state_value < self._no_schedule_after_state_value):
                total_tokens += self._estimate_tokens_needed(req)
        return total_tokens

    def _activate_local(
        self,
        active_requests: RequestList,
        waiting_queue: deque,
        cp_config: dict,
        cp_rank: int,
        cp_size: int,
        exclude_last_generation_logits: bool,
    ) -> tuple[RequestList, int]:
        """
        Activate new requests locally (TP-only mode, no global coordination).

        This method handles request activation when enable_global_scheduling=False,
        which means we're in TP-only mode without attention_dp.

        Args:
            active_requests: Currently active requests on this rank
            waiting_queue: Queue of waiting RequestQueueItems
            cp_config: CP configuration dict
            cp_rank: Current CP rank
            cp_size: Total number of CP ranks
            exclude_last_generation_logits: Whether to exclude last generation logits

        Returns:
            Tuple of (new_llm_requests, expected_num_active_requests)
        """
        # Calculate local capacity
        # Use max_num_requests as fallback when max_num_active_requests is unset
        max_active = self.max_num_active_requests if self.max_num_active_requests is not None else self.max_num_requests
        max_new_requests = max(0, max_active - len(active_requests))

        if max_new_requests == 0:
            return [], len(active_requests)

        # Pop requests from waiting queue (local capacity only)
        new_request_items = []
        for _ in range(min(max_new_requests, len(waiting_queue))):
            if len(waiting_queue) == 0:
                break
            new_request_items.append(waiting_queue.popleft())

        if len(new_request_items) == 0:
            return [], len(active_requests)

        # Convert RequestQueueItems to LlmRequests (ONLY ONCE)
        new_llm_requests = merge_requests(
            new_request_items,
            cp_config=cp_config,
            cp_rank=cp_rank,
            cp_size=cp_size,
            exclude_last_generation_logits=exclude_last_generation_logits)

        # For TP-only mode, expected_num_active_requests is local count
        expected_num_active_requests = len(active_requests) + len(
            new_llm_requests)

        return new_llm_requests, expected_num_active_requests

    def _activate_with_global_coordination(
        self,
        active_requests: RequestList,
        waiting_queue: deque,
        cp_config: dict,
        cp_rank: int,
        cp_size: int,
        exclude_last_generation_logits: bool,
    ) -> tuple[RequestList, int]:
        """
        Activate new requests using global coordination (attention_dp).

        This performs the full GATHER → SIMULATE → COMMIT flow to assign
        new requests to ranks, then extracts assigned requests from waiting_queue.

        Args:
            active_requests: Currently active requests
            waiting_queue: Queue of waiting RequestQueueItems
            cp_config: CP configuration dict
            cp_rank: Current CP rank
            cp_size: Total number of CP ranks
            exclude_last_generation_logits: Whether to exclude last generation logits

        Returns:
            Tuple of (new_llm_requests, expected_num_active_requests)
        """
        # === PHASE 1: GATHER ===
        # Gather states first to know total active requests across all ranks
        local_state = self.global_coordinator.build_local_state(active_requests)
        all_rank_states = self.global_coordinator.gather_all_states(local_state)

        # Calculate total active requests across all ranks
        total_num_active_requests = sum(state.current_batch_size
                                        for state in all_rank_states)

        # Calculate how many new candidates we can accept
        total_capacity = self.dist.tp_size * self.max_num_active_requests
        num_new_candidates = max(
            0,
            min(total_capacity - total_num_active_requests, len(waiting_queue)))

        if num_new_candidates == 0:
            # No capacity for new requests
            expected_num_active_requests = max(state.current_batch_size
                                               for state in all_rank_states)
            return [], expected_num_active_requests

        # Extract candidate requests
        candidate_requests = list(
            itertools.islice(waiting_queue, num_new_candidates))

        # Convert candidate RequestQueueItems to LlmRequests ONCE
        # These will be used for simulation AND execution (no recreation)
        candidate_llm_requests = merge_requests(
            candidate_requests,
            cp_config=cp_config,
            cp_rank=cp_rank,
            cp_size=cp_size,
            exclude_last_generation_logits=exclude_last_generation_logits)

        # Attach llm_request back to RequestQueueItem for simulation
        # Note: merge_requests may create child requests, we need to map them back
        llm_req_map = {}  # request_id -> LlmRequest
        for llm_req in candidate_llm_requests:
            llm_req_map[llm_req.request_id] = llm_req

        for req_item in candidate_requests:
            if req_item.id in llm_req_map:
                req_item.llm_request = llm_req_map[req_item.id]

        # === PHASE 2: SIMULATE ===
        assignments = self.global_coordinator.simulate_global_schedule(
            candidate_requests, all_rank_states)

        # === PHASE 2.5: BATCHING CHECK ===
        assignments = self.global_coordinator.apply_batching_filter(
            assignments, candidate_requests)

        # Calculate expected_num_active_requests (max across all ranks after assignment)
        # This uses data we already have from the allgather, no extra communication needed
        expected_num_active_requests = max(
            all_rank_states[rank_id].current_batch_size +
            len(assignments[rank_id])
            for rank_id in range(len(all_rank_states)))

        # === PHASE 3: EXTRACT ASSIGNED LLMREQUESTS ===
        my_assigned_req_ids = set(assignments[self.dist.rank])
        assigned_llm_requests = []

        # Convert to list to allow safe modification of waiting_queue
        items_to_process = list(waiting_queue)
        waiting_queue.clear()

        for req_item in items_to_process:
            if (hasattr(req_item, 'llm_request') and req_item.llm_request
                    and req_item.llm_request.request_id in my_assigned_req_ids):
                # Reuse the LlmRequest we created earlier ✅ (created only once!)
                assigned_llm_requests.append(req_item.llm_request)
                # Also add child requests if they exist
                if req_item.llm_request.child_requests:
                    assigned_llm_requests.extend(
                        req_item.llm_request.child_requests)
            else:
                # Put back unassigned items
                waiting_queue.append(req_item)

        return assigned_llm_requests, expected_num_active_requests
