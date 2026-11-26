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
import datetime
import enum
import gc
import json
import os
import time
import traceback
import uuid
import weakref
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Set

import torch

from tensorrt_llm.logger import logger

from .._torch.peft.lora.manager import LoraManager
from .._torch.pyexecutor.iteration_stats import IterationStatsRecord
from .._torch.pyexecutor.llm_request import release_undelivered_response
from .._utils import (global_mpi_rank, global_mpi_size, mpi_rank,
                      nvtx_range_debug)
from ..bindings import executor as tllm
from ..bindings.BuildInfo import check_cuda_architecture_supported
# submit and handle_for_worker are a frontend in the engine's process
from ..llmapi.engine_client import proto_wire
from ..llmapi.engine_client.executor import GenerationExecutor
from ..llmapi.engine_client.postproc_worker import PostprocWorker
from ..llmapi.engine_client.result import GenerationResult
from ..llmapi.llm_args import BaseLlmArgs, ExecutorMemoryType
from ..llmapi.tracer import global_tracer
from ..llmapi.utils import _SyncQueue, configure_cpu_affinity, logger_debug
from ..tokenizer import TokenizerBase
from . import engine_server
from ._load_balance_env import (LAUNCH_QUEUES_ENV, configure_moe_launch_queues,
                                uses_collective_shutdown)
from .ipc import IpcQueue
from .request import GenerationRequest, LoRARequest
from .startup_checks import no_visible_gpu_error
from .utils import (ErrorResponse, PostprocWorkerConfig, RequestError,
                    bucket_responses_by_frontend)
from .wire import StepFrame

if TYPE_CHECKING:
    from .._torch.disaggregation.kv_cache_transceiver import KvCacheTransceiver

__all__ = [
    "BaseWorker",
    "_init_hf_modules",
]


def _init_hf_modules():
    """Initialize cached HuggingFace modules for models with trust_remote_code=True.

    This is safe to call multiple times (idempotent) and should be called:
    1. At module import time (for main process and spawned subprocesses)
    2. At worker_main entry (for forked processes or external MPI ranks)

    References: https://github.com/vllm-project/vllm/pull/871
    """
    try:
        from transformers.dynamic_module_utils import init_hf_modules
        init_hf_modules()
        logger.debug("HF modules initialized")
    except ImportError as e:
        logger.warning(f"ImportError initializing HF modules: {e}")
    except Exception as e:
        logger.error(f"Exception initializing HF modules: {e}")


_init_hf_modules()


class BaseWorker(GenerationExecutor):

    class WorkerExit(GeneratorExit):
        pass

    def __init__(
        self,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
        is_llm_executor: Optional[bool] = None,
        hf_model_dir: Optional[Path] = None,
        tokenizer: Optional[TokenizerBase] = None,
        llm_args: Optional[BaseLlmArgs] = None,
    ) -> None:
        # Direct worker construction also bypasses BaseLLM/worker_main.
        launch_queues = os.environ.get(LAUNCH_QUEUES_ENV)
        if llm_args is not None and llm_args.env_overrides:
            os.environ.update(llm_args.env_overrides)
        if llm_args is not None and llm_args.backend == "pytorch":
            queue_overrides = configure_moe_launch_queues(
                getattr(llm_args, "moe_config", None),
                llm_args.env_overrides,
                initial_setting=launch_queues)
            if queue_overrides is not llm_args.env_overrides:
                llm_args.env_overrides = queue_overrides

        postproc_config = postproc_worker_config or PostprocWorkerConfig()
        super().__init__(
            num_postprocess_workers=postproc_config.num_postprocess_workers,
            postprocess_tokenizer_dir=postproc_config.postprocess_tokenizer_dir,
            is_llm_executor=is_llm_executor,
        )
        # GenerationExecutor.__init__ rebuilds postproc_config from the two
        # fields above, dropping the rest (e.g. post_processor_hook). Workers
        # that spawn their own postproc pool (RpcWorkerMixin) need the full
        # configuration, so keep the caller-provided one.
        self.postproc_config = postproc_config

        # inputs
        self._is_llm_executor = is_llm_executor
        self._hf_model_dir = hf_model_dir
        self._tokenizer = tokenizer
        self.llm_args = llm_args

        self.engine = None
        self.result_queue: Optional[IpcQueue] = None
        self.postproc_queues: Optional[List[IpcQueue]] = None
        # Multi-frontend serving: one result lane per frontend process,
        # selected by the frontend id in client_id's top bits.
        self.frontend_result_queues: Optional[List[IpcQueue]] = None
        self.rank = mpi_rank()
        self.global_rank = global_mpi_rank()
        # mapping: client_id -> rank 0's record of the request (engine_server.RequestEntry)
        self._results: Dict[int, "engine_server.RequestEntry"] = {}
        # mapping: client_id -> the GenerationResult a caller in this process waits on
        self._frontend_results: Dict[int, GenerationResult] = {}
        # mapping: client_id from Proxy -> request_id returned from runtime backend
        self._client_id_to_request_id: Dict[int, int] = {}
        # Requests engine_server answered with an error and aborted: their
        # prompt logprobs could not be computed (_prepare), or the native
        # publish could not say a response (_publish). Their remaining engine
        # responses are dropped.
        self._refused_client_ids: Set[int] = set()
        # The requests whose rows rank 0 prepares in Python (engine_server.py).
        self._prepared_client_ids: set[int] = set()
        self._await_response_helper = AwaitResponseHelper(weakref.proxy(self))
        self._backend = None if llm_args is None else llm_args.backend
        self._lora_config = llm_args.lora_config if llm_args is not None else None
        # Maps adapter ID to (canonical path, checkpoint source).
        self._lora_adapter_identities: dict[str, tuple[str, str]] = {}
        # Weak values avoid retaining locks for failed, one-off adapter IDs.
        self._lora_adapter_id_locks = weakref.WeakValueDictionary()
        # Ensures concurrent requests for one adapter ID share the same lock.
        self._lora_adapter_id_locks_guard = Lock()
        self._resource_governor_queue = None

        if global_mpi_size() > 1:
            logger.set_rank(self.global_rank)

    @property
    def resource_governor_queue(self):
        return self._resource_governor_queue

    def _configure_device(self):
        # Ranks launched on other nodes (mpirun, Slurm, trtllm-llmapi-launch)
        # are not covered by the GPU check in the LLM's own process.
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise no_visible_gpu_error()
        device_id = self.global_rank % device_count
        # Fail before loading the model: missing optimized kernels for this GPU otherwise
        # surface much later as unrelated errors such as running out of memory.
        check_cuda_architecture_supported(device_id)
        torch.cuda.set_device(device_id)
        configure_cpu_affinity(device_id)

    def setup_engine(self):
        """
        Setup the engine for the worker.
        """

        if self.llm_args is None:
            raise ValueError("llm_args is required to set up the worker engine")
        if self._backend != "pytorch":
            raise ValueError(f"Unsupported backend config: {self._backend}")


        from tensorrt_llm._torch.pyexecutor.model_loader import \
            construct_checkpoint_loader
        from tensorrt_llm._torch.pyexecutor.py_executor_creator import \
            create_py_executor

        self._configure_device()
        self.mapping = self.llm_args.parallel_config.to_mapping()
        self.checkpoint_loader = construct_checkpoint_loader(
            self.llm_args.checkpoint_loader,
            self.llm_args.checkpoint_format,
            mx_config=self.llm_args.mx_config,
            checkpoint_io_policy=self.llm_args.checkpoint_io_policy,
            load_format=self.llm_args.load_format,
            partial_model_loading=self.llm_args.is_partial_model_loading,
        )
        args = {
            "llm_args": self.llm_args,
            "checkpoint_dir": self._hf_model_dir,
            "tokenizer": self._tokenizer,
        }
        if self._resource_governor_queue is not None:
            args["resource_governor_queue"] = self._resource_governor_queue

        self.max_seq_len = self.llm_args.max_seq_len
        self.engine = create_py_executor(**args)
        # Model construction may update the maximum sequence length.
        if self.engine.max_seq_len is not None:
            self.max_seq_len = self.engine.max_seq_len

        self._lora_manager: Optional[LoraManager] = None

        if self._lora_config is not None:
            from tensorrt_llm._torch.pyexecutor.resource_manager import \
                ResourceManagerType
            peft_cache_manager = self.engine.resource_manager.resource_managers.get(
                ResourceManagerType.PEFT_CACHE_MANAGER)
            self._lora_manager = peft_cache_manager.get_lora_manager()
            lora_model_config = self.engine.model_engine.lora_model_config
            assert lora_model_config is not None
            self._lora_model_config = lora_model_config

    def await_responses(self, timeout: Optional[float] = None) -> list:
        return self.engine.await_responses(timeout=datetime.timedelta(
            seconds=timeout) if timeout is not None else None)

    def fetch_stats(self) -> list:
        return self.engine.get_latest_iteration_stats()

    def fetch_kv_cache_capacity(self) -> dict:
        if self.engine is None:
            return {}

        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        if isinstance(self.engine, PyExecutor):
            return self.engine.get_kv_cache_capacity()

        return {}

    def get_kv_cache_capacity(self) -> dict:
        """Read KV capacity directly when this worker serves as the executor."""
        return self.fetch_kv_cache_capacity()

    def fetch_openengine_discovery(self) -> dict:
        if self.engine is None:
            return {}

        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        if isinstance(self.engine, PyExecutor):
            return self.engine.get_openengine_discovery()

        return {}

    def get_openengine_discovery(self) -> dict:
        """Read OpenEngine discovery directly when this worker serves as the executor."""
        return self.fetch_openengine_discovery()

    def fetch_kv_cache_load(self) -> dict:
        if self.engine is None:
            return {}

        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        if isinstance(self.engine, PyExecutor):
            return self.engine.get_kv_cache_load()

        return {}

    def get_kv_cache_load(self) -> dict:
        """Read KV load directly when this worker serves as the executor."""
        return self.fetch_kv_cache_load()

    def fetch_kv_cache_events(self) -> list:
        return self.engine.get_latest_kv_cache_events()

    def set_result_queue(self, queue):
        """In multi-gpu mode, result_queue will be set here to communicate between the proxy and the worker 0 process."""
        assert self.postproc_queues is None
        assert self.frontend_result_queues is None
        self.result_queue = queue

    def set_postproc_queues(self,
                            queues: list["IpcQueue"],
                            *,
                            coexist_with_result_queue: bool = False) -> None:
        """ Set the IPC queues for feeding post-processing processes.

        coexist_with_result_queue: the classic proxy gives each PostprocWorker
        its own push lane to the frontend, so a result_queue must not exist
        there. Under RPC/Ray orchestration the finished PostprocWorker.Output
        records are collected back INTO the result queue (the single RPC
        response stream), so both queues legitimately coexist.
        """
        if not coexist_with_result_queue:
            assert self.result_queue is None
        assert self.frontend_result_queues is None
        self.postproc_queues = queues

    def set_frontend_result_queues(self, queues: List["IpcQueue"]):
        """Multi-frontend serving: one result lane per frontend process."""
        assert self.result_queue is None
        assert self.postproc_queues is None
        self.frontend_result_queues = queues

    def return_queue(self, client_id: int):
        """Return the queue used to deliver responses for ``client_id``.

        If a centralized result queue is registered (used for communication
        with the proxy) send the message there. Otherwise, push the result
        directly in the GenerationResult queue.
        """
        if self.result_queue is not None:
            return self.result_queue
        return self._frontend_results[client_id].queue

    def abort_request(self, client_id: int) -> None:
        # NOTE: the request_id is the request_id generated by cpp runtime, not the client_id
        can_cancel = getattr(self.engine, "_can_enqueue_cancellation", None)
        if (can_cancel() if callable(can_cancel) else
                self.engine.can_enqueue_requests()):
            request_id = self._client_id_to_request_id.get(client_id, None)
            if request_id is None:
                cancel_transfer = getattr(self.engine,
                                          "_cancel_context_transfer", None)
                if callable(cancel_transfer):
                    # The terminal result can precede KV retirement. Resolve
                    # this identity on the executor thread, without retaining
                    # every completed result's routing entry indefinitely.
                    cancel_transfer(client_id)
                    return
                logger.warning(
                    f"Request of client_id {client_id} is finished, cannot abort it."
                )
                return
            self.engine.cancel_request(request_id)

    def _has_background_error(self) -> bool:
        return not self._error_queue.empty()

    def _create_error_response(self, response: tllm.Response) -> ErrorResponse:
        bck_error = self._error_queue.get_nowait()
        assert isinstance(bck_error, Exception)
        return ErrorResponse(response.client_id, str(bck_error),
                             response.request_id)

    def start(self):
        raise NotImplementedError(
            "start method is not implemented in BaseWorker")

    def _load_lora_adapter(self, lora_request: LoRARequest) -> None:
        """Ensure the adapter is loaded into the local cache."""
        adapter_id = str(lora_request.adapter_id)
        with self._lora_adapter_id_locks_guard:
            adapter_lock = self._lora_adapter_id_locks.get(adapter_id)
            if adapter_lock is None:
                adapter_lock = Lock()
                self._lora_adapter_id_locks[adapter_id] = adapter_lock

        with adapter_lock:
            previous = self._lora_adapter_identities.get(adapter_id)
            identity = (lora_request.path, lora_request.ckpt_source)
            # An empty path retains the trusted API's ID-only cache lookup.
            if lora_request.path:
                same_source = (previous is not None
                               and previous[1] == lora_request.ckpt_source)
                if previous is None or (same_source and previous != identity):
                    # Preserve path-alias support while avoiding filesystem
                    # access for the common exact-identity reuse case.
                    identity = (str(Path(lora_request.path).resolve()),
                                lora_request.ckpt_source)
                if previous is not None and previous != identity:
                    logger.warning(
                        "Rejecting conflicting LoRA adapter identity for ID %s: "
                        "existing=%r, requested=%r", adapter_id, previous,
                        identity)
                    raise ValueError(
                        f"LoRA adapter ID {adapter_id} is already bound to a "
                        "different path or checkpoint source; use a new ID.")
            loaded = False
            try:
                self._lora_manager.load_from_ckpt(
                    [lora_request.path],
                    model_config=self._lora_model_config,
                    uids=[adapter_id],
                    ckpt_source=lora_request.ckpt_source)
                loaded = True
            finally:
                # Failed lookups are retryable unless the loader published UID
                # state. Keep bindings for partial loads and across eviction.
                if (loaded or adapter_id in self._lora_manager.cpp_lora_weights
                        or adapter_id in self._lora_manager.cpp_lora_config):
                    self._lora_adapter_identities.setdefault(
                        adapter_id, identity)

    def _load_lora_config(self, adapter_id: int, path: str,
                          ckpt_source: str) -> tllm.LoraConfig:
        """Load the adapter ``adapter_id`` on rank 0 and return its LoRA config.

        Each rank loads independently from disk via py_lora_path in
        PeftCacheManager.add_request_peft(); this warms rank 0's cache.

        Raises:
            RequestError: the adapter cannot be loaded.
        """
        try:
            self._load_lora_adapter(
                LoRARequest(str(adapter_id), adapter_id, path, ckpt_source))
            return tllm.LoraConfig(
                task_id=adapter_id,
                weights=None,
                config=self._lora_manager.cpp_lora_config[str(adapter_id)])
        except Exception as e:  # any failure fails the request, not the worker
            raise RequestError(f"Failed to load LoRA adapter: {e}") from e

    def submit(self, request: GenerationRequest) -> GenerationResult:
        """ Low-level API to the executor. Return a "future" GenerationResult which can be waited. """
        self.start()

        if self.rank != 0:
            raise RuntimeError(
                "Only rank 0 can submit requests.\n"
                "To fix this, ensure that the llm.generate(...) method is "
                "guarded with the `if __name__ == '__main__':` block.")

        client_id = request.id if request.id is not None else self._get_next_client_id(
        )
        if request.id is None:
            request.set_id(client_id)

        logprob_params = self._get_logprob_params(request)

        result = GenerationResult(
            request,
            background_error_handler=self._handle_background_error,
            executor=self,
            disaggregated_params=request.disaggregated_params,
            logprob_params=logprob_params)

        self._frontend_results[client_id] = result
        try:
            engine_server.submit(self,
                                 proto_wire.encode(request, postprocess=False),
                                 raise_errors=True)
        except BaseException:
            self._frontend_results.pop(client_id, None)
            raise

        self._handle_background_error()

        return result

    def _check_sleep_wakeup_preconditions(self, method: str) -> None:
        """Validate preconditions shared by sleep() and wakeup().

        Args:
            method: Name of the calling method (``"sleep"`` or ``"wakeup"``)
                used in error messages.

        Raises:
            ValueError: If the backend is not ``"pytorch"`` or
                ``sleep_config`` is not set.
        """
        if self._backend != "pytorch":
            raise ValueError(
                f"{method}() is only available for the PyTorch (TorchLLM) "
                "backend.")
        if self.llm_args is None or self.llm_args.sleep_config is None:
            raise ValueError(
                "Sleep feature is not enabled, please set sleep_config in "
                "the LLM arguments.")

    def _validate_sleep_tags(self, tags: list[ExecutorMemoryType]) -> None:
        validator = getattr(self.engine, "validate_sleep_tags", None)
        if validator is not None:
            validator(tags)

    def _invalidate_v1_prefix_cache_for_sleep(
            self, tags: list[ExecutorMemoryType]) -> None:
        invalidator = getattr(self.engine,
                              "invalidate_v1_prefix_cache_for_sleep", None)
        if invalidator is not None:
            invalidator(tags)

    def _multi_rank_sleep_wakeup(
        self,
        action: Literal["sleep", "wakeup"],
        tags: list[ExecutorMemoryType],
    ) -> None:
        """Coordinate a sleep or wakeup operation across all MPI ranks.

        Called on rank-0 only for ``world_size > 1`` deployments.

        Sequence:
        1. Enter ``control_action()`` to drain in-flight requests and pause
           rank-0's event loop.  Non-rank-0 event loops become idle (starved
           of NCCL collectives from rank-0) once the current iteration drains.
        2. Send PREPARE to every non-rank-0 rank via the dedicated
           ``_sleep_wakeup_comm`` communicator.  Peers quiesce and ACK without
           changing VMM state.
        3. When native MNNVL MoE resources are selected, send COMMIT to every
           prepared peer before rank-0 enters the subgroup checkpoint
           collectives. Otherwise execute the local VMM operation first.
        4. Execute the selected VMM and MNNVL operations and collect bounded
           COMMIT ACKs. If PREPARE or the initial rank-0 synchronization fails,
           send ABORT before any rank changes VMM state. Once COMMIT or local
           mutation starts, any error fail-stops the distributed worker because
           rank state can no longer be reconciled safely.
        5. Exit ``control_action()`` only after a successful operation or a
           recoverable pre-COMMIT abort. The public ``sleep()``/``wakeup()``
           wrapper owns the persistent admission transition around this helper.

        Args:
            action: ``"sleep"`` or ``"wakeup"``.
            tags: Parsed :class:`~tensorrt_llm.llmapi.llm_args.ExecutorMemoryType`
                values; forwarded verbatim to each rank's VMM call.
        """
        from tensorrt_llm._torch.pyexecutor.py_executor import (
            _SLEEP_WAKEUP_ACK_TIMEOUT_S, _recv_sleep_wakeup_ack_until,
            _SleepWakeupAction, _SleepWakeupTag)
        from tensorrt_llm._torch.virtual_memory import (materialize_with_tag,
                                                        release_with_tag)

        if self.rank != 0:
            raise RuntimeError(
                "_multi_rank_sleep_wakeup must only be called on rank 0")

        sleep_wakeup_comm = self.engine._sleep_wakeup_comm
        if sleep_wakeup_comm is None:
            raise RuntimeError(
                "_sleep_wakeup_comm not initialised; was start_worker() called?"
            )

        world_size = self.llm_args.parallel_config.world_size
        if world_size <= 1:
            raise ValueError(
                "_multi_rank_sleep_wakeup requires world_size greater than 1")

        tag_strings = [t.value for t in tags]
        op_id = uuid.uuid4().hex
        target_action = _SleepWakeupAction(action)
        has_mnnvl_resources = getattr(self.engine,
                                      "_has_mnnvl_checkpoint_resources",
                                      lambda _tags: False)(tags)
        prepare_msg = {
            "action": _SleepWakeupAction.PREPARE,
            "target_action": target_action,
            "tags": tag_strings,
            "op_id": op_id,
        }
        commit_msg = {
            "action": _SleepWakeupAction.COMMIT,
            "target_action": target_action,
            "tags": tag_strings,
            "op_id": op_id,
        }

        # Serialise concurrent sleep/wakeup calls.  control_action() uses an
        # Event-based barrier, not a mutex, so two concurrent callers can both
        # pass the barrier and then interleave sends/recvs on _sleep_wakeup_comm,
        # consuming the wrong ACKs or resuming the event loop prematurely.
        # _sleep_wakeup_lock turns the whole sequence into a critical section.
        with self.engine._sleep_wakeup_lock, self.engine.control_action(
                control_id=op_id):
            prepared_ranks = []
            errors = []
            local_error = None
            abort_sent = False
            commit_ranks = []
            commit_sent_early = False
            local_commit_started = False
            abort_incomplete = False

            def send_abort(reason: str,
                           ranks: Optional[list[int]] = None) -> list[int]:
                nonlocal abort_incomplete
                abort_ranks = []
                abort_dests = ranks if ranks is not None else range(
                    1, world_size)
                abort_msg = {
                    "action": _SleepWakeupAction.ABORT,
                    "tags": [],
                    "op_id": op_id,
                    "reason": reason,
                }
                for abort_dest in abort_dests:
                    try:
                        sleep_wakeup_comm.send(
                            abort_msg,
                            dest=abort_dest,
                            tag=_SleepWakeupTag.ACTION,
                        )
                        abort_ranks.append(abort_dest)
                    except Exception as abort_exc:
                        abort_incomplete = True
                        abort_error = (
                            "rank 0 failed to send sleep/wakeup abort "
                            f"to rank {abort_dest}: {abort_exc}")
                        errors.append(abort_error)
                        logger.error(
                            f"_multi_rank_sleep_wakeup: {abort_error}\n"
                            f"{traceback.format_exc()}")
                return abort_ranks

            def drain_acks(ranks: list[int],
                           phase: _SleepWakeupAction) -> list[dict]:
                nonlocal abort_incomplete
                received_acks = []
                ack_deadline = time.monotonic() + _SLEEP_WAKEUP_ACK_TIMEOUT_S
                for src in ranks:
                    try:
                        ack = _recv_sleep_wakeup_ack_until(sleep_wakeup_comm,
                                                           src,
                                                           ack_deadline,
                                                           expected_op_id=op_id,
                                                           expected_phase=phase)
                    except Exception as exc:
                        if phase == _SleepWakeupAction.ABORT:
                            abort_incomplete = True
                        errors.append(
                            f"rank 0 failed to receive {phase} ACK from "
                            f"rank {src}: {exc}")
                        logger.error(
                            "_multi_rank_sleep_wakeup: failed to receive "
                            f"{phase} ACK from rank {src}\n{traceback.format_exc()}"
                        )
                        continue
                    received_acks.append(ack)
                    if ack.get("status") != "ok":
                        if phase == _SleepWakeupAction.ABORT:
                            abort_incomplete = True
                        errors.append(
                            ack.get("error")
                            or f"rank {src} returned unknown {phase} ACK")
                return received_acks

            def send_commits() -> list[int]:
                sent_ranks = []
                failed_ranks = []
                for dest in prepared_ranks:
                    try:
                        sleep_wakeup_comm.send(
                            commit_msg,
                            dest=dest,
                            tag=_SleepWakeupTag.ACTION,
                        )
                        sent_ranks.append(dest)
                    except Exception as exc:
                        commit_error = (
                            f"rank 0 failed to send '{action}' commit to "
                            f"rank {dest}: {exc}")
                        errors.append(commit_error)
                        failed_ranks.append(dest)
                        logger.error(
                            f"_multi_rank_sleep_wakeup: {commit_error}\n"
                            f"{traceback.format_exc()}")
                if failed_ranks:
                    abort_ranks = send_abort("\n".join(errors),
                                             ranks=failed_ranks)
                    drain_acks(abort_ranks, _SleepWakeupAction.ABORT)
                return sent_ranks

            try:
                # Phase 1: prepare peers.  A prepared peer has reached the
                # control barrier and synchronized CUDA, but has not modified
                # VMM state yet.  This keeps send/local failures from leaving
                # a subset of ranks slept/woken while rank 0 did not commit.
                for dest in range(1, world_size):
                    try:
                        sleep_wakeup_comm.send(prepare_msg,
                                               dest=dest,
                                               tag=_SleepWakeupTag.ACTION)
                        prepared_ranks.append(dest)
                    except Exception as exc:
                        send_error = (
                            f"rank 0 failed to send '{action}' prepare to rank "
                            f"{dest}: {exc}")
                        errors.append(send_error)
                        logger.error(f"_multi_rank_sleep_wakeup: {send_error}\n"
                                     f"{traceback.format_exc()}")
                        abort_ranks = send_abort(send_error)
                        abort_sent = True
                        drain_acks(prepared_ranks, _SleepWakeupAction.PREPARE)
                        drain_acks(abort_ranks, _SleepWakeupAction.ABORT)
                        break

                if not errors:
                    prepare_acks = drain_acks(prepared_ranks,
                                              _SleepWakeupAction.PREPARE)
                    has_mnnvl_resources = has_mnnvl_resources or any(
                        ack.get("has_mnnvl_resources", False)
                        for ack in prepare_acks)

                if not errors:
                    # MNNVL resource hooks contain subgroup handle exchange,
                    # so peers must enter COMMIT before rank 0 executes them.
                    if has_mnnvl_resources:
                        # A failed MPI send has uncertain delivery, so any
                        # attempted COMMIT requires the bounded local phase.
                        commit_sent_early = bool(prepared_ranks)
                        commit_ranks = send_commits()

                if not errors or commit_sent_early:
                    torch.cuda.synchronize()
                    local_commit_started = True
                    if action == _SleepWakeupAction.SLEEP:
                        run_mnnvl = (getattr(
                            self.engine, "_run_mnnvl_checkpoint_resources",
                            None) if has_mnnvl_resources else None)
                        if run_mnnvl is not None:
                            run_mnnvl(target_action, tags)
                        self._invalidate_v1_prefix_cache_for_sleep(tags)
                        release_with_tag(*tags)
                        torch.cuda.synchronize()
                        gc.collect()
                        torch.cuda.empty_cache()
                    else:
                        materialize_with_tag(*tags)
                        torch.cuda.synchronize()
                        run_mnnvl = (getattr(
                            self.engine, "_run_mnnvl_checkpoint_resources",
                            None) if has_mnnvl_resources else None)
                        if run_mnnvl is not None:
                            run_mnnvl(target_action, tags)
            except Exception as exc:
                local_error = (f"rank 0 '{action}' failed: {exc}\n"
                               f"{traceback.format_exc()}")
                logger.error(f"_multi_rank_sleep_wakeup: {local_error}")
            finally:
                if local_error:
                    errors.append(local_error)

                if commit_sent_early:
                    drain_acks(commit_ranks, _SleepWakeupAction.COMMIT)

                if (errors and prepared_ranks and not abort_sent
                        and not commit_sent_early):
                    abort_ranks = send_abort("\n".join(errors))
                    drain_acks(abort_ranks, _SleepWakeupAction.ABORT)
                elif not errors and not commit_sent_early:
                    commit_ranks = send_commits()
                    drain_acks(commit_ranks, _SleepWakeupAction.COMMIT)

                if errors:
                    operation_error = RuntimeError(
                        f"{action}() failed on {len(errors)} rank(s):\n" +
                        "\n".join(errors))
                    if commit_sent_early or local_commit_started or abort_incomplete:
                        self._fail_stop_divergent_sleep_wakeup(operation_error)
                    raise operation_error

    def _fail_stop_divergent_sleep_wakeup(self, error: RuntimeError) -> None:
        """Stop every rank after a sleep/wakeup operation may have diverged."""
        from tensorrt_llm._torch.pyexecutor.hang_detector import \
            propagate_hard_kill

        self._set_fatal_error(error)
        if self.engine is not None:
            fail_transition = getattr(
                self.engine,
                "fail_sleep_wakeup_transition",
                None,
            )
            if fail_transition is not None:
                fail_transition()
            if getattr(self.engine, "_fatal_error", None) is None:
                self.engine._fatal_error = error
            self.engine.is_shutdown = True
        try:
            logger.critical("Distributed sleep/wakeup state may have diverged; "
                            f"hard-killing all ranks: {error}")
        finally:
            propagate_hard_kill()

    def get_memory_status(self) -> dict[str, object]:
        """Return the authoritative runtime-memory state for this worker."""
        self._check_sleep_wakeup_preconditions("get_memory_status")
        return self.engine.get_memory_status()

    def get_memory_report(self) -> list[str]:
        """Explain where GPU memory is going, as readable reports.

        Built by the GPU memory monitor (``tensorrt_llm._torch.memory.monitor``)
        in every monitor mode, including ``off``. This worker's own report comes
        first. Under the MPI executor only rank 0 is asked; it adds the other
        ranks' reports in the monitor's full mode, and otherwise one line per
        rank saying they were not gathered. A report that could not be built
        is replaced by a line saying so.
        """
        from tensorrt_llm._torch.memory.monitor import current_rank
        from tensorrt_llm._torch.memory.monitor.providers import NOT_MONITORED
        from tensorrt_llm._torch.memory.monitor.triggers import report_now

        token = getattr(self.engine, "_memory_monitor_token", None)
        if token is not None and not token.monitored:
            # Another executor in this process holds the monitor; a report
            # would describe that executor's memory.
            dist = getattr(self.engine, "dist", None)
            world_size = dist.world_size if dist is not None else 1
            return [
                f"rank {rank}: {NOT_MONITORED}" for rank in range(world_size)
            ]
        # Answered on an RPC thread, whose current device is not necessarily
        # the one this worker serves on; report_now selects it.
        own = report_now(tag="on_demand",
                         force=True,
                         device_index=getattr(self.engine, "device_id", None),
                         save=False)
        if own is None:
            dist = getattr(self.engine, "dist", None)
            rank = dist.rank if dist is not None else current_rank()
            own = (f"rank {rank}: the report could not be built; "
                   "see this rank's log")
        reports = [own]
        gather = getattr(self.engine, "gather_memory_reports", None)
        if gather is not None:
            reports.extend(gather())
        return reports

    def _prepare_sleep_tags(
        self,
        tags: list[ExecutorMemoryType],
    ) -> list[ExecutorMemoryType]:
        get_status = getattr(self.engine, "get_memory_status", None)
        if get_status is None:
            return tags

        status = get_status()
        state = status["state"]
        parked_tags = {ExecutorMemoryType(tag) for tag in status["parked_tags"]}
        requested_tags = set(tags)
        if state == "running":
            return tags
        if state == "parked":
            if requested_tags.issubset(parked_tags):
                return []
            additional_tags = requested_tags - parked_tags
            raise RuntimeError(
                "Runtime memory is already parked; resume it before releasing "
                f"additional tags: {sorted(tag.value for tag in additional_tags)}"
            )
        raise RuntimeError(
            f"Cannot release runtime memory while state is '{state}'.")

    def _prepare_wakeup_tags(
        self,
        tags: list[ExecutorMemoryType],
    ) -> list[ExecutorMemoryType]:
        get_status = getattr(self.engine, "get_memory_status", None)
        if get_status is None:
            return tags

        status = get_status()
        state = status["state"]
        if state == "running":
            return []
        if state != "parked":
            raise RuntimeError(
                f"Cannot resume runtime memory while state is '{state}'.")

        parked_tags = {ExecutorMemoryType(tag) for tag in status["parked_tags"]}
        return [tag for tag in tags if tag in parked_tags]

    def sleep(self, sleep_tags: list[str]) -> None:
        """Release GPU virtual memory for the specified memory type tags.

        Supports both single-rank (``world_size == 1``) and multi-rank
        (TP/PP > 1) MPI executor deployments.  For multi-rank, a lightweight
        control-listener thread on each non-rank-0 worker handles the local
        ``release_with_tag()`` call while rank-0 coordinates via a dedicated
        secondary MPI communicator.

        Uses ``PyExecutor.control_action()`` to drain in-flight requests and
        pause rank-0's event loop, matching the ``@control_action_decorator``
        behaviour used in the Ray executor.

        Only allocations backed by virtual memory (VMM) and registered under
        the active :class:`~tensorrt_llm.llmapi.llm_args.SleepConfig` are
        released.  Components using alternative memory management (e.g.
        ``LoadFormat.GMS``-managed weights) are not VMM-tagged and will be
        silently skipped by ``release_with_tag``.

        Args:
            sleep_tags: List of
                :class:`~tensorrt_llm.llmapi.llm_args.ExecutorMemoryType`
                value strings (e.g. ``["kv_cache"]``).

        Returns:
            None. The call is synchronous; when it returns all requested
            VMM-tagged allocations have been released on every rank and request
            admission remains closed until ``wakeup()`` succeeds.

        Raises:
            ValueError: If the backend is not ``"pytorch"`` or
                ``sleep_config`` is not set.
        """
        self._check_sleep_wakeup_preconditions("sleep")

        from tensorrt_llm._torch.virtual_memory import release_with_tag

        tags = [ExecutorMemoryType(tag) for tag in sleep_tags]
        tags = self._prepare_sleep_tags(tags)
        if not tags:
            return
        self._validate_sleep_tags(tags)
        logger.info(f"Sleep: {tags}")
        self.engine.begin_sleep_transition(tags)
        local_mutation_started = False
        try:
            if self.llm_args.parallel_config.world_size > 1:
                self._multi_rank_sleep_wakeup("sleep", tags)
            else:
                with self.engine._sleep_wakeup_lock, self.engine.control_action(
                ):
                    torch.cuda.synchronize()
                    local_mutation_started = True
                    self._invalidate_v1_prefix_cache_for_sleep(tags)
                    release_with_tag(*tags)
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
        except Exception:
            if local_mutation_started:
                self.engine.fail_sleep_wakeup_transition()
            self.engine.abort_sleep_transition()
            raise
        try:
            self.engine.complete_sleep_transition()
        except Exception:
            self.engine.fail_sleep_wakeup_transition()
            raise

    def wakeup(self, wakeup_tags: list[str]) -> None:
        """Materialize GPU virtual memory for the specified memory type tags.

        Supports both single-rank (``world_size == 1``) and multi-rank
        (TP/PP > 1) MPI executor deployments.  See :meth:`sleep` for details
        on VMM scope restrictions, backend prerequisites, and multi-rank
        coordination.

        Args:
            wakeup_tags: List of
                :class:`~tensorrt_llm.llmapi.llm_args.ExecutorMemoryType`
                value strings (e.g. ``["kv_cache"]``).

        Returns:
            None. The call is synchronous; when it returns all requested
            VMM-tagged allocations have been materialized on every rank.
            Request admission reopens once every tag from the corresponding
            sleep transition has been restored.

        Raises:
            ValueError: If the backend is not ``"pytorch"`` or
                ``sleep_config`` is not set.
        """
        self._check_sleep_wakeup_preconditions("wakeup")

        from tensorrt_llm._torch.virtual_memory import materialize_with_tag

        tags = [ExecutorMemoryType(tag) for tag in wakeup_tags]
        tags = self._prepare_wakeup_tags(tags)
        if not tags:
            return
        logger.info(f"Wakeup: {tags}")
        self.engine.begin_wakeup_transition(tags)
        local_mutation_started = False
        try:
            if self.llm_args.parallel_config.world_size > 1:
                self._multi_rank_sleep_wakeup("wakeup", tags)
            else:
                with self.engine._sleep_wakeup_lock, self.engine.control_action(
                ):
                    torch.cuda.synchronize()
                    local_mutation_started = True
                    materialize_with_tag(*tags)
                    torch.cuda.synchronize()
        except Exception:
            if local_mutation_started:
                self.engine.fail_sleep_wakeup_transition()
            self.engine.abort_wakeup_transition()
            raise
        try:
            self.engine.complete_wakeup_transition()
        except Exception:
            self.engine.fail_sleep_wakeup_transition()
            raise

    def shutdown(self):
        if self.doing_shutdown:
            return
        if not uses_collective_shutdown(getattr(self, "llm_args", None)):
            self._shutdown_rank_local()
            return
        self.doing_shutdown = True

        try:
            if self.engine is not None:
                engine = self.engine
                engine_shutdown_completed = bool(
                    getattr(self, "_engine_shutdown_completed", False))
                can_shutdown = getattr(engine, "can_shutdown", None)
                if can_shutdown is None:
                    can_shutdown = engine.can_enqueue_requests
                # PyExecutor separates leader-only shutdown publication from the
                # all-rank join/finalize path. Other backends retain their gate.
                # A retry must bypass that gate once shutdown already completed.
                if (engine_shutdown_completed
                        or getattr(engine, "shutdown_all_ranks", False)
                        or can_shutdown()):
                    shutdown_error = None
                    local_safe = engine_shutdown_completed
                    if not local_safe:
                        try:
                            engine.shutdown()
                            self._engine_shutdown_completed = True
                            local_safe = True
                        except BaseException as error:
                            shutdown_error = error
                    terminal_cleanup = getattr(engine, "terminal_cleanup", None)
                    if terminal_cleanup is not None:
                        try:
                            if getattr(engine, "shutdown_all_ranks", False):
                                terminal_cleanup(local_safe=local_safe)
                            elif local_safe:
                                terminal_cleanup()
                        except BaseException as error:
                            if shutdown_error is None:
                                shutdown_error = error
                    if shutdown_error is not None:
                        raise shutdown_error
                    self.engine = None
                    self._engine_shutdown_completed = False
        except BaseException:
            # Keep the engine reachable so a caller can retry a failed
            # collective terminal cleanup in this long-lived worker process.
            # Only an explicit call retries it; see __del__.
            self.doing_shutdown = False
            self._collective_shutdown_failed = True
            raise
        self._collective_shutdown_failed = False

    def _shutdown_rank_local(self):
        """Best-effort teardown for configurations without collective EP resources."""
        self.doing_shutdown = True

        if self.engine is not None:
            can_shutdown = getattr(self.engine, "can_shutdown", None)
            if can_shutdown is None:
                can_shutdown = self.engine.can_enqueue_requests
            if can_shutdown():
                self.engine.shutdown()
                if getattr(self.engine, "local_cleanup", None) is not None:
                    self.engine.local_cleanup()
                self.engine = None

    def get_disaggregated_params(self) -> dict:
        if self.engine is None or self.engine.kv_cache_transceiver is None:
            logger.warning("Engine or kv cache transceiver is not initialized")
            return {}
        return self.engine.kv_cache_transceiver.get_disaggregated_params()

    def get_cache_transceiver(self) -> Optional["KvCacheTransceiver"]:
        if self.engine is None:
            return None
        return self.engine.kv_cache_transceiver

    def get_data_transceiver_state(self) -> bytes:
        if self.engine is None or self.engine.kv_cache_transceiver is None:
            return b""
        return self.engine.kv_cache_transceiver.get_data_transceiver_state()

    def get_startup_metrics(self) -> dict:
        """Return rank-local startup metrics for the PyTorch backend."""
        if self.engine is None:
            return {}

        startup_metrics = {}
        executor_metrics = dict(getattr(self.engine, "metrics", {}))
        for model_engine_stage in ("initial_model_engine",
                                   "final_model_engine"):
            model_engine_metrics = executor_metrics.pop(model_engine_stage,
                                                        None)
            if model_engine_metrics is not None:
                startup_metrics[model_engine_stage] = dict(model_engine_metrics)
        startup_metrics["py_executor"] = executor_metrics

        # An executor without a model engine holds its loader itself.
        loader_owner = getattr(self.engine, "model_engine", self.engine)
        model_loader = getattr(loader_owner, "model_loader", None)
        if model_loader is not None:
            startup_metrics["model_loader"] = {
                **model_loader.metrics,
                **getattr(model_loader, "startup_metadata", {}),
            }

        draft_model_engine = getattr(self.engine, "draft_model_engine", None)
        draft_model_loader = getattr(draft_model_engine, "model_loader", None)
        if draft_model_loader is not None:
            startup_metrics["draft_model_loader"] = {
                **draft_model_loader.metrics,
                **getattr(draft_model_loader, "startup_metadata", {}),
            }

        return startup_metrics

    def get_engine_config(self) -> dict:
        """Return resolved model and KV-cache dtype metadata from the runtime."""
        model_engine = getattr(self.engine, "model_engine", None)
        model_config = getattr(getattr(model_engine, "model", None),
                               "model_config", None)
        if model_config is None:
            return {}

        from .._utils import torch_dtype_to_str
        dtype = model_config.torch_dtype
        pretrained_config = getattr(model_config, "pretrained_config", None)
        if dtype is None and pretrained_config is not None:
            dtype = pretrained_config.get_text_config().torch_dtype
        quant_config = getattr(model_config, "quant_config", None)
        kv_algo = getattr(quant_config, "kv_cache_quant_algo", None)
        quant_algo = getattr(quant_config, "quant_algo", None)
        return {
            "dtype": None if dtype is None else torch_dtype_to_str(dtype),
            "kv_cache_dtype": None if kv_algo is None else str(kv_algo),
            "quantization": None if quant_algo is None else str(quant_algo),
        }

    def start_profile(self,
                      output_dir: Optional[str] = None,
                      num_steps: Optional[int] = None,
                      start_step: int = 0,
                      activities: Optional[List[str]] = None) -> None:
        """Forward profiling request to the underlying PyExecutor engine.

        Logs a warning if the engine is uninitialized or does not expose
        the profiling API.
        """
        if self.engine is None:
            logger.warning(
                "start_profile called but engine is not initialized.")
            return
        start_fn = getattr(self.engine, "start_profile", None)
        if start_fn is None:
            logger.warning("Current engine does not support start_profile; "
                           "this API requires the PyTorch PyExecutor backend.")
            return
        start_fn(output_dir=output_dir,
                 num_steps=num_steps,
                 start_step=start_step,
                 activities=activities)

    def stop_profile(self) -> None:
        """Forward stop_profile request to the underlying PyExecutor engine."""
        if self.engine is None:
            logger.warning("stop_profile called but engine is not initialized.")
            return
        stop_fn = getattr(self.engine, "stop_profile", None)
        if stop_fn is None:
            logger.warning("Current engine does not support stop_profile; "
                           "this API requires the PyTorch PyExecutor backend.")
            return
        stop_fn()

    @staticmethod
    def _stats_serializer(stats) -> str:
        """Serialize named stats records, legacy tuples, or tagged per-rank rows."""
        # Per-rank path: stats is ("per_rank_dict", {..., "rank": N}).
        # Already serialized on the producing rank via allgather — just emit.
        if (isinstance(stats, tuple) and len(stats) == 2
                and stats[0] == "per_rank_dict"):
            return json.dumps(stats[1])
        # Attention-DP fanout rows are serialized when rank 0 appends them.
        if isinstance(stats, str):
            return stats

        entry = IterationStatsRecord.from_tuple(stats)
        return entry.to_json()

    @staticmethod
    def _kv_cache_capacity_serializer(capacity) -> str:
        return json.dumps(capacity)

    @staticmethod
    def _kv_cache_load_serializer(load) -> str:
        return json.dumps(load)

    # Define a Callable to serialize KV cache events
    @staticmethod
    def _kv_cache_events_serializer(events) -> str:
        from .._utils import KVCacheEventSerializer
        return json.dumps(KVCacheEventSerializer.serialize(events))

    def _pop_result(self, client_id: int):
        self._results.pop(client_id, None)
        self._frontend_results.pop(client_id, None)
        self._client_id_to_request_id.pop(client_id, None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.shutdown()
        return exc_type is None or exc_type == self.WorkerExit

    def __del__(self):
        # Finalizers run at a different time on each rank, so they must not
        # retry a failed collective shutdown.
        if getattr(self, "_collective_shutdown_failed", False):
            return
        self.shutdown()


class AwaitResponseHelper:
    ''' Multiple-implementations for await_response for performance. '''

    class HandlerKind(enum.Enum):
        unknown = 0
        single_process_worker = 1
        ipc_batched = 2

    def __init__(self, worker: "BaseWorker"):
        # TODO: make worker weakref
        self.worker = worker
        self.handler_kind: AwaitResponseHelper.HandlerKind = AwaitResponseHelper.HandlerKind.unknown
        self.enable_postprocprocess_parallel = self.worker.enable_postprocess_parallel
        # The error responses when submit request failed will be put here
        self.temp_error_responses = Queue()

    def _resolve_handler_kind(self) -> "AwaitResponseHelper.HandlerKind":
        """Determine (and memoise) which side of the IPC boundary we are on.

        Split out of ``responses_handler`` so the error path can ask the same
        question without having handled a response batch first — a crash
        during the very first ``await_responses`` leaves ``handler_kind``
        ``unknown`` otherwise.
        """
        HandlerKind = AwaitResponseHelper.HandlerKind

        if self.handler_kind is HandlerKind.unknown:
            has_ipc_queues = (self.worker.result_queue is not None
                              or self.worker.postproc_queues is not None
                              or self.worker.frontend_result_queues is not None)
            if not has_ipc_queues:
                logger_debug(f"creating await_response helper for Worker\n",
                             color="yellow")
                # When ExecutorBindingWorker is used in the main process
                # aka the single process mode
                self.handler_kind = HandlerKind.single_process_worker
            else:
                # The ExecutorBindingProxy is used
                logger_debug(f"creating await_response helper for IPC\n",
                             color="yellow")
                self.handler_kind = HandlerKind.ipc_batched
        return self.handler_kind

    def responses_handler(self, responses: List[tllm.Response]):
        HandlerKind = AwaitResponseHelper.HandlerKind

        self._resolve_handler_kind()

        match self.handler_kind:
            case HandlerKind.single_process_worker:
                return self.handle_for_worker(responses)
            case HandlerKind.ipc_batched:
                return self.handle_for_ipc_batched(responses)
            case _:
                raise NotImplementedError

    def process_responses(
            self, responses: List[tllm.Response]) -> List[tllm.Response]:
        """Filter empty engine responses, then append deferred submission errors."""
        responses = [response for response in responses if response is not None]

        # Drain with get_nowait(): this may run concurrently from the
        # ManagedThread and RPC fetch_responses(), and empty()+get() can
        # block forever if another consumer wins the race.
        while True:
            try:
                responses.append(self.temp_error_responses.get_nowait())
            except Empty:
                break

        return responses

    def process_and_handle_responses(
            self, responses: List[tllm.Response]) -> List[tllm.Response]:
        """Process engine responses and dispatch the client-visible results."""
        responses = self.process_responses(responses)
        with nvtx_range_debug(f"await_response-{len(responses)}",
                              color="red",
                              category="Worker"):
            self.responses_handler(responses)
        return responses

    def __call__(self, timeout: Optional[float] = None) -> bool:
        ''' This method should be called by a ManagedThread. '''
        timeout = timeout or 0.1
        try:
            responses = self.worker.engine.await_responses(
                timeout=datetime.timedelta(seconds=timeout))
        except Exception as e:
            # Defensive: with id=None, PyExecutor.await_responses routes
            # to _await_any_response, which does not raise on event-loop
            # crash — it returns [] silently and we detect the crash
            # via engine._event_loop_error after this block. But any
            # unexpected exception out of await_responses (e.g. from a
            # different engine implementation, or a future change to
            # _await_any_response) is also a clear signal to broadcast
            # and stop the thread.
            return self._broadcast_event_loop_error(e)
        self.process_and_handle_responses(responses)

        # Even when await_responses returned normally (e.g. via
        # _await_any_response, whose predicate already includes
        # is_shutdown but does not raise), an event-loop crash leaves
        # _event_loop_error stashed on the engine. Broadcast and stop the
        # thread in that case too — see nvbug 6038228.
        error = getattr(self.worker.engine, "_event_loop_error", None)
        if error is not None:
            # _broadcast_event_loop_error owns the delivery gate: it is the
            # only place that knows whether a client was actually woken.
            return self._broadcast_event_loop_error(error)
        return True

    def _broadcast_event_loop_error(self, error: BaseException) -> bool:
        """Wake every pending ``GenerationResult`` after an event-loop crash.

        Inject an ``ErrorResponse`` into every pending ``GenerationResult``
        queue so callers parked in ``queue.get()`` / ``aqueue.get()``
        (``LLM.generate``, ``generate_async`` + ``aresult``,
        ``trtllm-bench``) wake up with a meaningful error instead of
        hanging when the PyExecutor event loop dies.

        Returns ``False`` so the calling ``ManagedThread`` exits — there
        is no point polling a dead engine.

        Scope: single-process worker (the path that backs ``LLM.generate``
        and the bench async client). The IPC / proxy path tracks pending
        results on a different side of the boundary and would need a
        separate poison-pill on ``self.worker.result_queue``; that is left
        as a follow-up consistent with the PyExecutor-side fix.

        Because of that scope, this method also owns the rank-crash kill's
        delivery gate. The gate may only be set when a client verifiably
        woke: on ``ipc_batched`` the queues written below have no reader
        (responses travel via ``handle_for_ipc_batched``), so setting it
        there would stand the kill down while the peer ranks are still
        stranded — the case the kill exists for, in the default spawned-
        worker deployment. When delivery cannot be proven the gate stays
        clear and the kill fires, which is the safe direction: a spurious
        world-kill costs a traceback, a missed one costs the job.
        """
        error_msg = f"Event loop terminated with error: {error}"
        can_reach_client = (
            self._resolve_handler_kind()
            is AwaitResponseHelper.HandlerKind.single_process_worker)
        pending_client_ids = list(self.worker._results.keys())
        if not pending_client_ids:
            logger.error(
                f"Event-loop error with no pending results to wake: {error}")
            return False

        logger.error(
            f"Broadcasting event-loop error to {len(pending_client_ids)} "
            f"pending request(s): {error}")

        event_loop = None
        async_queues: List[_SyncQueue] = []
        # Counts queues a caller can actually read from. A _SyncQueue is only
        # readable once notify_many() has run, so those are counted there.
        woken = 0
        for client_id in pending_client_ids:
            try:
                queue = self.worker.return_queue(client_id)
            except KeyError:
                continue
            err_resp = ErrorResponse(
                client_id=client_id,
                error_msg=error_msg,
                request_id=client_id,
            )
            try:
                if isinstance(queue, _SyncQueue):
                    queue.put_nowait(err_resp)
                    async_queues.append(queue)
                    event_loop = event_loop or queue.loop
                elif queue is self.worker.result_queue:
                    # the result queue carries lists, as publish sends them
                    queue.put([err_resp])
                    woken += 1
                else:
                    queue.put(err_resp)
                    woken += 1
            except Exception as put_error:
                logger.error(f"Failed to push ErrorResponse for client_id="
                             f"{client_id}: {put_error}")
                continue
            self.worker._pop_result(client_id)

        if async_queues:
            try:
                _SyncQueue.notify_many(event_loop, async_queues)
                woken += len(async_queues)
            except Exception as notify_error:
                logger.error(
                    f"Failed to notify async queues on event-loop error: "
                    f"{notify_error}")

        if woken and can_reach_client:
            # A client is now holding the real error, so the crash is
            # reportable without killing the world: a symmetric crash (every
            # rank raised the same deterministic error, nobody stranded) ends
            # in N tracebacks rather than in MPI_Abort replacing them with a
            # bare exit 137.
            delivered = getattr(self.worker.engine,
                                "_event_loop_error_delivered", None)
            if delivered is not None:
                delivered.set()
        elif not can_reach_client:
            logger.error(
                "Event-loop error broadcast cannot reach the client on the "
                "IPC/proxy path; leaving the rank-crash hard kill armed so "
                "peer ranks are not stranded.")

        return False

    def handle_for_worker(self, responses: List[tllm.Response]) -> None:
        ''' Return the responses to asyncio.event_loop. '''
        event_loop = None
        async_queues = []
        results = self.worker._frontend_results
        for published in engine_server.publish_here(self.worker, responses):
            rows = (proto_wire.decode(published, results) if isinstance(
                published, StepFrame) else (published, ))
            for row in rows:
                result = results.get(row.client_id)
                if result is None:
                    proto_wire.release_undelivered(row)
                    continue
                queue = result.queue

                # For AsyncQueue.sync_q, we will batch the events to avoid too
                # many event notifications, thus put without wait here.
                if isinstance(queue, _SyncQueue):
                    global_tracer().log_instant("worker-rsp.put")
                    queue.put_nowait(row)
                    async_queues.append(queue)
                    # all the loops are identical
                    event_loop = event_loop or queue.loop
                else:
                    queue.put(row)

                # the last row ends the result here; rank 0 dropped its record
                # of the request when it published that row
                if isinstance(row, ErrorResponse) or row.result.is_final:
                    results.pop(row.client_id, None)

        # Notify the events in bulk for performance.
        if async_queues:
            _SyncQueue.notify_many(event_loop, async_queues)

    def handle_for_ipc_batched(self, responses: List[tllm.Response]) -> None:
        ''' Perform the IPC in batch explicitly. '''
        if self.worker._has_background_error():
            errors = []
            for response in responses:
                if isinstance(response, ErrorResponse):
                    errors.append(response)
                elif not engine_server.drop_refused_response(
                        self.worker, response):
                    release_undelivered_response(response)
                    errors.append(self.worker._create_error_response(response))
        else:
            errors = engine_server.publish(self.worker, responses,
                                           self.enable_postprocprocess_parallel)
        if not errors:
            return
        for error in errors:
            self.worker._pop_result(error.client_id)
        # Errors take the direct route to the frontends where one exists: under
        # RPC/Ray orchestration the worker holds the result_queue (the RPC
        # response stream feed) beside the postprocess workers' queues. The
        # classic proxy gives each postprocess worker its own lane to the
        # frontend instead, so there errors go through them.
        if (lanes := self.worker.frontend_result_queues) is not None:
            for frontend_id, sub_batch in enumerate(
                    bucket_responses_by_frontend(errors, len(lanes))):
                if sub_batch:
                    lanes[frontend_id].put(sub_batch)
        elif self.worker.result_queue is not None:
            self.worker.result_queue.put(errors)
        else:
            num_workers = self.worker.postproc_config.num_postprocess_workers
            for error in errors:
                self.worker.postproc_queues[error.client_id % num_workers].put(
                    PostprocWorker.Input(error))
