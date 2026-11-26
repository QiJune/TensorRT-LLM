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

import asyncio
import concurrent.futures
import ctypes
import os
import re
import sys
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Callable, List, NamedTuple, Optional, Union

from strenum import StrEnum

from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.llmapi.utils import enable_llm_debug, logger_debug

from ..bindings import executor as tllm
from ..logger import logger
from ..metrics import RequestEventTiming
from .mpi_session import (MpiCommSession, MpiPoolSession, MpiSession,
                          RemoteMpiCommSessionClient)


class LlmLauncherEnvs(StrEnum):
    # Spawn a process for the LLM-API Proxy
    TLLM_SPAWN_PROXY_PROCESS = "TLLM_SPAWN_PROXY_PROCESS"
    TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR = "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR"
    TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY = "TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY"


_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY: bytes | None = None
_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _scrub_process_env_value(key_name: str, value: str) -> None:
    if sys.platform != "linux":
        return

    libc = ctypes.CDLL(None, use_errno=True)
    libc.getenv.restype = ctypes.c_void_p
    value_ptr = libc.getenv(os.fsencode(key_name))
    if value_ptr:
        ctypes.memset(value_ptr, 0, len(os.fsencode(value)))


def _normalize_spawn_proxy_process_ipc_hmac_key(key: str) -> bytes:
    if not _SPAWN_PROXY_PROCESS_IPC_HMAC_KEY_PATTERN.fullmatch(key):
        raise ValueError("IPC HMAC key must be a 64-character hex string.")

    try:
        key_bytes = bytes.fromhex(key)
    except ValueError as exc:
        raise ValueError(
            "IPC HMAC key must be a 64-character hex string.") from exc

    if len(key_bytes) != 32:
        raise ValueError("IPC HMAC key must be 32 bytes.")
    return key_bytes


def get_spawn_proxy_process_ipc_addr_env() -> str | None:
    ''' Get the IPC address for the spawn proxy process dynamically. '''
    return os.getenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR)


def get_spawn_proxy_process_ipc_hmac_key_env() -> bytes:
    ''' Get the HMAC key for the spawn proxy process dynamically. '''
    global _SPAWN_PROXY_PROCESS_IPC_HMAC_KEY
    if _SPAWN_PROXY_PROCESS_IPC_HMAC_KEY is not None:
        return _SPAWN_PROXY_PROCESS_IPC_HMAC_KEY

    env_name = LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.value
    key = os.environ.get(env_name)
    if key is None:
        raise RuntimeError(
            f"{LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY} is not set. "
            "HMAC encryption is required for IPC communication.")
    _scrub_process_env_value(env_name, key)
    os.environ.pop(env_name, None)

    _SPAWN_PROXY_PROCESS_IPC_HMAC_KEY = (
        _normalize_spawn_proxy_process_ipc_hmac_key(key))
    return _SPAWN_PROXY_PROCESS_IPC_HMAC_KEY


def get_spawn_proxy_process_env() -> bool:
    ''' Get the environment variable for the spawn proxy process dynamically. '''
    return os.getenv(LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS) == "1"


def create_mpi_comm_session(
        n_workers: int) -> RemoteMpiCommSessionClient | MpiPoolSession:
    assert mpi_rank(
    ) == 0, f"create_mpi_comm_session must be called by rank 0, but it was called by rank {mpi_rank()}"
    if get_spawn_proxy_process_env():
        assert get_spawn_proxy_process_ipc_addr_env(
        ), f"{LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR} is not set."
        logger_debug(
            f"Using RemoteMpiPoolSessionClient to bind to external MPI processes at {get_spawn_proxy_process_ipc_addr_env()}\n",
            "yellow")
        hmac_key = get_spawn_proxy_process_ipc_hmac_key_env()
        return RemoteMpiCommSessionClient(
            addr=get_spawn_proxy_process_ipc_addr_env(), hmac_key=hmac_key)
    else:
        logger_debug(
            f"Using MpiCommSession to bind to external MPI processes\n",
            "yellow")
        return MpiCommSession(n_workers=n_workers)


def has_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class RequestError(RuntimeError):
    ''' The error raised when the request is failed. '''


# Machine-readable error code the OpenAI-compatible server attaches to
# over-length-prompt rejections (mirrors OpenAI's "context_length_exceeded").
CONTEXT_LENGTH_EXCEEDED_CODE = "context_length_exceeded"


def context_length_exceeded_message(max_context_length: int,
                                    num_prompt_tokens: int) -> str:
    """Client-facing message for a prompt that exceeds the model context.

    Mirrors OpenAI's wording so agent frameworks that pattern-match it (to
    trigger context compaction) work unchanged. Keep in sync with
    ``is_context_length_exceeded_message``.
    """
    return (
        f"This model's maximum context length is {max_context_length} tokens. "
        f"However, your messages resulted in {num_prompt_tokens} tokens. "
        "Please reduce the length of the messages.")


def check_prompt_capacity(num_prompt_tokens: int, max_prompt_len: int) -> None:
    """Raise ValueError with the context-length message for a prompt longer than max_prompt_len."""
    if num_prompt_tokens > max_prompt_len:
        raise ValueError(
            context_length_exceeded_message(
                max_context_length=max_prompt_len,
                num_prompt_tokens=num_prompt_tokens))


_CONTEXT_LENGTH_EXCEEDED_RE = re.compile(
    r"This model's maximum context length is \d+ tokens\. "
    r"However, your messages resulted in \d+ tokens\.")


def is_context_length_exceeded_message(message: str) -> bool:
    """True if ``message`` contains a ``context_length_exceeded_message``.

    Substring search rather than equality: the message crosses the executor
    IPC boundary as a plain string (``RequestError``) and may arrive wrapped
    in extra context.
    """
    return bool(_CONTEXT_LENGTH_EXCEEDED_RE.search(message))


class EngineDeadError(RuntimeError):
    """Raised by pending and new requests once the engine is known dead.

    Sticky and engine-level (unlike the per-request ``RequestError``): when a
    worker process dies, every queued ``GenerationResult`` is unblocked with this
    error and every subsequent ``submit()`` raises it immediately, instead of
    blocking forever on a response queue whose producer is gone.
    """

    def __init__(self, root_cause: Optional[BaseException] = None):
        msg = "Engine has died"
        if root_cause is not None:
            msg += f": {type(root_cause).__name__}: {root_cause}"
        super().__init__(msg)
        self.root_cause = root_cause


class ProcessPoolExecutorSession(MpiSession):
    # This process pool is introduced for better recoverable exceptions handling.
    # It replaces MpiPoolExecutor for single-gpu case.

    def __init__(self, n_workers: int, **kwargs):
        self.n_workers = n_workers
        self.mpi_pool = ProcessPoolExecutor(max_workers=self.n_workers,
                                            **kwargs)

    def submit(self, task: Callable, *args,
               **kwargs) -> List[concurrent.futures.Future]:
        return [
            self.mpi_pool.submit(task, *args, **kwargs)
            for i in range(self.n_workers)
        ]

    def submit_sync(self, task: Callable, *args, **kwargs) -> List[Any]:
        futures = [
            self.mpi_pool.submit(task, *args, **kwargs)
            for i in range(self.n_workers)
        ]
        return [future.result() for future in futures]

    def shutdown(self):
        self.mpi_pool.shutdown(wait=True)


class ErrorResponse(NamedTuple):
    client_id: int
    error_msg: str
    request_id: int


def _build_perf_metrics_dict(
    req_perf_metrics: tllm.RequestPerfMetrics
) -> dict[RequestEventTiming, float]:
    if not (req_perf_metrics and req_perf_metrics.timing_metrics):
        return {}
    return {
        RequestEventTiming.ARRIVAL_TIME:
        req_perf_metrics.timing_metrics.arrival_time.total_seconds(),
        RequestEventTiming.FIRST_TOKEN_TIME:
        req_perf_metrics.timing_metrics.first_token_time.total_seconds(),
        RequestEventTiming.FIRST_SCHEDULED_TIME:
        req_perf_metrics.timing_metrics.first_scheduled_time.total_seconds(),
        RequestEventTiming.LAST_TOKEN_TIME:
        req_perf_metrics.timing_metrics.last_token_time.total_seconds(),
        RequestEventTiming.KV_CACHE_TRANSFER_START:
        req_perf_metrics.timing_metrics.kv_cache_transfer_start.total_seconds(),
        RequestEventTiming.KV_CACHE_TRANSFER_END:
        req_perf_metrics.timing_metrics.kv_cache_transfer_end.total_seconds(),
        RequestEventTiming.KV_CACHE_SIZE:
        req_perf_metrics.timing_metrics.kv_cache_size,
    }


# Methods that are explicitly implemented for multi-rank MPI/IPC executor
# deployments and may be called via collective_rpc even when world_size > 1.
# sleep() and wakeup() coordinate across all ranks via a dedicated control
# communicator started in PyExecutor.start_worker(); rank-0 is the entry
# point, so routing through the rank-0 RPC shim is correct.
# get_memory_report() is answered by rank-0, which in the memory monitor's
# full mode gathers the other ranks' reports over its own communicator.
_MULTI_RANK_ALLOWED_METHODS: frozenset[str] = frozenset(
    {"get_memory_report", "get_memory_status", "sleep", "wakeup"})


def _check_collective_rpc_guard(
    model_world_size: int,
    unique_reply_rank: Optional[int],
    target_ranks: Optional[Union[int, List[int]]],
    method: str = "",
) -> None:
    """Validate collective_rpc preconditions shared by IPC and RPC proxies.

    Args:
        model_world_size: Total number of GPU worker processes.
        unique_reply_rank: Must be ``None`` for this shim.
        target_ranks: Must be ``None`` for this shim.
        method: Name of the method being invoked; used to allow selected
            multi-rank-capable methods past the world_size guard.

    Raises:
        NotImplementedError: If ``model_world_size > 1`` and ``method`` is not
            in :data:`_MULTI_RANK_ALLOWED_METHODS`, or if
            ``unique_reply_rank`` or ``target_ranks`` are provided.
    """
    if model_world_size > 1 and method not in _MULTI_RANK_ALLOWED_METHODS:
        raise NotImplementedError(
            f"MPI collective_rpc does not support model_world_size > 1 for "
            f"method '{method}'; use the Ray executor for general multi-rank "
            "deployments, or use sleep()/wakeup() which handle multi-rank "
            "coordination internally.")
    if unique_reply_rank is not None or target_ranks is not None:
        raise NotImplementedError(
            "unique_reply_rank and target_ranks are not supported; "
            "this shim only reaches rank-0.")


class IntraProcessQueue:
    ''' A Queue-like container for IPC within the same process. '''

    def __init__(self):
        self.queue = Queue()

    def put(self, obj: Any):
        self.queue.put(obj)

    def get(self, timeout=None) -> Any:
        return self.queue.get(timeout=timeout)

    def close(self):
        pass

    def drain(self) -> list:
        """Non-blocking drain: return all currently available messages."""
        results = []
        while True:
            try:
                results.append(self.queue.get_nowait())
            except Empty:
                break
        return results

    def poll(self, timeout=None) -> bool:
        with self.queue.not_empty:
            if self.queue._qsize() > 0:
                return True
            if timeout is not None and timeout > 0:
                self.queue.not_empty.wait(timeout=timeout)
                return self.queue._qsize() > 0
            return False


@dataclass
class PostprocWorkerConfig:
    """The config for the postprocess worker."""
    num_postprocess_workers: int = 0
    postprocess_tokenizer_dir: Optional[str] = None
    # Dotted import path of the user post-processing hook, or
    # None. NOTE: distinct from ``PostprocParams.post_processor``, which is the
    # per-endpoint response *formatter* (a Callable), not this hook.
    post_processor_hook: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self.num_postprocess_workers > 0


class WorkerCommIpcAddrs(NamedTuple):
    ''' IPC addresses (str) and HMAC keys (bytes) for communication with the worker processes. '''
    request_queue_addr: tuple[str, Optional[bytes]]
    worker_init_status_queue_addr: tuple[str, Optional[bytes]]
    result_queue_addr: tuple[str, Optional[bytes]]
    resource_governor_queue_addr: Optional[tuple[str, Optional[bytes]]] = None
    # Multi-frontend serving: one result lane per frontend. When set, the
    # rank0 worker BINDS the request queue (PULL) and routes responses to
    # these lanes; result_queue_addr then aliases lane 0 (the launcher).
    frontend_result_queue_addrs: Optional[list[tuple[str,
                                                     Optional[bytes]]]] = None
    # Worker -> proxy ack channel for synchronous control requests
    # (currently: StartProfileRequest, StopProfileRequest). Lets the
    # HTTP /start_profile and /stop_profile handlers wait for the worker
    # to actually finish processing the request before returning, so the
    # documented "trace is on disk by the time /stop_profile returns
    # 200" contract holds in the IPC-proxy deployment too.


# Multi-frontend client_id namespacing: a FRONTEND_ID_BITS-wide frontend id
# sits just below the sign bit -- bit 63 stays clear so ids remain positive
# in signed-int64 contexts -- and the low bits carry the per-frontend
# request counter. A stray un-namespaced (small) id reads as frontend 0,
# the launcher.
FRONTEND_ID_BITS = 6
# Keep llm_args.num_serve_frontends le= in sync (it cannot import this
# module; test_multi_frontend_routing pins the two together).
MAX_NUM_FRONTENDS = 1 << FRONTEND_ID_BITS
FRONTEND_ID_SHIFT = 63 - FRONTEND_ID_BITS
FRONTEND_COUNTER_MASK = (1 << FRONTEND_ID_SHIFT) - 1


def get_frontend_id(client_id: Optional[int]) -> int:
    """Extract the originating frontend id from a namespaced client id."""
    if not isinstance(client_id, int):
        return 0
    return client_id >> FRONTEND_ID_SHIFT


def namespace_client_id(frontend_id: int, client_id: int) -> int:
    """Embed frontend_id in the top bits of a per-frontend client id."""
    return (frontend_id << FRONTEND_ID_SHIFT) | (client_id
                                                 & FRONTEND_COUNTER_MASK)


def frontend_lane_index(client_id: Optional[int], num_lanes: int) -> int:
    """The result-lane index for a response's originating frontend.

    None (e.g. ADP dummy requests) and out-of-range frontend ids go to
    lane 0, the launcher, which silently discards them like today.
    """
    frontend_id = get_frontend_id(client_id)
    return frontend_id if frontend_id < num_lanes else 0


def bucket_responses_by_frontend(responses: list,
                                 num_frontends: int) -> list[list]:
    """Bucket responses by frontend_lane_index of their client_id."""
    buckets = [[] for _ in range(num_frontends)]
    for rsp in responses:
        buckets[frontend_lane_index(rsp.client_id, num_frontends)].append(rsp)
    return buckets


def multi_frontend_request_addr(ipc_dir: str) -> str:
    """The request ingress bound by the rank0 worker; frontends PUSH-connect."""
    return f"ipc://{os.path.join(ipc_dir, 'request.sock')}"


def multi_frontend_result_addr(ipc_dir: str, frontend_id: int) -> str:
    """The result lane bound by a frontend; worker/postproc PUSH-connect."""
    return f"ipc://{os.path.join(ipc_dir, f'result_{frontend_id}.sock')}"


def print_alive_threads():
    assert enable_llm_debug(
    ), "print_alive_threads must be called with enable_llm_debug() enabled"

    # Print all alive threads for debugging
    alive_threads = [t for t in threading.enumerate() if t.is_alive()]
    logger.info(
        f'All alive threads after shutdown: {[t.name for t in alive_threads]}\n',
        "red")
    for t in alive_threads:
        logger.info(f'Thread {t.name} (daemon={t.daemon}) is still alive')
        # Get the stack trace for this thread
        stack = sys._current_frames().get(t.ident)
        if stack is not None:
            logger.info(f'Stack trace for thread {t.name}:')
            traceback.print_stack(stack, file=sys.stdout)
            logger.info('')
