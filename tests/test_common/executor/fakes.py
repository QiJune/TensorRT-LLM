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
"""A fake rank 0 for CPU tests of the executor's wire (proto_wire.py, engine_server.py).

``fake_rank0`` is rank 0 without an engine or a process of its own: the state and the
``BaseWorker`` methods that ``engine_server`` and ``AwaitResponseHelper`` use, a ``FakeEngine``
that keeps the requests rank 0 enqueues, and ``Lane`` queues that keep what rank 0 sends.
``executor_requests`` turns what rank 0 enqueued into the executor's requests, which a test steps
as the sampler would (``add_new_token``, ``py_result``, ``set_finished_reason``). ``Rank0`` puts
them together for a frontend's requests: it submits them as the frontend does, then publishes the
executor's responses as rank 0 does and decodes the rows into the frontend's results.
``fake_postprocessor`` and ``run_postprocessor`` run a postprocess worker over what rank 0 put on
its queue.

A test that needs more of rank 0 extends these fakes, with a self-test in
``tests/unittest/executor/test_executor_test_helpers.py``, rather than keeping a fake of its own.
"""

import asyncio
import queue
import threading
from types import SimpleNamespace
from typing import Optional

from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequest,
    LlmRequestState,
    PyResult,
    executor_request_to_llm_request,
)
from tensorrt_llm._torch.pyexecutor.response_utils import prepare_response
from tensorrt_llm.executor import engine_server
from tensorrt_llm.executor.base_worker import AwaitResponseHelper, BaseWorker
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.utils import PostprocWorkerConfig
from tensorrt_llm.executor.wire import StepFrame
from tensorrt_llm.llmapi.engine_client import proto_wire
from tensorrt_llm.llmapi.engine_client.postproc_worker import PostprocWorker

__all__ = [
    "FakeEngine",
    "FakeRank0",
    "Lane",
    "Rank0",
    "executor_requests",
    "fake_postprocessor",
    "fake_rank0",
    "run_postprocessor",
]


class FakeEngine:
    """The executor rank 0 enqueues requests into: it keeps them in ``enqueued``.

    One with ``kv_cache_transceiver`` takes disaggregated requests.
    """

    is_generation_model = True

    def __init__(self, *, kv_cache_transceiver: bool = False):
        self.kv_cache_transceiver = object() if kv_cache_transceiver else None
        self.enqueued = []

    def enqueue_request(self, request, result_wait_queue=None) -> int:
        self.enqueued.append(request)
        return len(self.enqueued)


class Lane:
    """A queue rank 0 sends on: it keeps what it is given in ``items``."""

    def __init__(self):
        self.items = []

    def put(self, item) -> None:
        self.items.append(item)


class FakeRank0:
    """Rank 0 without an engine; ``fake_rank0`` makes one.

    It has the state and the ``BaseWorker`` methods that ``engine_server`` and
    ``AwaitResponseHelper`` use. ``aborted`` are the client ids rank 0 aborted, and ``popped``
    those whose results it dropped (``_pop_result``).
    """

    submit = BaseWorker.submit
    return_queue = BaseWorker.return_queue
    _get_logprob_params = BaseWorker._get_logprob_params
    _has_background_error = BaseWorker._has_background_error
    _create_error_response = BaseWorker._create_error_response
    _load_lora_adapter = BaseWorker._load_lora_adapter
    _load_lora_config = BaseWorker._load_lora_config

    def start(self) -> None:
        pass

    def abort_request(self, client_id: int) -> None:
        self.aborted.append(client_id)

    def _pop_result(self, client_id: int) -> None:
        self.popped.append(client_id)
        BaseWorker._pop_result(self, client_id)

    def _handle_background_error(self, error=None) -> None:
        """Nothing fails in the background; a ``GenerationResult`` keeps a WeakMethod of this."""


def fake_rank0(
    *,
    lanes: Optional[int] = None,
    postprocess_workers: int = 0,
    in_frontend_process: bool = False,
    max_seq_len: Optional[int] = 4096,
    kv_cache_transceiver: bool = False,
) -> FakeRank0:
    """Rank 0 as ``engine_server`` and its await helper see it, with a ``FakeEngine``.

    Rank 0 sends a frontend its rows and errors on ``result_queue``, a ``Lane``. With ``lanes``
    it sends each frontend's on that frontend's lane (``frontend_result_queues``) instead. With
    ``postprocess_workers`` it sends the rows on one lane per worker (``postproc_queues``) and the
    errors on ``result_queue``, as under RPC orchestration. ``in_frontend_process`` is rank 0 in
    the frontend's process: it has no queues and hands the rows to the ``GenerationResult``s it
    holds. ``max_seq_len`` is the engine's, None when it cannot be deduced.
    """
    worker = FakeRank0()
    worker.rank = 0
    worker.engine = FakeEngine(kv_cache_transceiver=kv_cache_transceiver)
    worker.mapping = SimpleNamespace(cp_size=1)
    worker.max_seq_len = max_seq_len
    worker.postproc_config = PostprocWorkerConfig(num_postprocess_workers=postprocess_workers)
    worker.enable_postprocess_parallel = postprocess_workers > 0
    worker.result_queue = None if in_frontend_process or lanes else Lane()
    worker.frontend_result_queues = [Lane() for _ in range(lanes)] if lanes else None
    worker.postproc_queues = (
        [Lane() for _ in range(postprocess_workers)] if postprocess_workers else None
    )
    worker._results = {}
    worker._frontend_results = {}
    worker._client_id_to_request_id = {}
    worker._refused_client_ids = set()
    worker._prepared_client_ids = set()
    worker._error_queue = queue.Queue()
    worker._lora_manager = None
    worker._lora_model_config = None
    worker._lora_adapter_identities = {}
    worker._lora_adapter_id_locks = {}
    worker._lora_adapter_id_locks_guard = threading.Lock()
    worker.aborted = []
    worker.popped = []
    # last: the helper reads the postprocess settings when it is made
    worker._await_response_helper = AwaitResponseHelper(worker)
    return worker


def _keep_logits_on_host(request: LlmRequest) -> None:
    """Keep ``request``'s logits in host memory, in one piece; children made later inherit it.

    The executor keeps them in device memory and moves generation logits to the host in chunks
    when the request ends.
    """
    request.py_return_logits_device_memory = False
    request.py_use_chunked_generation_logits = False
    request.py_result = PyResult(
        prompt_len=request.py_prompt_len,
        max_new_tokens=request.py_max_new_tokens,
        use_device_memory=False,
        streaming=request.streaming,
        return_log_probs=request.py_return_log_probs,
        return_context_logits=request.py_return_context_logits,
        return_generation_logits=request.py_return_generation_logits,
        use_chunked_generation_logits=False,
        additional_outputs=request.py_additional_outputs,
    )


def executor_requests(enqueued: list) -> list[list[LlmRequest]]:
    """The executor's requests for what rank 0 enqueued, generating; a family per request.

    A family is the request and, for one that samples more than one sequence, a child per
    further sequence, whose id is ``client_id * 10 + k`` for the k-th. They are built as the
    executor builds them (``executor_request_to_llm_request``), with their logits in host memory,
    where a test's sampler steps append them.
    """
    families = []
    for item in enqueued:
        client_id = item.request.client_id
        request = executor_request_to_llm_request(
            client_id, item, child_req_ids=[], exclude_last_generation_logits=False
        )
        _keep_logits_on_host(request)
        config = item.request.sampling_config
        # as the executor counts them: a beam search is one request
        sequences = 1 if config.beam_width > 1 else (config.num_return_sequences or 1)
        for k in range(1, sequences):
            request.create_child_request(client_id * 10 + k)
        family = [request, *request.child_requests]
        for member in family:
            member.state = LlmRequestState.GENERATION_IN_PROGRESS
        families.append(family)
    return families


class Rank0:
    """Rank 0 (``fake_rank0``) serving a frontend's requests, and that frontend.

    ``submit`` sends a request as the frontend does and returns the executor's requests it
    becomes, for a test to step. ``send`` hands their responses to rank 0's await helper, and
    ``publish`` also decodes the rows the frontend gets and hands each to its request's result.
    """

    def __init__(self, *, kv_cache_transceiver: bool = False):
        self.worker = fake_rank0(kv_cache_transceiver=kv_cache_transceiver)
        self.results = {}
        # what each request asked for, for the rows of one without a result
        self._requests = {}

    def submit(self, request: GenerationRequest, result=None) -> list[LlmRequest]:
        """Submit ``request``, whose rows go to ``result``; the executor's requests it becomes."""
        self._requests[request.id] = request
        if result is not None:
            self.results[request.id] = result
        frame = proto_wire.encode(request, postprocess=False)
        engine_server.submit(self.worker, frame, raise_errors=True)
        (family,) = executor_requests(self.worker.engine.enqueued[-1:])
        return family

    def send(self, *requests: LlmRequest) -> list:
        """What rank 0 sends the frontend for the responses of the executor's ``requests``.

        A request with nothing to send yet (a non-streaming one before it ends) has no response.
        """
        responses = [r for r in map(prepare_response, requests) if r is not None]
        self.worker._await_response_helper.process_and_handle_responses(responses)
        sent = list(self.worker.result_queue.items)
        self.worker.result_queue.items.clear()
        return sent

    def publish(self, *requests: LlmRequest) -> list:
        """The rows the frontend gets for the responses of ``requests``, each handed to its result.

        Errors come as rank 0 sent them.
        """
        rows = []
        for message in self.send(*requests):
            for item in message:
                if isinstance(item, StepFrame):
                    rows.extend(proto_wire.decode(item, self.results, self._requests))
                else:
                    rows.append(item)
        for row in rows:
            if (result := self.results.get(row.client_id)) is not None:
                result._handle_response(row)
        return rows


def fake_postprocessor(record_creator=PostprocWorker.default_record_creator) -> PostprocWorker:
    """A postprocess worker without its pipes, tokenizer or hook; ``run_postprocessor`` feeds it."""
    worker = object.__new__(PostprocWorker)
    worker._records = {}
    worker._record_creator = record_creator
    worker._tokenizer = None
    worker._post_processor_hook = None
    return worker


class _Pipe:
    """A postprocess worker's input queue: the messages rank 0 put on it, then its end."""

    def __init__(self, messages: list):
        self._messages = [*messages, None]

    async def get_async(self):
        return self._messages.pop(0)


def run_postprocessor(worker: PostprocWorker, messages: list) -> list:
    """What ``worker`` sends the frontends for ``messages``, the ones rank 0 put on its queue."""
    worker._pull_pipe = _Pipe(messages)
    worker._to_stop = asyncio.Event()
    out = []

    async def run():
        async for batch in worker._mainloop():
            if batch is None:
                break
            out.extend(batch)

    asyncio.run(run())
    return out
