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
"""Import directions between the engine and its frontend.

The engine is ``tensorrt_llm.executor`` (the engine host) and ``tensorrt_llm._torch``; the frontend
is ``tensorrt_llm.serve``, ``tensorrt_llm.llmapi``, ``tensorrt_llm.grpc`` and
``tensorrt_llm.commands``. The frontend imports the engine, including the requests, responses and
queues ``tensorrt_llm.executor`` defines for its clients; the engine does not import the frontend,
so that the frontend can be replaced as a whole. A contract forbids imports by module prefix; a
baseline allowance records the imports that existed when the contract was written and only shrinks.
"""

from pathlib import Path

import pytest
from test_common.serve.layering import (
    Contract,
    build_import_graph,
    check_contracts,
    stale_allowances,
)

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[3]

# Old paths that re-export moved names; test_shim_identity.py and test_shim_ray.py check them.
_SHIM_MODULES = (
    "tensorrt_llm._torch.async_llm",
    "tensorrt_llm.executor.postprocessor_hook",
    "tensorrt_llm.executor.ray.executor",
    "tensorrt_llm.executor.ray_executor",
    "tensorrt_llm.executor.result",
)

_CLIENT = "tensorrt_llm.llmapi.engine_client"
_ENGINE_INTERNALS = tuple(
    f"tensorrt_llm.executor.{module}"
    for module in (
        "base_worker",
        "ray",
        "rpc_worker",
        "rpc_worker_mixin",
        "worker",
        "worker_process_monitor",
    )
)

# The configuration and the helpers both sides use still live in the LLM API package; the engine
# reads them until the engine's configuration becomes part of its contract.
_SHARED_FOR_NOW = (
    "tensorrt_llm.llmapi.llm_args",
    "tensorrt_llm.llmapi.llm_utils",
    "tensorrt_llm.llmapi.tracer",
    "tensorrt_llm.llmapi.utils",
)

# The engine's other imports of the frontend. The worker is also the in-process GenerationExecutor:
# it returns GenerationResults and encodes and decodes the wire as a frontend does. Rank 0 starts
# the postprocess workers; a precompile pool reuses the process-group helpers of the multi-node
# launcher's guard script; PyExecutor creation starts the OpenEngine node service; rank 0 builds
# the thinking-budget logits processor a proto request asks for; and tensorrt_llm._torch.LLM is
# kept as an alias of the LLM class.
_ENGINE_USES_FRONTEND = frozenset(
    {
        ("tensorrt_llm.executor.base_worker", f"{_CLIENT}.executor", "GenerationExecutor"),
        (
            "tensorrt_llm.executor.engine_server",
            "tensorrt_llm.llmapi.thinking_budget",
            "ThinkingBudgetLogitsProcessor",
        ),
        ("tensorrt_llm.executor.base_worker", f"{_CLIENT}.postproc_worker", "PostprocWorker"),
        ("tensorrt_llm.executor.base_worker", f"{_CLIENT}.result", "GenerationResult"),
        ("tensorrt_llm.executor.base_worker", _CLIENT, "proto_wire"),
        ("tensorrt_llm.executor.ray.gpu_worker", _CLIENT, "proto_wire"),
        ("tensorrt_llm.executor.rpc_worker_mixin", f"{_CLIENT}.postproc_worker", "PostprocWorker"),
        (
            "tensorrt_llm.executor.rpc_worker_mixin",
            f"{_CLIENT}.postproc_worker",
            "postproc_worker_main",
        ),
        ("tensorrt_llm.executor.worker", f"{_CLIENT}.postproc_worker", "PostprocWorker"),
        ("tensorrt_llm.executor.worker", f"{_CLIENT}.postproc_worker", "postproc_worker_main"),
        (
            "tensorrt_llm._torch.attention.backends.fmha.prims_ts_precompile",
            "tensorrt_llm.llmapi._llmapi_process_guard",
            "_child_finished",
        ),
        (
            "tensorrt_llm._torch.attention.backends.fmha.prims_ts_precompile",
            "tensorrt_llm.llmapi._llmapi_process_guard",
            "_cleanup_group",
        ),
        (
            "tensorrt_llm._torch.pyexecutor.py_executor_creator",
            "tensorrt_llm.grpc.openengine.node",
            "initialize_node_discovery",
        ),
        ("tensorrt_llm._torch.llm", "tensorrt_llm.llmapi.llm", "_TorchLLM"),
    }
)

CONTRACTS = (
    Contract(
        name="the engine does not import the frontend",
        importers=("tensorrt_llm.executor", "tensorrt_llm._torch"),
        forbidden=(
            "tensorrt_llm.commands",
            "tensorrt_llm.grpc",
            "tensorrt_llm.llmapi",
            "tensorrt_llm.serve",
        ),
        allowed=_ENGINE_USES_FRONTEND,
        fix=(
            "The frontend (serve, llmapi, grpc, commands) imports the engine, not the other way "
            "round. Put what both use in the engine (tensorrt_llm.executor) or in a module outside "
            "the frontend packages. The baseline only shrinks: delete an allowance when its import "
            "goes away."
        ),
    ),
    Contract(
        name="serving frontends use only what the engine host defines for its clients",
        importers=("tensorrt_llm.serve", "tensorrt_llm.grpc"),
        forbidden=_ENGINE_INTERNALS,
        fix=(
            "Import requests, responses and errors from tensorrt_llm.executor, and executors and "
            "results from tensorrt_llm.llmapi.engine_client."
        ),
    ),
    Contract(
        name="the engine client does not import serving code",
        importers=(_CLIENT,),
        forbidden=("tensorrt_llm.serve",),
        fix="Serving code passes its formatters in through PostprocParams.",
    ),
)


def _reads_shared_module(edge) -> bool:
    return any(edge.imported == m or edge.imported.startswith(m + ".") for m in _SHARED_FOR_NOW)


@pytest.fixture(scope="module")
def edges():
    graph = build_import_graph(REPO_ROOT / "tensorrt_llm", "tensorrt_llm")
    return [
        edge
        for edge in graph
        if edge.importer not in _SHIM_MODULES and not _reads_shared_module(edge)
    ]


def test_contracts_hold(edges):
    assert check_contracts(edges, CONTRACTS) == []


def test_no_stale_allowances(edges):
    assert stale_allowances(edges, CONTRACTS) == []
