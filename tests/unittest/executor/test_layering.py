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
"""Import directions between the engine host and its clients.

``tensorrt_llm.executor`` is the engine host. It also defines what its clients exchange with it:
the requests, the responses and errors, and the queues and calls. The LLM API's client of the
engine (``tensorrt_llm.llmapi.engine_client``) and the serving frontends import those; the engine
does not import its clients. A contract forbids imports by module prefix; a baseline allowance
records the imports that existed when the contract was written and only shrinks.
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

# Old paths that re-export moved names; test_shim_identity.py checks them.
_SHIM_MODULES = (
    "tensorrt_llm.executor.postprocessor_hook",
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

# The engine's imports of the engine client. The worker is also the in-process
# GenerationExecutor and returns GenerationResults, rank 0 starts the postprocess workers, and
# Ray's executor still lives beside its GPU workers.
_ENGINE_USES_CLIENT = frozenset(
    {
        ("tensorrt_llm.executor.base_worker", f"{_CLIENT}.executor", "GenerationExecutor"),
        ("tensorrt_llm.executor.base_worker", f"{_CLIENT}.postproc_worker", "PostprocWorker"),
        ("tensorrt_llm.executor.base_worker", f"{_CLIENT}.result", "GenerationResult"),
        ("tensorrt_llm.executor.base_worker", f"{_CLIENT}.result", "ResponseWrapper"),
        ("tensorrt_llm.executor.base_worker", f"{_CLIENT}.result", "get_metrics_dict"),
        ("tensorrt_llm.executor.ray.executor", f"{_CLIENT}.executor", "GenerationExecutor"),
        ("tensorrt_llm.executor.ray.executor", f"{_CLIENT}.result", "GenerationResult"),
        ("tensorrt_llm.executor.ray.executor", f"{_CLIENT}.rpc_proxy_mixin", "RpcExecutorMixin"),
        ("tensorrt_llm.executor.ray.gpu_worker", f"{_CLIENT}.result", "GenerationResult"),
        ("tensorrt_llm.executor.rpc_worker_mixin", f"{_CLIENT}.postproc_worker", "PostprocWorker"),
        (
            "tensorrt_llm.executor.rpc_worker_mixin",
            f"{_CLIENT}.postproc_worker",
            "postproc_worker_main",
        ),
        ("tensorrt_llm.executor.worker", f"{_CLIENT}.postproc_worker", "PostprocWorker"),
        ("tensorrt_llm.executor.worker", f"{_CLIENT}.postproc_worker", "postproc_worker_main"),
        ("tensorrt_llm.executor.worker", f"{_CLIENT}.proxy", "_check_collective_rpc_guard"),
    }
)

CONTRACTS = (
    Contract(
        name="the engine does not import the engine client",
        importers=("tensorrt_llm.executor", "tensorrt_llm._torch"),
        forbidden=(_CLIENT,),
        allowed=_ENGINE_USES_CLIENT,
        fix=(
            "Define what the engine and its clients share in tensorrt_llm.executor (request.py, "
            "utils.py, logprobs.py) and import it from there. The baseline only shrinks: delete "
            "an allowance when its import goes away."
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


@pytest.fixture(scope="module")
def edges():
    graph = build_import_graph(REPO_ROOT / "tensorrt_llm", "tensorrt_llm")
    return [edge for edge in graph if edge.importer not in _SHIM_MODULES]


def test_contracts_hold(edges):
    assert check_contracts(edges, CONTRACTS) == []


def test_no_stale_allowances(edges):
    assert stale_allowances(edges, CONTRACTS) == []
