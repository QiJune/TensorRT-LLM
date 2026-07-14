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
"""Record per-test pytest verdicts into the GPU run artifact.

Only active for ``test_gpu_e2e.py`` runs with ``TLLM_ENGINE_CLIENT_GPU_REPORT``
set: every test's outcome is merged into the artifact so the report carries
the actual pytest verdicts, not just case payloads.
"""

import json
import os

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or "test_gpu_e2e" not in item.nodeid:
        return
    path = os.environ.get("TLLM_ENGINE_CLIENT_GPU_REPORT")
    if not path:
        return
    try:
        data = {}
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
        verdicts = data.setdefault("verdicts", {})
        verdicts[item.nodeid.split("::", 1)[-1]] = report.outcome
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass
