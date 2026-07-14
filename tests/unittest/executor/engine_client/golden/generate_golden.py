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
"""Regenerates the checked-in binary golden fixtures (*.msgpack).

Run from the repo root:
    python3 tests/unittest/executor/engine_client/golden/generate_golden.py

A diff in any .msgpack file is a wire-format change of the same protocol
version and must be treated as such (see ENGINE_CONTRACT.md).
"""

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from tensorrt_llm.executor.engine_client.codec import encode  # noqa: E402

HERE = pathlib.Path(__file__).parent


def golden_objects():
    """The canonical fixture payloads (shared with test_codec.py)."""
    sys.path.insert(0, str(HERE.parent))
    from test_codec import GOLDEN_OBJECTS
    return GOLDEN_OBJECTS


def main():
    for name, factory in golden_objects():
        data = encode(factory())
        path = HERE / f"{name}.msgpack"
        path.write_bytes(data)
        print(f"wrote {path.name}: {len(data)} bytes")


if __name__ == "__main__":
    main()
