# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Headless engine: run the model runtime behind the boundary socket.

The engine-only counterpart of ``trtllm-serve-frontend``: builds the model
runtime (the executor proxy owns MPI session startup, dispatch/error
threads, and the RPC control client, exactly as on the in-process path)
and serves the language-neutral boundary protocol. No HTTP server runs in
this process.
"""

import json

import click


@click.command("serve-engine")
@click.argument("model", type=str)
@click.option(
    "--endpoint",
    type=str,
    default="tcp://127.0.0.1:5555",
    help="ZMQ endpoint to bind the boundary socket to (localhost-only by "
    "default; see the protocol threat note before exposing further).",
)
@click.option(
    "--llm_args_json",
    type=str,
    default=None,
    help="JSON object of extra LLM argument overrides.",
)
def main(model: str, endpoint: str, llm_args_json: str) -> None:
    """Run a headless engine for MODEL on the boundary socket."""
    from tensorrt_llm.engine_api.engine_server import EngineServer, build_runtime_backend_factory
    from tensorrt_llm.logger import logger

    extra = json.loads(llm_args_json) if llm_args_json else None
    server = EngineServer(build_runtime_backend_factory(model, extra), endpoint=endpoint)
    logger.info(f"starting headless engine on {endpoint}")
    server.start(wait=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
