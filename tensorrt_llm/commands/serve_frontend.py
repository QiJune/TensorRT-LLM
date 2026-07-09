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
"""Detached OpenAI-compatible frontend: serve HTTP against a remote engine.

This entrypoint never loads the GPU runtime: it forces the lightweight
import mode before any ``tensorrt_llm`` import, connects to an engine
started with ``trtllm-serve-engine``, and serves chat/completions over the
boundary protocol.
"""

import os

# Must be set before any tensorrt_llm import in this process.
os.environ.setdefault("TLLM_LIGHTWEIGHT_IMPORT", "1")

import click  # noqa: E402


@click.command("serve-frontend")
@click.argument("engine_endpoint", type=str)
@click.option("--host", type=str, default="localhost", help="Hostname to bind.")
@click.option("--port", type=int, default=8000, help="Port to bind.")
@click.option(
    "--tool_parser",
    type=str,
    default=None,
    help="Tool parser used for chat tool calls.",
)
@click.option(
    "--handshake_timeout",
    type=float,
    default=120.0,
    help="Seconds to wait for the engine handshake before failing fast.",
)
def main(
    engine_endpoint: str,
    host: str,
    port: int,
    tool_parser: str,
    handshake_timeout: float,
) -> None:
    """Serve an OpenAI-compatible frontend against ENGINE_ENDPOINT.

    ENGINE_ENDPOINT is the ZMQ endpoint of a running engine, e.g.
    ``tcp://127.0.0.1:5555``.
    """
    import uvicorn

    from tensorrt_llm.serve.frontend.detached_app import DetachedFrontend, create_detached_app

    frontend = DetachedFrontend(
        engine_endpoint,
        handshake_timeout_seconds=handshake_timeout,
        tool_parser=tool_parser,
    )
    try:
        uvicorn.run(create_detached_app(frontend), host=host, port=port)
    finally:
        frontend.shutdown()


if __name__ == "__main__":
    main()
