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
"""OpenAI-compatible HTTP frontend detached from the engine.

This process speaks only the boundary protocol: it connects a
``RemoteEngineClient`` to an engine socket, builds its
``FrontendModelContext`` (and tokenizer) from the handshake, and serves
chat/completions through the frontend-owned pipeline. It never imports
``torch`` or the model runtime, never constructs a model/executor/proxy,
and rejects requests needing capabilities the engine did not advertise
with typed errors (there is no in-process path to fall back to).
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from tensorrt_llm.engine_api.contracts import EngineClientError, EngineErrorCode
from tensorrt_llm.serve.frontend.eligibility import PipelineDeploymentMode
from tensorrt_llm.serve.frontend.model_context import FrontendModelContext
from tensorrt_llm.serve.frontend.openai_pipeline import OpenAIServingPipeline
from tensorrt_llm.serve.frontend.request_processor import FrontendProcessor

__all__ = ["DetachedFrontend", "create_detached_app"]


class DetachedFrontend:
    """Detached serving frontend over a remote engine."""

    def __init__(
        self,
        engine_endpoint: str,
        *,
        handshake_timeout_seconds: float = 60.0,
        tool_parser: Optional[str] = None,
        tokenizer: Any = None,
    ) -> None:
        import time

        from tensorrt_llm.engine_api.contracts import EngineClientError, EngineError
        from tensorrt_llm.engine_api.contracts import EngineErrorCode as _Code
        from tensorrt_llm.engine_api.engine_server import RemoteEngineClient
        from tensorrt_llm.engine_api.protocol import ReadinessState

        self.client = RemoteEngineClient(
            engine_endpoint, handshake_timeout_seconds=handshake_timeout_seconds
        )
        # An engine may answer while still initializing; the model context is
        # only trustworthy once it is ready, so wait (within the handshake
        # budget) and refresh before building tokenizer and pipeline.
        deadline = time.monotonic() + handshake_timeout_seconds
        while self.client.readiness is not ReadinessState.READY and time.monotonic() < deadline:
            time.sleep(0.2)
            try:
                self.client.refresh_handshake(timeout=5.0)
            except EngineClientError:
                pass
        if self.client.readiness is not ReadinessState.READY:
            readiness = self.client.readiness
            self.client.shutdown()
            raise EngineClientError(
                EngineError(
                    code=_Code.ENGINE_UNAVAILABLE,
                    message=f"engine at {engine_endpoint} did not become ready "
                    f"within {handshake_timeout_seconds}s (last readiness: "
                    f"{readiness.value if readiness else 'unknown'})",
                )
            )
        self.model_context = FrontendModelContext.from_handshake(
            self.client.model_context, self.client.capabilities
        )
        if tokenizer is None:
            tokenizer = self.model_context.build_tokenizer()
        processor = FrontendProcessor(
            tokenizer,
            default_stream_interval=self.model_context.stream_interval,
            lightweight_templates=True,
        )
        self.pipeline = OpenAIServingPipeline(
            self.client,
            processor,
            model_label=self.model_context.model,
            mode=PipelineDeploymentMode.DETACHED,
            reasoning_parser=self.model_context.reasoning_parser,
            tool_parser=tool_parser,
        )

    def shutdown(self) -> None:
        self.client.shutdown()


def create_detached_app(frontend: DetachedFrontend) -> FastAPI:
    """Build the FastAPI app for a detached frontend."""
    app = FastAPI(title="TensorRT-LLM detached frontend")

    @app.get("/health")
    async def health() -> Response:
        healthy = frontend.client.check_health()
        return Response(status_code=HTTPStatus.OK if healthy else HTTPStatus.SERVICE_UNAVAILABLE)

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        return JSONResponse(
            content={
                "object": "list",
                "data": [{"id": frontend.model_context.model, "object": "model"}],
            }
        )

    @app.get("/version")
    async def version() -> JSONResponse:
        from tensorrt_llm.version import __version__

        return JSONResponse(content={"version": __version__})

    @app.get("/iteration_stats")
    async def iteration_stats() -> JSONResponse:
        try:
            stats = frontend.client.get_stats(timeout=2.0)
        except EngineClientError as e:
            return _typed_error_response(e)
        return JSONResponse(content=stats)

    @app.get("/kv_cache_events")
    async def kv_cache_events() -> JSONResponse:
        try:
            events = frontend.client.get_kv_events(timeout=2.0)
        except EngineClientError as e:
            return _typed_error_response(e)
        return JSONResponse(content=events)

    # Endpoints backed by collective RPC into the workers are engine-side
    # only: on a detached frontend (and on multi-rank headless engines) they
    # return the typed unsupported-capability error the capability matrix
    # declares.
    for route in ("/release_memory", "/resume_memory", "/update_weights"):

        async def collective_rpc_unsupported(
            raw_request: Request, _route: str = route
        ) -> JSONResponse:
            return JSONResponse(
                status_code=HTTPStatus.NOT_IMPLEMENTED,
                content={
                    "error": {
                        "message": f"{_route} is backed by collective RPC into the "
                        "workers and is not available on a detached frontend",
                        "type": "unsupported_capability",
                        "code": EngineErrorCode.UNSUPPORTED_CAPABILITY.value,
                        "param": None,
                    }
                },
            )

        app.add_api_route(route, collective_rpc_unsupported, methods=["POST"])

    @app.post("/v1/chat/completions")
    async def chat(raw_request: Request) -> Response:
        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

        request = ChatCompletionRequest(**(await raw_request.json()))
        return _as_response(await frontend.pipeline.try_chat(request, raw_request))

    @app.post("/v1/completions")
    async def completions(raw_request: Request) -> Response:
        from tensorrt_llm.serve.openai_protocol import CompletionRequest

        request = CompletionRequest(**(await raw_request.json()))
        return _as_response(await frontend.pipeline.try_completion(request, raw_request))

    return app


def _as_response(result: Any) -> Response:
    if isinstance(result, Response):
        return result
    return JSONResponse(content=result.model_dump(exclude_none=True))


def _typed_error_response(error: EngineClientError) -> JSONResponse:
    return JSONResponse(
        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        content={
            "error": {
                "message": error.error.message,
                "type": "engine_error",
                "code": error.error.code.value,
                "param": None,
            }
        },
    )
