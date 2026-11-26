"""The engine host: the workers that run the PyTorch executor on each rank.

It also defines what its clients exchange with it: the requests (``request``), the responses and
errors (``utils``, ``logprobs``) and the queues and calls (``ipc``, ``rpc``). The LLM API's client
of the engine is ``tensorrt_llm.llmapi.engine_client``; the names that moved there stay readable
from this package, with a ``FutureWarning``, until 1.6.0.
"""

import importlib
import warnings

from .request import *
from .utils import EngineDeadError, PostprocWorkerConfig, RequestError
from .worker import *

_LAZY_NAMES = {
    "GenerationExecutor": "tensorrt_llm.llmapi.engine_client.executor",
    "CppExecutorError": "tensorrt_llm.llmapi.engine_client.executor",
    "PostprocWorker": "tensorrt_llm.llmapi.engine_client.postproc_worker",
    "GenerationExecutorProxy": "tensorrt_llm.llmapi.engine_client.proxy",
    "GenerationExecutorFrontendProxy":
    "tensorrt_llm.llmapi.engine_client.proxy",
    "CompletionOutput": "tensorrt_llm.llmapi.engine_client.result",
    "GenerationResultBase": "tensorrt_llm.llmapi.engine_client.result",
    "DetokenizedGenerationResultBase":
    "tensorrt_llm.llmapi.engine_client.result",
    "GenerationResult": "tensorrt_llm.llmapi.engine_client.result",
    "IterationResult": "tensorrt_llm.llmapi.engine_client.result",
}


def __getattr__(name):
    if name in _LAZY_NAMES:
        warnings.warn(
            f"tensorrt_llm.executor.{name} has moved to {_LAZY_NAMES[name]} "
            "and will be removed in 1.6.0.",
            FutureWarning,
            stacklevel=2,
        )
        value = getattr(importlib.import_module(_LAZY_NAMES[name]), name)
        globals()[name] = value  # warn once: later reads skip __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PostprocWorkerConfig",
    "GenerationRequest",
    "LoRARequest",
    "PromptAdapterRequest",
    "GenerationExecutorWorker",
    "RequestError",
    "EngineDeadError",
]
