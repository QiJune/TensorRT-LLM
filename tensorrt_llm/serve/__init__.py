import os as _os

if _os.environ.get("TLLM_LIGHTWEIGHT_IMPORT", "0") == "1":
    # Lightweight import mode (detached serving frontend): skip the
    # package re-exports, which pull the GPU runtime. Light submodules
    # are imported individually by their users.
    __all__ = []
else:
    from .openai_disagg_server import OpenAIDisaggServer
    from .openai_server import OpenAIServer

    __all__ = ['OpenAIServer', 'OpenAIDisaggServer']
