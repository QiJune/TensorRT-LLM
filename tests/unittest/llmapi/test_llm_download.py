from tensorrt_llm.llmapi.utils import (download_hf_model,
                                       download_hf_pretrained_config)

# isort: off
from .test_llm import llama_model_path
# isort: on


# The legacy TensorRT backend (`tensorrt_llm._tensorrt_engine.LLM`) has been
# removed. This placeholder keeps the module importable so the backend-agnostic
# download tests still collect/run; TensorRT-only tests fail clearly if invoked.
class LLM:

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "The TensorRT backend has been removed; this TensorRT-only test "
            "is no longer supported. Use the PyTorch backend instead.")

prompts = ["A B C"]

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def test_llm_with_model_downloaded():
    llm = LLM(model=model_name, enable_build_cache=True)
    for output in llm.generate(prompts):
        print(output)


def test_llm_with_tokenizer_downloaded():
    llm = LLM(model=llama_model_path, tokenizer=model_name)
    for output in llm.generate(prompts):
        print(output)


def test_download_config():
    path0 = download_hf_pretrained_config(model_name)
    print(f"download config to {path0}")
    path1 = download_hf_model(model_name)
    print(f"download model to {path1}")
    assert path0 == path1


if __name__ == "__main__":
    test_download_config()
    test_llm_with_model_downloaded()
