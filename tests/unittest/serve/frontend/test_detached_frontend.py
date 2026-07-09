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
"""Reach-cutting audits for the detached serving frontend."""

import dataclasses
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]

# Every module on the detached-frontend path. The legacy adapter and the
# runtime backend factory are engine-side by design and excluded.
DETACHED_PATH_SOURCES = [
    "tensorrt_llm/engine_api/contracts.py",
    "tensorrt_llm/engine_api/protocol.py",
    "tensorrt_llm/engine_api/socket_transport.py",
    "tensorrt_llm/engine_api/engine_server.py",
    "tensorrt_llm/serve/frontend/eligibility.py",
    "tensorrt_llm/serve/frontend/request_processor.py",
    "tensorrt_llm/serve/frontend/response_assembler.py",
    "tensorrt_llm/serve/frontend/openai_formatters.py",
    "tensorrt_llm/serve/frontend/openai_pipeline.py",
    "tensorrt_llm/serve/frontend/model_context.py",
    "tensorrt_llm/serve/frontend/detached_app.py",
    "tensorrt_llm/commands/serve_frontend.py",
    "tensorrt_llm/tokenizer/chat_template.py",
    "tensorrt_llm/serve/tool_call_id.py",
]

# Module-level (column-0) imports that would pull the runtime.
FORBIDDEN_MODULE_IMPORTS = re.compile(
    r"^(import torch\b|from torch\b|import tensorrt_llm\._torch|"
    r"from tensorrt_llm\._torch|from tensorrt_llm import .*\bLLM\b|"
    r"from tensorrt_llm\.llmapi\.llm import|from \.\.llmapi\.llm import)",
    re.MULTILINE,
)

# Engine-side reaches that must not exist anywhere on the detached path.
FORBIDDEN_ACCESS_PATTERNS = [
    "generator.args",
    "generator._executor",
    "generator._hf_model_dir",
    "_torch.pyexecutor.config_utils",
]


class TestStaticAudits:
    @pytest.mark.parametrize("source", DETACHED_PATH_SOURCES)
    def test_no_module_level_runtime_imports(self, source):
        text = (REPO_ROOT / source).read_text()
        match = FORBIDDEN_MODULE_IMPORTS.search(text)
        assert match is None, f"{source} has a forbidden module-level import: {match.group(0)!r}"

    @pytest.mark.parametrize("source", DETACHED_PATH_SOURCES)
    def test_no_engine_side_reaches(self, source):
        text = (REPO_ROOT / source).read_text()
        for pattern in FORBIDDEN_ACCESS_PATTERNS:
            assert pattern not in text, f"{source} reaches into engine-side state: {pattern}"

    def test_audit_catches_reintroduced_reach(self, tmp_path):
        """The audit itself must flag forbidden patterns (self-test)."""
        offending = "value = generator.args.max_seq_len\nimport torch\n"
        assert any(pattern in offending for pattern in FORBIDDEN_ACCESS_PATTERNS)
        assert FORBIDDEN_MODULE_IMPORTS.search(offending) is not None


SMOKE_CHILD_SCRIPT = r"""
import json
import os
import sys

os.environ["TLLM_LIGHTWEIGHT_IMPORT"] = "1"

endpoint = sys.argv[1]

from tensorrt_llm.serve.frontend.detached_app import DetachedFrontend, create_detached_app


class MinimalTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    chat_template = "smoke-template"

    def encode(self, text, add_special_tokens=True, **kwargs):
        return [len(word) for word in text.split()]

    def decode(self, token_ids, **kwargs):
        if isinstance(token_ids, int):
            return f"<{token_ids}>"
        return "".join(f"<{t}>" for t in token_ids)

    def get_chat_template(self, chat_template=None, tools=None):
        return chat_template or self.chat_template

    def apply_chat_template(self, conversation=None, **kwargs):
        return " ".join(m["content"] for m in conversation)


frontend = DetachedFrontend(endpoint, tokenizer=MinimalTokenizer())
app = create_detached_app(frontend)

forbidden = sorted(
    name
    for name in sys.modules
    if name == "torch"
    or name.startswith("torch.")
    or name.startswith("tensorrt_llm._torch")
    or name == "tensorrt_llm.llmapi.llm"
    or name.startswith("tensorrt_llm.executor")
)
result = {
    "forbidden_modules": forbidden,
    "healthy": frontend.client.check_health(),
    "model": frontend.model_context.model,
    "capabilities": frontend.model_context.capabilities is not None,
}
frontend.shutdown()
print("SMOKE_RESULT:" + json.dumps(result))
"""


class TestProcessLevelSmoke:
    def test_detached_frontend_starts_without_runtime_modules(self):
        sys.path.insert(0, str(REPO_ROOT / "tests" / "unittest" / "engine_api"))
        from fake_engine import FakeEngine

        from tensorrt_llm.engine_api.socket_transport import EngineSocketServer

        endpoint = f"ipc:///tmp/detached_smoke_{uuid.uuid4().hex}.sock"
        server = EngineSocketServer(
            FakeEngine(),
            endpoint=endpoint,
            model_context={"model": "smoke-model", "tokenizer_dir": None},
        )
        server.start()
        try:
            completed = subprocess.run(
                [sys.executable, "-c", SMOKE_CHILD_SCRIPT, endpoint],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(REPO_ROOT),
            )
            marker = [
                line for line in completed.stdout.splitlines() if line.startswith("SMOKE_RESULT:")
            ]
            assert marker, (
                f"smoke child produced no result\nstdout: {completed.stdout[-2000:]}\n"
                f"stderr: {completed.stderr[-2000:]}"
            )
            result = json.loads(marker[0][len("SMOKE_RESULT:") :])
            assert result["forbidden_modules"] == []
            assert result["healthy"] is True
            assert result["model"] == "smoke-model"
            assert result["capabilities"] is True
        finally:
            server.shutdown()


class TestFrontendLauncher:
    """The lightweight flag must be established before the tensorrt_llm import."""

    def test_launcher_exports_flag_before_python(self):
        launcher = REPO_ROOT / "tensorrt_llm" / "commands" / "trtllm-serve-frontend"
        assert launcher.exists(), "the detached-frontend launcher script is missing"
        text = launcher.read_text()
        export_pos = text.index("export TLLM_LIGHTWEIGHT_IMPORT=1")
        python_pos = text.index("python3 -m tensorrt_llm.commands.serve_frontend")
        # The env export must precede the python invocation.
        assert export_pos < python_pos

    def test_frontend_not_registered_as_heavy_console_script(self):
        setup_text = (REPO_ROOT / "setup.py").read_text()
        # The console-script entry (which imports the package before the flag
        # is set) must not exist; the bash launcher replaces it.
        assert "trtllm-serve-frontend=tensorrt_llm.commands.serve_frontend:main" not in setup_text
        assert "tensorrt_llm/commands/trtllm-serve-frontend" in setup_text


class TestFrontendModelContext:
    def test_built_from_handshake(self):
        from tensorrt_llm.serve.frontend.model_context import FrontendModelContext

        context = FrontendModelContext.from_handshake(
            {
                "model": "m",
                "tokenizer_dir": "/models/m",
                "max_seq_len": 4096,
                "reasoning_parser": None,
                "return_perf_metrics": True,
                "stream_interval": 2,
            },
            capabilities={"generation": {"streaming": True}},
        )
        assert context.model == "m"
        assert context.tokenizer_dir == "/models/m"
        assert context.max_seq_len == 4096
        assert context.return_perf_metrics is True
        assert context.stream_interval == 2

    def test_context_is_read_only(self):
        from tensorrt_llm.serve.frontend.model_context import FrontendModelContext

        context = FrontendModelContext.from_handshake({"model": "m"})
        with pytest.raises(dataclasses.FrozenInstanceError):
            context.model = "other"

    def test_missing_tokenizer_source_fails_clearly(self):
        from tensorrt_llm.serve.frontend.model_context import FrontendModelContext

        context = FrontendModelContext.from_handshake({"model": "m"})
        with pytest.raises(ValueError, match="tokenizer source"):
            context.build_tokenizer()


class TestLightweightTemplates:
    def test_light_template_matches_full_pipeline_for_text(self):
        from tensorrt_llm.sampling_params import SamplingParams
        from tensorrt_llm.serve.frontend.request_processor import FrontendProcessor

        class TemplateTokenizer:
            eos_token_id = 2
            pad_token_id = 0
            chat_template = "tmpl"

            def encode(self, text, add_special_tokens=True, **kwargs):
                return [hash(w) % 997 for w in text.split()]

            def get_chat_template(self, chat_template=None, tools=None):
                return chat_template or self.chat_template

            def apply_chat_template(self, conversation=None, **kwargs):
                rendered = "".join(f"[{m['role']}]{m['content']}" for m in conversation)
                if kwargs.get("add_generation_prompt"):
                    rendered += "[assistant]"
                return rendered

        class ModelConfig:
            model_type = "llama"

        tokenizer = TemplateTokenizer()
        conversation = [
            {"role": "user", "content": "hello there"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "again"},
        ]
        params = SamplingParams(end_id=2, add_special_tokens=False)
        full = FrontendProcessor(tokenizer, model_config=ModelConfig()).process_chat(
            [dict(m) for m in conversation], params
        )
        light = FrontendProcessor(tokenizer, lightweight_templates=True).process_chat(
            [dict(m) for m in conversation], params
        )
        assert light.prompt == full.prompt
        assert light.prompt_token_ids == full.prompt_token_ids
