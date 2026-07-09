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
"""Import-light chat templating for text-only conversations.

Mirrors the text path of the full multimodal-aware template pipeline
(``tensorrt_llm.inputs.utils.apply_chat_template``) without its
torch/multimodal import graph: template resolution plus the HF tokenizer's
own ``apply_chat_template``. Used by frontends that must not load the
runtime (multimodal requests are rejected there, so the text path is the
whole surface).
"""

from typing import Any, Optional


def resolve_hf_chat_template(
    tokenizer: Any, processor: Any, chat_template: Optional[str], tools: Optional[list]
) -> Optional[str]:
    """Resolve the chat template with the same precedence as the full path."""
    # 1. If chat_template is not None, return it
    if chat_template is not None:
        return chat_template

    # 2. If tool is not provided, use the processor's default chat template
    if not tools and processor and hasattr(processor, "chat_template"):
        return processor.chat_template

    # 3. If tool is provided, use the tool
    try:
        return tokenizer.get_chat_template(chat_template, tools=tools)
    except Exception:
        return None


def apply_text_chat_template(
    *,
    tokenizer: Any,
    conversation: list,
    add_generation_prompt: bool,
    tools: Optional[list] = None,
    documents: Optional[list] = None,
    chat_template: Optional[str] = None,
    chat_template_kwargs: Optional[dict] = None,
) -> str:
    """Apply a chat template to a text-only conversation.

    Matches the full pipeline's behavior for flat-text messages: resolve the
    template, then delegate to the tokenizer's ``apply_chat_template``.
    """
    # Unwrap TransformersTokenizer-style wrappers to the HF tokenizer.
    hf_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    resolved_template = resolve_hf_chat_template(hf_tokenizer, None, chat_template, tools)
    if resolved_template is None:
        raise ValueError("No chat template found for the given tokenizer and tools.")
    return hf_tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        return_dict=False,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        documents=documents,
        chat_template=resolved_template,
        **(chat_template_kwargs or {}),
    )
