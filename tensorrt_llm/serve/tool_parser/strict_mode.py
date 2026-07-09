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
"""Strict-tool constrained-decoding synthesis, shared by both serving paths.

Kept import-light so the engine-client pipeline can evaluate request
eligibility with the same sampling-params mutations the historical chat
path applies (a strict tool turns into guided decoding, which the pipeline
must treat as ineligible this loop).
"""

from tensorrt_llm.logger import logger


def build_tool_strict_guided_decoding_params(tools, tool_parser_name):
    """Build GuidedDecodingParams with structural tags for tools with strict=True.

    When a tool has ``strict=True`` in its function definition, the server
    should use constrained decoding to guarantee that the generated tool call
    arguments exactly match the function's ``parameters`` JSON Schema.

    This function builds structural tag items from each tool parser's
    ``structure_info()`` and the tool's ``parameters`` schema, then returns
    a ``GuidedDecodingParams`` with the structural tag format.

    Returns None if no tool has strict=True or the parser doesn't support
    structural tags.
    """
    from tensorrt_llm.sampling_params import GuidedDecodingParams
    from tensorrt_llm.serve.openai_protocol import ResponseFormat
    from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory

    if not tools or not tool_parser_name:
        return None

    # Check if any tool has strict=True
    has_strict = any(tool.function.strict for tool in tools if tool.function.strict)
    if not has_strict:
        return None

    tool_parser_cls = ToolParserFactory.parsers.get(tool_parser_name.lower())
    if tool_parser_cls is None:
        logger.warning(
            "Tool parser '%s' not found, cannot enforce strict mode for tools.",
            tool_parser_name,
        )
        return None

    parser = tool_parser_cls()
    if not parser.supports_structural_tag():
        logger.warning(
            "Tool parser '%s' does not support structural tags, "
            "cannot enforce strict mode for tools.",
            tool_parser_name,
        )
        return None

    get_info = parser.structure_info()

    tags = []
    triggers = set()
    for tool in tools:
        info = get_info(tool.function.name)
        triggers.add(info.trigger)

        if tool.function.strict and tool.function.parameters:
            # Strict tool: constrain arguments to match the JSON Schema
            content = {
                "type": "json_schema",
                "json_schema": tool.function.parameters,
            }
        else:
            # Non-strict tool or no parameters: allow any text
            content = {"type": "any_text"}

        tags.append(
            {
                "begin": info.begin,
                "content": content,
                "end": info.end,
            }
        )

    stag_format = {
        "type": "triggered_tags",
        "triggers": sorted(triggers),
        "tags": tags,
    }

    resp_format = ResponseFormat(type="structural_tag", format=stag_format)
    return GuidedDecodingParams(
        structural_tag=resp_format.model_dump_json(by_alias=True, exclude_none=True)
    )
