"""MCP ↔ agent type conversion helpers.

The MCP SDK exposes Pydantic models (``types.Tool``, ``types.CallToolResult``,
``types.TextContent`` …). The agent's tool registry and the chat-completion
adapters all consume plain dicts with JSON-schema input shapes. These pure
functions sit on the boundary.

Two directions live here:

  * MCP SDK object → dict
      ``tool_to_dict`` / ``call_result_to_dict``
      Forward-compatible with SDK v2's ``structured_content``.

  * dict → agent tool schema
      ``mcp_schema_to_agent_schema``       — adds the ``mcp__{server}__{tool}``
                                             prefix, strips Bedrock-hostile
                                             keys (``default`` /
                                             ``additionalProperties``).
      ``extract_mcp_result``               — renders a CallToolResult dict as
                                             readable text.

Previously split across ``client.py`` (object→dict) and ``manager.py``
(dict→agent), which made it hard to reason about the full conversion
journey. Extracted 2026-05-12 as step 1 of the mcp/ cleanup.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import mcp.types as types


# ─── MCP SDK objects → dict ───────────────────────────────────────────────────

def tool_to_dict(tool: types.Tool) -> dict:
    """Convert a Pydantic ``Tool`` into the dict shape the agent uses.

    Always emits a non-empty ``inputSchema`` — MCP servers occasionally omit
    it for argument-less tools; we substitute the empty-object schema so the
    agent registry doesn't choke on ``None``.
    """
    return {
        "name": tool.name,
        "description": tool.description or f"MCP tool: {tool.name}",
        "inputSchema": tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}},
    }


def call_result_to_dict(result: types.CallToolResult) -> dict:
    """Convert a ``CallToolResult`` into a dict the runtime can serialise.

    Forward-compatible: the official SDK v2 will add a ``structured_content``
    field; we surface either spelling under ``structuredContent`` so callers
    can rely on one key regardless of SDK version.
    """
    content: list[dict] = []
    for item in (result.content or []):
        if isinstance(item, types.TextContent):
            content.append({"type": "text", "text": item.text})
        elif isinstance(item, types.ImageContent):
            content.append({
                "type": "image",
                "data": item.data,
                "mimeType": item.mimeType,
            })
        elif isinstance(item, types.EmbeddedResource):
            content.append({
                "type": "resource",
                "resource": item.resource.model_dump() if item.resource else {},
            })
        else:
            content.append(item.model_dump() if hasattr(item, "model_dump") else {"type": "unknown"})

    result_dict: dict[str, Any] = {
        "content": content,
        "isError": result.isError or False,
    }

    # Forward-compat — extract structured_content if SDK v2 adds it
    if hasattr(result, "structuredContent") and result.structuredContent:
        result_dict["structuredContent"] = result.structuredContent
    elif hasattr(result, "structured_content") and result.structured_content:
        result_dict["structuredContent"] = result.structured_content

    return result_dict


# ─── dict → agent ─────────────────────────────────────────────────────────────

def mcp_schema_to_agent_schema(mcp_tool: dict, tool_name: str) -> dict:
    """Convert an MCP tool schema to the agent's tool schema format.

    MCP format::

        {"name": "navigate_page", "description": "...",
         "inputSchema": {"type": "object", "properties": {...}}}

    Agent format::

        {"name": "mcp__chrome__navigate_page", "description": "...",
         "input_schema": {"type": "object", "properties": {...}}}

    Side effects on the schema (Bedrock compatibility):
      - ``default`` keys are stripped from every property.
      - ``additionalProperties`` is stripped from every nested object AND
        from the top level.
      - ``required`` is forced to ``[]`` when missing.

    Always deep-copies the incoming schema so the caller's dict is not
    mutated.
    """
    input_schema = mcp_tool.get("inputSchema", {"type": "object", "properties": {}})
    input_schema = copy.deepcopy(input_schema)

    if "required" not in input_schema:
        input_schema["required"] = []

    if "properties" in input_schema:
        for _prop_name, prop_def in input_schema["properties"].items():
            if isinstance(prop_def, dict):
                prop_def.pop("default", None)
                prop_def.pop("additionalProperties", None)

    input_schema.pop("additionalProperties", None)

    return {
        "name": tool_name,
        "description": mcp_tool.get("description", f"MCP tool: {mcp_tool.get('name', tool_name)}"),
        "input_schema": input_schema,
    }


def extract_mcp_result(result: dict) -> str:
    """Render an ``CallToolResult`` dict as readable plain text.

    Strategy:
      - If ``content`` is a list of typed blocks, concatenate ``text`` blocks
        and stub ``image`` blocks as ``[Image: <mime>]``. Unknown block types
        fall back to JSON-encoded form.
      - If ``content`` is already a plain string, return it.
      - As a last resort, dump the whole result as indented JSON so the
        agent at least sees something diagnosable.
    """
    content = result.get("content", [])
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "image":
                    texts.append(f"[Image: {item.get('mimeType', 'image')}]")
                elif "text" in item:
                    texts.append(item["text"])
                else:
                    texts.append(json.dumps(item, ensure_ascii=False))
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts) if texts else json.dumps(result, indent=2, ensure_ascii=False)
    elif isinstance(content, str):
        return content
    else:
        return json.dumps(result, indent=2, ensure_ascii=False)
