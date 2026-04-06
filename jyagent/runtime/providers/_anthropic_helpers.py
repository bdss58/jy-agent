"""Shared Anthropic helpers — pure functions for request/response/message transforms.

No side effects, no client creation, no adapter registration.
Used by both ``providers/anthropic.py`` (main adapter) and
``tools/subagent.py`` (legacy shim).
"""

from __future__ import annotations

from typing import Any, cast

from ..reasoning import build_anthropic_request_reasoning
from ..types import (
    AssistantMessage,
    Context,
    Message,
    ModelSpec,
    RuntimeOptions,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    compute_total_tokens,
)


# ─── Tool-call ID normalisation ──────────────────────────────────────────────

def normalize_anthropic_tool_call_id(tool_call_id: str) -> str:
    normalized = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in tool_call_id)
    return normalized[:64] or "tool_call"


# ─── Thinking → text fallback ────────────────────────────────────────────────

def thinking_to_text_block(block: ThinkingBlock) -> TextBlock | None:
    thinking = block.get("thinking", "").strip()
    if not thinking:
        return None
    return {
        "type": "text",
        "text": f"<thinking>\n{thinking}\n</thinking>",
    }


# ─── Orphaned tool-call repair ───────────────────────────────────────────────

def inject_missing_tool_results(messages: list[Message]) -> list[Message]:
    out: list[Message] = []
    pending_tool_calls: list[ToolCallBlock] = []
    existing_results: set[str] = set()

    def flush_pending() -> None:
        nonlocal pending_tool_calls, existing_results
        if not pending_tool_calls:
            return
        for tool_call in pending_tool_calls:
            if tool_call["id"] in existing_results:
                continue
            out.append({
                "role": "tool_result",
                "tool_call_id": tool_call["id"],
                "tool_name": tool_call["name"],
                "content": "No result provided",
                "is_error": True,
            })
        pending_tool_calls = []
        existing_results = set()

    for message in messages:
        role = message.get("role")
        if role == "assistant":
            flush_pending()
            assistant = cast(AssistantMessage, message)
            if assistant.get("stop_reason") in {"error", "aborted"}:
                continue
            out.append(assistant)
            tool_calls = [
                cast(ToolCallBlock, block)
                for block in assistant.get("content", [])
                if isinstance(block, dict) and block.get("type") == "tool_call"
            ]
            if tool_calls:
                pending_tool_calls = tool_calls
                existing_results = set()
        elif role == "tool_result":
            tool_result = cast(ToolResultMessage, message)
            existing_results.add(tool_result["tool_call_id"])
            out.append(tool_result)
        else:
            flush_pending()
            out.append(message)

    flush_pending()
    return out


# ─── Cross-model message normalisation ───────────────────────────────────────

def transform_messages_for_target(messages: list[Message], target: ModelSpec) -> list[Message]:
    tool_call_id_map: dict[str, str] = {}
    transformed: list[Message] = []

    for message in messages:
        role = message.get("role")
        if role == "user":
            transformed.append(message)
            continue

        if role == "tool_result":
            tool_result = cast(ToolResultMessage, dict(message))
            mapped = tool_call_id_map.get(tool_result["tool_call_id"])
            if mapped and mapped != tool_result["tool_call_id"]:
                tool_result["tool_call_id"] = mapped
            transformed.append(tool_result)
            continue

        assistant = cast(AssistantMessage, dict(message))
        same_model = assistant.get("provider") == target.provider and assistant.get("model") == target.model
        new_blocks = []
        for raw_block in assistant.get("content", []):
            if not isinstance(raw_block, dict):
                continue
            block_type = raw_block.get("type")
            if block_type == "text":
                new_blocks.append(raw_block)
                continue
            if block_type == "thinking":
                thinking_block_data = cast(ThinkingBlock, raw_block)
                if same_model:
                    new_blocks.append(thinking_block_data)
                else:
                    if thinking_block_data.get("redacted") or (
                        thinking_block_data.get("encrypted_content") and not thinking_block_data.get("thinking", "").strip()
                    ):
                        continue
                    text_block = thinking_to_text_block(thinking_block_data)
                    if text_block:
                        new_blocks.append(text_block)
                continue
            if block_type == "tool_call":
                tool_call = cast(ToolCallBlock, dict(raw_block))
                if target.provider == "anthropic":
                    normalized_id = normalize_anthropic_tool_call_id(tool_call["id"])
                    if normalized_id != tool_call["id"]:
                        tool_call_id_map[tool_call["id"]] = normalized_id
                        tool_call["id"] = normalized_id
                new_blocks.append(tool_call)

        assistant["content"] = new_blocks
        transformed.append(assistant)

    return inject_missing_tool_results(transformed)


def assistant_text(message: AssistantMessage) -> str:
    parts = [
        block.get("text", "")
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(parts)


# ─── Response normalisation ──────────────────────────────────────────────────

def usage_from_response(usage: Any) -> Usage:
    raw: Usage = {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }
    raw["total_tokens"] = compute_total_tokens(raw)
    return raw


def map_stop_reason(reason: str | None) -> str:
    if reason == "max_tokens":
        return "length"
    if reason == "tool_use":
        return "tool_use"
    if reason in {"end_turn", "stop_sequence", "pause_turn", None}:
        return "stop"
    if reason in {"refusal", "sensitive"}:
        return "error"
    return "error"


def thinking_block_from_sdk(block: Any) -> dict[str, Any]:
    if block.type == "redacted_thinking":
        return {
            "type": "thinking",
            "thinking": "",
            "signature": getattr(block, "data", ""),
            "redacted": True,
        }
    return {
        "type": "thinking",
        "thinking": getattr(block, "thinking", ""),
        "signature": getattr(block, "signature", ""),
    }


def assistant_from_response(model_spec: ModelSpec, response: Any) -> AssistantMessage:
    content: list[dict[str, Any]] = []
    message_id = getattr(response, "id", "")
    for block in response.content:
        if block.type == "text":
            content.append({"type": "text", "text": block.text})
        elif block.type in {"thinking", "redacted_thinking"}:
            content.append(thinking_block_from_sdk(block))
        elif block.type == "tool_use":
            content.append({
                "type": "tool_call",
                "id": block.id,
                "name": block.name,
                "arguments": block.input or {},
            })
    usage = usage_from_response(getattr(response, "usage", None))
    return {
        "role": "assistant",
        "content": content,
        "provider": model_spec.provider,
        "api": "anthropic-messages",
        "model": model_spec.model,
        "stop_reason": map_stop_reason(getattr(response, "stop_reason", None)),
        "usage": usage,
        "response_id": message_id,
        "id": message_id,
    }


def make_error_assistant_message(
    model_spec: ModelSpec,
    error: BaseException,
    partial_content: list[dict[str, Any]] | None = None,
) -> AssistantMessage:
    usage: Usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "role": "assistant",
        "content": partial_content or [],
        "provider": model_spec.provider,
        "api": "anthropic-messages",
        "model": model_spec.model,
        "stop_reason": "error",
        "usage": usage,
        "error_message": str(error),
    }


# ─── Request building ────────────────────────────────────────────────────────

def convert_assistant_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in message.get("content", []):
        block_type = block.get("type")
        if block_type == "text":
            blocks.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "thinking":
            if block.get("redacted") and block.get("signature"):
                blocks.append({"type": "redacted_thinking", "data": block["signature"]})
            elif block.get("signature"):
                blocks.append({
                    "type": "thinking",
                    "thinking": block.get("thinking", ""),
                    "signature": block["signature"],
                })
            elif block.get("thinking", "").strip():
                blocks.append({"type": "text", "text": block["thinking"]})
        elif block_type == "tool_call":
            blocks.append({
                "type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "input": block.get("arguments", {}),
            })
    return blocks


def convert_messages(model_spec: ModelSpec, messages: list[Message]) -> list[dict[str, Any]]:
    transformed = transform_messages_for_target(messages, model_spec)
    out: list[dict[str, Any]] = []
    idx = 0
    while idx < len(transformed):
        message = transformed[idx]
        role = message.get("role")
        if role == "user":
            out.append({"role": "user", "content": message.get("content", "")})
            idx += 1
            continue
        if role == "assistant":
            out.append({"role": "assistant", "content": convert_assistant_blocks(message)})
            idx += 1
            continue
        tool_results: list[dict[str, Any]] = []
        while idx < len(transformed) and transformed[idx].get("role") == "tool_result":
            tool_result = transformed[idx]
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_result["tool_call_id"],
                "content": tool_result["content"],
                "is_error": tool_result["is_error"],
            })
            idx += 1
        out.append({"role": "user", "content": tool_results})
    return out


def convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not tools:
        return []
    return [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("input_schema", {"type": "object", "properties": {}}),
        }
        for tool in tools
    ]


def build_request_kwargs(
    model_spec: ModelSpec,
    context: Context,
    options: RuntimeOptions,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model_spec.model,
        "max_tokens": options.max_output_tokens,
        "messages": convert_messages(model_spec, context.get("messages", [])),
    }
    if options.reasoning is not None:
        thinking, output_config = build_anthropic_request_reasoning(
            options.reasoning,
            model=model_spec.model,
        )
        if thinking is not None:
            kwargs["thinking"] = thinking
        if output_config is not None:
            kwargs["output_config"] = output_config
    if context.get("system_prompt"):
        kwargs["system"] = context["system_prompt"]
    tools = convert_tools(context.get("tools"))
    if tools:
        kwargs["tools"] = tools
    if options.tool_choice is not None:
        kwargs["tool_choice"] = options.tool_choice
    return {k: v for k, v in kwargs.items() if v is not None}
