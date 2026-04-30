"""Shared Anthropic helpers — pure functions for request/response/message transforms.

No side effects, no client creation, no adapter registration.
Used by both ``providers/anthropic.py`` (main adapter) and
``tools/subagent.py`` (legacy shim).
"""

from __future__ import annotations

from typing import Any, cast

from ._anthropic_reasoning import build_anthropic_request_reasoning
from ...config import (
    ANTHROPIC_PROMPT_CACHE_ENABLED,
    ANTHROPIC_PROMPT_CACHE_TTL,
)
from ..messages import (  # noqa: F401 — re-export for backward compat
    assistant_text,
    inject_missing_tool_results,
    normalize_anthropic_tool_call_id,
    thinking_to_text_block,
    transform_messages_for_target,
)
from ..types import (
    AssistantMessage,
    Context,
    Message,
    ModelSpec,
    LLMOptions,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    compute_total_tokens,
)


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
    options: LLMOptions,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model_spec.model,
        "max_tokens": options.max_output_tokens,
        "messages": convert_messages(model_spec, context.get("messages", [])),
    }
    # ── Prompt caching (top-level automatic mode) ────────────────────
    # A single top-level cache_control field tells the Messages API to
    # auto-place the cache breakpoint at the last cacheable block and
    # slide it forward as the conversation grows. Without this, NO
    # caching happens — verified against current docs (2026-04).
    if ANTHROPIC_PROMPT_CACHE_ENABLED:
        cc: dict[str, Any] = {"type": "ephemeral"}
        if ANTHROPIC_PROMPT_CACHE_TTL and ANTHROPIC_PROMPT_CACHE_TTL != "5m":
            cc["ttl"] = ANTHROPIC_PROMPT_CACHE_TTL
        kwargs["cache_control"] = cc
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
