"""Shared OpenAI helpers — pure functions for request/response/message transforms.

No side effects, no client creation, no adapter registration.
Used by ``providers/openai.py`` (OpenAI Chat Completions adapter).
"""

from __future__ import annotations

import json
from typing import Any, cast

from ..messages import transform_messages_for_target
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


# ─── Reasoning-model detection ─────────────────────────────────────────────

_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    """Return True for reasoning-series models (o1, o3, o4-*)."""
    normalized = model.strip().lower()
    return any(normalized.startswith(p) for p in _REASONING_MODEL_PREFIXES)


# ─── Response normalisation ────────────────────────────────────────────────

def usage_from_response(usage: Any) -> Usage:
    """Extract a normalized Usage from an OpenAI response usage object."""
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0

    cache_read = 0
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_details is not None:
        cache_read = getattr(prompt_details, "cached_tokens", 0) or 0

    raw: Usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read,
    }
    raw["total_tokens"] = compute_total_tokens(raw)
    return raw


def map_stop_reason(finish_reason: str | None) -> str:
    """Map OpenAI finish_reason to normalized stop reason."""
    if finish_reason == "stop":
        return "stop"
    if finish_reason == "length":
        return "length"
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "content_filter":
        return "error"
    if finish_reason is None:
        return "stop"
    return "error"


def assistant_from_response(model_spec: ModelSpec, response: Any) -> AssistantMessage:
    """Build an AssistantMessage from a non-streaming OpenAI ChatCompletion response."""
    choice = response.choices[0]
    message = choice.message
    content: list[dict[str, Any]] = []

    # Text content
    if message.content:
        content.append({"type": "text", "text": message.content})

    # Tool calls
    if message.tool_calls:
        for tc in message.tool_calls:
            arguments = tc.function.arguments or "{}"
            try:
                parsed_args = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                parsed_args = {}
            content.append({
                "type": "tool_call",
                "id": tc.id,
                "name": tc.function.name,
                "arguments": parsed_args,
            })

    usage = usage_from_response(getattr(response, "usage", None))
    response_id = getattr(response, "id", "")

    return {
        "role": "assistant",
        "content": content,
        "provider": model_spec.provider,
        "api": "openai-chat",
        "model": model_spec.model,
        "stop_reason": map_stop_reason(getattr(choice, "finish_reason", None)),
        "usage": usage,
        "response_id": response_id,
        "id": response_id,
    }


# ─── Request building — messages ───────────────────────────────────────────

def _convert_assistant_blocks(message: dict[str, Any]) -> dict[str, Any]:
    """Convert a normalized AssistantMessage to OpenAI chat format."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in message.get("content", []):
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "thinking":
            # Cross-model thinking: already converted to <thinking> text by
            # transform_messages_for_target, but if same-model thinking blocks
            # survive, wrap them.
            if block.get("redacted"):
                continue  # drop redacted thinking
            thinking_text = block.get("thinking", "").strip()
            if thinking_text:
                text_parts.append(f"<thinking>\n{thinking_text}\n</thinking>")
        elif block_type == "tool_call":
            arguments = block.get("arguments", {})
            if isinstance(arguments, str):
                args_str = arguments
            else:
                args_str = json.dumps(arguments)
            tool_calls.append({
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": args_str,
                },
            })

    result: dict[str, Any] = {"role": "assistant"}
    combined_text = "".join(text_parts)
    if combined_text:
        result["content"] = combined_text
    else:
        result["content"] = None
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def convert_messages(model_spec: ModelSpec, messages: list[Message]) -> list[dict[str, Any]]:
    """Transform normalized messages to OpenAI Chat Completions format.

    Calls transform_messages_for_target() first for cross-model normalization.
    """
    transformed = transform_messages_for_target(messages, model_spec)
    out: list[dict[str, Any]] = []

    for message in transformed:
        role = message.get("role")

        if role == "user":
            out.append({"role": "user", "content": message.get("content", "")})

        elif role == "assistant":
            out.append(_convert_assistant_blocks(message))

        elif role == "tool_result":
            tool_result = cast(ToolResultMessage, message)
            content = tool_result.get("content", "")
            if tool_result.get("is_error"):
                content = f"[ERROR] {content}"
            out.append({
                "role": "tool",
                "tool_call_id": tool_result["tool_call_id"],
                "content": content,
            })

    return out


# ─── Request building — tools ──────────────────────────────────────────────

def _add_additional_properties_false(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively add "additionalProperties": false to schemas that have "properties".

    This is required for OpenAI strict mode / structured outputs.
    """
    schema = dict(schema)  # shallow copy
    if "properties" in schema:
        schema["additionalProperties"] = False
        # Recurse into nested properties
        new_props = {}
        for key, prop in schema["properties"].items():
            if isinstance(prop, dict):
                new_props[key] = _add_additional_properties_false(prop)
            else:
                new_props[key] = prop
        schema["properties"] = new_props
    # Handle items in arrays
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        schema["items"] = _add_additional_properties_false(schema["items"])
    return schema


def convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert normalized tool definitions to OpenAI function calling format."""
    if not tools:
        return []
    result = []
    for tool in tools:
        parameters = tool.get("input_schema", {"type": "object", "properties": {}})
        parameters = _add_additional_properties_false(parameters)
        result.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": parameters,
                "strict": True,
            },
        })
    return result


# ─── Request building — tool_choice ────────────────────────────────────────

def convert_tool_choice(tool_choice: dict[str, Any] | None) -> str | dict[str, Any] | None:
    """Map normalized ToolChoice to OpenAI tool_choice parameter."""
    if tool_choice is None:
        return None
    tc_type = tool_choice.get("type")
    if tc_type == "auto":
        return "auto"
    if tc_type == "any":
        return "required"
    if tc_type == "none":
        return "none"
    if tc_type == "tool":
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return None


# ─── Request building — full kwargs ────────────────────────────────────────

def build_request_kwargs(
    model_spec: ModelSpec,
    context: Context,
    options: RuntimeOptions,
) -> dict[str, Any]:
    """Build the kwargs dict for client.chat.completions.create()."""
    is_reasoning = _is_reasoning_model(model_spec.model)

    openai_messages = convert_messages(model_spec, context.get("messages", []))

    # System prompt handling
    system_prompt = context.get("system_prompt", "")
    if system_prompt:
        if is_reasoning:
            # Reasoning models don't support system messages; prepend as user message
            openai_messages.insert(0, {
                "role": "user",
                "content": f"[System]\n{system_prompt}",
            })
        else:
            openai_messages.insert(0, {
                "role": "system",
                "content": system_prompt,
            })

    kwargs: dict[str, Any] = {
        "model": model_spec.model,
        "messages": openai_messages,
    }

    # max_output_tokens -> max_completion_tokens (reasoning models) or max_tokens
    if options.max_output_tokens is not None:
        if is_reasoning:
            kwargs["max_completion_tokens"] = options.max_output_tokens
        else:
            kwargs["max_tokens"] = options.max_output_tokens

    # Tools
    tools = convert_tools(context.get("tools"))
    if tools:
        kwargs["tools"] = tools

    # Tool choice
    if options.tool_choice is not None:
        tc = convert_tool_choice(options.tool_choice)
        if tc is not None:
            kwargs["tool_choice"] = tc

    # Reasoning effort (for o1/o3/o4 models)
    if options.reasoning is not None and isinstance(options.reasoning, dict):
        effort = options.reasoning.get("effort")
        if effort and is_reasoning:
            kwargs["reasoning_effort"] = effort

    return kwargs


__all__ = [
    "assistant_from_response",
    "build_request_kwargs",
    "convert_messages",
    "convert_tool_choice",
    "convert_tools",
    "map_stop_reason",
    "usage_from_response",
]
