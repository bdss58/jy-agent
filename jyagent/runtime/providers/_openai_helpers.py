"""Shared OpenAI helpers — pure functions for request/response/message transforms.

No side effects, no client creation, no adapter registration.
Used by ``providers/openai.py`` (OpenAI Responses API adapter).
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


# ─── Model capability detection ─────────────────────────────────────────────

_OPENAI_LEGACY_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4")
_OPENAI_REASONING_EFFORT_MODEL_PREFIX = "gpt-5.4"

_OPENAI_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh"}


def validate_openai_reasoning(reasoning: Any, *, model: str | None = None) -> dict[str, Any]:
    """Validate an OpenAI reasoning config dict."""
    if not isinstance(reasoning, dict):
        raise ValueError(f"OpenAI reasoning config must be a dict, got {type(reasoning).__name__}.")
    effort = reasoning.get("effort")
    if effort is not None and effort not in _OPENAI_REASONING_EFFORTS:
        allowed = ", ".join(sorted(_OPENAI_REASONING_EFFORTS))
        raise ValueError(f"OpenAI reasoning effort must be one of: {allowed}. Got '{effort}'.")
    if model and effort and not supports_openai_reasoning_effort(model):
        raise ValueError(
            f"OpenAI reasoning effort is not supported by model '{model}'. "
            "Use 'gpt-5.4' or newer."
        )
    return reasoning


def supports_openai_reasoning_effort(model: str) -> bool:
    """Return True for GPT-5.4 models that accept reasoning_effort."""
    normalized = model.strip().lower()
    return normalized == _OPENAI_REASONING_EFFORT_MODEL_PREFIX or normalized.startswith(
        f"{_OPENAI_REASONING_EFFORT_MODEL_PREFIX}-"
    )


def uses_openai_legacy_reasoning_transport(model: str) -> bool:
    """Return True for o-series models that require transport quirks.

    In the Responses API, o-series models are natively supported and
    ``instructions`` works normally.  The only remaining quirk is that
    temperature is not supported for these models.
    """
    normalized = model.strip().lower()
    return any(normalized.startswith(p) for p in _OPENAI_LEGACY_REASONING_MODEL_PREFIXES)


# ─── Response normalisation ────────────────────────────────────────────────

def usage_from_response(usage: Any) -> Usage:
    """Extract a normalized Usage from an OpenAI Responses API usage object.

    The Responses API ``ResponseUsage`` exposes ``input_tokens``,
    ``output_tokens``, ``total_tokens``, and
    ``input_tokens_details.cached_tokens``.
    """
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0

    cache_read = 0
    input_details = getattr(usage, "input_tokens_details", None)
    if input_details is not None:
        cache_read = getattr(input_details, "cached_tokens", 0) or 0

    raw: Usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read,
    }
    raw["total_tokens"] = compute_total_tokens(raw)
    return raw


def map_stop_reason(status: str | None, has_tool_calls: bool = False) -> str:
    """Map a Responses API ``response.status`` to a normalized stop reason.

    Parameters
    ----------
    status:
        One of ``"completed"``, ``"incomplete"``, ``"failed"``, or ``None``.
    has_tool_calls:
        Whether the response output contains any ``function_call`` items.
    """
    if status == "completed":
        return "tool_use" if has_tool_calls else "stop"
    if status == "incomplete":
        return "length"
    if status == "failed":
        return "error"
    if status is None:
        return "stop"
    return "error"


def assistant_from_response(model_spec: ModelSpec, response: Any) -> AssistantMessage:
    """Build an AssistantMessage from an OpenAI Responses API ``Response`` object."""
    content: list[dict[str, Any]] = []
    has_tool_calls = False

    for item in getattr(response, "output", []):
        item_type = getattr(item, "type", None)

        if item_type == "message":
            # Extract text from message content parts
            for part in getattr(item, "content", []):
                part_type = getattr(part, "type", None)
                if part_type == "output_text":
                    text = getattr(part, "text", "")
                    if text:
                        content.append({"type": "text", "text": text})

        elif item_type == "function_call":
            has_tool_calls = True
            arguments_str = getattr(item, "arguments", "") or "{}"
            try:
                parsed_args = json.loads(arguments_str)
            except (json.JSONDecodeError, TypeError) as exc:
                import logging
                logging.getLogger("jyagent.runtime").warning(
                    "Malformed tool-call arguments from OpenAI (call_id=%s, name=%s): %s",
                    getattr(item, "call_id", "?"), getattr(item, "name", "?"), exc,
                )
                parsed_args = {"_parse_error": str(exc)}
            content.append({
                "type": "tool_call",
                "id": getattr(item, "call_id", ""),
                "name": getattr(item, "name", ""),
                "arguments": parsed_args,
            })

        elif item_type == "reasoning":
            # Preserve reasoning/thinking output if available
            summary_parts = getattr(item, "summary", None) or []
            summary_texts: list[str] = []
            for sp in summary_parts:
                text = getattr(sp, "text", "") if not isinstance(sp, str) else sp
                if text:
                    summary_texts.append(text)
            if summary_texts:
                content.append({
                    "type": "thinking",
                    "thinking": "\n".join(summary_texts),
                    "summary": summary_texts,
                })

    usage = usage_from_response(getattr(response, "usage", None))
    response_id = getattr(response, "id", "")
    status = getattr(response, "status", None)

    return {
        "role": "assistant",
        "content": content,
        "provider": model_spec.provider,
        "api": "openai-responses",
        "model": model_spec.model,
        "stop_reason": map_stop_reason(status, has_tool_calls),
        "usage": usage,
        "response_id": response_id,
        "id": response_id,
    }


# ─── Request building — messages (Responses API input items) ──────────────

def _convert_assistant_to_input_items(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a normalized AssistantMessage to Responses API input items.

    An assistant message may expand to multiple items:
    - An ``EasyInputMessage`` with role ``"assistant"`` for any text content.
    - One ``function_call`` item per tool call.
    """
    text_parts: list[str] = []
    items: list[dict[str, Any]] = []

    for block in message.get("content", []):
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "thinking":
            # Cross-model thinking: wrap in <thinking> text for replay.
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
            items.append({
                "type": "function_call",
                "call_id": block["id"],
                "name": block["name"],
                "arguments": args_str,
            })

    # Emit an assistant message item for text content (before tool calls)
    combined_text = "".join(text_parts)
    if combined_text:
        items.insert(0, {"role": "assistant", "content": combined_text})
    elif not items:
        # No text and no tool calls — emit empty assistant message
        items.append({"role": "assistant", "content": ""})

    return items


def convert_messages(model_spec: ModelSpec, messages: list[Message]) -> list[dict[str, Any]]:
    """Transform normalized messages to OpenAI Responses API input items.

    Calls ``transform_messages_for_target()`` first for cross-model normalization.

    Returns a list suitable for the ``input`` parameter of
    ``client.responses.create()``.
    """
    transformed = transform_messages_for_target(messages, model_spec)
    out: list[dict[str, Any]] = []

    for message in transformed:
        role = message.get("role")

        if role == "user":
            out.append({"role": "user", "content": message.get("content", "")})

        elif role == "assistant":
            out.extend(_convert_assistant_to_input_items(message))

        elif role == "tool_result":
            tool_result = cast(ToolResultMessage, message)
            content = tool_result.get("content", "")
            if tool_result.get("is_error"):
                content = f"[ERROR] {content}"
            out.append({
                "type": "function_call_output",
                "call_id": tool_result["tool_call_id"],
                "output": content,
            })

    return out


# ─── Request building — tools (Responses API format) ─────────────────────

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
    """Convert normalized tool definitions to OpenAI Responses API function tool format.

    In the Responses API, tools are flat objects (not nested under ``function``):
    ``{"type": "function", "name": ..., "description": ..., "parameters": ..., "strict": ...}``
    """
    if not tools:
        return []
    result = []
    for tool in tools:
        parameters = tool.get("input_schema", {"type": "object", "properties": {}})
        parameters = _add_additional_properties_false(parameters)
        result.append({
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": parameters,
            "strict": True,
        })
    return result


# ─── Request building — tool_choice (Responses API format) ───────────────

def convert_tool_choice(tool_choice: dict[str, Any] | None) -> str | dict[str, Any] | None:
    """Map normalized ToolChoice to OpenAI Responses API tool_choice parameter."""
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
        return {"type": "function", "name": tool_choice["name"]}
    return None


# ─── Request building — full kwargs for client.responses.create() ────────

def build_request_kwargs(
    model_spec: ModelSpec,
    context: Context,
    options: RuntimeOptions,
) -> dict[str, Any]:
    """Build the kwargs dict for ``client.responses.create()`` / ``.stream()``.

    The Responses API uses ``input`` instead of ``messages``,
    ``instructions`` instead of a system message, and
    ``max_output_tokens`` for all models.
    """
    is_o_series = uses_openai_legacy_reasoning_transport(model_spec.model)
    supports_reasoning = supports_openai_reasoning_effort(model_spec.model)

    input_items = convert_messages(model_spec, context.get("messages", []))

    kwargs: dict[str, Any] = {
        "model": model_spec.model,
        "input": input_items,
        "store": False,
    }

    # System prompt → instructions (works for all models in Responses API,
    # including o-series — no more user-message hack needed).
    system_prompt = context.get("system_prompt", "")
    if system_prompt:
        kwargs["instructions"] = system_prompt

    # max_output_tokens (Responses API uses this for all models)
    if options.max_output_tokens is not None:
        kwargs["max_output_tokens"] = options.max_output_tokens

    # Temperature — o-series models do not support it
    # (skipped for o-series; other models get it if set in options metadata)

    # Tools
    tools = convert_tools(context.get("tools"))
    if tools:
        kwargs["tools"] = tools

    # Tool choice
    if options.tool_choice is not None:
        tc = convert_tool_choice(options.tool_choice)
        if tc is not None:
            kwargs["tool_choice"] = tc

    # Reasoning config → reasoning parameter
    if options.reasoning is not None and isinstance(options.reasoning, dict):
        validated = validate_openai_reasoning(options.reasoning, model=model_spec.model)
        effort = validated.get("effort")
        if effort and supports_reasoning:
            kwargs["reasoning"] = {"effort": effort}

    return kwargs


__all__ = [
    "assistant_from_response",
    "build_request_kwargs",
    "convert_messages",
    "convert_tool_choice",
    "convert_tools",
    "map_stop_reason",
    "supports_openai_reasoning_effort",
    "usage_from_response",
    "uses_openai_legacy_reasoning_transport",
    "validate_openai_reasoning",
]
