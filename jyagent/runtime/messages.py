"""Provider-neutral message utilities.

These functions operate on the normalized message types (AssistantMessage,
ToolCallBlock, etc.) and are NOT specific to any provider SDK.
"""

from __future__ import annotations

from typing import Any, cast

from .types import (
    AssistantMessage,
    Message,
    ToolCallBlock,
    ToolResultMessage,
)


def assistant_text(message: AssistantMessage) -> str:
    """Extract concatenated text content from an AssistantMessage."""
    parts = [
        block.get("text", "")
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(parts)


def inject_missing_tool_results(messages: list[Message]) -> list[Message]:
    """Ensure every tool_call block in the history has a matching tool_result.

    If an assistant message contains tool_call blocks but subsequent messages
    don't include results for all of them, synthetic error results are injected.
    This prevents API validation errors for any provider.
    """
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


__all__ = [
    "assistant_text",
    "inject_missing_tool_results",
]
