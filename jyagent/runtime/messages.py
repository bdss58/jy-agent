"""Provider-neutral message utilities.

These functions operate on the normalized message types (AssistantMessage,
ToolCallBlock, etc.) and are NOT specific to any provider SDK.
"""

from __future__ import annotations

from typing import Any, cast

from .types import (
    AssistantMessage,
    Message,
    ModelSpec,
    TextBlock,
    ThinkingBlock,
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


# ─── Tool-call ID normalisation ──────────────────────────────────────────────

def normalize_anthropic_tool_call_id(tool_call_id: str) -> str:
    """Sanitise a tool_call id to satisfy Anthropic's character constraints."""
    normalized = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in tool_call_id)
    return normalized[:64] or "tool_call"


# ─── Thinking → text fallback ────────────────────────────────────────────────

def thinking_to_text_block(block: ThinkingBlock) -> TextBlock | None:
    """Convert a thinking block to a text block (for cross-model replay)."""
    thinking = block.get("thinking", "").strip()
    if not thinking:
        return None
    return {
        "type": "text",
        "text": f"<thinking>\n{thinking}\n</thinking>",
    }


# ─── Cross-model message normalisation ───────────────────────────────────────

def transform_messages_for_target(messages: list[Message], target: ModelSpec) -> list[Message]:
    """Normalise a message history for replay to a (possibly different) model.

    - Thinking blocks from the same model are preserved; from other models they
      are converted to ``<thinking>`` text blocks or dropped if redacted.
    - Tool-call IDs are sanitised when the target is Anthropic.
    - Missing tool_results are synthesised for dangling tool_calls.
    """
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
        new_blocks: list[Any] = []
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
    "normalize_anthropic_tool_call_id",
    "thinking_to_text_block",
    "transform_messages_for_target",
]
