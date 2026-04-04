# Conversation compaction — inspired by Claude Code's /compact command.
#
# Key design:
#   1. Token-based trigger instead of message count
#   2. Structured summary: preserves files modified, key decisions, pending tasks
#   3. Memory re-injection after compaction

import sys
from typing import Optional

from ..config import (
    COMPACT_TOKEN_THRESHOLD, SUMMARIZE_KEEP_RECENT, SUMMARIZE_THRESHOLD,
    DEFAULT_MAX_TOKENS,
)
from .conversation import ConversationMemory
COMPACT_PROMPT = """\
You are compacting a conversation to free up context space. \
Analyze the conversation below and produce a STRUCTURED summary that preserves critical information.

Your summary MUST include these sections (skip any section that has no relevant content):

## Task Context
What is the user working on? What's the overall goal?

## Files Modified
List every file that was created, modified, or read during this conversation, with a brief note on what was done:
- `path/to/file.py` — added function X, fixed bug Y

## Key Decisions
Important choices made and WHY (e.g., "Chose approach A over B because..."):
- Decision: ...
- Rationale: ...

## Technical Details
Code patterns, configurations, error messages, or technical facts that would be needed to continue the work:
- ...

## Current State
What was accomplished? What's the current status?

## Pending Tasks
What remains to be done? Any next steps the user mentioned?
- [ ] ...

---
CONVERSATION TO COMPACT:

{conversation}
"""


def compact_conversation(
    conversation: ConversationMemory,
    runtime_owner,
    keep_recent: int = None,
    custom_instruction: str = "",
) -> dict:
    """Compact a conversation by summarizing older messages.

    Keeps the most recent `keep_recent` messages intact, and replaces older
    messages with a structured summary.
    """
    if keep_recent is None:
        keep_recent = SUMMARIZE_KEEP_RECENT

    if len(conversation) < keep_recent + 2:
        return {"compacted": False, "before_tokens": conversation.estimated_tokens(),
                "after_tokens": conversation.estimated_tokens(), "summary": ""}

    before_tokens = conversation.estimated_tokens()

    try:
        messages_to_compact = conversation.messages[:-keep_recent]
        recent_messages = conversation.messages[-keep_recent:]

        conversation_text = _format_messages_for_compact(messages_to_compact)

        prompt = COMPACT_PROMPT.format(conversation=conversation_text)
        if custom_instruction:
            prompt += f"\n\nAdditional instruction: {custom_instruction}"

        summary = runtime_owner.complete_text(
            prompt,
            max_output_tokens=min(2048, DEFAULT_MAX_TOKENS),
        )

        if not summary.strip():
            return {"compacted": False, "before_tokens": before_tokens,
                    "after_tokens": before_tokens, "summary": ""}

        conversation.messages = [
            {
                "role": "user",
                "content": (
                    "[Conversation compacted — structured summary of previous "
                    f"{len(messages_to_compact)} messages below]\n\n{summary}"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Understood. I have the compacted context above. "
                    "I'll continue with full awareness of our previous work."
                ),
            },
        ] + recent_messages

        after_tokens = conversation.estimated_tokens()

        return {
            "compacted": True,
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "summary": summary,
        }

    except Exception as e:
        return {"compacted": False, "before_tokens": before_tokens,
                "after_tokens": before_tokens, "summary": "",
                "error": str(e)}


def _format_messages_for_compact(messages: list) -> str:
    """Format messages into readable text for the compact prompt."""
    MAX_TOOL_RESULT_CHARS = 2000

    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if isinstance(content, str):
            lines.append(f"**{role}**: {content}")

        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(block.get("text", ""))
                    elif btype in {"tool_use", "tool_call"}:
                        name = block.get("name", "?")
                        inp = str(block.get("arguments", block.get("input", {})))
                        if len(inp) > 500:
                            inp = inp[:500] + "..."
                        parts.append(f"[Tool call: {name}({inp})]")
                    elif btype == "thinking":
                        thinking = block.get("thinking", "")
                        if thinking:
                            parts.append(f"[Thinking: {thinking[:500]}]")
                    elif btype == "tool_result":
                        result = str(block.get("content", ""))
                        if len(result) > MAX_TOOL_RESULT_CHARS:
                            result = result[:MAX_TOOL_RESULT_CHARS] + f"... ({len(result)} chars total, truncated)"
                        parts.append(f"[Tool result: {result}]")
                    else:
                        s = str(block)
                        if len(s) > 500:
                            s = s[:500] + "..."
                        parts.append(s)
                else:
                    parts.append(str(block))
            lines.append(f"**{role}**: {' '.join(parts)}")
        else:
            lines.append(f"**{role}**: {str(content)}")

    return "\n\n".join(lines)


def summarize_if_needed(
    conversation: ConversationMemory,
    runtime_owner,
    system_prompt_rebuilder=None,
    threshold_tokens: int = None,
    keep_recent: int = None,
) -> Optional[dict]:
    """Auto-compact when conversation exceeds token threshold."""
    if threshold_tokens is None:
        threshold_tokens = COMPACT_TOKEN_THRESHOLD
    if keep_recent is None:
        keep_recent = SUMMARIZE_KEEP_RECENT

    estimated = conversation.estimated_tokens()

    token_trigger = estimated >= threshold_tokens
    count_trigger = len(conversation) >= SUMMARIZE_THRESHOLD * 2

    if not token_trigger and not count_trigger:
        return None

    trigger = "tokens" if token_trigger else "messages"
    sys.stdout.write(
        f"\033[2m  ⚡ Auto-compacting conversation "
        f"({trigger}: ~{estimated} tokens, {len(conversation)} msgs)...\033[0m\n"
    )
    sys.stdout.flush()

    result = compact_conversation(conversation, runtime_owner, keep_recent)

    if result.get("compacted"):
        sys.stdout.write(
            f"\033[2m  ✅ Compacted: ~{result['before_tokens']} → "
            f"~{result['after_tokens']} tokens "
            f"(saved ~{result['before_tokens'] - result['after_tokens']} tokens)\033[0m\n"
        )
        sys.stdout.flush()

        if system_prompt_rebuilder:
            try:
                system_prompt_rebuilder()
            except Exception:
                pass

    return result
