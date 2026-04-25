# Conversation compaction — inspired by Claude Code's /compact command.
#
# Key design:
#   1. Token-based trigger instead of message count
#   2. Structured 9-section summary: preserves errors, environment, hypotheses
#   3. Cache-friendly: reuse system prompt so the prefix stays cached
#   4. File re-injection: restore recently accessed file contents after compaction
#   5. Memory re-injection after compaction

import os
import sys
from typing import Optional

from ..config import (
    COMPACT_TOKEN_THRESHOLD, SUMMARIZE_KEEP_RECENT,
    DEFAULT_MAX_TOKENS,
    FILE_REINJECTION_COUNT, FILE_REINJECTION_MAX_TOKENS,
    CHARS_PER_TOKEN,
)
from .conversation import ConversationMemory, estimate_tokens


# ─── 9-section structured summary prompt (Phase 2: P1.6) ─────────────────────
# Compared to the original 6-section prompt, this adds:
#   - Errors & Failures (preserves failure signals — critical per JetBrains research)
#   - Environment State (working directory, active connections)
#   - Working Hypotheses (current approach being tried)

COMPACT_PROMPT = """\
You are compacting a conversation to free up context space. \
Analyze the conversation below and produce a STRUCTURED summary that preserves \
all critical information needed to continue the work seamlessly.

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

## Errors & Failures
What went wrong? How was it resolved (or is it still unresolved)? Include error messages verbatim if they are needed for debugging:
- Error: ...
- Resolution / Status: ...

## Technical Details
Code patterns, configurations, API specifics, or technical facts that would be needed to continue the work:
- ...

## Environment State
Working directory, active connections (MCP servers, databases), relevant environment variables, tool versions:
- ...

## Current State
What was accomplished? What's the current status of the work?

## Working Hypotheses
What approach is currently being tried? Any theories about remaining issues?
- ...

## Pending Tasks
What remains to be done? Any next steps the user mentioned?
- [ ] ...

---
CONVERSATION TO COMPACT:

{conversation}
"""


# ─── File access tracker ──────────────────────────────────────────────────────

class FileAccessTracker:
    """Track files read/written during a session for post-compaction re-injection.

    Maintains an ordered list (most recent last) of unique file paths that were
    accessed via read_file, write_file, or edit_file tool calls.
    """

    def __init__(self):
        self._accessed: list[str] = []  # ordered, most recent last
        self._set: set[str] = set()     # for O(1) dedup

    def record(self, path: str) -> None:
        """Record a file access. Moves existing entries to the end (most recent)."""
        if path in self._set:
            self._accessed.remove(path)
        else:
            self._set.add(path)
        self._accessed.append(path)

    def recent(self, n: int = 5) -> list[str]:
        """Return the N most recently accessed file paths (newest first)."""
        return list(reversed(self._accessed[-n:]))

    def clear(self) -> None:
        """Reset all tracked accesses."""
        self._accessed.clear()
        self._set.clear()

    def __len__(self) -> int:
        return len(self._accessed)


# Module-level singleton — survives compaction, reset on new session.
_file_tracker = FileAccessTracker()


def get_file_tracker() -> FileAccessTracker:
    """Return the module-level file access tracker."""
    return _file_tracker


def record_file_access(path: str) -> None:
    """Convenience: record a file access on the global tracker."""
    _file_tracker.record(path)


# ─── File re-injection (Phase 2: P1.4) ───────────────────────────────────────

def _build_file_reinjection_content() -> str:
    """Build a context block containing recently accessed file contents.

    Reads up to FILE_REINJECTION_COUNT files, capped at
    FILE_REINJECTION_MAX_TOKENS total estimated tokens.  Skips files that
    no longer exist or cannot be read.
    """
    tracker = get_file_tracker()
    recent_files = tracker.recent(FILE_REINJECTION_COUNT)
    if not recent_files:
        return ""

    parts = []
    total_chars = 0
    max_chars = FILE_REINJECTION_MAX_TOKENS * CHARS_PER_TOKEN  # rough token→char

    for fpath in recent_files:
        if total_chars >= max_chars:
            break
        try:
            if not os.path.isfile(fpath):
                continue
            size = os.path.getsize(fpath)
            # Skip huge files (>100KB) — they'd dominate the budget
            if size > 100_000:
                parts.append(f"### `{fpath}` (skipped — {size:,} bytes, too large)")
                continue
            with open(fpath, "r", errors="replace") as f:
                content = f.read()
            remaining = max_chars - total_chars
            if len(content) > remaining:
                content = content[:remaining] + f"\n[... truncated at reinjection budget ...]"
            parts.append(f"### `{fpath}`\n```\n{content}\n```")
            total_chars += len(content)
        except Exception:
            continue  # skip unreadable files silently

    if not parts:
        return ""

    header = (
        "[Post-compaction context — recently accessed files re-injected "
        f"for continuity ({len(parts)} files)]\n\n"
    )
    return header + "\n\n".join(parts)


# ─── Core compaction ─────────────────────────────────────────────────────────

def compact_conversation(
    conversation: ConversationMemory,
    runtime_owner,
    keep_recent: int = None,
    custom_instruction: str = "",
    system_prompt: str = "",
) -> dict:
    """Compact a conversation by summarizing older messages.

    Keeps the most recent `keep_recent` messages intact, and replaces older
    messages with a structured summary.

    **Cache-friendly** (P0.3): When ``system_prompt`` is provided, the
    compaction call reuses it so the prompt cache prefix stays warm.
    Without this, every compaction is a guaranteed cache miss on the system
    prompt (measured at 98% miss rate in Claude Code's early iterations).

    **File re-injection** (P1.4): After compaction, re-injects the contents
    of recently accessed files so the LLM retains working context.
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

        compact_instruction = COMPACT_PROMPT.format(conversation=conversation_text)
        if custom_instruction:
            compact_instruction += f"\n\nAdditional instruction: {custom_instruction}"

        # ── Cache-friendly compaction (P0.3) ──────────────────────────
        # Reuse the existing system prompt so the prompt cache prefix stays
        # warm.  The compaction instruction is sent as a user message, not
        # as a standalone prompt.  This mirrors Claude Code's approach.
        if system_prompt:
            summary = _complete_with_system_prompt(
                runtime_owner, system_prompt, compact_instruction,
            )
        else:
            # Fallback: standalone prompt (cache miss, but works)
            summary = runtime_owner.complete_text(
                compact_instruction,
                max_output_tokens=min(4096, DEFAULT_MAX_TOKENS),
            )

        if not summary.strip():
            return {"compacted": False, "before_tokens": before_tokens,
                    "after_tokens": before_tokens, "summary": ""}

        # ── Build compacted conversation ──────────────────────────────
        new_messages = [
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
        ]

        # ── File re-injection (P1.4) ─────────────────────────────────
        file_context = _build_file_reinjection_content()
        if file_context:
            new_messages.append({
                "role": "user",
                "content": file_context,
            })
            new_messages.append({
                "role": "assistant",
                "content": (
                    "I've noted the re-injected file contents. "
                    "I'll use these as reference for continued work."
                ),
            })

        conversation.messages = new_messages + recent_messages

        after_tokens = conversation.estimated_tokens()

        return {
            "compacted": True,
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "summary": summary,
            "files_reinjected": len(get_file_tracker().recent(FILE_REINJECTION_COUNT)) if file_context else 0,
        }

    except Exception as e:
        return {"compacted": False, "before_tokens": before_tokens,
                "after_tokens": before_tokens, "summary": "",
                "error": str(e)}


def _complete_with_system_prompt(
    runtime_owner, system_prompt: str, user_prompt: str,
) -> str:
    """Run a completion reusing the existing system prompt for cache friendliness.

    Instead of ``complete_text(standalone_prompt)`` which creates a fresh
    system prompt (cache miss), this sends the compaction instruction as a
    user message under the existing system prompt — so the prompt prefix
    stays cached.
    """
    from ..llm.types import LLMOptions
    from ..config import get_reasoning_config_for_provider

    message = runtime_owner.complete(
        {
            "system_prompt": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        options=LLMOptions(
            max_output_tokens=min(4096, DEFAULT_MAX_TOKENS),
            timeout=120.0,
            reasoning=get_reasoning_config_for_provider(
                runtime_owner.model_spec.provider,
                max_output_tokens=min(4096, DEFAULT_MAX_TOKENS),
                model=runtime_owner.model_spec.model,
            ),
            metadata={"component": "compaction", "mode": "cache_friendly"},
        ),
    )
    parts = []
    for block in message.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def _format_messages_for_compact(messages: list) -> str:
    """Format messages into readable text for the compact prompt.

    Thinking blocks are omitted entirely (they were already pruned by the
    in-loop compaction, and the summary doesn't need them).
    Tool results are truncated to keep the compaction prompt manageable.
    """
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
                        # Omit thinking blocks from compaction input —
                        # they're verbose and the summary captures intent.
                        continue
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

            if parts:
                lines.append(f"**{role}**: {' '.join(parts)}")
        else:
            lines.append(f"**{role}**: {content}")

    return "\n\n".join(lines)


def summarize_if_needed(
    conversation: ConversationMemory,
    runtime_owner,
    keep_recent: int = SUMMARIZE_KEEP_RECENT,
    system_prompt: str = "",
    system_prompt_rebuilder=None,
) -> dict | None:
    """Auto-compact when the conversation exceeds COMPACT_TOKEN_THRESHOLD.

    Called between turns (after user input, before the next API call).
    Returns the compaction result dict, or None if no compaction was needed.
    """
    estimated = conversation.estimated_tokens()
    if estimated < COMPACT_TOKEN_THRESHOLD:
        return None
    if len(conversation) < keep_recent + 4:
        return None

    sys.stdout.write(
        f"\033[2m  ⚡ Auto-compacting conversation "
        f"(~{estimated} tokens, {len(conversation)} msgs)...\033[0m\n"
    )
    sys.stdout.flush()

    result = compact_conversation(
        conversation, runtime_owner, keep_recent,
        system_prompt=system_prompt,
    )

    if result.get("compacted"):
        files_note = ""
        if result.get("files_reinjected", 0):
            files_note = f", {result['files_reinjected']} files re-injected"
        sys.stdout.write(
            f"\033[2m  ✅ Compacted: ~{result['before_tokens']} → "
            f"~{result['after_tokens']} tokens "
            f"(saved ~{result['before_tokens'] - result['after_tokens']} tokens"
            f"{files_note})\033[0m\n"
        )
        sys.stdout.flush()

        if system_prompt_rebuilder:
            try:
                system_prompt_rebuilder()
            except Exception:
                pass

    return result
