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


# ─── 9-section structured summary prompt ─────────────────────────────────────
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


# ─── Reflection prompt — emit memory-write candidates ────────────────────────
# After the structured summary is produced, we ask the model a second, much
# smaller question: "do any durable lessons fall out of this session?".
# Candidates are written to the journal rather than directly into MEMORY.md —
# auto-promotion is a slippery slope (would erode the 200-line cap we defend
# against context rot). The agent reviews the reflection entry and explicitly
# promotes anything worth keeping with `manage_memory remember`.

REFLECTION_PROMPT = """\
You are a reflection agent. Below is a structured summary of a long \
conversation that was just compacted. Identify at most 3 durable learnings \
that would prevent future mistakes — things a fresh agent starting a new \
session would benefit from knowing.

HARD CONSTRAINTS:
- Return at most 3 items.
- Each item must pass the test: "would removing this cause future mistakes?"
- Skip anything that is just a record of what happened today (that is already
  in the summary). We want rules, gotchas, environment facts — not a diary.
- If nothing qualifies, return exactly: NONE

Output format — one directive per line, no prose:
  [category] <one-line durable fact>
Categories: correction | preference | gotcha | tip | workflow | user_stated

COMPACTION SUMMARY:
{summary}
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


# ─── File re-injection ───────────────────────────────────────────────────────

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


# ─── Tool-pair boundary safety ────────────────────────────────────────────────

def _starts_with_orphan_tool_result(msg: dict) -> bool:
    """Return True if this message is (or leads with) a tool_result block.

    A ``tool_result`` kept in the recent suffix whose matching assistant
    ``tool_use`` was summarised away becomes an ORPHAN — the provider
    either rejects the request (strict validation) or has to synthesize a
    fake tool_use, both are fragile.  We detect both shapes:

    * top-level ``role == "tool_result"`` (the normalised shape used by
      ``ConversationMemory``),
    * Anthropic-style user message whose content list contains a
      ``tool_result`` block (added by the provider transform after
      normalisation).
    """
    if msg.get("role") == "tool_result":
        return True
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return True
    return False


def _choose_split_point(messages: list, keep_recent: int) -> int:
    """Return the index of the first message to KEEP in the recent suffix.

    Nominally ``len(messages) - keep_recent``.  Walks leftward while that
    boundary message is an orphan-producing tool_result — pulling the
    preceding assistant ``tool_use`` message into the recent suffix so
    the tool-call ↔ tool-result pair stays intact after summarisation.

    Bounded by ``split > 0`` so we never return a negative index.  If the
    whole conversation is tool_results (pathological), the caller's
    normal ``len(conversation) < keep_recent + 2`` guard already no-ops.
    """
    n = len(messages)
    split = max(0, n - keep_recent)
    while split > 0 and _starts_with_orphan_tool_result(messages[split]):
        split -= 1
    return split


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

    **Cache-friendly**: When ``system_prompt`` is provided, the
    compaction call reuses it so the prompt cache prefix stays warm.
    Without this, every compaction is a guaranteed cache miss on the system
    prompt (measured at 98% miss rate in Claude Code's early iterations).

    **File re-injection**: After compaction, re-injects the contents
    of recently accessed files so the LLM retains working context.
    """
    if keep_recent is None:
        keep_recent = SUMMARIZE_KEEP_RECENT

    if len(conversation) < keep_recent + 2:
        return {"compacted": False, "before_tokens": conversation.estimated_tokens(),
                "after_tokens": conversation.estimated_tokens(), "summary": ""}

    before_tokens = conversation.estimated_tokens()

    try:
        # Tool-pair boundary safety: extend the keep window backward so we
        # don't split an assistant tool_use from its matching tool_result
        # (which would orphan the result and risk provider validation
        # failures on the next turn).  See ``_choose_split_point``.
        split = _choose_split_point(conversation.messages, keep_recent)
        if split == 0:
            # Walked past the beginning — conversation is almost entirely
            # tool_result-shaped (pathological, or keep_recent larger than
            # any assistant-text boundary).  No safe summarisation cut —
            # skip this round rather than produce an empty summary or
            # risk structural damage.
            return {"compacted": False, "before_tokens": before_tokens,
                    "after_tokens": before_tokens, "summary": ""}
        messages_to_compact = conversation.messages[:split]
        recent_messages = conversation.messages[split:]

        conversation_text = _format_messages_for_compact(messages_to_compact)

        compact_instruction = COMPACT_PROMPT.format(conversation=conversation_text)
        if custom_instruction:
            compact_instruction += f"\n\nAdditional instruction: {custom_instruction}"

        # ── Cache-friendly compaction ────────────────────────────────
        # Reuse the existing system prompt so the prompt cache prefix stays
        # warm.  The compaction instruction is sent as a user message, not
        # as a standalone prompt.  This mirrors Claude Code's approach.
        if system_prompt:
            summary = _complete_with_system_prompt(
                runtime_owner, system_prompt, compact_instruction,
                session_id=conversation.session_id,
            )
        else:
            # Fallback: standalone prompt (cache miss, but works)
            summary = runtime_owner.complete_text(
                compact_instruction,
                max_output_tokens=min(4096, DEFAULT_MAX_TOKENS),
                metadata={"component": "compaction", "session_id": conversation.session_id},
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

        # ── File re-injection ────────────────────────────────────────
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

        # ── Reflection pass (P3) ──────────────────────────────────────
        # Runs after compaction succeeds, never raises. Returns the count
        # of memory-write candidates written to the journal for the caller
        # to include in its status line.
        reflections = _run_reflection_pass(
            runtime_owner, summary, system_prompt=system_prompt,
        )

        return {
            "compacted": True,
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "summary": summary,
            "files_reinjected": len(get_file_tracker().recent(FILE_REINJECTION_COUNT)) if file_context else 0,
            "reflections": reflections,
        }

    except Exception as e:
        return {"compacted": False, "before_tokens": before_tokens,
                "after_tokens": before_tokens, "summary": "",
                "error": str(e)}


def _run_reflection_pass(
    runtime_owner,
    summary: str,
    *,
    system_prompt: str = "",
) -> int:
    """Ask the LLM to extract durable lessons from a compaction summary.

    Successful candidates land in the journal as ``[reflection]`` entries —
    NOT in MEMORY.md. Auto-promotion to the always-loaded index would erode
    the 200-line cap and reintroduce the context-rot symptoms we documented
    in topics/memory-design.md. The agent reviews and explicitly promotes.

    Returns the number of reflection candidates written. Always returns 0 on
    failure rather than raising — reflection is a best-effort enrichment of
    the compaction step, not part of its critical path.
    """
    if not summary or not summary.strip():
        return 0

    # Lazy import: append_journal lives in operations, which is heavy; keep
    # the module-level imports of compaction.py minimal so test suites can
    # patch operations.* late.
    try:
        from .operations import append_journal
    except Exception:
        return 0

    prompt = REFLECTION_PROMPT.format(summary=summary[:8000])
    try:
        if system_prompt:
            raw = _complete_with_system_prompt(
                runtime_owner, system_prompt, prompt,
            )
        else:
            raw = runtime_owner.complete_text(prompt, max_output_tokens=512)
    except Exception:
        return 0

    if not raw or raw.strip().upper() == "NONE":
        return 0

    candidates: list[str] = []
    cat_re = __import__("re").compile(
        r"^\s*\[(?P<cat>[a-z_]+)\]\s*(?P<body>.+?)\s*$", __import__("re").IGNORECASE,
    )
    for line in raw.strip().splitlines():
        m = cat_re.match(line)
        if not m:
            continue
        cat = m.group("cat").lower()
        body = m.group("body").strip()
        if len(body) < 10 or len(body) > 200:
            continue
        candidates.append(f"[{cat}] {body}")
        if len(candidates) >= 3:
            break

    if not candidates:
        return 0

    # Single journal entry, multiple candidates inside — the agent can scan
    # one entry to decide what (if anything) to promote with `remember`.
    body = (
        "Compaction reflection — candidates for `manage_memory remember` "
        "(do NOT promote blindly; apply the 'would removing this cause "
        "future mistakes?' filter):\n\n" + "\n".join(f"- {c}" for c in candidates)
    )
    try:
        append_journal(body, "reflection")
    except Exception:
        return 0
    return len(candidates)


def _complete_with_system_prompt(
    runtime_owner, system_prompt: str, user_prompt: str, session_id: str | None = None,
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
            metadata={
                "component": "compaction",
                "mode": "cache_friendly",
                **({"session_id": session_id} if session_id else {}),
            },
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
        reflect_note = ""
        if result.get("reflections", 0):
            reflect_note = (
                f", {result['reflections']} reflection candidate(s) "
                f"in latest journal entry"
            )
        sys.stdout.write(
            f"\033[2m  ✅ Compacted: ~{result['before_tokens']} → "
            f"~{result['after_tokens']} tokens "
            f"(saved ~{result['before_tokens'] - result['after_tokens']} tokens"
            f"{files_note}{reflect_note})\033[0m\n"
        )
        sys.stdout.flush()

        if system_prompt_rebuilder:
            try:
                system_prompt_rebuilder()
            except Exception:
                pass

    return result
