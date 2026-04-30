# Proactive memory extraction — auto-extract facts from conversation turns.
#
# After each assistant response, scan the latest user<>assistant exchange for
# user preferences, corrections, stated facts, and environment details worth
# remembering.  Runs in a background thread to avoid blocking the main loop.
#
# Reconciliation (2026-04-25): instead of blind-appending extracted facts,
# the LLM is shown the most-similar existing MEMORY.md lines (via BM25 over
# the index itself) and must emit one of ADD / UPDATE / NOOP per candidate.
# This is the Mem0 pipeline, scaled down: we don't build a vector store, we
# just BM25 the always-loaded index which is capped at 200 lines.

import re
import sys
import threading
from typing import Optional

from ..config import CHARS_PER_TOKEN
from .operations import (
    append_journal, forget_from_memory_md, read_memory_md, remember,
    _MEMORY_MD_LOCK,
)


# Minimum user message length to trigger extraction (skip short commands)
_MIN_USER_MSG_CHARS = 30

# Maximum chars of exchange to send for analysis
_MAX_EXCHANGE_CHARS = 4000

# Cooldown: extract every N user messages (not every turn)
_EXTRACTION_INTERVAL = 4

# How many neighbour lines we show the LLM as reconciliation context.
_NEIGHBOUR_LINES = 4

# Module-level state
_messages_since_extraction = 0
_extraction_lock = threading.Lock()

EXTRACTION_PROMPT = """\
You are a memory reconciler for an AI agent. Analyze the conversation exchange \
below and decide what (if anything) should be written to long-term memory.

The always-loaded memory (MEMORY.md) already contains the lines shown under \
EXISTING MEMORY. For every candidate fact you extract from the EXCHANGE, emit \
ONE of the following directives on its own line:

  ADD::[category] <new fact>
      → append as a fresh durable rule. Use for information not represented
        in EXISTING MEMORY at all.

  UPDATE::<old_keyword>::[category] <new fact>
      → the user corrected or refined an existing line. <old_keyword> is a
        short substring that uniquely identifies the line to replace
        (case-insensitive). The old line is deleted from MEMORY.md and
        archived to the current month's journal so the audit trail is
        preserved without bloating the always-loaded index.

  NOOP::<reason>
      → the candidate is already covered or not worth remembering. Use this
        liberally — over-writing memory hurts more than missing a fact.

Rules:
- Extract at most 3 candidates per exchange.
- Each candidate must be one line, under 120 chars.
- Categories: correction | preference | gotcha | tip | workflow | user_stated
- Prefer UPDATE over ADD when an existing line covers the same topic but is
  wrong, stale, or narrower than the new fact.
- If nothing is worth writing, return exactly: NONE

EXISTING MEMORY (do not duplicate — UPDATE if you are correcting one of these):
{existing_memory}

---
EXCHANGE TO ANALYZE:

User: {user_message}
Assistant: {assistant_message}
"""


def _extract_text(content) -> str:
    """Pull plain text from a message content field (str or list-of-blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def should_extract(user_message: str) -> bool:
    """Decide whether this turn warrants extraction."""
    global _messages_since_extraction
    with _extraction_lock:
        _messages_since_extraction += 1
        if _messages_since_extraction < _EXTRACTION_INTERVAL:
            return False
        if len(user_message) < _MIN_USER_MSG_CHARS:
            return False
        # Reset counter — we'll extract this turn
        _messages_since_extraction = 0
        return True


def _build_reconciliation_context(user_message: str, assistant_message: str) -> str:
    """Render MEMORY.md for the LLM with relevance hints.

    We surface the full index (capped below 200 lines / 25 KB by design) but
    annotate the lines most relevant to the current exchange with a ``*``
    prefix so the reconciler focuses there first. BM25 scoring reuses the
    same tokenizer as the topic search — keeping one definition of
    "significant word" across the codebase.
    """
    existing = read_memory_md()
    if not existing:
        return "(MEMORY.md is empty)"

    # Truncate total payload. MEMORY.md is capped at 25 KB but we're paying
    # per-token on every extraction call, so keep the payload tight.
    if len(existing) > 3500:
        existing = existing[:3500] + "\n... (truncated — full index not shown)"

    try:
        # Lazy import — search imports operations, so we avoid any risk of
        # cycles at module-load time.
        from .search import _tokenize  # noqa: PLC0415
    except Exception:
        return existing

    exchange = f"{user_message}\n{assistant_message}"
    q_tokens = set(_tokenize(exchange))
    if not q_tokens:
        return existing

    annotated: list[str] = []
    for line in existing.splitlines():
        line_tokens = set(_tokenize(line))
        overlap = len(q_tokens & line_tokens)
        # 2+ shared significant tokens is a cheap but effective "this is
        # plausibly about the same thing" signal for single-line facts.
        marker = "*" if overlap >= 2 else " "
        annotated.append(f"{marker} {line}")
    return "\n".join(annotated)


_ADD_RE = re.compile(r"^\s*ADD::\s*(?:\[(?P<cat>[a-z_]+)\]\s*)?(?P<body>.+)$", re.IGNORECASE)
_UPDATE_RE = re.compile(
    r"^\s*UPDATE::\s*(?P<old>.+?)::\s*(?:\[(?P<cat>[a-z_]+)\]\s*)?(?P<body>.+)$",
    re.IGNORECASE,
)
_NOOP_RE = re.compile(r"^\s*NOOP::", re.IGNORECASE)


# Minimum unique-substring length for an UPDATE directive's <old_keyword>.
# Short substrings (e.g. "k8s") would steamroll many unrelated lines on
# common tokens. Mirror the old supersede floor.
_MIN_UPDATE_KEYWORD_LEN = 6

# Section headers and lines under them are protected from UPDATE-driven
# replacement. The LLM extraction pipeline is a prompt-injection lever; we
# don't let it rewrite Behavioral Rules, User Preferences, or User Profile
# from a single conversation turn.
_PROTECTED_SECTION_HEADERS = {
    "behavioral rules (critical)",
    "behavioral rules",
    "user preferences",
    "user profile",
}


def _replace_line(old_keyword: str, new_text: str, category: str) -> tuple[str, str]:
    """Replace MEMORY.md line(s) matching ``old_keyword`` with ``new_text``.

    Replaces the supersede() function the agent used to expose. Behavior
    differs in tier placement, not in semantics:

      - Old line(s) are removed from MEMORY.md (Tier 1 stays lean — no
        strikethrough accretion that costs prompt-cache hits forever).
      - The old line(s) are archived to ``data/memory/journal/YYYY-MM.md``
        with category ``memory_revision`` so "what did this used to say?"
        questions can still be answered.
      - The new line is appended via ``remember`` so soft-cap warnings,
        category validation and prefix formatting stay in one place.

    Same safety rails as the old supersede:
      - Keyword must be ≥ ``_MIN_UPDATE_KEYWORD_LEN`` chars.
      - Header / Behavioral-Rules / User-Profile / User-Preferences lines
        are skipped silently.
      - The whole read-modify-write runs under ``_MEMORY_MD_LOCK``.

    Returns (status, message). ``status`` is ``"update"`` on success or
    ``"skip"`` if no eligible line matched (mirroring the contract the
    extraction loop relied on).
    """
    if len(old_keyword) < _MIN_UPDATE_KEYWORD_LEN:
        return ("skip", f"Error: UPDATE keyword too short ({len(old_keyword)} < {_MIN_UPDATE_KEYWORD_LEN})")

    with _MEMORY_MD_LOCK:
        content = read_memory_md()
        if not content:
            return ("skip", f"No entries matched '{old_keyword}' — MEMORY.md empty")

        # Identify protected line indices (headers + lines inside protected
        # sections). Keep this logic local to extraction.py — it's only
        # needed by this code path now that supersede is gone.
        lines = content.split("\n")
        protected: set[int] = set()
        inside_protected_section = False
        for i, line in enumerate(lines):
            stripped_line = line.lstrip()
            if stripped_line.startswith("#"):
                protected.add(i)
                heading = stripped_line.lstrip("#").strip().lower()
                inside_protected_section = heading in _PROTECTED_SECTION_HEADERS
                continue
            if inside_protected_section:
                protected.add(i)

        keyword_lower = old_keyword.lower()
        matched_lines: list[str] = []
        skipped_protected = 0
        for i, line in enumerate(lines):
            if keyword_lower not in line.lower():
                continue
            if i in protected:
                skipped_protected += 1
                continue
            matched_lines.append(line)

        if not matched_lines:
            if skipped_protected:
                return (
                    "skip",
                    f"No entries matched '{old_keyword}' outside protected "
                    f"sections ({skipped_protected} header/rule hit(s) ignored)",
                )
            return ("skip", f"No entries matched '{old_keyword}'")

        # Archive the old line(s) to journal BEFORE removing them, so a
        # crash between the two writes leaves the audit trail intact rather
        # than an unrecoverable delete.
        archive_body = "\n".join(f"  - {ln.strip()}" for ln in matched_lines)
        append_journal(
            f"Replaced via UPDATE directive (keyword='{old_keyword}'):\n"
            f"{archive_body}\n"
            f"  → [{category or 'note'}] {new_text}",
            category="memory_revision",
        )

        # Now remove the old line(s) and append the new one. forget() runs
        # its own write under the same RLock — safe.
        forget_from_memory_md(old_keyword)
        remember(new_text, category, suppress_warning=True)

        skipped_note = (
            f" ({skipped_protected} protected line(s) preserved)"
            if skipped_protected else ""
        )
        return (
            "update",
            f"♻️ Replaced {len(matched_lines)} line(s) matching '{old_keyword}'"
            f"{skipped_note} with new [{category or 'note'}] entry "
            "(old archived to journal)",
        )


def _apply_directive(line: str) -> tuple[str, str] | None:
    """Interpret a single directive line. Returns (action, message) or None.

    Unknown lines are dropped silently — the LLM sometimes emits prose
    commentary alongside directives, and we prefer resilience to strict
    parsing here.

    Post-LLM validation (code-review H3): bodies are clamped to one line and
    120 chars, and lines that look like markdown headers are rejected. This
    closes the prompt-injection path where the LLM is coaxed into emitting a
    directive that turns into a new heading inside MEMORY.md.
    """
    stripped = line.strip()
    if not stripped or _NOOP_RE.match(stripped):
        return None

    def _sanitize_body(body: str) -> str | None:
        # One-line only; drop anything after an embedded newline.
        body = body.splitlines()[0].strip().lstrip("-").strip()
        if len(body) < 10 or len(body) > 120:
            return None
        # Don't let the LLM smuggle a new heading into MEMORY.md.
        if body.lstrip().startswith(("#", "~~")):
            return None
        return body

    m = _UPDATE_RE.match(stripped)
    if m:
        old_keyword = m.group("old").strip().strip("`'\"")
        category = (m.group("cat") or "").strip().lower()
        body = _sanitize_body(m.group("body"))
        if not (old_keyword and body):
            return None
        return _replace_line(old_keyword, body, category)

    m = _ADD_RE.match(stripped)
    if m:
        category = (m.group("cat") or "").strip().lower()
        body = _sanitize_body(m.group("body"))
        if not body:
            return None
        remember(body, category)
        return ("add", f"added [{category or 'note'}] {body[:80]}")

    return None


def extract_and_remember(runtime_owner, user_message: str, assistant_message: str) -> None:
    """Run extraction in background thread. Non-blocking, best-effort."""
    # Quick pre-filter: skip if messages are too short to contain extractable facts
    if len(user_message) < _MIN_USER_MSG_CHARS:
        return

    def _do_extract():
        try:
            context = _build_reconciliation_context(user_message, assistant_message)
            user_text = user_message[:_MAX_EXCHANGE_CHARS]
            asst_text = assistant_message[:_MAX_EXCHANGE_CHARS]

            prompt = EXTRACTION_PROMPT.format(
                existing_memory=context,
                user_message=user_text,
                assistant_message=asst_text,
            )

            result = runtime_owner.complete_text(prompt, max_output_tokens=384)

            if not result or not result.strip() or result.strip().upper() == "NONE":
                return

            adds = 0
            updates = 0
            for line in result.strip().splitlines():
                outcome = _apply_directive(line)
                if not outcome:
                    continue
                kind, _msg = outcome
                if kind == "add":
                    adds += 1
                elif kind == "update":
                    updates += 1
                # "skip" means the LLM emitted UPDATE/ADD but the underlying
                # call refused (e.g. supersede no-match) — don't count it
                # toward the cap so a real candidate later in the response
                # still gets a chance.
                if adds + updates >= 3:
                    break

            if adds or updates:
                bits = []
                if adds:
                    bits.append(f"+{adds} added")
                if updates:
                    bits.append(f"♻️{updates} replaced")
                sys.stderr.write(
                    f"\033[2m  🧠 Memory reconciled ({', '.join(bits)})\033[0m\n"
                )
                sys.stderr.flush()

        except Exception:
            pass  # Best-effort, never crash the agent

    thread = threading.Thread(target=_do_extract, daemon=True, name="memory-extraction")
    thread.start()
