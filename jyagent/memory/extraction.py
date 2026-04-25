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
from .operations import read_memory_md, remember, supersede


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
        short substring that uniquely identifies the line to supersede
        (case-insensitive). The old line will be marked ~~struck-through~~
        and the new one appended.

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
        result = supersede(old_keyword, body, category)
        # supersede() is fallible: if nothing matched, or the keyword was
        # too short, or only protected lines were hit, it returns a message
        # that starts with "No entries matched" / "Error:". In that case
        # we must NOT count this as a successful update — otherwise a
        # hallucinated UPDAte crowds out real ADDs (code-review C3).
        if result.lstrip().startswith(("No entries matched", "Error:")):
            return ("skip", result)
        return ("update", result)

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
                    bits.append(f"♻️{updates} superseded")
                sys.stderr.write(
                    f"\033[2m  🧠 Memory reconciled ({', '.join(bits)})\033[0m\n"
                )
                sys.stderr.flush()

        except Exception:
            pass  # Best-effort, never crash the agent

    thread = threading.Thread(target=_do_extract, daemon=True, name="memory-extraction")
    thread.start()
