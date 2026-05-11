# Extraction security — structural defenses against prompt injection on the
# auto-extraction write path.
#
# Auto-extraction sees the raw User: / Assistant: text of each exchange. Any
# pasted web page, tool output, or shell log can try to smuggle a "future
# instruction" into always-loaded MEMORY.md via the reconciler LLM. We defend
# in two layers:
#
#   Layer 1 — the SECURITY paragraph inside EXTRACTION_PROMPT (see
#             ``extraction.py``). This is the prompt-level instruction to the
#             LLM to treat EXCHANGE content as data, not instructions.
#   Layer 2 — THIS module. Even if the LLM is fooled, we reject candidates
#             whose ADD/UPDATE body matches a known injection shape, or
#             contains URLs / HTML tags / code fences that have no place in
#             a durable one-line rule.
#
# Manual ``remember`` calls bypass both layers — the human user is the trust
# boundary there.

from __future__ import annotations

import re


# Minimum durable-memory body length after sanitisation. Shorter candidates
# are too generic to be worth remembering.
_MIN_BODY_CHARS = 10

# Maximum durable-memory body length — matches the 120-char target used in
# EXTRACTION_PROMPT.
_MAX_BODY_CHARS = 120


_INJECTION_PATTERNS = re.compile(
    r"(?:"
    # "ignore/disregard/forget [the/all/any/my/these] [previous/above/prior/...]
    # [instruction/rule/...]" — allow up to 3 filler words between the verb
    # and the target so "disregard the above rules" still catches.
    r"(?:ignore|disregard|forget) (?:\w+ ){0,3}?(?:previous|above|prior|earlier|preceding|prompt|instruction|directive|rule|memory)"
    r"|you are (?:now |a new |no longer )"
    r"|act as (?:if you|a|an) "
    r"|pretend (?:to be|you are) "
    # "from now on" as a sentence opener almost always introduces an override.
    # Catch it broadly — legitimate facts rarely need this phrase.
    r"|\bfrom now on\b"
    r"|\bnew (?:instruction|directive|rule|system prompt)\b"
    r"|\bsystem\s*:\s*(?:you|reply|respond|do)"
    r"|</?system\b"
    r"|</?assistant\b"
    r"|</?developer\b"
    r"|\[/?INST\]"
    r"|<\|im_(?:start|end)\|>"
    r")",
    re.IGNORECASE,
)

# URLs and HTML tags are content, not durable rules. Auto-extraction rejects
# them; manual `remember` still accepts them.
_URL_PATTERN = re.compile(r"\bhttps?://\S+", re.IGNORECASE)
_HTML_TAG_PATTERN = re.compile(r"<[a-zA-Z][a-zA-Z0-9_-]{0,20}(?:\s[^<>]*)?/?>")
# Fenced code blocks (backtick triplets) — never a durable rule.
_CODE_FENCE_PATTERN = re.compile(r"`{3,}")


def _looks_like_injection(body: str) -> bool:
    """Return True if ``body`` matches any known prompt-injection shape.

    Used only on the auto-extraction write path. Manual ``remember`` calls
    bypass this because the user is the trust boundary there.
    """
    if _INJECTION_PATTERNS.search(body):
        return True
    if _URL_PATTERN.search(body):
        return True
    if _HTML_TAG_PATTERN.search(body):
        return True
    if _CODE_FENCE_PATTERN.search(body):
        return True
    return False


def sanitize_body(body: str) -> str | None:
    """Structural validator for one directive body line.

    Returns the cleaned body on success, or ``None`` if the candidate must
    be dropped (too short/long, markdown heading, strikethrough, or
    injection-shaped).

    Rules:
      - one line only (anything after an embedded newline is discarded)
      - leading hyphens stripped (the LLM sometimes emits bullet lists)
      - length clamp ``_MIN_BODY_CHARS`` … ``_MAX_BODY_CHARS``
      - reject leading ``#`` (markdown heading) / ``~~`` (strikethrough) —
        those would smuggle structure into the always-loaded prompt.
      - reject injection-shaped content per ``_looks_like_injection``.

    Auto-extraction is the trust boundary, so this function biases toward
    over-rejection — a legitimate fact that gets filtered just gets
    another shot on the next extraction tick.
    """
    body = body.splitlines()[0].strip().lstrip("-").strip()
    if len(body) < _MIN_BODY_CHARS or len(body) > _MAX_BODY_CHARS:
        return None
    if body.lstrip().startswith(("#", "~~")):
        return None
    if _looks_like_injection(body):
        return None
    return body
