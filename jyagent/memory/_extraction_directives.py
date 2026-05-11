# Extraction directive grammar — pure regex parsing for the lines the
# extraction LLM emits. No I/O, no MEMORY.md mutation, no security
# filtering (that lives in ``_extraction_security``). Just regex match
# definitions and ``parse_directive`` which returns a structured tuple.
#
# Directive shapes (see EXTRACTION_PROMPT in extraction.py):
#
#   ADD::[category] <new fact>
#   UPDATE::<old_keyword>::[category] <new fact>
#   NOOP::<reason>
#
# Each category is optional; missing category becomes "" (the renderer
# substitutes "note" later).

from __future__ import annotations

import re
from typing import Optional


# Minimum unique-substring length for an UPDATE directive's <old_keyword>.
# Short substrings (e.g. "k8s") would steamroll many unrelated lines on
# common tokens. Mirror the old ``supersede`` floor.
_MIN_UPDATE_KEYWORD_LEN = 6


_ADD_RE = re.compile(
    r"^\s*ADD::\s*(?:\[(?P<cat>[a-z_]+)\]\s*)?(?P<body>.+)$",
    re.IGNORECASE,
)
_UPDATE_RE = re.compile(
    r"^\s*UPDATE::\s*(?P<old>.+?)::\s*(?:\[(?P<cat>[a-z_]+)\]\s*)?(?P<body>.+)$",
    re.IGNORECASE,
)
_NOOP_RE = re.compile(r"^\s*NOOP::", re.IGNORECASE)


# Parsed directive — returned as a tagged tuple for the orchestrator to
# dispatch on. Tuples (not dataclasses) so the contract stays trivial
# and pattern-matchable; this module is only re-imported by the
# extraction orchestrator and a few tests.
#
#   ("add",    body, category)               — ADD directive
#   ("update", old_keyword, body, category)  — UPDATE directive
#   ("noop",)                                — NOOP directive (or unparseable)
ParsedDirective = tuple


def parse_directive(line: str) -> Optional[ParsedDirective]:
    """Parse a single directive line into a tagged tuple.

    Returns ``None`` for blank lines or lines that don't match any directive
    shape (the orchestrator drops these — the LLM sometimes emits prose
    commentary alongside directives, and we prefer resilience to strict
    parsing here).

    NOOP lines are recognised explicitly (returned as ``("noop",)``) so the
    caller can distinguish "LLM said don't do anything" from "garbage line
    we should ignore".

    Body and old_keyword are returned RAW — sanitisation lives in
    ``_extraction_security.sanitize_body``. This separation keeps the
    grammar layer pure (no security policy) and the security layer pure
    (no grammar).
    """
    stripped = line.strip()
    if not stripped:
        return None
    if _NOOP_RE.match(stripped):
        return ("noop",)

    m = _UPDATE_RE.match(stripped)
    if m:
        old_keyword = m.group("old").strip().strip("`'\"")
        category = (m.group("cat") or "").strip().lower()
        body = m.group("body")
        return ("update", old_keyword, body, category)

    m = _ADD_RE.match(stripped)
    if m:
        category = (m.group("cat") or "").strip().lower()
        body = m.group("body")
        return ("add", body, category)

    return None
