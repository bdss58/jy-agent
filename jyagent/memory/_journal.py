# Tier 3 — Journal (chronological, append-only, never auto-loaded).
#
# Owns: monthly journal-file CRUD (`list_journals`, `read_journal`,
# `append_journal`) + the `_JOURNAL_LOCK` that serialises appends from
# multiple threads.
#
# Extracted from ``operations.py`` (2026-05-06) as part of the split that
# made each tier its own module.  Existing callers continue to import via
# ``jyagent.memory.operations`` (which re-exports this).

from __future__ import annotations

import os
import re
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

from .. import config as _cfg
from ._paths import ensure_dirs


_JOURNAL_LOCK = threading.Lock()

# Same timezone constant the topics module uses; kept module-local so each
# tier file is self-contained.
ASIA_SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


# ─── Journal tier (Tier 3: never auto-loaded) ────────────────────────────────
#
# Why a separate tier? Anthropic Claude Code docs and the consensus across
# Letta/Mem0/LangMem/Zep/A-MEM are unanimous: chronological "what I did today"
# notes do NOT belong in the always-loaded memory file. They cause:
#   - prompt-cache invalidation (mutating the cached prefix → ~12× cost penalty)
#   - context rot (NoLiMa: Claude 3.5 Sonnet effective length only ~4K tokens)
#   - "lost in the middle" attention degradation (Liu et al., TACL 2023)
#
# Journal entries live under data/memory/journal/YYYY-MM.md and are read on
# demand only (e.g. "what did we work on last Tuesday?"). They are NOT injected
# into the system prompt.

_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def _journal_path(month: str | None = None) -> str:
    """Return the journal file path for the given month (YYYY-MM).

    Validates ``month`` to prevent path traversal (``../../etc/passwd``) —
    low-severity in a single-user agent, but cheap to close the door on a
    future caller passing user-controlled strings.
    """
    if month is None:
        month = datetime.now(ASIA_SHANGHAI_TZ).strftime("%Y-%m")
    if not _MONTH_RE.match(month):
        raise ValueError(f"journal month must match YYYY-MM, got {month!r}")
    return os.path.join(_cfg.JOURNAL_DIR, f"{month}.md")


def list_journals() -> list[str]:
    """List all journal months (YYYY-MM strings), newest first."""
    ensure_dirs()
    months = []
    if os.path.exists(_cfg.JOURNAL_DIR):
        for f in os.listdir(_cfg.JOURNAL_DIR):
            if f.endswith(".md") and _MONTH_RE.match(f[:-3]):
                months.append(f[:-3])
    return sorted(months, reverse=True)


def read_journal(month: str | None = None) -> str:
    """Read a single journal month. Empty string if no entries."""
    ensure_dirs()
    path = _journal_path(month)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


# Journal categories live in markdown headers ("## YYYY-MM-DD HH:MM [cat]")
# so a category containing "[", "]", "\n", or other markdown control chars
# breaks the journal structure (and the `## ` header parser used by
# search.py::_split_sections). Validate against a permissive identifier
# regex — more relaxed than _ALLOWED_MEMORY_CATEGORIES because journal
# categories are freeform tags ("ship", "debug", "refactor", "session",
# "memory_revision", "codex_review", ...). Invalid input falls back to
# "note" silently — journal writes are best-effort and we don't want a
# malformed tag to lose the actual entry body.
_JOURNAL_CATEGORY_RE = re.compile(r"^[a-z][a-z0-9_]{0,30}$")


def _sanitize_journal_category(raw: str | None) -> str:
    """Normalize a journal category for safe use in a markdown header.

    Returns "note" for any input that fails validation. Lowercases the
    input and strips surrounding whitespace before checking the pattern.
    """
    if not raw:
        return "note"
    cat = raw.strip().lower()
    if not cat or not _JOURNAL_CATEGORY_RE.match(cat):
        return "note"
    return cat


def append_journal(text: str, category: str = "note") -> str:
    """Append a dated entry to the current month's journal file.

    Returns the journal-relative path that was written, so callers (and the
    agent) know where the note lives without having to guess.

    The ``category`` is sanitized (``_sanitize_journal_category``) before
    being interpolated into the markdown header — a malformed tag from a
    user-supplied facade call would otherwise break the journal section
    structure that ``search.py::_split_sections`` relies on.

    Concurrency: local threads serialize the header+body append under
    ``_JOURNAL_LOCK``. The header still uses O_CREAT|O_EXCL so a separate
    process racing on the first entry of a fresh month cannot produce a
    duplicate ``# Journal —`` heading.
    """
    ensure_dirs()
    safe_category = _sanitize_journal_category(category)
    now = datetime.now(ASIA_SHANGHAI_TZ)
    month = now.strftime("%Y-%m")
    timestamp = now.strftime("%Y-%m-%d %H:%M")
    path = _journal_path(month)

    with _JOURNAL_LOCK:
        # Race-free header install: whoever wins the O_EXCL create writes the
        # header; everyone else raises FileExistsError and skips straight to the
        # append step.
        header = (
            f"# Journal — {month}\n\n"
            "Tier 3 / append-only / NEVER auto-loaded. Chronological session "
            "notes live here so they don't pollute MEMORY.md (the always-loaded "
            "index) or topic files (curated knowledge).\n\n"
        )
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                os.write(fd, header.encode("utf-8"))
            finally:
                os.close(fd)
        except FileExistsError:
            pass  # another writer already installed the header — safe to append

        with open(path, "a", encoding="utf-8") as f:
            f.write(f"## {timestamp} [{safe_category}]\n{text.rstrip()}\n\n")

    return f"data/memory/journal/{month}.md"

