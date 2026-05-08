# Cross-tier memory verbs — the only thing that genuinely lives ABOVE
# the tier modules.
#
# Tier-aligned implementation lives in:
#   _index.py          — Tier 1 (MEMORY.md, always-loaded index)
#   _topics.py         — Tier 2 (curated topic files, on-demand)
#   _journal.py        — Tier 3 (chronological journal, on-demand)
#   _consolidation.py  — read-only dedup / size analysis
#
# Public API surface for memory: ``jyagent.memory`` package (see __init__.py).
# Importers should use ``from jyagent.memory import …`` rather than reaching
# in here. This module is kept solely as the home of the dispatcher verbs
# ``remember`` / ``forget`` / ``show_memory`` (used by the manage_memory tool)
# because they compose primitives across tiers and don't belong to any single
# tier module.

from __future__ import annotations

from ._index import (
    _validate_memory_entry,
    append_memory_md as _append_memory_md,
    forget_from_memory_md as _forget_from_memory_md,
    memory_index_size_warning as _memory_index_size_warning,
    read_memory_md as _read_memory_md,
)
from ._topics import (
    list_topics as _list_topics,
    read_topic as _read_topic,
    read_topic_meta as _read_topic_meta,
)
from ._journal import (
    list_journals as _list_journals,
    read_journal as _read_journal,
)


__all__ = ["remember", "forget", "show_memory"]


def remember(text: str, category: str = "", *, suppress_warning: bool = False) -> str:
    """Remember a durable fact or learning by appending to MEMORY.md.

    NOTE: This appends to the always-loaded index. Use only for data-independent
    rules / facts that prevent future mistakes. Ephemeral task notes ("today I
    finished X") belong in the journal tier — call ``append_journal`` instead.

    The return string concatenates a soft-cap warning when MEMORY.md is near
    its load limit. Set ``suppress_warning=True`` for programmatic callers /
    tests that parse the return value.
    """
    entry, cat = _validate_memory_entry(text, category)
    prefix = f"[{cat}] " if cat else "- "
    _append_memory_md(f"{prefix}{entry}")
    msg = f"Remembered: {entry[:100]}"
    if not suppress_warning:
        warning = _memory_index_size_warning()
        if warning:
            msg += "\n" + warning
    return msg


def forget(keyword: str) -> str:
    """Forget memories matching a keyword.

    Wraps ``_forget_from_memory_md`` with default safeties (≥6-char keyword,
    skips lines under Behavioral Rules / User Profile / User Preferences and
    ``#`` headings). Returns a human-readable preview of what was removed
    plus a count of protected lines that were preserved.
    """
    try:
        removed, preview, protected_skipped = _forget_from_memory_md(keyword)
    except ValueError as e:
        return f"Refused to forget: {e}"

    suffix = ""
    if protected_skipped:
        suffix = (
            f" ({protected_skipped} protected line(s) preserved — Behavioral "
            f"Rules / User Profile / User Preferences are never deleted by keyword)"
        )

    if removed == 0:
        if protected_skipped:
            return (
                f"No entries removed — all {protected_skipped} match(es) were in "
                "protected sections (Behavioral Rules / User Profile / User Preferences)"
            )
        return f"No entries found matching '{keyword}'"

    # Show up to 3 removed lines so the user can see what was lost.
    sample = "\n".join(f"  - {ln.strip()[:120]}" for ln in preview[:3])
    more = f"\n  …and {removed - 3} more" if removed > 3 else ""
    return (
        f"Removed {removed} entries matching '{keyword}'{suffix}:\n{sample}{more}"
    )


def show_memory() -> str:
    """Show all memory contents."""
    parts = []

    content = _read_memory_md()
    if content:
        line_count = len(content.splitlines())
        display = content[:2000]
        if len(content) > 2000:
            display += f"\n... ({line_count} total lines)"
        parts.append(f"🧠 MEMORY.md ({line_count} lines):\n{display}")

    topics = _list_topics()
    if topics:
        topic_lines = []
        for t in topics:
            tc = _read_topic(t)
            meta = _read_topic_meta(t)
            size = len(tc)
            lines = len(tc.split("\n"))
            updated = meta.get("updated", "")
            ts_suffix = f", updated {updated}" if updated else ""
            topic_lines.append(f"  📄 {t}.md ({lines} lines, {size} chars{ts_suffix})")
        parts.append(f"📂 TOPIC FILES ({len(topics)} topics):\n" + "\n".join(topic_lines))

    journals = _list_journals()
    if journals:
        j_lines = []
        for m in journals[:6]:  # show at most 6 most-recent months
            jc = _read_journal(m)
            j_lines.append(f"  📓 {m}.md ({len(jc.split(chr(10)))} lines, {len(jc)} chars)")
        more = "" if len(journals) <= 6 else f"\n  … and {len(journals) - 6} older months"
        parts.append(f"📓 JOURNAL ({len(journals)} months, on-demand only):\n" + "\n".join(j_lines) + more)

    if not parts:
        return "🧠 Memory is empty. I'll start learning about you as we interact!"

    warning = _memory_index_size_warning()
    suffix = ("\n\n" + warning) if warning else ""
    return "🧠 SELF-USE MEMORY\n" + "\n\n".join(parts) + suffix
