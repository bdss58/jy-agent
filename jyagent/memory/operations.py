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
    _MEMORY_MD_LOCK,
    _compute_protected_indices,
    _validate_memory_entry,
    append_memory_md as _append_memory_md,
    forget_from_memory_md as _forget_from_memory_md,
    memory_index_size_warning as _memory_index_size_warning,
    read_memory_md as _read_memory_md,
    write_memory_md as _write_memory_md,
)
from ._topics import (
    list_topics as _list_topics,
    read_topic as _read_topic,
    read_topic_meta as _read_topic_meta,
)
from ._journal import (
    append_journal as _append_journal,
    list_journals as _list_journals,
    read_journal as _read_journal,
)
from ._extraction_directives import _MIN_UPDATE_KEYWORD_LEN


__all__ = ["remember", "forget", "show_memory", "replace_memory_entry"]


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


def replace_memory_entry(
    old_keyword: str,
    new_text: str,
    category: str = "",
) -> tuple[str, str]:
    """Replace MEMORY.md line(s) matching ``old_keyword`` with ``new_text``.

    This is the cross-tier UPDATE transaction that backs the LLM-driven
    ``UPDATE::`` directive emitted by auto-extraction:

      - Old line(s) are removed from MEMORY.md (Tier 1 stays lean — no
        ``~~strikethrough~~`` accretion that costs prompt-cache hits forever).
      - The old line(s) are archived to ``data/memory/journal/YYYY-MM.md``
        with category ``memory_revision`` so "what did this used to say?"
        questions can still be answered.
      - The new line is appended via ``remember`` so soft-cap warnings,
        category validation and prefix formatting stay in one place.

    The whole read-modify-write (read MEMORY.md → archive → delete → append)
    runs under ``_MEMORY_MD_LOCK``. Archive-to-journal happens BEFORE the
    MEMORY.md write so a crash between the two steps leaves the audit trail
    intact rather than an unrecoverable delete.

    Safety rails (unchanged from the extraction-owned implementation):
      - Keyword must be ≥ ``_MIN_UPDATE_KEYWORD_LEN`` chars.
      - Header / Behavioral-Rules / User-Profile / User-Preferences lines
        are skipped silently (via ``_compute_protected_indices``).

    Returns ``(status, message)``. ``status`` is ``"update"`` on success or
    ``"skip"`` if no eligible line matched. The contract mirrors the old
    ``extraction._replace_line`` (which now forwards here) so the extraction
    loop's result-counting logic keeps working.

    Lives in ``operations`` rather than any single tier module because it
    writes across Tier 1 (MEMORY.md) AND Tier 3 (journal). Previously lived
    in ``extraction.py`` which was the wrong home — that module is supposed
    to be an orchestrator, not the owner of a cross-tier transaction.
    """
    if len(old_keyword) < _MIN_UPDATE_KEYWORD_LEN:
        return (
            "skip",
            f"Error: UPDATE keyword too short ({len(old_keyword)} < {_MIN_UPDATE_KEYWORD_LEN})",
        )

    with _MEMORY_MD_LOCK:
        content = _read_memory_md()
        if not content:
            return ("skip", f"No entries matched '{old_keyword}' — MEMORY.md empty")

        # Identify protected line indices via the shared helper. Headers +
        # lines inside Behavioral Rules / User Profile / User Preferences are
        # never eligible for replacement.
        lines = content.split("\n")
        protected = _compute_protected_indices(lines)

        keyword_lower = old_keyword.lower()
        matched_indices: list[int] = []
        matched_lines: list[str] = []
        skipped_protected = 0
        for i, line in enumerate(lines):
            if keyword_lower not in line.lower():
                continue
            if i in protected:
                skipped_protected += 1
                continue
            matched_indices.append(i)
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
        _append_journal(
            f"Replaced via UPDATE directive (keyword='{old_keyword}'):\n"
            f"{archive_body}\n"
            f"  → [{category or 'note'}] {new_text}",
            category="memory_revision",
        )

        # Delete ONLY the eligible matched indices in-place. A previous
        # implementation used ``forget_from_memory_md(old_keyword)`` here,
        # which was a substring delete with no protection list — a keyword
        # appearing in both an eligible line and a protected line would have
        # wiped the protected line too (Codex review 2026-05-05, CRITICAL).
        kept = [ln for i, ln in enumerate(lines) if i not in set(matched_indices)]
        _write_memory_md("\n".join(kept))
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