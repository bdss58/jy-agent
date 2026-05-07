# MEMORY.md and topic file operations — package facade.
#
# This module is the stable public entrypoint for the manage_memory tool
# and for legacy callers (tests, extraction.py, tools/facades.py).  The
# underlying implementation was split (2026-05-06) along the documented
# 3-tier memory design:
#
#   ``_index.py``         — Tier 1 (MEMORY.md): the always-loaded index
#   ``_topics.py``        — Tier 2 (curated topic files, on-demand)
#   ``_journal.py``       — Tier 3 (chronological, never auto-loaded)
#   ``_consolidation.py`` — read-only dedup / size analysis report
#
# This file:
#   1. Re-exports the public API from each tier so existing
#      ``from jyagent.memory.operations import …`` callers keep working
#      with zero changes.
#   2. Holds the high-level ``remember`` / ``forget`` / ``show_memory``
#      verbs that the manage_memory tool dispatches to — these compose
#      across tiers and naturally live at the facade.
#
# Internal call sites (e.g. ``extraction.py``) also import
# ``_MEMORY_MD_LOCK``, ``_PROTECTED_SECTION_HEADERS`` and
# ``_compute_protected_indices`` from here; those re-exports are
# preserved below.

from __future__ import annotations

# ─── Re-exports (back-compat) ────────────────────────────────────────────────
# Import order matters: ``_index`` defines shared lock + dirs; the other tiers
# depend on it.

from ._index import (  # noqa: F401
    # Public API
    ensure_dirs,
    read_memory_md,
    read_memory_index,
    write_memory_md,
    append_memory_md,
    forget_from_memory_md,
    memory_index_size_warning,
    # Constants kept on the facade for stability of test imports.
    MAX_MEMORY_INDEX_LINES,
    MAX_MEMORY_INDEX_BYTES,
    MAX_DURABLE_MEMORY_TEXT_CHARS,
    MIN_FORGET_KEYWORD_LEN,
    # Privates re-exported for in-tree callers (extraction.py).
    _ALLOWED_MEMORY_CATEGORIES,
    _MEMORY_CATEGORY_RE,
    _MEMORY_MD_LOCK,
    _PROTECTED_SECTION_HEADERS,
    _compute_protected_indices,
    _validate_memory_entry,
)

from ._topics import (  # noqa: F401
    list_topics,
    read_topic,
    read_topic_body,
    read_topic_meta,
    read_topic_section,
    list_topic_sections,
    write_topic,
    delete_topic,
    # Privates re-exported for tests / advanced callers.
    _topic_path,
    _parse_frontmatter,
    _build_frontmatter,
    _now_iso,
    _sanitize_topic_description,
    _extract_topic_description,
    _upsert_topic_index_entry,
    _add_topic_index_entry,
    _remove_topic_index_entry,
    # Module-level constants — re-exported so any caller doing
    # ``from jyagent.memory.operations import _FRONTMATTER_SEP`` (etc.)
    # keeps working with zero source changes.
    _FRONTMATTER_SEP,
    ASIA_SHANGHAI_TZ,
    _VALID_TOPIC_NAME_RE,
    _SECTION_HEADER_RE,
    _TOPIC_INDEX_HEADING,
    _MAX_TOPIC_DESC_CHARS,
    _CONTROL_CHARS_RE,
)

from ._journal import (  # noqa: F401
    list_journals,
    read_journal,
    append_journal,
    # Privates re-exported for tests.
    _journal_path,
    _sanitize_journal_category,
    _JOURNAL_LOCK,
    _MONTH_RE,
    _JOURNAL_CATEGORY_RE,
)

from ._consolidation import (  # noqa: F401
    consolidate_memory,
    _STOP_WORDS,
)


# ─── High-level operations (used by manage_memory tool) ──────────────────────
#
# These compose primitives across tiers and live here at the facade because
# none of the tier modules is the natural owner.

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
    append_memory_md(f"{prefix}{entry}")
    msg = f"Remembered: {entry[:100]}"
    if not suppress_warning:
        warning = memory_index_size_warning()
        if warning:
            msg += "\n" + warning
    return msg


def forget(keyword: str) -> str:
    """Forget memories matching a keyword.

    Wraps ``forget_from_memory_md`` with default safeties (≥6-char keyword,
    skips lines under Behavioral Rules / User Profile / User Preferences and
    ``#`` headings). Returns a human-readable preview of what was removed
    plus a count of protected lines that were preserved.
    """
    try:
        removed, preview, protected_skipped = forget_from_memory_md(keyword)
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

    content = read_memory_md()
    if content:
        line_count = len(content.splitlines())
        display = content[:2000]
        if len(content) > 2000:
            display += f"\n... ({line_count} total lines)"
        parts.append(f"🧠 MEMORY.md ({line_count} lines):\n{display}")

    topics = list_topics()
    if topics:
        topic_lines = []
        for t in topics:
            tc = read_topic(t)
            meta = read_topic_meta(t)
            size = len(tc)
            lines = len(tc.split("\n"))
            updated = meta.get("updated", "")
            ts_suffix = f", updated {updated}" if updated else ""
            topic_lines.append(f"  📄 {t}.md ({lines} lines, {size} chars{ts_suffix})")
        parts.append(f"📂 TOPIC FILES ({len(topics)} topics):\n" + "\n".join(topic_lines))

    journals = list_journals()
    if journals:
        j_lines = []
        for m in journals[:6]:  # show at most 6 most-recent months
            jc = read_journal(m)
            j_lines.append(f"  📓 {m}.md ({len(jc.split(chr(10)))} lines, {len(jc)} chars)")
        more = "" if len(journals) <= 6 else f"\n  … and {len(journals) - 6} older months"
        parts.append(f"📓 JOURNAL ({len(journals)} months, on-demand only):\n" + "\n".join(j_lines) + more)

    if not parts:
        return "🧠 Memory is empty. I'll start learning about you as we interact!"

    warning = memory_index_size_warning()
    suffix = ("\n\n" + warning) if warning else ""
    return "🧠 SELF-USE MEMORY\n" + "\n\n".join(parts) + suffix
