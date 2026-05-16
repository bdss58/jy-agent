# Cross-tier memory verbs — the only thing that genuinely lives ABOVE
# the tier modules.
#
# Tier-aligned implementation lives in:
#   _index.py          — Tier 1 (MEMORY.md, always-loaded index)
#   _topics.py         — Tier 2 (curated topic files, on-demand)
#   _journal.py        — Tier 3 (chronological journal, on-demand)
#
# Public API surface for memory: ``jyagent.memory`` package (see __init__.py).
# Importers should use ``from jyagent.memory import …`` rather than reaching
# in here. This module is kept solely as the home of the dispatcher verbs
# ``remember`` / ``forget`` / ``show_memory`` / ``replace_memory_entry`` /
# ``consolidate_memory`` (used by the manage_memory tool) because they
# compose primitives across tiers and don't belong to any single tier module.

from __future__ import annotations

import re

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


__all__ = [
    "remember",
    "forget",
    "show_memory",
    "replace_memory_entry",
    "consolidate_memory",
]


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


# ─── Consolidate (read-only dedup analysis, no LLM call) ─────────────────────
#
# Folded back into operations.py 2026-05-17 (was extracted to a private
# ``_consolidation.py`` on 2026-05-06, but the split had a single caller —
# the package facade — and the operation is the same shape as the other
# cross-tier verbs here. See data/memory/topics/simplification-audit-2026-05.md
# verdict 3.1 for the rationale.

def consolidate_memory() -> str:
    """Analyze MEMORY.md for likely-duplicate entries and report groupings.

    Pure heuristic / read-only — does NOT delete anything. The agent can use
    the report to decide which entries to merge or move out manually.

    Heuristics:
      1. Group lines by category tag (`[gotcha]`, `[tip]`, `[note]`, etc.)
      2. Flag categories with >5 entries (consolidation candidates)
      3. Within each large category, flag pairs whose significant-token sets
         overlap by >=4 (likely related topics). Token extraction covers
         ASCII identifiers, version numbers, and CJK words so the heuristic
         works on bilingual memory.
      4. Always flag any line longer than 400 chars (belongs in a topic file)
      5. Always flag any categorized line with a YYYY-MM-DD date in its body
         (likely belongs in the journal tier, not the index)
    """
    content = _read_memory_md()
    if not content:
        return "MEMORY.md is empty — nothing to consolidate."

    lines = content.splitlines()
    cat_re = re.compile(r"^\[(?P<cat>[a-z_]+)\]\s*(?P<body>.*)$", re.IGNORECASE)

    by_cat: dict[str, list[tuple[int, str]]] = {}
    long_lines: list[tuple[int, str]] = []
    journal_candidates: list[tuple[int, str, str]] = []  # (line_no, category, preview)
    date_re = re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b")

    for idx, line in enumerate(lines, start=1):
        m = cat_re.match(line.strip())
        if not m:
            continue
        cat = m.group("cat").lower()
        body = m.group("body")
        by_cat.setdefault(cat, []).append((idx, body))
        if len(line) > 400:
            long_lines.append((idx, body[:80]))
        # Dated entries in ANY category are journal candidates — not only [note].
        # A dated [gotcha] or [tip] is equally an activity-log smell.
        if date_re.search(body):
            journal_candidates.append((idx, cat, body[:80]))

    report = ["📊 MEMORY.md consolidation analysis", ""]
    issues_found = False

    # 1. Per-category counts
    report.append("## Categories")
    for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
        if len(by_cat[cat]) > 5:
            report.append(f"  [{cat}]: {len(by_cat[cat])} entries  ⚠️ consider consolidating")
            issues_found = True
        else:
            report.append(f"  [{cat}]: {len(by_cat[cat])} entries")

    # 2. Lines that are too long for the always-loaded tier
    if long_lines:
        issues_found = True
        report.append("")
        report.append("## Lines > 400 chars (move to topics/<name>.md)")
        for ln, preview in long_lines:
            report.append(f"  L{ln}: {preview}…")

    # 3. Dated entries in any category → belong in journal
    if journal_candidates:
        issues_found = True
        report.append("")
        report.append("## Dated entries (move to journal/YYYY-MM.md)")
        for ln, cat, preview in journal_candidates:
            report.append(f"  L{ln} [{cat}]: {preview}")

    # 4. Cheap overlap dedup hints inside large categories.
    # Token extraction covers three shapes:
    #   • ASCII identifiers of 3+ letters with ._- internal punctuation so
    #     version numbers and dotted paths stay intact ("Python3.14",
    #     "jyagent.tools.facades").
    #   • CJK character bigrams (Chinese/Japanese/Korean). CJK has no
    #     inter-word whitespace, so "用户偏好中文" as one "run" is useless for
    #     dedup — instead we emit "用户", "户偏", "偏好", "好中", "中文" and
    #     let the set-overlap heuristic do its job. Stop-words are ASCII-only
    #     (no curated CJK stop list; bigram noise is low in practice).
    ascii_tok = re.compile(r"[A-Za-z][\w.\-]{2,}")
    cjk_run = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]{2,}")

    def _sig_words(s: str) -> set[str]:
        out = {
            tok.lower()
            for tok in ascii_tok.findall(s)
            if tok.lower() not in _STOP_WORDS
        }
        for run in cjk_run.findall(s):
            for i in range(len(run) - 1):
                out.add(run[i : i + 2])
        return out

    overlap_hints: list[str] = []
    for cat, entries in by_cat.items():
        if len(entries) < 4:
            continue
        word_sets = [(ln, body, _sig_words(body)) for ln, body in entries]
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                ln_a, body_a, ws_a = word_sets[i]
                ln_b, body_b, ws_b = word_sets[j]
                shared = ws_a & ws_b
                if len(shared) >= 4:
                    overlap_hints.append(
                        f"  [{cat}] L{ln_a} ↔ L{ln_b} share: "
                        f"{', '.join(sorted(shared)[:6])}"
                    )

    if overlap_hints:
        issues_found = True
        report.append("")
        report.append("## Possible duplicate pairs (>=4 shared content words)")
        report.extend(overlap_hints[:20])
        if len(overlap_hints) > 20:
            report.append(f"  … and {len(overlap_hints) - 20} more pairs")

    warning = _memory_index_size_warning()
    if warning:
        issues_found = True
        report.append("")
        report.append(warning)

    if not issues_found:
        report.append("")
        report.append("✅ No obvious consolidation candidates found.")

    return "\n".join(report)


_STOP_WORDS = {
    "with", "from", "that", "this", "have", "when", "then", "than",
    "they", "them", "their", "there", "where", "which", "what",
    "your", "yours", "been", "being", "would", "could", "should", "must",
    "will", "shall", "does", "doesn", "didn", "isn", "wasn", "aren", "weren",
    "about", "above", "below", "between", "after", "before", "during",
    "while", "until", "because", "since", "though", "although",
    "also", "only", "even", "just", "still", "very", "more", "most", "less",
    "least", "other", "another", "some", "many", "each", "every", "both",
    "either", "neither", "such", "same", "different",
    "into", "onto", "over", "under", "through", "across", "around",
    "always", "never", "often", "sometimes", "usually",
    "make", "made", "makes", "making", "take", "taken", "takes", "taking",
    "give", "gave", "given", "gives", "giving", "want", "wants", "need",
    "needs", "see", "saw", "seen", "look", "looks", "looking",
    "good", "better", "best", "bad", "worse", "worst",
    "true", "false", "none", "null",
}
