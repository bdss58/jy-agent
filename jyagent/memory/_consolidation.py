# Memory consolidation (read-only dedup analysis).
#
# Owns: the `consolidate_memory` analyzer used by the manage_memory tool to
# surface likely-duplicate entries, oversized lines, and dated entries that
# would belong in the journal tier instead of the always-loaded index.
# Pure heuristic / read-only — never deletes anything; outputs a report the
# agent (or user) can act on.
#
# Extracted from ``operations.py`` (2026-05-06).  Existing callers continue
# to import via ``jyagent.memory.operations``.

from __future__ import annotations

import re

from ._index import read_memory_md, memory_index_size_warning


# ─── Consolidate (dedup analysis, no LLM call) ───────────────────────────────

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
    content = read_memory_md()
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

    warning = memory_index_size_warning()
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


