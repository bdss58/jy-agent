# Memory System Design — Three-Tier Architecture

**Index pointer in MEMORY.md.** Background and rationale here so the design
doesn't drift back to the bad pattern.

## Tier model

| Tier | Path | Loaded? | Purpose | Hard cap |
|---|---|---|---|---|
| 1 | `data/memory/MEMORY.md` | ALWAYS (every turn) | Durable rules / facts that prevent future mistakes | 200 lines / 25 KB |
| 2 | `data/memory/topics/<name>.md` | On demand (read_file) | Curated extended knowledge — architecture, library quirks, ongoing project state | none |
| 3 | `data/memory/journal/YYYY-MM.md` | NEVER auto-loaded | Append-only chronological notes — "what I did today", debug session logs | none |

## What goes where — the routing rule

Apply Anthropic's litmus test:
> *"Would removing this cause Claude to make mistakes? If not, cut it."*

| Information | Tier |
|---|---|
| Behavioral rule, gotcha, environment constant | 1 (`remember`) |
| Architecture detail, library bug, project plan | 2 (`topic` write) |
| "Today I shipped X", session debugging trace, commit-style narrative | 3 (`journal`) |
| Anything git already records | None — drop it |

## Why this design (consensus across the field)

- **Anthropic Claude Code docs** explicitly separate `CLAUDE.md` (instructions
  & rules) from `MEMORY.md` (auto-managed learnings) with a hard 200-line /
  25-KB load cap and a topic-files split for detail.
- **Letta** memory blocks default to **2000 chars** per always-loaded block.
- **Mem0 / LangMem / A-MEM** all use **extract → reconcile → consolidate**;
  none blind-append.
- **LangMem** exact rule: "Profiles UPDATE; collections may APPEND with
  consolidation".
- **Zep** assembles per-turn context from a temporal graph rather than
  persisting an always-loaded blob.

## Why bloating Tier 1 hurts (empirical)

- **Prompt-cache invalidation:** mutating cached prefix → write 1.25× vs hit
  0.10× → ~12× per-token penalty. Anthropic's own pattern is to inject
  dynamic content as a `<system-reminder>` on the tail user message — never
  edit the system prompt. (We document this rule separately in MEMORY.md.)
- **Context rot (Chroma 2025, 18 models):** even one distractor measurably
  hurts; needles that *blend* with surrounding text become unretrievable.
  Fifty dated `[note]` bullets blend.
- **NoLiMa (ICML 2025):** Claude 3.5 Sonnet's *effective* context length is
  only ~4 K tokens. Pushing past it degrades reasoning, not just retrieval.
- **Lost in the Middle (Liu, TACL 2023):** U-shaped accuracy. Middle of long
  context is a black hole.
- **Anthropic "attention budget":** *"Every new token depletes the attention
  budget. n² pairwise relationships … gets stretched thin."*

## API surface (`manage_memory` tool)

| Action | Tier | Purpose |
|---|---|---|
| `remember` | 1 | Append a 1-line durable rule. Returns size warning if approaching cap. |
| `forget` | 1 | Remove lines matching a keyword (destructive). |
| `search` | 2+3 | BM25 over topic + journal bodies, section-chunked. Returns top-5 hits. Use this BEFORE reading whole topic files. |
| `topic` (`list` / `read:` / `read:<name>#<section>` / `sections:` / `write:` / `delete:`) | 2 | Curated knowledge files; auto-indexed in MEMORY.md. Section subaction reads one `##`/`###` block (with nested sub-sections). Topic names validated against `^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$`. |
| `journal` | 3 | Append a dated session note to `data/memory/journal/YYYY-MM.md`. |
| `note` | 3 | **Deprecated alias** for `journal`. Old call sites still work but are routed to Tier 3 (was Tier 1 — that was the bug). |
| `consolidate` | 1 | Read-only analyzer: dedup candidates, oversized lines, dated notes that belong in journal. |
| `goal` | 1 | Append/complete a `[goal]` line in MEMORY.md. |
| `show` | all | Display memory + size warnings. |

## Soft warnings

`memory_index_size_warning()` triggers when MEMORY.md crosses:
- `MEMORY_INDEX_WARN_LINES` — default 150 (cap 200)
- `MEMORY_INDEX_WARN_BYTES` — default 18 KB (cap 25 KB)

The warning is included in:
- `manage_memory(action='remember', ...)` return value
- `manage_memory(action='show')` footer
- `manage_memory(action='consolidate')` report

## Migration history

- **2026-04-30** — **Removed `supersede` action** (this commit).
  After 6+ months of single-digit usage, the public `supersede` action was
  removed. Its tier placement was wrong: keeping ``~~struck~~`` audit lines
  in the always-loaded MEMORY.md doubled the line's footprint, accreted
  forever, and invalidated the prompt cache on every revision. The
  LLM-driven UPDATE directive in `extraction.py` now calls
  `_replace_line()` instead, which: (a) archives the old line(s) to the
  current month's journal under `[memory_revision]`, (b) calls
  `forget_from_memory_md` to remove from Tier 1, (c) calls `remember` to
  append the new line. Same safety rails (≥6-char keyword, protected
  sections, RLock-guarded RMW) but audit history is now where it belongs
  (Tier 3, on-demand, never auto-loaded). Recommended pattern for manual
  revisions: `journal` (record the change) → `forget` (drop the old) →
  `remember` (add the new).
- **2026-04-25** — **Five upgrades + post-review hardening**.
  Five new capabilities (`search_memory` / reconciliation in `extraction.py` /
  non-destructive `supersede` / reflection pass at compaction /
  section-level topic reads) were added in `feat/memory-upgrades`.
  Code review surfaced 3 CRITICAL bugs and 3 HIGH security issues; all
  fixed before merge.
  - **C1**: `append_memory_md` now heals a missing trailing newline so
    auto-extracted entries don't get glued onto the previous last line of
    a hand-edited MEMORY.md.
  - **C2**: All MEMORY.md read-modify-write paths now hold a reentrant
    `_MEMORY_MD_LOCK`. Closes the race between the background extraction
    thread and synchronous `manage_memory` calls.
  - **C3**: `_apply_directive` now distinguishes a real UPDATE from a
    no-match replacement (returns `("skip", msg)` instead of counting it
    as success). Hallucinated UPDATEs no longer crowd out real ADDs
    against the per-turn cap. *(Originally enforced inside `supersede()`;
    after the 2026-04-30 removal the same contract is preserved by
    `_replace_line()` in `extraction.py`.)*
  - **H1**: Introduced `_topic_path(name)` with strict regex
    `^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$` + `realpath` containment check.
    Closes path-traversal in `read_topic` / `write_topic` /
    `delete_topic` / `read_topic_section`.
  - **H2**: `supersede()` enforced a 6-char minimum keyword and skipped
    markdown headers + lines inside Behavioral Rules / User Preferences /
    User Profile sections, so the LLM extraction reconciler could not be
    coaxed into striking through hard agent rules. *(After the 2026-04-30
    removal these rails were lifted into `_replace_line()` unchanged.)*
  - **H3**: `_apply_directive` now sanitizes bodies to one line, ≤120
    chars, no leading `#` or `~~`. Closes the residual prompt-injection
    surface in the reconciliation pipeline.
  - 113 tests pass (28 feature tests + 11 hardening tests + the
    pre-existing 49 + 28 + 25 across phase1/tiers/compaction).
- **2026-04-24** — Tier system introduced. Migrated the
  `[note] 2026-04-18 Agent-loop upgrade` 1500-char dump into
  `topics/agent-loop-changelog.md`. Trimmed `[gotcha] Skill LLM router`,
  `[workflow] GFW fallback` and `[tip] run_background hardening` to 1-line
  index entries pointing at topic files. Added `journal/` tier; redirected
  `action='note'`. Net MEMORY.md size: ~9 KB → ~3 KB.
- Tests: `tests/test_memory_tiers.py` (29 tests) plus the original
  `test_memory_phase1.py` (28 tests) plus `test_memory_upgrades.py`
  (40 tests covering BM25 search, reconciliation, UPDATE replacement, reflection,
  section reads, and the post-review hardening).
