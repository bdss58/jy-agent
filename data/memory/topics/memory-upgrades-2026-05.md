---
created: 2026-05-17T19:03:44+08:00
status: active — curated reference for jyagent/memory/search.py
updated: 2026-05-17T19:03:44+08:00
---
# Memory Subsystem Upgrades — 2026-05-17

Curated companion to `jyagent/memory/search.py`. Captures the design
rationale, the *non*-features chosen deliberately, the synonym-curation
criteria, and the deferred roadmap. Read this BEFORE making invasive
changes to memory retrieval.

## What shipped today

| Commit | Change | Files | Tests added |
|---|---|---|---|
| `69f7998` | Recency boost on journal chunks | search.py +70 LOC | 4 |
| `b92ed73` | Curated query expansion (synonym aliases) | search.py +92 LOC | 7 |

Together: 11 new tests, 0 regressions across 1029-test repo suite.

## Design context — the deep-research review that motivated this

Today's deep-research pass on modern agent-memory mechanisms (Mem0, Letta,
Zep/Graphiti, HippoRAG 2, MemoryBank, Anthropic context-engineering, Claude
Code memory) concluded that jy-agent's three-tier markdown-on-disk design is
already field-current. The structural choices that the field has converged on
— always-loaded core / on-demand topics / append-only journal, Mem0-style
ADD/UPDATE/NOOP reconciliation on writes, 200-line / 25 KB cap, structured
compaction with file re-injection, prompt-cache stability as a first-class
constraint — are all already in `jyagent/memory/`.

The gaps were in retrieval and lifecycle. Today closes the two cheapest
gaps:

1. **No time decay on journal entries** — fixed by recency boost.
2. **No paraphrase recall** — fixed by curated query expansion.

## Recency boost — algorithm summary

```
boost = RECENCY_FLOOR + (1 - RECENCY_FLOOR) * exp(-age_days / HALF_LIFE_DAYS)
        where RECENCY_FLOOR = 0.5, HALF_LIFE_DAYS = 90
```

- Applied to journal chunks only. Topic files are curated knowledge, NOT
  events — they get `multiplier = 1.0` (no decay).
- Floor of 0.5 keeps old unique-keyword matches surfacing — we down-rank,
  we don't bury.
- Date extraction prefers section-header prefix `## YYYY-MM-DD ...`,
  falls back to the filename month for undated/preamble sections.
- Toggleable via `search_memory(..., recency_boost=False)` (off in
  reconciliation / tests where pure-BM25 is wanted).

## Query expansion — curation criteria

**Build path:** synonym groups defined in source as natural-language tokens,
canonicalised through the same `_stem` used by `_tokenize` at import time.
Keys in `_EXPANSION` are guaranteed to match what `_tokenize` emits.

**The KEEP list (9 groups, all corpus-backed surface-form aliases):**

```
postgres | postgresql | pg
kubernetes | k8s
boolean | bool
typescript | ts
javascript | js
repository | repo
environment | env
python | py
markdown | md
```

**The DROP list (each rejected by Codex pre-review, then re-confirmed
against corpus):**

| Dropped group | Why dropped |
|---|---|
| `configuration / configure / config` | Verb-vs-noun ≠ surface-form alias |
| `application / app` | "app" too broad (web app, Application class, etc.) |
| `command / cmd` | Context-dependent (shell cmd vs `cmd.exe` vs Tk command) |
| `documentation / docs / doc` | "doc" overloaded (Python `__doc__`, .doc files, doc-string) |
| `directory / dir / folder` | "dir" is the shell builtin in many contexts |
| `database / db` | "db" overloaded (decibel, .tsx db ref, etc.) |
| `tcc / accessibility` | **DIFFERENT TCC permission buckets per MEMORY.md gotcha** — conflation would corrupt a load-bearing technical rule |

**Future additions** must clear all three bars:
1. Both terms appear in the actual corpus (`grep -ciw <term> data/memory/...`).
2. They are TRUE surface-form aliases — same referent, different spelling.
3. NOT meaning-expanding (verb↔noun, broad↔narrow, etc.).
4. Pinned in `test_query_expansion_dropped_groups_do_not_expand` if rejected.

## Deliberate non-features (do NOT add without a measured failure case)

1. **Vector / embedding retrieval.** Rejected explicitly 2026-05-17 by the
   user as "not for me" (single-user laptop). `sentence-transformers` ≈
   500 MB; `fastembed` + ONNX weights ≈ 200 MB; corpus is ~few KB of text.
   Letta's LoCoMo result: 74% with grep/BM25 over a text filesystem.
   *Premise re-test before reconsidering:* would I be able to demonstrate a
   concrete query that BM25 + recency + expansion misses on the real
   corpus? If not, the threat model still fails.
2. **Corpus-learned synonym expansion.** Corpus is too small for reliable
   co-occurrence stats. Curated stays.
3. **WordNet / NLTK / spaCy** for morphological expansion. Adding a 30 MB
   lexicon to handle a 30-line problem is the failure mode MEMORY.md
   warns against.
4. **Phrase / n-gram expansion.** Tokenisation stays single-token.
5. **Index caching by mtime.** Bodies are small; the index is rebuilt on
   every call. Profile before optimising.

## Deferred roadmap (lower priority than non-memory work)

| # | Item | Effort | Why deferred |
|---|---|---|---|
| 1 | **Sleep-time consolidation** — background promotion of frequently-recalled or thematically-clustered journal entries → curated topic files (Letta sleep-time-agent pattern) | ~half day | Best-in-class next feature, but wants a fresh attention chunk. |
| 2 | **Bi-temporal `replace_memory_entry`** — append `[superseded YYYY-MM-DD]` marker instead of overwriting | ~2 hours | Marginal: the journal-archive workflow already half-does this. |
| 3 | **Hybrid semantic retrieval** — only if (a) a concrete BM25+expansion miss appears in practice AND (b) we accept the install footprint | ≥1 day | See non-features §1. |

## Useful queries to validate retrieval after future changes

These are the smoke tests that proved both upgrades on the live corpus
(2026-05-17). Re-run after any change to `search.py`:

```python
from jyagent.memory.search import search_memory

# Recency boost — newer entries should rank above older on tied text
search_memory("Codex review", top_k=3)
search_memory("agent.py refactor", top_k=3)

# Query expansion — aliased queries should hit ≥1 doc that uses the other form
search_memory("pg", top_k=3)         # finds journal entries that say "postgres"
search_memory("md tier", top_k=3)    # md→markdown lifts topic-file hits

# Topic files must NOT be decayed (must outrank old journals for unique keys)
search_memory("TCC accessibility", top_k=3)  # topic ahead of journals
```

## Cross-references

- `docs/design/2026-05-17-memory-query-expansion.md` — design doc + Codex
  pre-review verdict + the original (pre-pruning) synonym list, preserved
  as audit trail.
- `data/memory/journal/2026-05.md` — chronological ship notes for both
  commits, plus the deep-research delivery note.
- `jyagent/memory/search.py` — the actual implementation (look for
  `_SYNONYM_GROUPS`, `_recency_multiplier`, `expand_query_tokens`).
