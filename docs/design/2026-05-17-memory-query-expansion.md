# Design — Query Expansion for `search_memory`

**Author:** jy-agent · **Date:** 2026-05-17 · **Status:** for Codex pre-review
**Scope:** ~50 LOC + curated synonym map + tests. Closes the BM25 paraphrase
gap without adding embedding deps.

## Problem statement

Current `search_memory` (jyagent/memory/search.py) uses Okapi BM25 with a
small stop-word set and a cheap plural stemmer. It fails on **paraphrased
queries** that use a different surface form for the same concept:

- Query "Postgres booleans" misses the journal entry phrased as
  "SQLAlchemy `server_default=sa.text('0')` Boolean column on Postgres".
  (`postgres` vs no occurrence; `booleans` vs `Boolean`.)
- Query "k8s pod restart" misses an entry phrased "kubernetes CrashLoopBackOff".
- Query "config file" misses "configuration".

The user has explicitly **rejected** the embedding/RRF route for this
single-user personal-laptop use case (sentence-transformers ≈ 500 MB; fastembed
+ ONNX weights ≈ 200 MB; Letta's LoCoMo result showed BM25 over a text
filesystem hits ~74%; my own search.py header cites this).

## Design

Add a **query-time** expansion step. Index stays clean; only the query
tokens are expanded with a small, curated, bidirectional alias map.

### Algorithm

```python
def expand_query_tokens(tokens: list[str]) -> list[str]:
    out = list(tokens)
    seen = set(tokens)
    for tok in tokens:
        for alias in _EXPANSION.get(tok, ()):
            if alias not in seen:
                out.append(alias)
                seen.add(alias)
    return out
```

`_EXPANSION` is built once at import time from a curated `_SYNONYM_GROUPS`
list of equivalence classes:

```python
_SYNONYM_GROUPS: list[set[str]] = [
    {"postgres", "postgresql", "pg"},
    {"kubernetes", "k8s"},
    {"boolean", "bool"},
    {"typescript", "ts"},
    {"javascript", "js"},
    {"repository", "repo"},
    {"documentation", "docs", "doc"},
    {"configuration", "config", "configure"},
    {"environment", "env"},
    {"python", "py"},
    {"markdown", "md"},
    {"application", "app"},
    {"directory", "dir", "folder"},
    {"database", "db"},
    {"command", "cmd"},
    # … target ~20–30 entries, NOT a generic dictionary
]
```

`_EXPANSION: dict[str, frozenset[str]]` is built once:

```python
_EXPANSION = {tok: frozenset(group - {tok})
              for group in _SYNONYM_GROUPS for tok in group}
```

### Integration

Single line change in `search_memory`:

```python
q_tokens = _tokenize(query)
q_tokens = expand_query_tokens(q_tokens)          # ← new
```

Behavior contract:

- **Bidirectional**: `pg` expands to `postgres`+`postgresql`, and `postgres`
  expands to `pg`+`postgresql`. Symmetric.
- **Idempotent**: expanding a query that already contains all aliases is a
  no-op (no duplicates added).
- **Conservative**: never expand stop-words (they're filtered before
  expansion already, so this is automatic).
- **Curated only**: NO automatic morphological expansion beyond the existing
  `_stem` plural stripper. We will NOT learn synonyms from the corpus.
  Reason: the corpus is too small to derive reliable co-occurrence stats,
  and accidental conflations (e.g. `automation` ↔ `accessibility`) would
  silently degrade precision on technical queries.
- **Opt-out**: new kwarg `expand_query: bool = True` so tests / power users
  can disable it.

### What this is NOT

1. NOT domain-specific aliasing for the user's vocabulary. The map covers
   only generic CS abbreviations that are unambiguous in any technical
   context. Domain terms (e.g. `OpenClaw`, `cashclaw`, `hooksToken`)
   already match literally and don't need synonyms.
2. NOT a phrase/n-gram expander. Single-token equivalence only.
3. NOT WordNet / NLTK. Adding a 30 MB lexicon to handle a 30-line problem
   is exactly the kind of "import everything" failure mode MEMORY.md warns
   against.

## Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Over-expansion lowers precision (false positives) | Medium | Curated map ≤30 groups, all genuine equivalences. Stop-words excluded. |
| `bool` matches Boolean-as-language-feature when user wants Python `bool`-the-type | Low | BM25 IDF naturally favors the more-specific term when both appear. |
| Map drifts out of sync with corpus vocabulary | Low | Easy to audit (single file, single list). Add new groups when a missed query is observed in practice, NOT preemptively. |
| Stem interaction (`configuration` stems to `configuration`, `config` stays `config`) | Low — but worth verifying | Test: build the map AFTER `_stem` so groups match what `_tokenize` actually emits. |
| Performance regression | Negligible | Expansion is O(|query tokens|) hashtable lookups. Query tokens are ≤20 in practice. |

## Testing

Add tests in `tests/test_memory_upgrades.py` (alongside the recency-boost
tests added today):

1. `test_query_expansion_postgres_pg_bidirectional` — write a topic whose
   only relevant token is `postgres`; query "pg replication" finds it.
   Reverse direction too.
2. `test_query_expansion_disabled_by_kwarg` — same corpus, query "pg", with
   `expand_query=False` → no hit. With default `True` → hit.
3. `test_query_expansion_does_not_duplicate_existing_token` — query
   "postgres pg replication" expands to a set, not a multiset. Assert no
   spurious score inflation vs the `postgres replication` baseline.
4. `test_query_expansion_unknown_token_passthrough` — random unknown token
   yields exactly the same hits with and without expansion.
5. `test_synonym_map_uses_post_stem_tokens` — every key in `_EXPANSION`
   passes through `_stem` unchanged. (Catches the "configuration → configur"
   edge case if the stemmer ever evolves.)
6. `test_query_expansion_composes_with_recency_boost` — expanded match in
   recent journal still beats older expanded match.

## Files touched

| File | Change |
|---|---|
| `jyagent/memory/_synonyms.py` (new) | ~40 LOC: `_SYNONYM_GROUPS`, `_EXPANSION`, `expand_query_tokens` |
| `jyagent/memory/search.py` | +2 lines (import + call), +1 kwarg |
| `tests/test_memory_upgrades.py` | +6 tests, ~80 LOC, registered in legacy runner list |

Total: ~120 LOC, no new dependencies, no architectural change.

## Open questions for reviewer

1. **Stem-canonical form**: should I store `configuration`/`config` directly,
   or store `configur`/`config` (post-stem)? My instinct: store the
   natural-language form and pass each group through `_stem` at build time
   to get the actual matching form. That way the source is readable.
2. **`anthropic` ↔ `claude`?** They are not synonyms (company vs product)
   but they ARE highly co-occurrent in this corpus. Skip — the precision
   risk is real ("Claude Code" vs "Anthropic's Claude").
3. Should the expansion map live in code or in `data/memory/synonyms.json`
   (user-editable)? My recommendation: code, for now. Move to data when /
   if it grows past ~50 entries.

## Verdict requested

`ship` | `fold` | `redraft` — and any single most-important suggestion.
