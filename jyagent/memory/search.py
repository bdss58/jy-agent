# Retrieval over Tier-2 (topics) and Tier-3 (journal) — BM25 + section chunking.
#
# Why BM25 and not vectors?
#   - The corpus is tiny (a few thousand lines across a handful of files).
#   - No external deps, no embedding API call on every search.
#   - Letta's LoCoMo result (74% with grep/BM25 over a text filesystem) shows
#     naive retrieval can match or beat specialized vector memory at this scale.
#
# What we index
#   - Each topic file (data/memory/topics/<name>.md) is split into **sections**
#     by markdown `##` headers. A topic with no `##` headers becomes a single
#     chunk.
#   - Each journal month (data/memory/journal/YYYY-MM.md) is split by `##`
#     headers (each entry already starts with `## YYYY-MM-DD HH:MM [cat]`).
#   - Every chunk is a `SearchChunk(source, section, body)` where `source` is
#     human-readable ("topics/foo.md" or "journal/2026-04.md") and `section`
#     is the header text (or "" for the preamble).

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

# Markdown chunking + frontmatter — shared with _topics. Keep underscored
# aliases for the test suite (test_memory_upgrades imports _split_sections).
from ._markdown import (
    H2_H3_HEADER_RE as _H2_H3,
    split_sections as _split_sections,
    strip_frontmatter as _strip_frontmatter,
)

from .. import config as _cfg
from ._topics import list_topics, read_topic
from ._journal import list_journals, read_journal


# ─── Tokenization ─────────────────────────────────────────────────────────────

_ASCII_TOK = re.compile(r"[A-Za-z][\w.\-]{1,}")
# Version / numeric tokens like "3.14", "1.2.3", "2026-04-25" — they carry
# real signal for technical search and are missed by _ASCII_TOK because that
# pattern requires a letter prefix.
_NUM_TOK = re.compile(r"\b\d+(?:[.\-]\d+){1,}\b")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]+")

# Shared stop-word set — kept very small so BM25 IDF does most of the filtering.
_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "at", "for", "with",
    "by", "from", "as", "this", "that", "these", "those", "it", "its",
    "do", "does", "did", "done", "has", "have", "had", "not", "no", "so",
    "too", "very", "can", "will", "would", "should", "could", "may", "might",
}


def _stem(token: str) -> str:
    """Cheapest plural stripper that doesn't make words worse.

    Rules (applied in order):
      - contains ``.`` or ``-`` → leave alone (dotted paths, versions,
        identifiers like "jyagent.tools.facades" must not lose their tail
        characters)
      - len < 4 → leave alone (e.g. "is", "uv", "k8s" — these are content)
      - ends in "ies" → "y" (queries → query, but not "ties" → "ty"; that
        false stem doesn't matter for BM25 because it only loses you a hit)
      - ends in "es" with a non-vowel before → drop "es" (fixes "boxes",
        "watches"; "tomatoes" mis-stems to "tomato" which is correct)
      - ends in single "s" preceded by anything except "s" → drop "s"
        (producers → producer; class stays class)
    """
    if "." in token or "-" in token:
        return token
    if len(token) < 4:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("es") and token[-3] not in "aeiou":
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    """Extract normalized tokens from text.

    Rules:
      - ASCII identifiers >=2 chars (keeps "k8s", "uv", "ls", but drops "a")
      - Preserves dotted paths / version numbers ("jyagent.tools", "3.14.3")
      - Numeric/version tokens with internal . or - ("3.14", "2026-04-25")
      - CJK text → character bigrams so Chinese matches work
      - Lower-cased, stop-words dropped, plurals stemmed by ``_stem``
    """
    out: list[str] = []
    for tok in _ASCII_TOK.findall(text):
        t = tok.lower()
        if t in _STOP:
            continue
        out.append(_stem(t))
    for tok in _NUM_TOK.findall(text):
        out.append(tok)
    for run in _CJK_RUN.findall(text):
        # bigrams: "用户偏好" -> 用户, 户偏, 偏好
        if len(run) == 1:
            out.append(run)
            continue
        for i in range(len(run) - 1):
            out.append(run[i : i + 2])
    return out



# ─── Index + BM25 ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SearchChunk:
    source: str        # e.g. "topics/agent-loop-changelog.md" or "journal/2026-04.md"
    section: str       # header text, "" for preamble
    body: str          # raw markdown body of the chunk


@dataclass(frozen=True)
class SearchHit:
    chunk: SearchChunk
    score: float


def _collect_chunks(
    include_topics: bool = True,
    include_journal: bool = True,
    journal_months: int | None = None,
) -> list[SearchChunk]:
    """Enumerate all indexable chunks from topics and journals.

    ``journal_months`` caps how many months are read (newest first). ``None``
    means all months. This is a cheap safety valve for very long-running
    agents; the default is "all" because bodies are small.
    """
    chunks: list[SearchChunk] = []

    if include_topics:
        for name in list_topics():
            body = read_topic(name)
            if not body:
                continue
            # Strip YAML frontmatter if present — we don't want to score against
            # metadata keys like "updated:" clobbering real content.
            body = _strip_frontmatter(body)
            for header, text in _split_sections(body):
                chunks.append(SearchChunk(
                    source=f"topics/{name}.md",
                    section=header,
                    body=text,
                ))

    if include_journal:
        months = list_journals()
        if journal_months is not None:
            months = months[:journal_months]
        for m in months:
            body = read_journal(m)
            if not body:
                continue
            for header, text in _split_sections(body):
                chunks.append(SearchChunk(
                    source=f"journal/{m}.md",
                    section=header,
                    body=text,
                ))

    return chunks



def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    doc_freq: dict[str, int],
    n_docs: int,
    avg_dl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Classic Okapi BM25 (Robertson/Zaragoza 2009).

    Implemented inline to avoid a dependency. For tiny corpora (<1000 chunks)
    this is microseconds and never a bottleneck.
    """
    if not doc_tokens:
        return 0.0

    # Term frequencies in this doc
    tf: dict[str, int] = {}
    for t in doc_tokens:
        tf[t] = tf.get(t, 0) + 1

    dl = len(doc_tokens)
    score = 0.0
    for qt in query_tokens:
        if qt not in tf:
            continue
        # IDF with the +1 smoothing so single-doc corpora stay non-negative.
        df = doc_freq.get(qt, 0)
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        freq = tf[qt]
        norm = freq * (k1 + 1.0) / (freq + k1 * (1.0 - b + b * dl / avg_dl))
        score += idf * norm
    return score


def search_memory(
    query: str,
    top_k: int = 5,
    *,
    include_topics: bool = True,
    include_journal: bool = True,
    journal_months: int | None = None,
    min_score: float = 0.0,
) -> list[SearchHit]:
    """Rank topic + journal chunks against ``query`` with BM25.

    Returns at most ``top_k`` hits with score > ``min_score`` (default 0, i.e.
    any hit with a non-zero overlap). The index is rebuilt on every call
    because bodies are small and memory files change often; if this ever
    becomes a hotspot, cache by mtime.
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    chunks = _collect_chunks(
        include_topics=include_topics,
        include_journal=include_journal,
        journal_months=journal_months,
    )
    if not chunks:
        return []

    # Tokenize once, compute doc-frequency across the corpus
    tokenized: list[list[str]] = [_tokenize(c.body) for c in chunks]
    doc_freq: dict[str, int] = {}
    for toks in tokenized:
        for t in set(toks):
            doc_freq[t] = doc_freq.get(t, 0) + 1
    n_docs = len(chunks)
    total_dl = sum(len(toks) for toks in tokenized)
    avg_dl = (total_dl / n_docs) if n_docs else 1.0

    hits: list[SearchHit] = []
    for chunk, toks in zip(chunks, tokenized):
        score = _bm25_score(q_tokens, toks, doc_freq, n_docs, avg_dl)
        if score > min_score:
            hits.append(SearchHit(chunk=chunk, score=score))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


# ─── Rendering for the manage_memory tool ─────────────────────────────────────

def render_hits(hits: list[SearchHit], max_body_chars: int = 400) -> str:
    """Format search hits for display to the agent.

    Each hit shows source:section, score, and a body snippet (truncated).
    """
    if not hits:
        return "🔎 No matching memory found."
    lines = [f"🔎 {len(hits)} hit(s):"]
    for i, h in enumerate(hits, 1):
        header = f"{h.chunk.source}" + (f"#{h.chunk.section}" if h.chunk.section else "")
        body = h.chunk.body
        if len(body) > max_body_chars:
            body = body[:max_body_chars].rstrip() + " …"
        lines.append(f"\n{i}. **{header}**  (score={h.score:.2f})\n{body}")
    return "\n".join(lines)
