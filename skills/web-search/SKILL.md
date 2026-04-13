---
name: web-search
description: >-
  Search the web efficiently using search engines. Use this skill whenever the
  user asks to look up information, find answers, check current events/news,
  compare options, fact-check claims, or research any topic that requires
  current/external knowledge. TRIGGER on: "search for", "look up", "find out",
  "what is the latest", "current news", "research X", any factual question about
  real-world state, prices, events, people, comparisons. DO NOT TRIGGER on:
  fetching a known URL (web_fetch handles that), browser automation tasks,
  or questions clearly answerable from training data alone.
metadata:
  author: jy-agent
  version: "4.0"
---

# Web Search

Search the web using the `web_search` tool (primary) or `web_fetch` with
search URLs (advanced). Combines DuckDuckGo for fast results and Codex for
deep research with multi-source synthesis.

> **Scope**: This skill is about *searching* — constructing queries, picking
> engines, and synthesizing results. For *fetching* a known URL, use
> `web_fetch()` directly.

## Step 0: Always Verify Date

```
run_shell("date")  # NEVER assume the year — agent defaults can be wrong
```

## Decision Tree: Which Approach?

```
What does the user need?
├─ Quick fact / single answer
│   → web_search(query="...", engine="ddg")
│   → Fast, free, always works
│
├─ General search / find resources
│   → web_search(query="...", engine="auto")
│   → DDG first; Codex fallback if results are poor
│
├─ Deep research / multi-source synthesis
│   → web_search(query="...", engine="codex")
│   → Best quality, includes synthesis, 40-80K tokens per call
│   → Or dispatch_agent with a research task for parallel searches
│
├─ Current events / breaking news
│   → web_search(query="...", engine="codex")
│   → Codex web_search has real-time access, synthesizes across sources
│
├─ Chinese topic (中文内容)
│   → web_search(query="中文查询", engine="ddg") first
│   → Then web_fetch("https://www.baidu.com/s?wd=查询") for cross-reference
│
├─ Specific search engine features needed
│   │  (time filters, site:, filetype:, etc.)
│   → Use web_fetch with search URLs directly (see Advanced section)
│
└─ Need raw page content from results
    → web_search to find URLs, then web_fetch each URL
```

## Primary Tool: `web_search`

### Quick lookup
```python
web_search(query="python 3.14 new features")
# → DDG results with titles, URLs, snippets
```

### Best quality (Codex-powered)
```python
web_search(query="comparison of vLLM vs TGI for LLM inference 2026", engine="codex")
# → Structured results + synthesis paragraph from multiple sources
```

### Auto mode (recommended default)
```python
web_search(query="kubernetes pod security standards", engine="auto")
# → Tries DDG first (fast). If < 3 results, falls back to Codex
```

### Engines at a glance

| Engine | Speed | Cost | Quality | Synthesis | When to use |
|--------|-------|------|---------|-----------|-------------|
| `ddg` | ~2s | Free | Good | No | Quick lookups, known topics |
| `codex` | ~15-30s | ~40-80K tokens | Excellent | Yes | Deep research, current events, comparisons |
| `auto` | ~2-30s | Free→expensive | Good→Excellent | Maybe | Default — fast path with quality fallback |

## Advanced: Search Engine URLs via `web_fetch`

When you need specific search engine features (time filters, operators),
use `web_fetch` with search URLs directly.

### Google (needs Chrome MCP for best results)

```python
# Basic search — goes through web_fetch cascade, Chrome handles anti-bot
web_fetch("https://www.google.com/search?q=your+query+here")

# Time filter: past week
web_fetch("https://www.google.com/search?q=query&tbs=qdr:w")

# News search
web_fetch("https://www.google.com/search?q=query&tbm=nws")
```

**Key operators:** `"exact phrase"`, `site:domain`, `filetype:pdf`, `-exclude`,
`after:YYYY-MM-DD`, `intitle:`, `OR`

→ Full operator reference: [references/search-operators.md](references/search-operators.md)

### DuckDuckGo (always works, no Chrome needed)

```python
web_fetch("https://duckduckgo.com/html/?q=your+query+here")
```

### 百度 (Chinese content)

```python
web_fetch("https://www.baidu.com/s?wd=你的搜索词")
```

## Query Construction Tips

### Be specific, not conversational
```
❌ "what's the best way to deploy a python app to kubernetes"
✅ "python kubernetes deployment best practices 2025"
```

### Use error messages verbatim
```
❌ python ssl certificate error
✅ "ssl: CERTIFICATE_VERIFY_FAILED" python 3.14
```

### Iterate: broad → narrow
```
1st: web_search(query="vllm performance tuning")
2nd: web_search(query="vllm tensor parallel vs pipeline parallel A100")
3rd: web_fetch a specific result URL for deep content
```

### Add context qualifiers
```
"docker compose v2 migration 2025"    # year for freshness
"python 3.14 breaking changes"        # version
"kubernetes ingress nginx vs traefik"  # comparison framing
```

## Research Workflows

### Quick lookup (single fact)
```python
result = web_search(query="current Python latest stable version")
# Read top result → answer with citation
```

### Standard research (most cases)
```python
# 1. Search for results
result = web_search(query="FastAPI vs Django REST framework comparison 2026")
# 2. Fetch top 3-5 URLs from results
web_fetch("https://result-url-1.com/...")
web_fetch("https://result-url-2.com/...")
# 3. Cross-reference → synthesize with citations
```

### Deep research (comparisons, decisions)
```python
# Option A: Let Codex do the heavy lifting
result = web_search(
    query="AWS vs GCP GPU pricing comparison for LLM inference 2026",
    engine="codex",
)
# Codex searches multiple sources and provides synthesis

# Option B: Parallel sub-agent research
dispatch_agent(task="Search for AWS GPU instance pricing for LLM inference...")
dispatch_agent(task="Search for GCP GPU instance pricing for LLM inference...")
# Combine results yourself
```

### Breaking news / current events
```python
# Codex excels here — real-time search with synthesis
result = web_search(query="latest AI regulation news April 2026", engine="codex")
```

## Anti-Patterns

❌ **Don't** answer "from training data" when user asks for current info
✅ **Do** always search, even if you think you know — things change

❌ **Don't** search once and trust a single source
✅ **Do** cross-reference 2-3 sources for important claims

❌ **Don't** use `engine="codex"` for simple factual lookups — it's overkill
✅ **Do** use `engine="ddg"` or `"auto"` for quick lookups; reserve Codex for deep research

❌ **Don't** construct Google search URLs when `web_search` would suffice
✅ **Do** prefer `web_search(query="...")` — it handles engine selection and parsing

❌ **Don't** dump raw search results to the user
✅ **Do** fetch top results, synthesize, and cite URLs

❌ **Don't** forget to check the date of your sources
✅ **Do** note publication dates — a 2021 article may be outdated for a 2026 question

## Reference Files

- [📋 Search Operators](references/search-operators.md) — Google, DuckDuckGo, 百度 advanced syntax & power combos
- [🔍 Source Evaluation](references/source-evaluation.md) — How to assess credibility, domain tier list, red flags
