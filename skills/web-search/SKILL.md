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
  version: "3.0"
---

# Web Search

Use search engines effectively to find, verify, and synthesize information.

> **Scope**: This skill is about *searching* — constructing queries, picking engines,
> and synthesizing results. For *fetching* a known URL, just use `web_fetch()` directly.

## Step 0: Always Verify Date

```
run_shell("date")  # NEVER assume the year — agent defaults can be wrong
```

## Decision Tree: Which Engine?

```
What are you searching for?
├─ General / English topic
│   → Google (primary), DuckDuckGo (fallback)
│
├─ Chinese topic (中文内容)
│   → Google + 百度 (cross-reference both)
│
├─ Technical / programming
│   ├─ Official docs → Go to docs site directly (skip search)
│   ├─ Error messages → Google exact error string in quotes
│   └─ Code examples → Google with "site:github.com" or "site:stackoverflow.com"
│
├─ Current events / news
│   ├─ English → Google News (tbm=nws) or Google with time filter (tbs=qdr:d)
│   ├─ Chinese → 百度 + Google
│   └─ Cross-reference 2-3 sources minimum
│
└─ Product / comparison
    → Google, fetch 3+ review/comparison pages, build table
```

## Search Engine Cheat Sheet

### Google (preferred — most comprehensive)

```python
# Basic search
web_fetch("https://www.google.com/search?q=your+query+here")

# NOTE: Google blocks cffi/httpx, auto-cascade will use Chrome tier.
# Chrome MCP must be connected. If not available, fall back to DuckDuckGo.
```

**Key operators:** `"exact phrase"`, `site:domain`, `filetype:pdf`, `-exclude`, `after:YYYY-MM-DD`, `intitle:`, `OR`
**Time filters:** `&tbs=qdr:d` (day), `qdr:w` (week), `qdr:m` (month), `qdr:y` (year)
**News:** `&tbm=nws`

→ Full operator reference: [references/search-operators.md](references/search-operators.md)

### DuckDuckGo (fallback — no anti-bot)

```python
# HTML version — works with cffi tier, no Chrome needed
web_fetch("https://duckduckgo.com/html/?q=your+query+here")
```
- ✅ Always works (no anti-bot)
- ❌ Less comprehensive than Google
- Use when: Chrome MCP not connected, or Google is being difficult

### 百度 (Chinese content)

```python
web_fetch("https://www.baidu.com/s?wd=你的搜索词")
```
- ✅ Best for Chinese-language content, domestic sites
- ❌ Heavy ads, lower signal-to-noise
- Use when: Searching for Chinese-specific topics, domestic news, Chinese tech

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
1st search: "vllm performance tuning"
2nd search: "vllm tensor parallel vs pipeline parallel A100"  (narrowed by findings)
3rd search: site:github.com/vllm-project/vllm "tensor_parallel"  (specific repo)
```

### Add context qualifiers
```
# Add year for freshness
"docker compose v2 migration 2025"

# Add technology version
"python 3.14 breaking changes"

# Add platform
"kubernetes ingress nginx vs traefik comparison"
```

## Research Workflow

### Quick lookup (single fact)
1. One Google search → read top 1-2 results → answer with citation

### Standard research (most cases)
1. Google search → identify 3-5 relevant URLs from results
2. `web_fetch()` each URL → extract key facts
3. Cross-reference findings → note agreements and contradictions
4. Synthesize with citations

### Deep research (comparisons, decisions)
1. Search from multiple angles (2-3 different queries)
2. Fetch 5-8 sources across queries
3. Build comparison table or structured summary
4. Note source dates and potential biases
5. Present findings with confidence levels

## Anti-Patterns

❌ **Don't** answer "from training data" when user asks for current info
✅ **Do** always search, even if you think you know — things change

❌ **Don't** search once and trust a single source
✅ **Do** cross-reference 2-3 sources for important claims

❌ **Don't** use vague queries like "tell me about X"
✅ **Do** use specific, keyword-rich queries with operators

❌ **Don't** dump raw search results to the user
✅ **Do** fetch top results, synthesize, and cite URLs

❌ **Don't** forget to check the date of your sources
✅ **Do** note publication dates — a 2021 article may be outdated for a 2025 question

❌ **Don't** keep retrying Google if Chrome MCP isn't connected
✅ **Do** fall back to DuckDuckGo immediately if Chrome is unavailable

## Reference Files

- [📋 Search Operators](references/search-operators.md) — Google, DuckDuckGo, 百度 advanced syntax & power combos
- [🔍 Source Evaluation](references/source-evaluation.md) — How to assess credibility, domain tier list, red flags
