---
name: web-search
description: >-
  Use this skill when users need current web information or real-world
  lookups. TRIGGER immediately for: information seeking ("find out about X",
  "look up Y"), current events and updates ("what's the latest on Z", "what
  changed with A"), research requests ("research B topic"), verification
  queries ("is X still Y", "check if C is true"), comparisons needing live
  data, factual questions about people/companies/products/prices, and anything
  requiring fresh external knowledge beyond training data. DO NOT TRIGGER for:
  known URL fetching (use web_fetch), weather/time queries (use appropriate
  APIs), questions answerable from existing knowledge alone, or comprehensive
  multi-source reports (use deep-research). This skill provides efficient web
  search with smart query breakdown and cited answers. Backed by a resilient
  multi-engine cascade (SearxNG → DuckDuckGo).
metadata:
  author: jy-agent
  version: "6.1"
---

# Web Search

Search the web using `web_search` (SearxNG → DDG cascade), `web_fetch` (page-level),
and `dispatch_agent` (parallel). Inspired by how Google, OpenAI, and Anthropic
implement search in their AI agents — with query decomposition, iterative
multi-hop search, dynamic filtering, and citation-first output.

> **Scope**: This skill handles searching and synthesizing. For *fetching*
> a known URL, use `web_fetch()` directly. For comprehensive multi-source
> research reports (5+ min), escalate to the **deep-research** skill.

## Step 0: Always Verify Date

```python
run_shell("date")  # NEVER assume the year — model defaults can be wrong
```

## Decision Tree: Which Approach?

```
What does the user need?
│
├─ Quick fact / single answer (who, what, when)
│   → Single-Shot Search
│
├─ General search / find resources
│   → Single-Shot Search
│
├─ Multi-faceted question or comparison
│   → Decompose & Multi-Hop Search
│
├─ Current events / breaking news
│   → Multi-Hop Search with recency in the query (e.g. "2026")
│
├─ Chinese topic (中文内容)
│   → DDG first + 百度 cross-reference (web_fetch the 百度 search URL)
│
├─ Deep research report (10+ sources, comprehensive analysis)
│   → ESCALATE to deep-research skill
│   → Tell user: "This needs deep research — launching multi-agent investigation"
│
├─ Specific search features needed (time filters, site:, filetype:)
│   → Advanced: web_fetch with search engine URLs
│
└─ Need raw page content from search results
    → Search → web_fetch each URL
```

## Core Pattern: Search → Filter → Cite

All search follows this 3-step loop (inspired by Google's grounding pipeline
and Anthropic's dynamic filtering):

```
1. SEARCH — Generate optimized query, execute search
2. FILTER — Evaluate results: relevance, authority, freshness; fetch top sources
3. CITE   — Synthesize answer with inline citations [title](url)
```

## Single-Shot Search

For straightforward questions with a single clear answer:

```python
# Quick fact
web_search(query="python 3.14 release date")

# General search
web_search(query="kubernetes pod security standards")

# Current events — add the year to bias toward fresh results
web_search(query="latest AI regulation news 2026")
```

### Backend

`web_search` runs a **multi-engine cascade** (first non-empty wins):

1. **SearxNG** — only if `SEARXNG_URL` env is set (self-hosted meta-search; aggregates Google/Bing/Brave/etc.). Best quality.
2. **DuckDuckGo HTML** — universal fallback, no auth needed.

Force a single engine with `WEB_SEARCH_ENGINE=searxng|ddg`.
Returns up to `max_results` (default 10) hits with title, URL, snippet.

For deeper coverage, the agent does the orchestration itself:
**search → pick promising URLs → `web_fetch` each → synthesize.**
This is cheaper and more controllable than delegating to a subagent
search engine. For genuinely large investigations, escalate to the
**deep-research** skill which dispatches parallel `dispatch_agent` workers.

## Query Decomposition & Reformulation

Before searching, transform the user's question into optimized queries.
All three major AI providers do this — it is the single biggest quality lever.

### Reformulation Rules

```
User's natural language → Search-optimized query

├─ Drop conversational filler
│   "Can you help me find out what the best..." → "best ..."
│
├─ Add specificity qualifiers
│   Version numbers, year, platform, context
│   "python ssl error" → '"ssl: CERTIFICATE_VERIFY_FAILED" python 3.14'
│
├─ Use exact error messages verbatim (in quotes)
│   "I got some timeout error" → ask user for exact message, then quote it
│
├─ Split compound questions into multiple queries
│   "Compare vLLM vs TGI performance and pricing"
│   → Query 1: "vLLM vs TGI inference performance benchmark 2026"
│   → Query 2: "vLLM TGI pricing comparison cloud deployment"
│
└─ Add domain qualifiers for authoritative sources
    "kubernetes networking" → "site:kubernetes.io networking CNI"
```

### Decomposition Pattern

For multi-faceted questions, decompose BEFORE searching:

```python
# User asks: "Should I use vLLM or TGI for my LLM inference setup?"
# Decompose into sub-queries:
queries = [
    "vLLM vs TGI inference throughput latency benchmark 2026",
    "vLLM features model support GPU compatibility",
    "TGI text-generation-inference features limitations",
    "vLLM production deployment best practices",
]
# Execute top 2-3 most important queries, then fetch key pages
```

## Iterative Multi-Hop Search

For questions where the first search informs what to search next — like
how OpenAI's o3 chains dozens of searches and Anthropic's Claude pivots
based on intermediate findings.

### The Loop

```
Start with broad query
  ↓
Review results → Extract leads (names, terms, URLs)
  ↓
Formulate narrower follow-up queries using new terms
  ↓
Fetch specific pages for detailed content
  ↓
Repeat until: answer is clear OR 3-4 hops done OR diminishing returns
```

### Example: Multi-Hop in Practice

```python
# Hop 1: Broad search
results = web_search(query="fastest open-source LLM inference engine 2026")
# → Learn about vLLM, TGI, SGLang, TensorRT-LLM

# Hop 2: Narrow based on findings
results = web_search(query="SGLang vs vLLM benchmark throughput A100 2026")
# → Find specific benchmark page

# Hop 3: Fetch the actual benchmark data
content = web_fetch("https://benchmark-page-url.com/results")
# → Extract specific numbers

# Hop 4 (if needed): Verify with second source
content2 = web_fetch("https://another-benchmark.com/llm-inference")
```

### When to Multi-Hop

- First search results are generic or insufficient
- Found a specific term/name that needs drilling into
- Results contradict each other — need more sources
- Paywalled or empty results — pivot to alternative query

### When to STOP

- 2+ reliable sources agree on the answer
- 3-4 hops done with diminishing returns
- User asked a simple question — don't over-research

## Parallel Search (for comparisons & multi-aspect queries)

When sub-queries are independent, search in parallel using dispatch_agent:

```python
# User: "Compare AWS vs GCP GPU pricing for LLM inference"
dispatch_agent(
    task='Search for "AWS GPU instance pricing p5 p4d LLM inference 2026". '
         'Return: instance types, GPU models, hourly prices, spot prices.',
    background=True
)
dispatch_agent(
    task='Search for "GCP GPU instance pricing A100 H100 LLM inference 2026". '
         'Return: instance types, GPU models, hourly prices, spot prices.',
    background=True
)
# Poll both, then synthesize into comparison table
```

## Citation-First Output

Every search-based answer MUST include citations. This is how Google,
OpenAI, and Anthropic all handle it — inline source attribution.

### Citation Format

```markdown
## Answer

According to the official benchmarks, vLLM achieves 2.3x higher throughput
than TGI on A100 GPUs for Llama 3 70B [vLLM Benchmarks](https://url1.com).
However, TGI offers better integration with Hugging Face's ecosystem
[TGI Docs](https://url2.com).

### Sources
1. [vLLM Official Benchmarks](https://url1.com) — Retrieved 2026-04-17
2. [TGI Documentation](https://url2.com) — Retrieved 2026-04-17
```

### Rules

- Every factual claim gets a citation — no uncited assertions
- Use `[Title](URL)` inline, plus a Sources section at the end
- Note when information comes from training data vs live search
- If sources conflict, present both sides with citations
- Include retrieval date for time-sensitive information

## Source Quality Evaluation

Evaluate every source before citing. Anthropic found their early agents
chose SEO-optimized content farms over authoritative sources — adding
quality heuristics to prompts fixed this.

### Quick Credibility Check

```
Source found → Evaluate in order:
├─ 1. Domain authority
│   ├─ Official docs (*.readthedocs.io, docs.*.com) → High trust
│   ├─ Known tech (SO, GitHub, MDN, HN) → High trust
│   ├─ Major publications (Reuters, NYT, BBC) → High trust
│   ├─ Personal engineering blogs → Medium (check author)
│   ├─ Content farms (w3schools, geeksforgeeks) → Low trust
│   └─ Unknown domain → Low trust (must cross-reference)
│
├─ 2. Freshness — is it current enough?
│   ├─ Software/API → within 1-2 years
│   ├─ News/events → days/weeks
│   └─ Concepts/theory → older OK
│
├─ 3. Does it directly answer the question?
│
└─ 4. Do other sources confirm it?
    ├─ 2+ agree → high confidence
    ├─ Sources conflict → present both, note disagreement
    └─ Single source → qualify with "according to [source]"
```

→ Detailed guide: [references/source-evaluation.md](references/source-evaluation.md)

## Advanced: Search Engine URLs via web_fetch

When you need specific search engine features, use web_fetch with URLs:

```python
# Google with time filter (past week)
web_fetch("https://www.google.com/search?q=query&tbs=qdr:w")

# Google News
web_fetch("https://www.google.com/search?q=query&tbm=nws")

# DuckDuckGo (always works, no Chrome needed)
web_fetch("https://duckduckgo.com/html/?q=your+query+here")

# 百度 (Chinese content)
web_fetch("https://www.baidu.com/s?wd=你的搜索词")
```

→ Full operator reference: [references/search-operators.md](references/search-operators.md)

## Escalation to Deep Research

Recognize when a question exceeds web-search scope and escalate:

```
Escalation signals:
├─ User asks for "comprehensive", "thorough", "detailed report"
├─ Question has 3+ independent aspects needing separate investigation
├─ Answer requires synthesizing 10+ sources
├─ Comparison across many dimensions (pricing, features, benchmarks, etc.)
├─ User explicitly asks for "deep research" or "research report"
└─ First search attempt reveals the topic is much bigger than expected
```

When escalating, tell the user and switch to the deep-research skill pattern.

## Anti-Patterns

❌ **Don't** answer from training data when user asks about current state
✅ **Do** always search — things change; verify even "known" facts

❌ **Don't** use a single query for multi-faceted questions
✅ **Do** decompose into sub-queries, search the most important ones

❌ **Don't** cite sources without checking credibility
✅ **Do** prefer official docs and Tier 1 sources; cross-reference claims

❌ **Don't** search once and give up if results are poor
✅ **Do** iterate: reformulate query, try different terms, pivot strategy

❌ **Don't** dump raw search results to the user
✅ **Do** synthesize, cite inline, and add a Sources section

❌ **Don't** forget to check publication dates
✅ **Do** note when sources are old — a 2022 article may be wrong for 2026

❌ **Don't** over-research simple questions (5+ searches for "what year was X")
✅ **Do** match search depth to question complexity — 1 search for simple facts

## Reference Files

- [📋 Search Operators](references/search-operators.md) — Google, DuckDuckGo, 百度 advanced syntax
- [🔍 Source Evaluation](references/source-evaluation.md) — Credibility tiers, red flags, presentation formats
