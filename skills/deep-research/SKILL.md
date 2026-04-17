---
name: deep-research
description: >-
  Conduct comprehensive, multi-agent deep research that produces detailed cited
  reports. Use this skill whenever the user asks for thorough investigation,
  comprehensive analysis, detailed comparison, literature review, market
  research, survey, or any task requiring synthesis across 10+ sources.
  TRIGGER on: "deep research", "comprehensive report", "thorough analysis",
  "investigate thoroughly", "research report on", "detailed comparison of",
  "survey of X", "do a survey", "landscape analysis", "state of the art",
  "literature review", "market research", "competitive analysis", or when a
  web-search escalates due to complexity. DO NOT TRIGGER on: quick lookups,
  single-fact questions, or simple searches (use web-search skill instead).
metadata:
  author: jy-agent
  version: "1.0"
---

# Deep Research

Conduct multi-agent research investigations that produce comprehensive,
cited reports. Modeled after how Google (Gemini Deep Research), OpenAI
(o3 Deep Research), and Anthropic (Claude Research) build their research
agents — adapted to jy-agent's tools.

**Key insight from industry**: Token usage explains 80% of research quality
variance (Anthropic). Multi-agent systems outperform single agents by 90%+
on research tasks. The architecture: an orchestrator plans the work, parallel
subagents explore independently, and the orchestrator synthesizes findings.

> **Time expectation**: Deep research takes 3-15 minutes. Tell the user
> upfront: "This will take several minutes — I'm launching parallel research
> agents to investigate thoroughly."

## Decision Tree: Is This Deep Research?

```
User request →
├─ Quick fact or single-source answer
│   → NOT deep research. Use web-search skill.
│
├─ Multi-faceted comparison (3+ dimensions)
│   → YES. Use Comparison Research template.
│
├─ "Research X thoroughly" / "comprehensive report on Y"
│   → YES. Use General Research template.
│
├─ Technology/product landscape survey
│   → YES. Use Landscape Survey template.
│
├─ Question where web-search found insufficient/contradictory results
│   → YES. Escalate from web-search.
│
└─ Already doing web-search but realizing scope is bigger than expected
    → Pivot to deep research. Tell user.
```

## The 5-Phase Research Process

Inspired by OpenAI's deep research pipeline and Anthropic's multi-agent system:

```
Phase 1: CLARIFY   — Understand what the user actually needs
Phase 2: PLAN      — Decompose into research questions, design strategy
Phase 3: SEARCH    — Parallel subagents investigate independently
Phase 4: SYNTHESIZE — Merge findings, resolve conflicts, build narrative
Phase 5: DELIVER   — Produce cited report with structured sections
```

### Phase 1: CLARIFY (30 seconds)

Before launching research, ensure the query is specific enough.
OpenAI uses an intermediate model to ask clarifying questions.

```
Check:
├─ Is the scope clear? ("compare LLM frameworks" — which ones? for what use case?)
├─ Is there a specific angle? (technical, business, security, cost?)
├─ What output format? (comparison table, report, recommendations?)
├─ Any constraints? (time period, geography, specific products?)
└─ How deep? (overview vs exhaustive?)
```

If ambiguous, ask 1-2 targeted clarifying questions. Don't over-ask —
infer from context when possible.

### Phase 2: PLAN (1 minute)

Decompose the query into 3-5 independent research threads.
Each thread becomes a subagent task.

```python
# Example: "Compare vLLM vs TGI vs SGLang for production LLM inference"
research_plan = {
    "question": "Which LLM inference engine is best for production?",
    "threads": [
        {
            "id": "performance",
            "query": "Performance benchmarks: throughput, latency, GPU utilization",
            "search_terms": [
                "vLLM vs TGI vs SGLang benchmark throughput latency 2026",
                "LLM inference engine performance comparison A100 H100",
            ]
        },
        {
            "id": "features",
            "query": "Feature comparison: model support, quantization, batching",
            "search_terms": [
                "vLLM features continuous batching PagedAttention",
                "TGI SGLang feature comparison model support",
            ]
        },
        {
            "id": "production",
            "query": "Production readiness: stability, scaling, monitoring, community",
            "search_terms": [
                "vLLM production deployment experience 2026",
                "TGI SGLang production issues stability",
            ]
        },
        {
            "id": "cost",
            "query": "Cost analysis: licensing, cloud pricing, operational overhead",
            "search_terms": [
                "vLLM TGI SGLang cloud deployment cost comparison",
                "LLM inference engine licensing open source",
            ]
        }
    ],
    "output_format": "comparison report with recommendation"
}
```

**Planning principles** (from Anthropic's research):
- Each thread should be independently explorable
- 3-5 threads is the sweet spot (more = diminishing returns)
- Threads should cover different ANGLES, not just different queries
- Include at least one thread for "contrarian/critical" perspective

### Phase 3: SEARCH — Parallel Subagent Dispatch

This is the core innovation. Launch 3-5 background subagents simultaneously,
each exploring one research thread.

#### Subagent Prompt Template

Each subagent gets a self-contained task with clear instructions:

```python
dispatch_agent(
    task="""RESEARCH THREAD: Performance Benchmarks

GOAL: Find concrete performance data comparing vLLM, TGI, and SGLang
for LLM inference on modern GPUs (A100, H100).

SEARCH STRATEGY:
1. Search for: "vLLM vs TGI vs SGLang benchmark throughput latency 2026"
2. Search for: "LLM inference engine performance comparison A100 H100"
3. Fetch the top 3-5 most relevant result pages
4. If results are thin, try: "vLLM benchmark", "TGI benchmark", "SGLang benchmark" separately

WHAT TO COLLECT:
- Specific numbers: requests/sec, tokens/sec, latency p50/p99
- Test conditions: model size, GPU type, batch size, precision
- Date of benchmarks (reject anything older than 12 months)

OUTPUT FORMAT:
Return a structured summary with:
1. Key findings (bullet points with specific numbers)
2. Sources (URL + title + date for each)
3. Confidence level (high/medium/low based on source quality)
4. Gaps (what you couldn't find)

RULES:
- Prefer official benchmarks and reputable tech blogs over random posts
- If sources conflict, note both with citations
- Do NOT make up numbers — if data is unavailable, say so
- Spend your full budget searching — thoroughness matters""",
    background=True,
    timeout=600
)
```

#### Dispatch Pattern

```python
# Launch all subagents in one call block (they run in parallel)
agent_ids = []
for thread in research_plan["threads"]:
    result = dispatch_agent(
        task=f"""RESEARCH THREAD: {thread['query']}
        ... (prompt template above, customized per thread) ...""",
        background=True,
        timeout=600
    )
    agent_ids.append(result["agent_id"])

# Poll until all complete (check every 30s)
# Use check_agent(agent_id) for each
```

#### Subagent Design Principles

From Anthropic's engineering lessons:

1. **Self-contained tasks**: Each subagent gets ALL context it needs.
   No references to "the conversation" or "what we discussed."

2. **Structured output**: Tell subagents exactly what format to return.
   This makes synthesis much easier.

3. **Search budget**: Each subagent should do 3-8 searches and fetch
   2-5 full pages. More tokens = better results.

4. **Source quality rules**: Include source evaluation criteria in the
   prompt so subagents don't return content-farm garbage.

5. **Failure handling**: Subagents should report gaps, not hallucinate.
   "Could not find X" is a valid and useful output.

### Phase 4: SYNTHESIZE (2-3 minutes)

Once all subagents return, the orchestrator (you) synthesizes findings.

#### Synthesis Checklist

```
For each research thread:
├─ Extract key findings with citations
├─ Assess source quality and confidence
├─ Identify agreements across threads
├─ Flag contradictions (investigate if critical)
├─ Note gaps where information was missing
│
Cross-thread synthesis:
├─ Build unified narrative from all threads
├─ Resolve conflicts (more sources = stronger claim)
├─ Create comparison tables where applicable
├─ Formulate recommendations based on evidence
└─ Rank sources by quality for the citation list
```

#### Handling Contradictions

When subagents return conflicting information:
- Check source authority (official docs > blog posts)
- Check source freshness (newer usually wins for tech)
- If still unclear, do a targeted follow-up search
- Present both viewpoints with citations in the final report

#### Follow-Up Searches

After initial synthesis, you may need 1-2 targeted follow-up searches
to fill gaps or resolve contradictions. This is the "backtracking" that
OpenAI's deep research agent does.

```python
# Gap identified: no pricing data for SGLang
web_search(query="SGLang deployment cost cloud pricing 2026")
web_fetch("https://specific-source-found.com/pricing")
```

### Phase 5: DELIVER — The Research Report

#### Report Template

```markdown
# [Research Topic]

*Deep research report — [date] — [N] sources consulted*

## Executive Summary
[2-3 paragraph overview of key findings and recommendation]

## Key Findings

### [Thread 1 Title]
[Findings with inline citations]
- Finding A [Source 1](url1)
- Finding B [Source 2](url2)

### [Thread 2 Title]
[Findings with inline citations]

### [Thread 3 Title]
[Findings with inline citations]

## Comparison Table (if applicable)
| Dimension | Option A | Option B | Option C |
|-----------|----------|----------|----------|
| ... | ... [src] | ... [src] | ... [src] |

## Analysis & Recommendation
[Synthesized analysis with clear reasoning]

## Limitations & Gaps
- [What couldn't be verified]
- [Areas needing more investigation]
- [Potential biases in available sources]

## Sources
1. [Title](URL) — [domain] — [date] — Used for: [what]
2. [Title](URL) — [domain] — [date] — Used for: [what]
...
```

#### Report Quality Criteria

From Anthropic's LLM-judge evaluation rubric:

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Factual accuracy | High | Claims match cited sources |
| Citation accuracy | High | Citations actually support the claims |
| Completeness | Medium | All requested aspects covered |
| Source quality | Medium | Primary/authoritative sources preferred |
| Tool efficiency | Low | Right number of searches (not wasteful) |

## Research Templates

### Comparison Research

```python
# Best for: "A vs B vs C" questions
# Threads: one per comparison dimension
# Output: comparison table + recommendation
threads = [
    "Performance/benchmarks for A, B, C",
    "Features/capabilities for A, B, C",
    "Cost/pricing for A, B, C",
    "Community/ecosystem/maturity for A, B, C",
]
```

### Landscape Survey

```python
# Best for: "What are the options for X?"
# Threads: one per category/segment
# Output: landscape map + top picks
threads = [
    "Category 1 players and their strengths",
    "Category 2 players and their strengths",
    "Recent entrants and emerging trends",
    "Industry analyst reports and rankings",
]
```

### Technical Deep Dive

```python
# Best for: "How does X work?" at expert level
# Threads: one per technical layer
# Output: technical explanation + architecture diagrams
threads = [
    "Architecture and design principles",
    "Implementation details and key algorithms",
    "Performance characteristics and limitations",
    "Comparison with alternative approaches",
]
```

### Current Events / Rapidly Evolving Topic

```python
# Best for: "What's happening with X?"
# Threads: temporal + perspective
# Output: timeline + analysis
threads = [
    "Latest developments (past week/month)",
    "Key stakeholder positions and reactions",
    "Historical context and trajectory",
    "Expert predictions and analysis",
]
```

## Anti-Patterns

❌ **Don't** launch deep research for simple questions
✅ **Do** use web-search for quick lookups; reserve deep research for complex queries

❌ **Don't** give subagents vague prompts ("research vLLM")
✅ **Do** give specific goals, search terms, output format, and quality criteria

❌ **Don't** skip the planning phase — jumping straight to search wastes tokens
✅ **Do** spend 1 minute planning threads before dispatching subagents

❌ **Don't** trust a single subagent's findings without cross-referencing
✅ **Do** look for agreement across threads; investigate contradictions

❌ **Don't** produce reports without citations
✅ **Do** cite every factual claim; include a full Sources section

❌ **Don't** report gaps as failures
✅ **Do** include a "Limitations & Gaps" section — knowing what's unknown is valuable

❌ **Don't** over-parallelize (10+ subagents) — coordination overhead exceeds benefit
✅ **Do** use 3-5 subagents; add targeted follow-ups if gaps remain

❌ **Don't** let subagents choose SEO content farms as primary sources
✅ **Do** include source quality rules in subagent prompts

## Reference Files

- [📋 Research Patterns](references/research-patterns.md) — Detailed subagent prompt templates and output schemas
- [🔍 Source Evaluation](../web-search/references/source-evaluation.md) — Credibility tiers and red flags (shared with web-search)
