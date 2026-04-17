# Research Patterns — Detailed Reference

## Subagent Prompt Template (Full Version)

Use this template when constructing subagent tasks. Customize the
bracketed sections for each research thread.

```
RESEARCH THREAD: [Thread Title]

CONTEXT: [Brief description of the overall research question and where
this thread fits in the larger investigation]

GOAL: [Specific objective for this thread — what information to find]

SEARCH STRATEGY (execute in order):
1. Primary search: "[main search query with specific terms]"
2. Secondary search: "[alternative angle or narrower query]"
3. If results are thin: "[fallback query with different terms]"
4. Fetch the top 3-5 most relevant pages from search results
5. For each fetched page, extract only the relevant sections

WHAT TO COLLECT:
- [Specific data point 1 — e.g., performance numbers]
- [Specific data point 2 — e.g., feature lists]
- [Specific data point 3 — e.g., pricing information]
- Publication dates for all sources
- Author/organization credentials where visible

SOURCE QUALITY RULES:
- Prefer: official docs, peer-reviewed papers, reputable tech blogs
- Accept: well-known engineering blogs, conference talks, GitHub READMEs
- Avoid: content farms (w3schools, tutorialspoint), undated articles
- Reject: sources older than [timeframe] unless for historical context

OUTPUT FORMAT (return EXACTLY this structure):

## Findings
- [Bullet point findings with specific data]
- Each finding cites its source: [claim] (Source: [title], [url])

## Sources
| # | Title | URL | Domain | Date | Trust |
|---|-------|-----|--------|------|-------|
| 1 | ... | ... | ... | ... | High/Medium/Low |

## Confidence
- Overall confidence: [High/Medium/Low]
- Reasoning: [why this confidence level]

## Gaps
- [What you searched for but couldn't find]
- [Areas that need further investigation]

BUDGET: Use up to [N] web_search calls and [M] web_fetch calls.
Do NOT stop early if you haven't found enough — exhaust your search budget.
```

## Research Thread Design Patterns

### Pattern 1: Dimension-Per-Thread (Comparisons)

Split by analysis dimension. Each thread compares ALL options on ONE axis.

```
Thread 1: Performance → Compare A, B, C on speed/throughput
Thread 2: Features   → Compare A, B, C on capabilities
Thread 3: Cost       → Compare A, B, C on pricing/licensing
Thread 4: Ecosystem  → Compare A, B, C on community/support
```

**Why**: Produces clean comparison tables. Each thread is fully independent.

### Pattern 2: Option-Per-Thread (Deep Evaluation)

Split by option. Each thread deeply investigates ONE option.

```
Thread 1: Deep dive on Option A (all dimensions)
Thread 2: Deep dive on Option B (all dimensions)
Thread 3: Deep dive on Option C (all dimensions)
Thread 4: Cross-cutting analysis (industry trends, expert opinions)
```

**Why**: Better for complex options where you need deep understanding of each.
Risk: threads may miss cross-cutting comparisons.

### Pattern 3: Temporal-Per-Thread (Current Events)

Split by time period or perspective.

```
Thread 1: Latest developments (past month)
Thread 2: Historical context (past 1-2 years)
Thread 3: Expert analysis and predictions
Thread 4: Stakeholder reactions and positions
```

### Pattern 4: Source-Per-Thread (High-Trust Research)

Split by source type to ensure diverse sourcing.

```
Thread 1: Academic papers and official reports
Thread 2: Industry analyst reports and surveys
Thread 3: Practitioner blogs and case studies
Thread 4: News coverage and public discourse
```

## Synthesis Strategies

### Agreement Scoring

When multiple subagents report on the same claim:

```
3+ subagents with independent sources agree → State as fact with citations
2 subagents agree, 1 silent → State with moderate confidence
Subagents disagree → Present both viewpoints with citations
1 subagent only → Qualify with "according to [source]"
```

### Source Deduplication

Multiple subagents may find the same source. During synthesis:
- Merge duplicate sources (same URL)
- Keep the most detailed extraction
- Note when multiple threads independently found the same source (strengthens credibility)

### Gap Analysis

After synthesis, identify:
- Questions that NO subagent answered → targeted follow-up search
- Answers backed by only 1 low-trust source → verification search
- Contradictions between threads → resolution search

## Output Schema (JSON — for programmatic use)

When producing structured output for downstream processing:

```json
{
  "title": "Research Report: ...",
  "date": "2026-04-17",
  "executive_summary": "...",
  "threads": [
    {
      "id": "performance",
      "title": "Performance Benchmarks",
      "findings": [
        {
          "claim": "vLLM achieves 2.3x throughput vs TGI on A100",
          "confidence": "high",
          "sources": ["https://url1.com", "https://url2.com"]
        }
      ],
      "gaps": ["No data found for H200 GPUs"]
    }
  ],
  "comparison_table": { ... },
  "recommendation": "...",
  "sources": [
    {
      "url": "https://...",
      "title": "...",
      "domain": "...",
      "date": "...",
      "trust_tier": 1,
      "used_in_threads": ["performance", "features"]
    }
  ],
  "limitations": ["..."]
}
```

## Timing Guidelines

| Phase | Expected Duration | Budget |
|-------|-------------------|--------|
| Clarify | 30 seconds | 0 tool calls |
| Plan | 1 minute | 0-1 tool calls |
| Search (parallel) | 3-8 minutes | 3-5 subagents × 5-8 searches each |
| Follow-up | 1-2 minutes | 2-4 targeted searches |
| Synthesize + Write | 2-3 minutes | 0 tool calls |
| **Total** | **5-15 minutes** | **15-50 searches total** |

## Failure Modes & Recovery

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Subagent returns empty | No findings in output | Retry with broader search terms |
| Subagent hallucinates | Claims without URLs | Discard uncited claims |
| Rate limiting | API errors in subagents | Reduce parallelism, add delays |
| All sources are old | Dates > 12 months | Add "after:YYYY" to queries |
| Topic too niche | < 3 sources found | Broaden scope, check academic sources |
| Subagent stuck | Background agent times out | Kill and use partial results |
