# Source Evaluation Guide

How to assess whether a search result is trustworthy and useful.

## Quick Credibility Check (30 seconds)

```
Source found → Check these in order:
├─ 1. Domain: Is it authoritative?
│   ├─ Official docs (*.readthedocs.io, docs.*.com) → High trust
│   ├─ Known tech sites (SO, GitHub, HN, MDN) → High trust
│   ├─ Major publications (BBC, Reuters, NYT) → High trust
│   ├─ Personal blogs → Medium trust (check author credentials)
│   ├─ Content farms (w3schools, geeksforgeeks) → Low trust (verify elsewhere)
│   └─ Unknown domain → Low trust (cross-reference required)
│
├─ 2. Date: Is it current enough?
│   ├─ Software/API docs → Must be within 1-2 years (APIs change fast)
│   ├─ News/events → Must be recent (days/weeks)
│   ├─ Concepts/theory → Older is OK (fundamentals don't change)
│   └─ No date shown → Suspect (may be outdated)
│
├─ 3. Specificity: Does it actually answer the question?
│   ├─ Directly addresses the query → Use it
│   ├─ Tangentially related → Extract specific facts only
│   └─ Generic/filler content → Skip
│
└─ 4. Agreement: Do other sources confirm?
    ├─ 2+ sources agree → High confidence
    ├─ Sources conflict → Note the disagreement, present both sides
    └─ Only 1 source → Qualify with "according to [source]"
```

## Domain Tier List

### Tier 1: High Trust (use directly)
- **Official docs**: python.org, docs.docker.com, kubernetes.io, developer.mozilla.org
- **Primary sources**: GitHub repos, RFCs, academic papers
- **Reputable tech**: stackoverflow.com (high-voted answers), news.ycombinator.com
- **Major news**: Reuters, AP, BBC, NYT, 新华社, 人民日报

### Tier 2: Good but Verify
- **Tech blogs**: medium.com (varies by author), dev.to, personal engineering blogs
- **Aggregators**: InfoQ, The New Stack, 36kr, CSDN (top articles)
- **Company blogs**: AWS/GCP/Azure blogs (may be biased toward own products)

### Tier 3: Low Trust (always cross-reference)
- **Content farms**: w3schools, tutorialspoint, geeksforgeeks (often shallow/outdated)
- **SEO-optimized**: Sites that seem to exist purely for ad revenue
- **Anonymous**: No author, no date, no credentials shown
- **AI-generated**: Increasingly common; may contain hallucinations

## Red Flags
- 🚩 Article date doesn't match content (recycled/updated old articles)
- 🚩 "Top 10 best X in 2025" published in 2023 (SEO title manipulation)
- 🚩 No code examples for a technical claim
- 🚩 Contradicts official documentation
- 🚩 Single source making an extraordinary claim
- 🚩 Affiliate links everywhere (biased recommendations)

## How to Present Findings

### High confidence (2+ reliable sources agree)
> "X works this way [source1] [source2]"

### Medium confidence (single good source)
> "According to [source], X works this way"

### Low confidence (conflicting or weak sources)
> "Sources disagree on this. [Source A] says X, while [Source B] says Y. The official docs don't address this directly."

### Unknown (couldn't verify)
> "I couldn't find reliable information on this. The closest I found was [source], but it's [old/unverified/tangential]."
