# Eval-Driven Skill Iteration

Source: https://agentskills.io/skill-creation/evaluating-skills.md

You wrote a skill, tried it on a prompt, it seemed to work. But does it work
*reliably*, across varied prompts, in edge cases, *better than no skill at all*?
Structured evals answer that question and give you a feedback loop.

## Test-Case Format (`evals/evals.json`)

```json
{
  "skill_name": "csv-analyzer",
  "evals": [
    {
      "id": 1,
      "prompt": "I have a CSV of monthly sales data in data/sales_2026.csv. Find the top 3 months by revenue and make a bar chart.",
      "expected_output": "A bar chart image showing the top 3 months by revenue, with labeled axes and values.",
      "files": ["evals/files/sales_2026.csv"]
    },
    {
      "id": 2,
      "prompt": "there's a csv in my downloads called customers.csv, some rows have missing emails — can you clean it up and tell me how many were missing?",
      "expected_output": "A cleaned CSV with missing emails handled, plus a count of how many were missing.",
      "files": ["evals/files/customers.csv"]
    }
  ]
}
```

**Tips for writing test prompts:**

- Start with **2-3 cases**. Don't over-invest before seeing the first results.
- **Vary phrasing**: casual ("hey can you clean up this csv") vs precise
  ("Parse the CSV at data/input.csv, drop rows where column B is null").
- **Cover edge cases**: malformed input, unusual requests, ambiguous instructions.
- **Use realistic context**: file paths, column names, personal asides. "Process
  this data" is too vague to test anything useful.
- Add **assertions later**, after you see what the first runs produce. You
  often don't know what "good" looks like until the skill has run once.

## Workspace Structure

Keep the skill repo clean by writing results to a sibling workspace.
Each iteration through the eval loop gets its own directory.

```
csv-analyzer/
├── SKILL.md
└── evals/
    ├── evals.json
    └── files/
        ├── sales_2026.csv
        └── customers.csv

csv-analyzer-workspace/
└── iteration-1/
    ├── eval-top-months-chart/
    │   ├── with_skill/
    │   │   ├── outputs/       # files produced by the run
    │   │   ├── timing.json    # {total_tokens, duration_ms}
    │   │   └── grading.json   # assertion results
    │   └── without_skill/
    │       ├── outputs/
    │       ├── timing.json
    │       └── grading.json
    ├── eval-clean-missing-emails/
    │   ├── with_skill/  …
    │   └── without_skill/ …
    └── benchmark.json            # aggregated stats
```

Only `evals/evals.json` is hand-authored. Everything else is produced by the
eval runs.

## The Core Eval Pattern

Run **each test case twice**: once **with the skill** and once **without it**
(or with the previous version of the skill). The baseline is what makes the
data meaningful — a skill that produces "good" output is uninteresting if
the base model produced equally good output without it.

Spawn each run with a **clean context** (a subagent in Claude Code, a fresh
session elsewhere). For each run, provide:

- Skill path (or none, for the baseline)
- The test prompt
- Input files
- The output directory

Example task spec for a single with-skill run:

```
Execute this task:
- Skill path: /path/to/csv-analyzer
- Task: I have a CSV of monthly sales data in data/sales_2026.csv. Find the top 3 months by revenue and make a bar chart.
- Input files: evals/files/sales_2026.csv
- Save outputs to: csv-analyzer-workspace/iteration-1/eval-top-months-chart/with_skill/outputs/
```

When improving an existing skill, snapshot the old version
(`cp -r ./csv-analyzer ./csv-analyzer-snapshot`), use it as the baseline,
and save those outputs to `old_skill/outputs/` instead of `without_skill/`.

### Capture Timing Per Run

```json
{ "total_tokens": 84852, "duration_ms": 23332 }
```

Token-cost matters. A skill that improves quality but triples token usage
is a different trade-off than one that's both better and cheaper.

## Assertions

Add after the first run, once you've seen real output. Assertions are
verifiable statements about the output.

**Good assertions:**
- `"The output file is valid JSON"` — programmatically verifiable.
- `"The bar chart has labeled axes"` — specific and observable.
- `"The report includes at least 3 recommendations"` — countable.

**Weak assertions:**
- `"The output is good"` — too vague to grade.
- `"The output uses exactly the phrase 'Total Revenue: $X'"` — too brittle.

Not every quality needs an assertion. Writing style, visual design, and
"feel" are often better judged by a grader agent (see `agents/grader.md`).

## Trigger-Rate Testing (separate from output evals)

For **description optimization**, you also test triggering: does the agent
*decide to load* the skill on relevant prompts and *decline to load* it on
irrelevant ones?

- Build a **trigger eval set**: ~20 queries, 8-10 should-trigger and 8-10
  should-not-trigger.
- Include **near-misses** in the negatives — prompts that share keywords
  with the skill but actually need something different.
- Model behavior is nondeterministic: run each query **3+ times** and
  compute a *trigger rate*, not a binary pass/fail.

Our `scripts/test_trigger.py` automates this.

## Iteration Loop

```
1. Run all test cases with-skill AND without-skill → capture outputs
2. Grade outputs (assertions + grader agent)
3. Read execution traces — where did the agent waste time? get confused?
4. Generalize fixes (don't add narrow patches for one example)
5. Keep the skill lean — if pass rates plateau while adding rules, the
   skill may be over-constrained. Try removing instructions.
6. Re-run. Stop when iteration-N matches iteration-(N-1) on all metrics.
```
