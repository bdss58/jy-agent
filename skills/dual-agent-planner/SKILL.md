---
name: dual-agent-planner
description: >-
  Orchestrate Claude Code and Codex CLI together for complex tasks. Use this
  skill whenever the user asks to plan a complex task, architect a solution,
  compare approaches, or wants both agents' perspectives before implementing.
  TRIGGER on: "plan this", "how should we approach", "architect this",
  "design the solution", "both agents", "dual agent", "orchestrate",
  "compare plans", "get both opinions", "consult both", "plan and implement",
  "complex task planning", "multi-agent", "what's the best approach for",
  or any complex task that benefits from multiple perspectives before coding.
  Also triggers on substantial feature requests, migrations, refactors, or
  architecture decisions where the user hasn't specified a single tool.
  DO NOT TRIGGER on: simple edits, trivial bug fixes, questions about a
  specific tool's config, web search, browser tasks, or when the user
  explicitly asks for just one agent.
metadata:
  author: jy-agent
  version: "2.0"
---

# Dual-Agent Planner

Get plans from both Claude Code and Codex CLI, compare their strengths,
synthesize the best approach, then orchestrate execution. Think of yourself
as a tech lead receiving proposals from two senior engineers.

## Decision Tree: When to Use Dual Planning

```
User has a task →
├─ Trivial? (typo, config, one-liner)
│   → Skip. Do it yourself with edit_file.
│
├─ Simple and clear? (single file, obvious approach)
│   → Pick one agent. Claude Code for Anthropic-model tasks,
│     Codex for OpenAI-model tasks, or user's preference.
│
├─ Complex but well-defined? (multi-file refactor, new feature)
│   → Consider dual planning if time allows.
│   → If time-pressed, pick the better-fit agent.
│
├─ Complex AND ambiguous? (architecture decision, migration strategy,
│   design tradeoffs, multiple valid approaches)
│   → ★ DUAL PLAN. This is the sweet spot.
│
└─ User explicitly asks for both opinions?
    → ★ DUAL PLAN. Always honor the request.
```

## Core Workflow

### Phase 1: Frame the Problem

Before consulting either agent, write a clear problem statement:

```
CONTEXT: [Project, codebase, current state]
GOAL: [What needs to change]
CONSTRAINTS: [Budget, time, compatibility, don't-touch zones]
SUCCESS CRITERIA: [How to verify the result]
```

This becomes the shared prompt for both agents.

### Phase 2: Collect Plans (Parallel)

Run both agents in plan-only mode. Use `&` to run them simultaneously:

```bash
# Claude Code — plan only
claude -p --model sonnet --bare \
  "PLAN ONLY — do not implement. Analyze this task and produce a detailed
   execution plan with: (1) approach summary, (2) files to change,
   (3) step sequence, (4) risks/tradeoffs, (5) testing strategy.

   TASK: <problem statement>" > /tmp/plan_claude.txt 2>&1 &

# Codex — plan only
codex exec --sandbox read-only \
  "PLAN ONLY — do not implement. Analyze this task and produce a detailed
   execution plan with: (1) approach summary, (2) files to change,
   (3) step sequence, (4) risks/tradeoffs, (5) testing strategy.

   TASK: <problem statement>" > /tmp/plan_codex.txt 2>&1 &

wait  # Wait for both to finish
```

**Fallback**: If one agent fails (timeout, rate limit, not installed), proceed
with the other agent's plan alone. A single informed plan is better than
blocking. Note the failure to the user.

### Phase 3: Compare & Synthesize

Present both plans and analyze divergence:

```
## Where They Agree (high confidence — proceed boldly)
- ...

## Where They Differ (needs your judgment)
| Aspect | Claude Code | Codex | My Recommendation |
|--------|------------|-------|-------------------|
| ...    | ...        | ...   | ...               |

## Synthesized Plan
1. ...
```

The "My Recommendation" column is key — don't just present options, make a call
and explain why. You're the tech lead, not a neutral reporter.

### Phase 4: Execute

Assign to the best-fit agent:

| Task characteristic | Best agent | Why |
|---|---|---|
| Deep multi-file refactor | Claude Code | Stronger at cross-file reasoning |
| Needs sandbox safety | Codex | OS-level sandbox (Seatbelt/Landlock) |
| CI/CD integration | Codex | Native `codex exec`, GitHub Action |
| Security audit / review | Claude Code | `--allowedTools "Read"` is precise |
| Quick / cost-sensitive | Codex | `--profile fast` + mini model |
| Hard reasoning | Claude Code | Opus model for toughest problems |

Or split work: one implements backend, other does frontend. One codes, other tests.

### Phase 5: Cross-Review (Optional but Powerful)

Use the OTHER agent to review the implementation:

```bash
# Codex implemented → Claude Code reviews
claude -p --model sonnet --bare --allowedTools "Read" \
  "Review the changes in the last commit against this plan: <plan>. 
   Check for bugs, edge cases, deviations."

# Claude Code implemented → Codex reviews
codex exec --sandbox read-only \
  "Review the changes since last commit against this plan: <plan>"
```

This catches blind spots — each model has different failure modes.

## Error Handling

```
Agent call fails →
├─ Timeout (>120s)
│   → Retry once with a simpler prompt. If still fails, proceed with other agent.
│
├─ Rate limit (429)
│   → Wait 60s and retry. If persistent, proceed with other agent.
│
├─ Agent not installed
│   → Tell the user. Fall back to the available agent.
│   → Check: `which claude` / `which codex`
│
├─ One plan is clearly low quality
│   → Discard it. Note why. Use the better plan.
│
└─ Both agents fail
    → Fall back to your own analysis. You've read the codebase.
    → Be transparent: "Both sub-agents failed, here's my own analysis..."
```

## Cost Awareness

Dual planning costs ~2x a single agent. Worth it when:
- **High stakes**: wrong approach = days of wasted work
- **Genuinely ambiguous**: you can't tell which approach is better
- **Architecture decisions**: hard to reverse later

Not worth it when:
- Clear best approach already exists
- Time pressure > quality need
- Task is well-scoped and mechanical

To control costs:
- Use `--model sonnet` (not opus) for planning phase
- Use `--max-budget-usd 0.50` per agent for planning
- Save opus/xhigh reasoning for execution of the hardest subtask only

## Anti-Patterns

❌ **Don't** dual-plan trivial tasks — it's overkill for a typo fix
✅ **Do** reserve dual planning for complex, ambiguous, or high-stakes work

❌ **Don't** present both plans without a recommendation — that pushes the decision back to the user
✅ **Do** synthesize and make a call: "I recommend X because..."

❌ **Don't** blindly merge both plans — they may have contradictory assumptions
✅ **Do** identify consensus (high confidence) vs divergence (needs judgment)

❌ **Don't** let both agents execute on the same files simultaneously
✅ **Do** use separate branches or split files clearly between agents

❌ **Don't** block if one agent fails — the other's plan is still valuable
✅ **Do** fall back gracefully and note the failure

❌ **Don't** skip the problem framing step — garbage in, garbage out
✅ **Do** write a clear problem statement before consulting either agent

❌ **Don't** ignore consensus — if both agents agree, that's high confidence
✅ **Do** focus your judgment on the points where they diverge

## Reference Files

- [📋 Strength Matrix](references/strength-matrix.md) — Detailed Claude Code vs Codex comparison
- [📋 Prompt Templates](references/prompt-templates.md) — Ready-to-use templates for architecture, migration, features
- [📋 Orchestration Patterns](references/orchestration-patterns.md) — Consult→Execute, Split, Cross-Review, Competitive
- [📋 Worked Example](references/worked-example.md) — Full end-to-end walkthrough of dual planning
