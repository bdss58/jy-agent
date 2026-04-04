---
name: claude-code
description: >-
  Delegate coding tasks to Claude Code (Anthropic's CLI agent). Use this skill
  whenever the user asks to write code, refactor, debug, review, or implement
  features — especially multi-file changes, complex refactors, or tasks that
  benefit from a dedicated coding agent. TRIGGER on: "write code", "implement",
  "refactor", "fix this bug", "code review", "add a feature", "migrate",
  "write tests", "delegate to Claude Code", "use Claude Code", or any
  substantial coding task. Also triggers when jy-agent decides a task is better
  handled by a specialized coding agent rather than manual edit_file calls.
  DO NOT TRIGGER on: simple one-line edits, reading files, running shell
  commands, web search, browser automation, or non-coding tasks.
metadata:
  author: jy-agent
  version: "1.0"
---

# Claude Code Delegation

Delegate coding tasks to Claude Code (`claude -p`) as a sub-agent.
jy-agent plans and orchestrates; Claude Code implements.

## Decision Tree: Self-Do vs Delegate

```
User wants a code change →
├─ Trivial? (typo, rename, one-liner, config tweak)
│   → Do it yourself with edit_file. Don't spin up Claude Code.
│
├─ Medium? (single-file logic change, add a function, write a test)
│   → Could go either way. Prefer self if you understand the code well.
│   → Delegate if the file is complex or you'd need multiple iterations.
│
├─ Complex? (multi-file refactor, new feature, migration, architecture change)
│   → Delegate to Claude Code. This is its sweet spot.
│
├─ Review/audit? (find bugs, security review, code quality)
│   → Delegate with read-only tools. Claude Code excels at deep analysis.
│
└─ Bulk/parallel? (rename across 50 files, migrate API calls, batch changes)
    → Fan-out: generate task list, run parallel claude -p instances.
```

## Core Invocation Patterns

### Pattern A: Quick Task (most common)

```bash
echo "TASK_DESCRIPTION" | claude -p --model sonnet --bare
```

- `--bare` skips hooks, plugins, CLAUDE.md auto-discovery → fast, deterministic
- `--model sonnet` for speed/cost balance; use `opus` for hard problems
- Pipe the task via stdin for multi-line prompts

### Pattern B: Scoped Task (with tool permissions)

```bash
claude -p --model sonnet \
  --allowedTools "Read" "Edit" "Write" "Bash(npm test)" \
  "Refactor the auth middleware in src/auth.py to use JWT. Run tests after."
```

- `--allowedTools` auto-approves listed tools, blocks everything else
- Always include a **verification step** in the prompt ("run tests", "check types")

### Pattern C: Budget-Controlled

```bash
claude -p --model sonnet --max-budget-usd 0.50 --max-turns 10 \
  "Fix the failing test in tests/test_api.py"
```

- `--max-budget-usd` caps spending (prevents runaway)
- `--max-turns` limits back-and-forth iterations

### Pattern D: Structured Output

```bash
claude -p --model sonnet --output-format json \
  --json-schema '{"type":"object","properties":{"summary":{"type":"string"},"files_changed":{"type":"array","items":{"type":"string"}}}}' \
  "Review src/main.py and summarize issues"
```

- Use when you need to parse Claude Code's output programmatically

### Pattern E: Continuation

```bash
# First call
claude -p --model sonnet "implement the user login feature"

# Follow-up (continues same session)
claude -p --model sonnet --continue "now add rate limiting to the login endpoint"
```

### Pattern F: Fan-Out (parallel bulk tasks)

```bash
# Generate task list, then parallelize
cat task_list.txt | xargs -P 4 -I {} claude -p --bare --model sonnet "{}"
```

- Use git worktrees if tasks modify overlapping files
- See [references/advanced-patterns.md](references/advanced-patterns.md)

## Prompt Engineering for Delegation

### The Golden Template

```
CONTEXT: [What the project is, relevant architecture decisions]
TASK: [Specific what-to-do, reference exact files/functions]
CONSTRAINTS: [Style rules, don't-touch zones, compatibility requirements]  
VERIFY: [How to confirm it worked — tests, linter, specific behavior check]
```

### Key Principles

1. **Always include verification** — "Run `pytest` after. Fix failures before reporting."
   This is the single biggest quality multiplier (2-3x improvement).

2. **Reference specific files** — "Look at `src/auth.py:L45-L80`" not "the auth code"

3. **Point to patterns** — "Follow the same pattern as `src/handlers/user.py`"

4. **Scope tightly** — Unbounded tasks → poor results. "Fix all bugs" bad. "Fix the null pointer in `parse_config()` when input is empty" good.

5. **Include context Claude Code can't see** — Error messages, user requirements, design decisions that aren't in the code

## Model Selection

| Model | When to use | Cost | Speed |
|-------|-------------|------|-------|
| `haiku` | Trivial lookups, formatting, grep-like tasks | $ | Fast |
| `sonnet` | **Default.** Most coding tasks, reviews, refactors | $$ | Medium |
| `opus` | Hard problems: complex architecture, subtle bugs, novel algorithms | $$$$ | Slow |

Rule of thumb: Start with `sonnet`. Escalate to `opus` only if sonnet fails or the task requires deep reasoning.

## Reading Claude Code Output

When Claude Code returns output via `run_shell`:
- Check exit code (0 = success)
- Look for "I've made the following changes:" pattern
- If `--output-format json`: parse the JSON for `result`, `session_id`, `cost_usd`
- If it failed: read the error, adjust prompt, retry (max 2 retries before changing approach)

## Anti-Patterns

❌ **Don't** delegate trivial edits — spinning up Claude Code for a typo wastes time and tokens
✅ **Do** use `edit_file` directly for simple, well-understood changes

❌ **Don't** give vague unbounded prompts — "improve the codebase" will wander aimlessly
✅ **Do** scope precisely: specific files, specific behavior, specific verification

❌ **Don't** skip `--bare` when calling programmatically — local CLAUDE.md and hooks can cause unexpected behavior
✅ **Do** use `--bare` for deterministic, predictable sub-agent behavior

❌ **Don't** trust output without verification — Claude Code can introduce subtle bugs
✅ **Do** always include a verify step in the prompt, and spot-check the results yourself

❌ **Don't** retry the same failing prompt — if it failed twice, the prompt is the problem
✅ **Do** rewrite with more context, tighter scope, or escalate to a better model

❌ **Don't** forget `--max-budget-usd` for experimental/exploratory tasks
✅ **Do** set a budget cap to prevent cost surprises

❌ **Don't** use `--dangerously-skip-permissions` outside of sandboxed environments
✅ **Do** use `--allowedTools` to grant specific permissions safely

## Reference Files

- [📋 Advanced Patterns](references/advanced-patterns.md) — Fan-out, agent teams, worktree isolation, review chains
- [🛠️ Flag Reference](references/flag-reference.md) — Complete CLI flag reference with examples
