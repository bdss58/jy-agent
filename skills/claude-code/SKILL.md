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

## Timeout Policy: `run_shell` vs `run_background`

Claude Code can be slow on complex tasks. Choose the right execution mode:

```
How long will this Claude Code call take?
├─ Preflight (which claude, claude --version)
│   → run_shell(cmd, timeout=60)
│
├─ Quick task — scoped review, simple fix, focused refactor (<5 min likely)
│   → run_shell(cmd, timeout=600)
│
├─ Medium/heavy task — multi-file refactor, large codebase analysis, broad review
│   → Use run_background(cmd) + check_background(pid) polling
│   → Fallback: try run_shell(timeout=600) first, switch to background on timeout
```

### Background execution for Claude Code

```python
# Start with a deadline so a runaway job gets auto-killed
run_background(
    'claude -p --bare "Comprehensive refactor of the auth module"',
    timeout_seconds=1800,   # 30 min hard cap
)
# → {"pid": 12345, "output_file": "/tmp/jyagent_bg_xxx.out", "status": "started"}

# Poll progress — ALWAYS use tail=20..50 while the job is running
# (tail=0 floods your context with the whole log every poll)
check_background(12345, tail=30)
# → {"status": "running", "elapsed_seconds": 180.5, "output": "...",
#    "deadline_seconds_remaining": 1619.5}

# Prefer `wait` over tight polling loops — each poll costs a model turn
check_background(12345, action="wait", wait_timeout_seconds=120)
# Blocks up to 120 s waiting for exit, then returns status.

# Read full output once the job is finished
check_background(12345)
# → {"status": "succeeded", "exit_code": 0, "output": "..."}
#   Status values: running | succeeded | failed | killed | timed_out
#   ALWAYS check exit_code — "succeeded" just means exit-0, not that the
#   task actually did what you wanted.
```

**Concurrency cap**: at most 8 live background jobs. If you hit the cap,
`run_background` returns `{"status":"rejected","reason":"concurrency_cap"}` —
kill or wait for existing jobs before launching more.

When `run_shell(timeout=600)` times out, do NOT retry the same command.
Switch to `run_background` or narrow the task scope.

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
echo "TASK_DESCRIPTION" | claude -p --bare
```

- From `jy-agent`, invoke this as `run_shell("<claude command>", timeout=600)`.
- `--bare` skips hooks, plugins, CLAUDE.md auto-discovery → fast, deterministic
- Let the configured default model apply unless the task explicitly needs an override
- Pipe the task via stdin for multi-line prompts

### Pattern B: Scoped Task (with tool permissions)

```bash
claude -p \
  --allowedTools "Read" "Edit" "Write" "Bash(npm test)" \
  "Refactor the auth middleware in src/auth.py to use JWT. Run tests after."
```

- From `jy-agent`, invoke this as `run_shell("<claude command>", timeout=600)`.
- `--allowedTools` auto-approves listed tools, blocks everything else
- Always include a **verification step** in the prompt ("run tests", "check types")

### Pattern C: Structured Output

```bash
claude -p --output-format json \
  --json-schema '{"type":"object","properties":{"summary":{"type":"string"},"files_changed":{"type":"array","items":{"type":"string"}}}}' \
  "Review src/main.py and summarize issues"
```

- From `jy-agent`, invoke this as `run_shell("<claude command>", timeout=600)`.
- Use when you need to parse Claude Code's output programmatically

### Pattern D: Continuation

```bash
# First call
claude -p "implement the user login feature"

# Follow-up (continues same session)
claude -p --continue "now add rate limiting to the login endpoint"
```

- From `jy-agent`, invoke each call as `run_shell("<claude command>", timeout=600)`.

### Pattern E: Fan-Out (parallel bulk tasks)

```bash
# Generate task list, then parallelize
cat task_list.txt | xargs -P 4 -I {} claude -p --bare "{}"
```

- From `jy-agent`, invoke the orchestration shell command as `run_shell("<claude command>", timeout=600)`.
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

Use the Claude Code model configured in your local tool config by default.
Only pass an explicit model override when the task genuinely needs a different tier.

| Model | When to use | Cost | Speed |
|-------|-------------|------|-------|
| `haiku` | Trivial lookups, formatting, grep-like tasks | $ | Fast |
| `sonnet` | Most coding tasks, reviews, refactors | $$ | Medium |
| `opus` | Hard problems: complex architecture, subtle bugs, novel algorithms | $$$$ | Slow |

Rule of thumb: rely on config for the normal path. Override to `opus` only when a cheaper/default model is not enough, or to `haiku` for intentionally lightweight tasks.

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

❌ **Don't** use `--dangerously-skip-permissions` outside of sandboxed environments
✅ **Do** use `--allowedTools` to grant specific permissions safely

## Reference Files

- [📋 Advanced Patterns](references/advanced-patterns.md) — Fan-out, agent teams, worktree isolation, review chains
- [🛠️ Flag Reference](references/flag-reference.md) — Complete CLI flag reference with examples
