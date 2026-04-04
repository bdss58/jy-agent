---
name: codex-cli
description: >-
  Delegate coding work to Codex CLI, the local Codex command-line agent. Use this skill
  whenever the user explicitly asks for Codex, Codex CLI, `codex exec`,
  `codex review`, or a Codex-specific workflow such as "delegate to Codex",
  "review with Codex", or "use Codex to analyze this repo". Also use it when
  jy-agent has already chosen Codex for a follow-up and should continue that
  same Codex thread. TRIGGER on: "use Codex", "Codex CLI", "run codex exec",
  "run codex review", "delegate to Codex", "review with Codex", "continue the
  Codex session". DO NOT TRIGGER on: generic coding tasks, trivial direct
  edits, non-programming information lookups, or requests that explicitly ask for Claude or
  another agent.
metadata:
  author: jy-agent
  version: "1.0"
---

# Codex CLI Delegation

Delegate coding tasks to the local Codex CLI from inside `jy-agent`.
Keep this skill narrow: it is for Codex-specific delegation, not for every
coding request.

## Decision Tree: Should Codex Handle This?

```
User wants help with code →
├─ They explicitly ask for Codex / Codex CLI / codex exec / codex review
│  → Use this skill.
│
├─ They want a code review of repo changes
│  → Prefer `codex review`.
│
├─ They want analysis, planning, implementation, or a follow-up Codex turn
│  → Use `codex exec`.
│
├─ The task is trivial and already clear to jy-agent
│  → Do it yourself. Spinning up Codex adds latency without much gain.
│
└─ The request is a broad coding task but does not mention Codex
   → Do not force this skill. Let a broader coding skill or direct tool use handle it.
```

## Decision Tree: `codex review` vs `codex exec`

```
What does the user need?
├─ Review staged / unstaged / untracked changes
│  → `codex review --uncommitted`
│
├─ Review relative to a branch
│  → `codex review --base <branch>`
│
├─ Review a specific commit
│  → `codex review --commit <sha>`
│
├─ Analyze, plan, refactor, fix, implement, or explain code
│  → `codex exec`
│
└─ Continue the same Codex task with new instructions
   → `codex exec resume --last`
```

## Decision Tree: Read-Only vs Workspace-Write

```
How much access should Codex get?
├─ Repo exploration, diagnosis, planning, review, or read-only analysis
│  → `codex exec --ask-for-approval never --sandbox read-only`
│
├─ Intended code edits inside the repo
│  → `codex exec --ask-for-approval never --sandbox workspace-write`
│
└─ Anything that sounds like "just give it full access"
   → Push back unless the environment is already externally sandboxed.
      Least privilege is the default because it narrows blast radius and makes failures clearer.
```

## Core Workflow

### 1. Preflight

Confirm Codex is available before delegating:

```python
run_shell("which codex && codex --version")
```

If Codex is missing or not authenticated enough to run, fail fast and say so.
Do not pretend the delegation happened.

### 2. Build a good Codex prompt

Use this shape for most non-trivial tasks:

```text
Goal: [the outcome Codex should produce]
Context: [repo facts, relevant files, prior decisions, error text]
Constraints: [style rules, files to avoid, compatibility limits, validation rules]
Done when: [tests/checks or concrete finish criteria]
```

Why this works: Codex performs better when the task boundary and completion
criteria are explicit. Put durable repo-wide rules in `AGENTS.md` instead of
repeating them in every prompt.

### 3. Pick the right command

- Use `codex review` for reviewing changes.
- Use `codex exec` for analysis, planning, implementation, and follow-up work.
- Prefer piping a multi-line prompt via stdin or heredoc. It avoids shell-quoting
  bugs and keeps prompts readable.

Detailed command recipes live in
[references/command-patterns.md](references/command-patterns.md).

### 4. Verify and report

When using `codex exec` for implementation, tell Codex exactly how to verify the
result in the `Done when` section. After it returns, summarize:

- what Codex changed or found
- what verification it claims to have run
- any open risks or follow-up needed

## Core Invocation Patterns

### Pattern A: Read-only analysis or planning

Use `codex exec` in read-only mode when the user wants investigation, design,
or the smallest viable fix before any file changes.

### Pattern B: Implementation

Use `codex exec` in `workspace-write` mode when edits are intended. Keep the
prompt scoped to the exact change and include concrete verification.

### Pattern C: Review

Use `codex review` when the user wants findings about uncommitted work, a branch
diff, or a commit. This is the cleanest match for review-style tasks.

### Pattern D: Follow-up on the same Codex task

Use `codex exec resume --last` when the user wants to continue or refine the
most recent Codex run instead of starting from scratch.

### Pattern E: Structured output

If jy-agent needs machine-readable results, provide a JSON Schema file and use
`codex exec --output-schema <file>`. This works well for issue lists, migration
plans, or other outputs jy-agent will post-process.

## Anti-Patterns

❌ **Don't** activate this skill for every coding request
✅ **Do** keep it Codex-specific so it does not overlap with `claude-code`

❌ **Don't** use `workspace-write` for tasks that only need reading or planning
✅ **Do** start with `read-only` unless edits are actually intended

❌ **Don't** give Codex a vague prompt like "fix this project"
✅ **Do** define `Goal`, `Context`, `Constraints`, and `Done when`

❌ **Don't** restate stable repo policy in every Codex prompt
✅ **Do** keep durable instructions in `AGENTS.md` and task-specific instructions in the prompt

❌ **Don't** pass long multi-line prompts through brittle inline quoting
✅ **Do** use stdin or a heredoc so the prompt stays readable and exact

❌ **Don't** claim delegation succeeded if `codex` is unavailable
✅ **Do** fail fast, report the environment problem, and stop

## Reference Files

- [Command Patterns](references/command-patterns.md) — safe shell recipes for preflight, analysis, implementation, review, resume, and structured output
