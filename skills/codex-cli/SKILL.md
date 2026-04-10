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
  version: "1.1"
---

# Codex CLI Delegation

Delegate coding tasks to the local Codex CLI from inside `jy-agent`.
Keep this skill narrow: it is for Codex-specific delegation, not for every
coding request.

## Timeout Policy: `run_shell` vs `run_background`

Codex CLI can be slow. Choose the right execution mode:

```
How long will this Codex call take?
├─ Preflight (which codex, codex --version)
│   → run_shell(cmd, timeout=60)
│
├─ Quick task — scoped review, small analysis, simple fix (<5 min likely)
│   → run_shell(cmd, timeout=600)
│
├─ Medium task — multi-file analysis, complex implementation
│   → Prefer run_background(cmd), then poll with check_background(pid)
│   → Fallback: try run_shell(timeout=600), switch to background if it times out
│
└─ Heavy task — large repo scan, architecture review, broad refactor
    → Always use run_background(cmd) + check_background(pid) polling loop
```

### Background execution pattern

```python
# Step 1: Start in background (returns instantly)
run_background('codex exec --sandbox read-only "Analyze the full codebase"')
# → {"pid": 12345, "output_file": "/tmp/jyagent_bg_xxx.out", "status": "started"}

# Step 2: Poll progress (repeat until done)
check_background(12345, tail=20)
# → {"pid": 12345, "status": "running", "elapsed_seconds": 45.2, "output": "...last 20 lines..."}

# Step 3: Read final output when done
check_background(12345)
# → {"pid": 12345, "status": "done", "exit_code": 0, "output": "...full output..."}

# If stuck: kill it
check_background(12345, action="kill")
```

### When run_shell times out at 600s

Do NOT just retry the same command. Instead:
1. Switch to `run_background` for the same command, OR
2. Narrow the task scope and try `run_shell` again with a tighter prompt

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
│  → `codex exec --sandbox read-only`
│
├─ Intended code edits inside the repo
│  → `codex exec --sandbox workspace-write`  (or `--full-auto`)
│
└─ Anything that sounds like "just give it full access"
   → Push back unless the environment is already externally sandboxed.
      Least privilege is the default because it narrows blast radius and makes failures clearer.
```

## Core Workflow

### 1. Preflight

Confirm Codex is available before delegating:

```python
run_shell("which codex && codex --version", timeout=60)
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

## ⚠️ Critical CLI Syntax Rules

**The prompt is a POSITIONAL argument, not a flag.** Get this wrong and the
command fails or misinterprets your intent.

```bash
# ✅ CORRECT — prompt is a bare positional arg (quote the string)
codex exec --sandbox read-only "Analyze the auth module"

# ✅ CORRECT — multi-line prompt via stdin heredoc
cat <<'EOF' | codex exec --sandbox read-only -
Goal: Analyze the auth module.
Done when: Return a summary.
EOF

# ❌ WRONG — -p is --profile, NOT prompt
codex exec --sandbox read-only -p "Analyze the auth module"

# ❌ WRONG — -q does not exist
codex exec -q "Analyze the auth module"
```

### Key flags (from `codex exec --help`)

| Flag | Meaning |
|------|---------|
| `-s, --sandbox <MODE>` | `read-only`, `workspace-write`, `danger-full-access` |
| `-C, --cd <DIR>` | Working directory for the agent |
| `-m, --model <MODEL>` | Override model |
| `-p, --profile <NAME>` | Config profile (⚠️ NOT prompt!) |
| `--full-auto` | Alias for `-a on-request --sandbox workspace-write` |
| `--output-schema <FILE>` | JSON Schema for structured output |
| `-o, --output-last-message <FILE>` | Write last agent message to file |
| `--json` | Print JSONL events to stdout |
| `-` (as PROMPT) | Read prompt from stdin |

### Flags that DO NOT exist

Do not hallucinate these: `-q`, `--quiet`, `--bare`, `--max-budget-usd`.
These are Claude Code flags, not Codex flags.

## Core Invocation Patterns

### Pattern A: Read-only analysis or planning

```python
run_shell('codex exec --sandbox read-only -C /path/to/repo "Analyze X and suggest a fix"', timeout=600)
```

For multi-line prompts, use a heredoc:

```python
run_shell("""cat <<'EOF' | codex exec --sandbox read-only -C /path/to/repo -
Goal: Analyze the auth module.
Context: JWT header parsing fails on empty input.
Done when: Return diagnosis and recommended fix.
EOF""", timeout=600)
```

### Pattern B: Implementation

```python
run_shell('codex exec --sandbox workspace-write -C /path/to/repo "Fix the null pointer in parse_config and run tests"', timeout=600)
```

### Pattern C: Review

```python
run_shell('codex review --uncommitted', timeout=600)
# or with a base branch:
run_shell('codex review --base main', timeout=600)
```

### Pattern D: Follow-up on the same Codex task

```python
run_shell('codex exec resume --last "Now add regression tests for the fix"', timeout=600)
```

### Pattern E: Structured output

```python
run_shell('codex exec --sandbox read-only --output-schema /tmp/schema.json "Review the auth layer"', timeout=600)
```

### Pattern F: Background execution (for slow tasks)

Use `run_background` + `check_background` when the task may exceed 600 seconds.
This is the **preferred pattern for complex Codex tasks**.

```python
# Start — returns instantly with PID
run_background('codex exec --sandbox read-only -C /path/to/repo "Comprehensive architecture review"')
# → {"pid": 12345, "output_file": "/tmp/jyagent_bg_xxx.out", "status": "started"}

# Poll every 30-60s — use tail to avoid output flood
check_background(12345, tail=30)
# → {"status": "running", "elapsed_seconds": 120.5, "output": "...last 30 lines..."}

# When status == "done", read full output
check_background(12345)
# → {"status": "done", "exit_code": 0, "output": "...complete result..."}
```

For implementation tasks in background:
```python
run_background('codex exec --full-auto -C /path/to/repo "Refactor the auth module to use JWT"')
```

## Anti-Patterns

❌ **Don't** use `-p` to pass the prompt — `-p` is `--profile` (config profile selection)
✅ **Do** pass the prompt as a bare positional argument: `codex exec --sandbox read-only "your prompt"`

❌ **Don't** hallucinate Claude Code flags (`-q`, `--quiet`, `--bare`, `--max-budget-usd`) — they don't exist in Codex
✅ **Do** check `codex exec --help` if unsure about a flag

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

❌ **Don't** retry `run_shell(timeout=600)` when it times out — the task is too slow for synchronous execution
✅ **Do** switch to `run_background` + `check_background` polling for slow tasks

❌ **Don't** poll `check_background` with `tail=0` every few seconds on a running process — it floods context with repeated full output
✅ **Do** use `tail=20` or `tail=30` while polling; only use `tail=0` for the final read after status is "done"

## Reference Files

- [Command Patterns](references/command-patterns.md) — safe shell recipes for preflight, analysis, implementation, review, resume, and structured output
