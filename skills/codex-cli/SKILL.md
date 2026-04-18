---
name: codex-cli
description: >-
  Delegate tasks to Codex CLI — coding, code review, image/vision analysis, and
  structured output. Use this skill whenever the user explicitly asks for Codex,
  Codex CLI, `codex exec`, `codex review`, or a Codex-specific workflow such as
  "delegate to Codex", "review with Codex", or "use Codex to analyze this repo".
  Also TRIGGER on image/vision tasks: "analyze this image", "describe this
  screenshot", "what's in this picture", "read this diagram/chart", "compare
  these screenshots" — because jy-agent has no native vision and Codex provides
  it via the `-i` flag. TRIGGER on: "use Codex", "Codex CLI", "codex exec",
  "codex review", "delegate to Codex", "analyze/describe image/screenshot",
  "what's in this image", "read this chart/diagram". DO NOT TRIGGER on: generic
  coding tasks without Codex mention, trivial direct edits, or requests that
  explicitly ask for Claude or another agent.
metadata:
  author: jy-agent
  version: "2.0"
---

# Codex CLI Delegation

Delegate tasks to the local Codex CLI from inside `jy-agent`.
Codex handles coding, code review, **image/vision analysis**, and structured
output with JSON schema enforcement.

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
# Step 1: Start with a deadline so a runaway task gets auto-killed
run_background(
    'codex exec --sandbox read-only "Analyze the full codebase"',
    timeout_seconds=1200,   # 20 min hard cap
)
# → {"pid": 12345, "output_file": "/tmp/jyagent_bg_xxx.out", "status": "started"}

# Step 2: Poll progress — ALWAYS use tail>0 while the job is running.
# tail=0 returns the last ~50 KB every poll and floods your context.
check_background(12345, tail=30)
# → {"pid": 12345, "status": "running", "elapsed_seconds": 45.2,
#    "output": "...", "deadline_seconds_remaining": 1154.8}

# Step 3: Prefer `wait` over tight polling — it saves a model turn per poll.
check_background(12345, action="wait", wait_timeout_seconds=120)
# Blocks up to 120 s (hard cap 300); returns early when the job finishes.

# Step 4: Final collection — tail=0 only AFTER the job has finished
check_background(12345)
# → status values: running | succeeded | failed | killed | timed_out
#   ALWAYS inspect exit_code — "succeeded" is exit-0, not task success.

# If you SUSPECT it's stuck, read more output before killing:
check_background(12345, tail=50)
# Only kill after confirming it's truly looping or idle across polls:
check_background(12345, action="kill")
# (kill is a no-op if the process already exited; status reflects reality.)
```

**Concurrency cap**: at most 8 live background jobs. `run_background` will
return `{"status":"rejected","reason":"concurrency_cap"}` if you exceed it.

### Diagnosing "stuck" Codex processes

Codex research/analysis tasks routinely take 3-10 minutes. Don't assume
a running process is stuck just because it's been going for a few minutes.

```
Codex still running after N seconds →
├─ N < 300s (5 min) — Probably fine. Poll with tail=20, be patient.
│
├─ N = 300-600s — Read a larger window: check_background(pid, tail=50)
│   ├─ Output shows progress (new content each poll) → Let it run
│   ├─ Output is identical across 2-3 polls → Likely stuck, kill it
│   └─ Output shows repeated identical tool calls → Stuck in a loop, kill it
│
└─ N > 600s — Read full output (tail=0), extract whatever it produced,
    then kill and use the partial results.
```

**Key rule**: Always read substantial output (`tail=50`) before deciding
to kill. Never judge from `tail=3` alone — that's not enough to see patterns.

### When run_shell times out at 600s

Do NOT just retry the same command. Instead:
1. Switch to `run_background` for the same command, OR
2. Narrow the task scope and try `run_shell` again with a tighter prompt

## Decision Tree: Should Codex Handle This?

```
User wants help →
├─ They explicitly ask for Codex / Codex CLI / codex exec / codex review
│  → Use this skill.
│
├─ They want a code review of repo changes
│  → Prefer `codex review`.
│
├─ They want analysis, planning, implementation, or a follow-up Codex turn
│  → Use `codex exec`.
│
├─ They ask about an image, screenshot, diagram, or chart
│  → Use `codex exec -i <file>` (jy-agent has NO native vision).
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

**Sandbox scope**: The `--sandbox` flag controls **filesystem access** for
model-generated shell commands only. It does NOT restrict network access,
web search, or any other Codex capability. Don't blame sandbox mode when
Codex has trouble with web searches — the issue is elsewhere.

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

## Vision / Image Analysis

jy-agent has **no native image understanding**. Codex fills this gap via the
`-i <file>` flag, which attaches images to the prompt. Use this for any task
involving visual content.

### When to use

- User says "analyze/describe this image/screenshot/diagram/chart"
- Need to read text from an image (OCR-like)
- Compare two UI screenshots for differences
- Extract data from a chart or table image
- Verify visual changes after CSS/UI edits

### Patterns

```bash
# Describe an image
echo "Describe this image in detail." | codex exec --sandbox read-only -i /path/to/image.png -

# Analyze a screenshot for UI issues
echo "Review this UI screenshot. Identify layout issues, accessibility problems, or visual bugs." | codex exec --sandbox read-only -i screenshot.png -

# Read text / extract data from an image
echo "Extract all visible text from this image." | codex exec --sandbox read-only -i photo.jpg -

# Compare two screenshots
echo "Compare these two screenshots and describe what changed." | codex exec --sandbox read-only -i before.png -i after.png -

# Extract structured data from a chart image
echo "Extract the data from this chart as a markdown table." | codex exec --sandbox read-only --output-schema /tmp/schema.json -i chart.png -
```

### Image flag gotchas

- `-i` must come **before** the prompt argument or use stdin (`-`)
- Multiple `-i` flags for multiple images: `-i img1.png -i img2.png`
- Supports PNG, JPG, JPEG, GIF, WebP
- Large images work but increase token cost; resize if possible

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

❌ **Don't** kill a Codex process after checking only `tail=3` and assuming it's stuck — 3 lines is not enough to diagnose anything
✅ **Do** read `tail=50` before deciding to kill; compare output across 2-3 polls to detect real loops vs. normal progress

❌ **Don't** blame sandbox mode for web search or network issues — sandbox only restricts filesystem access for shell commands
✅ **Do** investigate the actual cause (e.g., model looping, prompt issue, service outage) and don't fabricate explanations

## Reference Files

- [Command Patterns](references/command-patterns.md) — safe shell recipes for preflight, analysis, implementation, review, resume, and structured output
