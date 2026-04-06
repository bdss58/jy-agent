# Command Patterns

Concrete shell recipes for delegating from `jy-agent` to local Codex CLI.
Use these patterns as templates and fill in the task-specific prompt content.

## ⚠️ Flag Quick Reference

The **prompt is a positional argument** — never use `-p` for it (`-p` = `--profile`).

```bash
# One-liner (most common from jy-agent)
codex exec --sandbox read-only "Your prompt here"

# Multi-line via heredoc
cat <<'EOF' | codex exec --sandbox read-only -
Your multi-line prompt here.
EOF
```

**Flags that exist:** `-s/--sandbox`, `-C/--cd`, `-m/--model`, `-p/--profile`,
`--full-auto`, `--output-schema`, `-o/--output-last-message`, `--json`, `-i/--image`

**Flags that DO NOT exist (Claude Code only):** `-q`, `--quiet`, `--bare`,
`--max-budget-usd`, `--allowedTools`

## 1. Preflight

Check that Codex exists before delegating:

```bash
which codex && codex --version
```

If this fails, stop and report the environment problem instead of pretending the
task ran.

## 2. Read-only analysis with `codex exec`

Use this for repo exploration, debugging, design, and planning:

```bash
cat <<'EOF' | codex exec \
  --sandbox read-only \
  -C /absolute/path/to/repo \
  -
Goal: Analyze the auth middleware and suggest the smallest safe fix.
Context: The failure happens when the JWT header is missing. Focus on existing patterns in the repo.
Constraints: Do not modify files. Reference the files and functions that matter most.
Done when: Return a concise diagnosis, the likely fix, and the exact files that should change.
EOF
```

## 3. Implementation with `codex exec`

Use this when edits are intended inside the repo:

```bash
cat <<'EOF' | codex exec \
  --sandbox workspace-write \
  -C /absolute/path/to/repo \
  -
Goal: Fix the auth middleware so missing JWT headers return a clean 401.
Context: The current failure path raises an unhandled exception.
Constraints: Touch only the auth middleware and its tests. Follow existing error-handling conventions.
Done when: Update the code, run the targeted tests, and report the exact verification performed.
EOF
```

## 4. Review current changes with `codex review`

Review local staged, unstaged, and untracked changes:

```bash
codex review --uncommitted
```

Review against a branch:

```bash
codex review --base main
```

Review a specific commit:

```bash
codex review --commit <sha>
```

Use a custom prompt when you want the review to emphasize something specific:

```bash
cat <<'EOF' | codex review --uncommitted -
Review these changes for logic errors, missing edge cases, and test gaps.
Report findings in priority order with file references.
EOF
```

## 5. Continue the latest Codex thread

Use this when the user wants to refine or continue the most recent Codex run:

```bash
cat <<'EOF' | codex exec resume --last -
Goal: Continue the previous Codex task and add regression coverage.
Context: Keep the earlier implementation approach unless a test failure proves it wrong.
Constraints: Stay within the same feature area.
Done when: Update the tests and summarize what changed.
EOF
```

## 6. Structured output with `--output-schema`

Use this when jy-agent needs machine-readable results:

```bash
cat <<'EOF' >/tmp/codex-issues-schema.json
{
  "type": "object",
  "properties": {
    "summary": { "type": "string" },
    "issues": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "file": { "type": "string" },
          "line": { "type": "integer" },
          "severity": { "type": "string" },
          "description": { "type": "string" }
        },
        "required": ["file", "severity", "description"]
      }
    }
  },
  "required": ["summary", "issues"]
}
EOF

cat <<'EOF' | codex exec \
  --sandbox read-only \
  --output-schema /tmp/codex-issues-schema.json \
  -C /absolute/path/to/repo \
  -
Goal: Review the auth layer for correctness risks.
Context: Focus on request validation, auth failures, and error handling.
Constraints: Read-only analysis.
Done when: Return a JSON object that matches the schema exactly.
EOF
```

## 7. Prompt quality checklist

Before running Codex, make sure the prompt includes:

- a specific goal, not a vague aspiration
- enough local context to avoid guesswork
- clear constraints on files, behavior, and validation
- a concrete `Done when` section that defines success
