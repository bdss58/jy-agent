# Advanced Patterns

## 1. Writer-Reviewer Chain

Two Claude Code instances: one writes, one reviews.

```bash
# Step 1: Implement
claude -p --model sonnet --bare \
  --allowedTools "Read" "Edit" "Write" "Bash(pytest *)" \
  "Implement feature X in src/feature.py. Run tests after."

# Step 2: Review (read-only, different perspective)
claude -p --model sonnet --bare \
  --allowedTools "Read" "Bash(grep *)" \
  "Review the recent changes in src/feature.py. Check for:
   - Logic errors
   - Missing edge cases  
   - Security issues
   - Style violations
   Report findings as a bullet list."
```

Why this works: the reviewer has a fresh context, unbiased by implementation decisions.

## 2. Fan-Out with Task Lists

For bulk changes across many files:

```bash
# Generate task list (jy-agent does this part)
cat > /tmp/tasks.txt << 'EOF'
Rename all instances of 'old_api' to 'new_api' in src/handlers/user.py
Rename all instances of 'old_api' to 'new_api' in src/handlers/auth.py
Rename all instances of 'old_api' to 'new_api' in src/handlers/billing.py
EOF

# Fan out to parallel Claude Code instances
cat /tmp/tasks.txt | xargs -P 4 -I {} \
  claude -p --bare --model haiku --allowedTools "Read" "Edit" "{}"
```

- Use `haiku` for repetitive mechanical tasks (fast, cheap)
- `-P 4` = 4 parallel workers (adjust based on rate limits)
- Each instance gets its own context — no interference

### When files overlap: use git worktrees

```bash
# Create isolated worktrees for each worker
git worktree add /tmp/worker-1 -b worker-1
git worktree add /tmp/worker-2 -b worker-2

# Run Claude Code in each worktree
cd /tmp/worker-1 && claude -p --bare "task 1" &
cd /tmp/worker-2 && claude -p --bare "task 2" &
wait

# Merge results
git merge worker-1 worker-2
git worktree remove /tmp/worker-1
git worktree remove /tmp/worker-2
```

## 3. Explore-Plan-Implement Pipeline

Three-phase delegation for complex tasks:

```bash
# Phase 1: Explore (cheap, read-only)
ANALYSIS=$(claude -p --model haiku --bare \
  --allowedTools "Read" "Bash(find *)" "Bash(grep *)" \
  "Analyze the codebase structure for auth-related code. 
   List all files, key functions, and dependencies.")

# Phase 2: Plan (medium, uses exploration output)  
PLAN=$(echo "$ANALYSIS" | claude -p --model sonnet --bare \
  "Based on this analysis, create a detailed plan to migrate 
   from session-based auth to JWT. Output as numbered steps.")

# Phase 3: Implement (full access, follows plan)
echo "$PLAN" | claude -p --model sonnet --bare \
  --allowedTools "Read" "Edit" "Write" "Bash(pytest *)" \
  "Execute this plan step by step. Run tests after each major change."
```

## 4. Incremental Continuation

For tasks that exceed a single context window:

```bash
# First chunk
claude -p --model sonnet \
  "Migrate files in src/handlers/ from REST to GraphQL. Start with user.py and auth.py."

# Continue where it left off
claude -p --model sonnet --continue \
  "Continue the migration. Do billing.py and orders.py next."

# Final chunk
claude -p --model sonnet --continue \
  "Finish migration: update router, tests, and docs. Run full test suite."
```

## 5. Structured Extraction + Action

Use Claude Code to analyze, return structured data, then act on it:

```bash
# Step 1: Analyze and return structured findings
ISSUES=$(claude -p --model sonnet --bare --output-format json \
  --json-schema '{
    "type": "object",
    "properties": {
      "issues": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "file": {"type": "string"},
            "line": {"type": "integer"},
            "severity": {"type": "string", "enum": ["critical","warning","info"]},
            "description": {"type": "string"}
          }
        }
      }
    }
  }' \
  "Scan src/ for security vulnerabilities. Return structured findings.")

# Step 2: jy-agent processes the JSON, prioritizes, delegates fixes
echo "$ISSUES" | jq '.issues[] | select(.severity == "critical")' | ...
```

## 6. Custom Subagents via .claude/agents/

For repeated specialized tasks, define reusable agents:

```yaml
# .claude/agents/security-reviewer.md
---
description: "Security-focused code reviewer"
model: sonnet
tools:
  - Read
  - Bash(grep *)
  - Bash(find *)
---
You are a security reviewer. Focus on:
- SQL injection, XSS, CSRF vulnerabilities
- Authentication and authorization flaws
- Hardcoded secrets or credentials
- Insecure dependencies
- Input validation gaps

Report each finding with: file, line, severity (critical/high/medium/low), 
description, and recommended fix.
```

Invoke with:
```bash
claude -p --agent security-reviewer "Review the recent PR changes"
```
