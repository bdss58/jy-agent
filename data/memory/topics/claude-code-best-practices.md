# Claude Code Best Practices — Research Summary

Research date: 2026-04-04
Sources: Anthropic official docs, builder.io (50 tips), community guides

## Repo-Specific Update (2026-04-05)

- When jy-agent delegates to Claude Code, prefer `claude -p --bare` without `--model` and let local Claude config choose the default model
- Use explicit `--model` only for deliberate tier changes, such as `opus` for difficult architecture/debugging work or `haiku` for lightweight cheap tasks

## Core Principle: Context Window Management

Claude Code's **#1 constraint** is the context window. Performance degrades as it fills.
Everything else flows from managing this resource well.

## 1. CLAUDE.md — The Foundation

- Place `CLAUDE.md` at project root. Claude reads it at every session start.
- Run `/init` to generate a starter version, then **ruthlessly prune**.
- Rule of thumb: ~150-200 instruction budget before compliance drops.
- For each line, ask: "Would removing this cause Claude to make mistakes?" If not, cut it.

### What to include:
- Build/test/lint commands Claude can't guess
- Code style rules that differ from defaults
- Repo etiquette (branch naming, PR conventions)
- Architecture decisions specific to your project
- Developer environment quirks (required env vars)
- Common gotchas

### What to exclude:
- Anything Claude can figure out by reading code
- Standard language conventions
- Long explanations or tutorials
- File-by-file descriptions

### Organization:
- Use `.claude/rules/` for topic-specific rules with `paths` frontmatter
- Use `@imports` like `@docs/solutions.md` to reference detailed docs without bloating CLAUDE.md
- When Claude makes a mistake: "update the CLAUDE.md so this doesn't happen again"

## 2. Plan Mode — Think Before Coding

- Use **Shift+Tab** to cycle: Normal → Auto-Accept → Plan Mode
- Plan Mode = Claude reads files, analyzes, but **doesn't make changes**
- Best for: multi-file changes, unfamiliar code, architectural decisions
- Skip for: small clear-scope tasks (typo fix, rename, one-liner)

**Workflow:**
1. **Explore** (Plan Mode): Claude reads files, answers questions
2. **Plan**: Ask for detailed implementation plan, Ctrl+G to edit it
3. **Implement** (Normal Mode): Let Claude code, verify against plan

## 3. Give Claude Verification (2-3x Quality Boost)

- Always provide a way for Claude to verify its work (tests, linter, bash check)
- "Run the existing test suite after making changes. Fix any failures before calling it done."
- Install **LSP plugins** (`/plugin install typescript-lsp@claude-plugins-official`) — single highest-impact plugin
- Use Playwright MCP for UI verification

## 4. Prompt Engineering

- Be specific: reference files, mention constraints, point to example patterns
- "Look at how existing widgets are implemented... follow the pattern"
- Scope the task: which file, what scenario, testing preferences
- For bugs: provide symptom, likely location, what "fixed" looks like

### Interview pattern for complex tasks:
```
I want to build [brief description]. Interview me in detail
using the AskUserQuestion tool. Ask about technical implementation,
edge cases, concerns, and tradeoffs. Keep interviewing until we've
covered everything, then write a complete spec to SPEC.md.
```

## 5. Subagents — Divide and Conquer

- Built-in: Explore (Haiku, read-only), Plan, General-purpose
- Custom: create in `.claude/agents/` or `~/.claude/agents/`
- Use subagents to **preserve main context** — exploration happens in separate context
- Key config: `model`, `tools`, `permissionMode`, `mcpServers`, `isolation`
- `isolation: worktree` for agents needing their own file system

### Useful custom agents:
- Security reviewer (Opus, read-only tools)
- Quick search (Haiku, for speed)
- Code reviewer (Sonnet, read-only)

## 6. Headless / SDK Mode (`-p` flag)

- `claude -p "prompt"` — non-interactive, outputs result and exits
- `--bare` — skip hooks, plugins, MCP, CLAUDE.md auto-discovery (fast, deterministic)
- `--output-format json` — structured output with session ID and metadata
- `--json-schema` — force structured output matching a JSON Schema
- `--stream-json` — real-time streaming
- `--allowedTools` — auto-approve specific tools without prompting
- `--max-budget-usd` — cap spending
- `--model <tier>` — optional override when you intentionally do not want the configured default
- `--continue` / `--resume` — continue conversations

### Key patterns:
```bash
# Pipe data in
cat error.log | claude -p "explain these errors"

# Structured output
claude -p --output-format json --json-schema '{"type":"object",...}' "query"

# Auto-approve tools
claude -p --allowedTools "Bash(npm test)" "Read" "Edit" "Write" "prompt"

# Create commit
git diff --staged | claude -p --allowedTools "Bash(git commit *)" "create a commit with an appropriate message"
```

## 7. Parallel Sessions & Scaling

- Run multiple sessions for different tasks (writer + reviewer pattern)
- Fan-out: generate task list, then `xargs -P` with `claude -p`
- Use `--name` / `/rename` for session identification
- Git worktrees for file-level isolation between sessions

## 8. Hooks — Automation

### PostToolUse: Auto-format after edits
```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{"type": "command", "command": "npx prettier --write \"$CLAUDE_FILE_PATH\" 2>/dev/null || true"}]
    }]
  }
}
```

### PreToolUse: Block destructive commands
```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Bash",
      "type": "command",
      "command": "if echo \"$TOOL_INPUT\" | grep -qE 'rm -rf|drop table|truncate'; then echo 'BLOCKED' >&2; exit 2; fi"
    }]
  }
}
```

### Notification: Re-inject context after compaction
- Hook with `compact` matcher to re-inject task description, modified files, constraints

## 9. Context Hygiene

- `/clear` between unrelated tasks
- After 2 failed corrections, `/clear` and rewrite the prompt
- Scope investigations narrowly or use subagents
- `Esc` to stop, `Esc+Esc` or `/rewind` to rewind
- `/branch` to try risky approaches without losing main conversation

## 10. Common Anti-Patterns

1. **Kitchen sink session**: mixing unrelated tasks → `/clear` between tasks
2. **Correcting over and over**: failed context pollution → `/clear` + better prompt
3. **Over-specified CLAUDE.md**: too long, Claude ignores half → prune ruthlessly
4. **Trust-then-verify gap**: no tests → always provide verification
5. **Infinite exploration**: unscoped "investigate" → scope or use subagents

## 11. Tips for jy-agent Integration

As jy-agent (me) calling Claude Code as a sub-tool:

### Best approach: `claude -p` (headless/SDK mode)
```bash
echo "task description" | claude -p --bare --allowedTools "Read" "Edit" "Write" "Bash(*)"
```

### Key flags for programmatic use:
- `--bare` — deterministic, no local config interference
- `--output-format json` — parseable output
- `--max-turns N` — limit execution steps
- `--max-budget-usd N` — cap cost
- `--allowedTools` — specify exactly what it can do
- `--model <tier>` — optional override; default to local config unless the task needs a specific tier
- `--dangerously-skip-permissions` — full autonomy (sandbox only!)

### Delegation patterns:
1. **Code tasks**: jy-agent plans, Claude Code implements
2. **Review**: Claude Code writes, second Claude Code instance reviews
3. **Parallel fan-out**: jy-agent distributes tasks to multiple Claude Code instances
4. **Specialized agents**: use `--agent` for pre-configured subagent types
