# Claude Code CLI Flag Reference

Complete reference for `claude` CLI flags relevant to programmatic use.

## Execution Mode Flags

| Flag | Description | Example |
|------|-------------|---------|
| `-p, --print` | Non-interactive mode. Print response and exit. | `claude -p "query"` |
| `--bare` | Skip hooks, plugins, MCP servers, CLAUDE.md auto-discovery. Fast, deterministic. | `claude -p --bare "query"` |
| `-c, --continue` | Continue the most recent conversation. | `claude -p -c "follow up"` |
| `-r, --resume ID` | Resume a specific session by ID or name. | `claude -r "my-session" "query"` |
| `--name NAME` | Name the current session for later resumption. | `claude --name "refactor-auth"` |

## Model & Budget

| Flag | Description | Example |
|------|-------------|---------|
| `--model MODEL` | Select model: `haiku`, `sonnet`, `opus` | `--model sonnet` |
| `--max-budget-usd N` | Maximum spend cap in USD. Aborts if exceeded. | `--max-budget-usd 1.00` |

## Output Format

| Flag | Description | Example |
|------|-------------|---------|
| `--output-format FORMAT` | `text` (default), `json`, `stream-json` | `--output-format json` |
| `--json-schema SCHEMA` | Force output to match a JSON Schema. Implies `--output-format json`. | `--json-schema '{"type":"object",...}'` |

### JSON output structure:
```json
{
  "result": "Claude's response text",
  "session_id": "abc123",
  "cost_usd": 0.042,
  "duration_ms": 3200,
  "num_turns": 5
}
```

## Permission & Tool Control

| Flag | Description | Example |
|------|-------------|---------|
| `--allowedTools TOOLS...` | Auto-approve specific tools. Others require confirmation. | `--allowedTools "Read" "Edit" "Bash(npm test)"` |
| `--permission-mode MODE` | Set permission mode: `default`, `acceptEdits`, `plan`, `bypassPermissions`, `auto`. | `--permission-mode plan` |
| `--dangerously-skip-permissions` | Skip ALL permission prompts. **Only use in sandboxes.** | `--dangerously-skip-permissions` |
| `--effort LEVEL` | Set effort level: `low`, `medium`, `high`, `max`. | `--effort high` |

### Tool name patterns for `--allowedTools`:
- `Read` — read files
- `Edit` — edit files  
- `Write` — create/overwrite files
- `Bash(pattern)` — shell commands matching glob: `Bash(npm *)`, `Bash(git commit *)`, `Bash(*)` (all)
- `WebSearch` — web search
- `mcp__servername__toolname` — specific MCP tool

## Context & System Prompt

| Flag | Description | Example |
|------|-------------|---------|
| `--append-system-prompt TEXT` | Append to the system prompt. Useful for role customization. | `--append-system-prompt "You are a security reviewer."` |
| `--system-prompt TEXT` | Replace the entire system prompt. **Use with caution.** | Rarely needed |

## Input Methods

```bash
# Direct argument
claude -p "your prompt here"

# Stdin pipe (supports multi-line, binary data)
echo "prompt" | claude -p
cat file.txt | claude -p "analyze this file"
git diff | claude -p "review these changes"

# File content via stdin
claude -p "explain this code" < src/complex.py
```

## Useful Combinations

### Safe CI/CD task
```bash
claude -p --bare --model sonnet \
  --max-budget-usd 0.50 \
  --allowedTools "Read" "Edit" "Bash(npm test)" \
  "Fix lint errors in src/. Run npm test to verify."
```

### Deep review (read-only, high-quality)
```bash
claude -p --model opus --bare \
  --allowedTools "Read" "Bash(grep *)" "Bash(find *)" \
  "Security audit src/auth/. Report vulnerabilities with severity ratings."
```

### Quick format/cleanup (cheap, fast)
```bash
claude -p --model haiku --bare \
  --allowedTools "Read" "Edit" \
  "Add docstrings to all public functions in src/utils.py"
```

### Structured analysis
```bash
claude -p --model sonnet --bare --output-format json \
  --json-schema '{"type":"object","properties":{"score":{"type":"integer"},"issues":{"type":"array","items":{"type":"string"}}}}' \
  "Rate the code quality of src/main.py on a scale of 1-10 and list issues."
```

### Git commit message
```bash
git diff --staged | claude -p --bare --model haiku \
  --allowedTools "Bash(git commit *)" \
  "Create a conventional commit with an appropriate message for these changes."
```
