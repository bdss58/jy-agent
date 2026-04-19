# JY Agent 🤖

A personal AI assistant with a **provider-neutral runtime** (Anthropic + OpenAI), sub-agent
orchestration, MCP integration, durable cross-session memory, and a skills system.

## Features

### Runtime
- **Provider-neutral core**: single conversation state, pluggable adapters for
  Anthropic Messages (`claude-*`) and — optionally — OpenAI Responses (`gpt-*`, `o*`).
  The Anthropic SDK is a hard dependency; the `openai` SDK is optional and only
  required when `AGENT_PROVIDER=openai` (install with `pip install openai`).
- **Live provider/model switching**: `/model <provider> <model>` swaps the active
  runtime for subsequent turns without restarting.
- **Streaming tool-use loop** with parallel-safe tool execution, cooperative
  cancellation (Ctrl-C), and automatic token-budget retries.
- **Automatic long-context compaction**: conversation summarization with
  observation masking and post-compaction file re-injection
  (`jyagent/memory/compaction.py`).
- **Extended reasoning** support (Anthropic thinking / OpenAI reasoning).
- **Optional quality harness**: JSONL run tracing (`AGENT_TRACE_ENABLED`) and a
  pre-completion verification gate (`AGENT_VERIFICATION_ENABLED`).

### Tools (built-in)
- **Filesystem**: `read_file`, `write_file`, `edit_file`, `list_directory`,
  `glob_files`, `grep_files`.
- **Shell**: `run_shell` (foreground, bounded timeout) and
  `run_background` / `check_background` — long-running jobs with
  `timeout_seconds` auto-kill, `cwd`, `stdin_null`, `action="wait"`,
  and a global 8-job concurrency cap.
- **Web**: `web_fetch` (5-tier anti-blocking cascade: curl_cffi → httpx →
  Jina Reader → Chrome → diagnostics) and `web_search` (DDG → Codex fallback).
- **Sub-agents**: `dispatch_agent` / `check_agent` — spawn focused workers in
  foreground or background with soft handoff, grace-period cancellation, and
  isolated context windows.
- **MCP**: `mcp` tool for connecting/managing servers; Chrome DevTools MCP
  pre-configured in `.mcp.json`.
- **Self-use**: `manage_memory`, `manage_skills`.

### Memory
- `MEMORY.md` as a concise always-loaded index.
- Detailed topic files under `data/memory/topics/<name>.md`, read on demand.
- Auto-synced topic index when topic files are written or deleted.

### Skills (agentskills.io standard)
- Auto-activate on matching queries; manually controllable via `/skills`,
  `/skill <name>`, `/skill -<name>`.
- Ships with: `web-search`, `deep-research`, `browser-automation`,
  `claude-code`, `codex-cli`, `git-workflow`, `create-skill`.

### CLI
- Rich markdown rendering, syntax highlighting, multi-line input.
- Session stats with per-provider cost tracking (prompt / output / cache-read /
  cache-write tokens, sub-agent attribution) using a static pricing table in
  `jyagent/session_stats.py`. Models without a pricing entry are reported as
  unknown-cost; extend via `set_model_pricing(...)`.
- Commands: `/help /quit /new /continue /history /tools /model /multi /markdown /stats /skills /skill <name>`.

## Quick Start

Requires **Python ≥ 3.12** (tested on 3.14).

```bash
# Clone and enter
cd ~/jy-agent

# Install (editable)
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# — or, if you use uv:
#   uv sync

# Configure
cp .env.example .env
# Edit .env — set AGENT_PROVIDER, AGENT_MODEL, and the matching API key
# (ANTHROPIC_API_KEY or OPENAI_API_KEY).

# Run
jy-agent
# or: python -m jyagent
```

### Switching providers mid-session

```
/model anthropic claude-sonnet-4-6
/model openai    gpt-5
```

The runtime preserves conversation state across switches.

## Project Layout

```
jyagent/
  agent.py              # Top-level loop, slash commands, runtime owner
  cli.py                # Rich CLI, history rendering, help
  runtime/
    core.py             # Provider-neutral conversation state
    providers/
      anthropic.py      # Messages API adapter
      openai.py         # Responses API adapter
  tools/
    core.py             # Filesystem + run_shell + run_background
    subagent.py         # dispatch_agent / check_agent
    web_fetch.py        # 5-tier anti-blocking fetch
    web_search_tool.py  # DDG + Codex search
    mcp_tool.py         # MCP bridge
  registry.py           # Tool registry with parallel-safe / timeout metadata
data/
  memory/               # MEMORY.md index + topics/
skills/                 # Agent skills (loaded by jyagent/skills.py)
tests/                  # pytest suite (background, subagent, runtime, memory, …)
.mcp.json               # Default MCP servers (Chrome DevTools)
```

## Development

```bash
# Lint
ruff check jyagent tests

# Tests
pytest -q
```

## Origin

Bootstrapped from a single Claude API call in the `ai-agent-boot` experiment —
the agent wrote itself, then graduated to this standalone project. The
provider-neutral runtime, sub-agent system, and background-task layer were all
added through self-directed iteration.
