# JY Agent 🤖

A personal AI assistant with a **provider-neutral runtime** (Anthropic + OpenAI), sub-agent
orchestration, MCP integration, durable cross-session memory, and a skills system.

## Features

### Runtime
- **Provider-neutral core**: single conversation state, pluggable adapters for
  Anthropic Messages (`claude-*`) and — optionally — OpenAI Responses (`gpt-*`, `o*`).
  The Anthropic SDK is a hard dependency; the `openai` SDK is optional and only
  required when `AGENT_PROVIDER=openai` (install with `uv sync --extra openai`
  or `pip install -e ".[openai]"`).
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
  Jina Reader → Chrome → diagnostics) and `web_search` (DuckDuckGo HTML).
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
  `jyagent/runtime/stats.py`. Models without a pricing entry are reported as
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
  agent.py              # Reference application: top-level loop, slash commands, runtime owner
  cli.py                # Rich CLI, history rendering, help
  __main__.py           # python -m jyagent entry point

  runtime/              # ─── Reusable agent runtime (library surface) ───
    __init__.py         #   Public API: AgentLoop, LoopConfig, LoopResult, LoopCallbacks,
                        #               get_registry, ToolResult, get_stats, SessionStats
    loop/
      engine.py         #   AgentLoop — streaming tool-use loop, dispatch, retries
      callbacks.py      #   LoopCallbacks protocol — UI/observer seam
      config.py         #   LoopConfig, LoopResult dataclasses
      phases.py         #   Multi-phase planning helper
      reflection.py     #   Periodic self-reflection
      checkpoint.py     #   Conversation checkpointing
      todos.py          #   In-loop TODO scratchpad
      verification.py   #   Pre-completion self-check gate
      remediation.py    #   Tool-error enrichment
      tracing.py        #   JSONL run tracing
    tools/
      registry.py       #   Tool registry (parallel-safe / timeout metadata)
      result.py         #   ToolResult value type
      validation.py     #   Tool input validation
    stats.py            #   SessionStats + cost tracking
    skills.py           #   Skill manager (agentskills.io)

  llm/                  # ─── Provider-neutral LLM layer ───
    core.py             #   Single conversation state, pluggable adapters
    providers/
      anthropic.py      #   Messages API adapter
      openai.py         #   Responses API adapter

  tools/                # ─── Built-in tool implementations ───
    core.py             #   Filesystem + run_shell + run_background
    subagent.py         #   dispatch_agent / check_agent
    web_fetch.py        #   5-tier anti-blocking fetch
    web_search.py       #   SearxNG → DDG cascade
    mcp_tool.py         #   MCP bridge

  memory/               # Conversation compaction, session save/load, extraction
  mcp/                  # MCP protocol — client + server lifecycle / tool registration
    client.py           #   Sync wrapper around the official MCP SDK
    manager.py          #   MCPManager (lifecycle, keepalive, Chrome helpers)
  terminal_ux.py        # CLI-side LoopCallbacks implementation
  config.py             # Env-driven config (overridable by library users)

data/
  memory/               # MEMORY.md index + topics/ + journal/
skills/                 # Agent skills (loaded by jyagent/skills.py)
tests/                  # pytest suite (background, subagent, runtime, memory, …)
.mcp.json               # Default MCP servers (Chrome DevTools)
```

## Using `jyagent.runtime` as a library

The runtime is decoupled from the CLI / reference app and can be embedded:

```python
from jyagent.runtime import AgentLoop, LoopConfig, LoopCallbacks, get_registry
from jyagent.llm import LLMOwner
from jyagent.llm.types import ModelSpec
import jyagent.tools  # noqa: F401 — triggers built-in tool registration

# 1. Pick a provider/model.
runtime_owner = LLMOwner(ModelSpec(provider="anthropic", model="claude-sonnet-4-6"))

# 2. Build observer hooks.  `LoopCallbacks` is a dataclass — every field is an
#    Optional[Callable]; pass only the hooks you care about.  None = silent.
callbacks = LoopCallbacks(
    on_text_delta=lambda t: print(t, end="", flush=True),
    on_tool_start=lambda name, inp: print(f"\n[tool] {name}({inp})"),
    on_tool_end=lambda name, content, is_error: print(f"[tool/{name} ok={not is_error}]"),
)

# 3. Tool source: a callable that returns (schemas, functions) on each iteration.
#    `registry.freeze()` returns an immutable ToolBatch with `.schemas` and
#    `.functions` attributes.  Mutate the registry between turns and the loop
#    will pick it up.
registry = get_registry()
def tool_source():
    batch = registry.freeze()
    return batch.schemas, batch.functions

# 4. Run.  `messages` is mutated in-place; pass an empty list for a fresh turn,
#    or a prior conversation to continue.
loop = AgentLoop(
    runtime_owner,
    LoopConfig(streaming=True, max_working_tokens=120_000),
    callbacks=callbacks,
    tool_source=tool_source,
)
messages = [{"role": "user", "content": "Summarize all TODO comments in this repo."}]
result = loop.run(system_prompt="You are a focused code-spelunking assistant.",
                  messages=messages)
print(f"\nstatus={result.status}  steps={result.steps}  "
      f"tokens={result.total_input_tokens}/{result.total_output_tokens}")
```

For a full-featured integration (slash commands, MCP, skills, persistence,
session stats, cancellation, sub-agent attribution, …) see `jyagent/agent.py` —
the reference application built on this exact API.

Public API surface (everything else is internal):

| Import | Purpose |
|---|---|
| `AgentLoop`, `LoopConfig`, `LoopResult` | Core loop |
| `LoopCallbacks` | Observer dataclass — set the hooks you need, leave the rest `None` |
| `get_registry`, `ToolResult` | Tool registration & return type |
| `get_stats`, `SessionStats` | Per-session telemetry & cost tracking |

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
