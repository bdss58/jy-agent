# Architecture

## Module Map (post-refactor 2025-07)

| Module | Lines | Role |
|--------|-------|------|
| `agent.py` | ~493 | Main REPL loop, command handlers, system prompt, LoopResult handling |
| `loop_engine.py` | ~798 | Headless agentic tool-use loop (shared by planner + subagent). Owns: AgentLoop, LoopConfig, LoopCallbacks, LoopResult, message compaction, tool execution, tool-input truncation, fallback-on-max-steps |
| `terminal_ux.py` | ~298 | Terminal presentation layer: ANSI colors, ThinkingSpinner, tool call headers/icons, result previews, diff rendering, `build_streaming_callbacks()` factory |
| `planner.py` | 2 | Deprecated re-export shim (`from .terminal_ux import build_streaming_callbacks`) |
| `cli.py` | ~530 | Rich-based CLI (prompt, panels, markdown rendering, turn summaries) |
| `config.py` | ~280 | All tunables (tokens, timeouts, model specs, reasoning config) |
| `skills.py` | ~700 | agentskills.io skill discovery, activation, context injection |
| `mcp_client.py` | ~670 | MCP protocol client (stdio transport) |
| `mcp_manager.py` | ~730 | MCP server lifecycle manager (connect/disconnect/reconnect) |
| `session_stats.py` | ~350 | Per-session token/cost tracking |
| `tools/` | ~2500 | Native tool implementations (core, web_fetch, subagent, facades, schemas, search, mcp_tool) |
| `runtime/` | ~800 | Provider-neutral LLM runtime (Anthropic + OpenAI adapters) |
| `memory/` | ~450 | Conversation memory, MEMORY.md/topic-file operations, compaction |

## Data Flow (per turn)

```
user input → agent.py
  → build system prompt (memory + skills)
  → build LoopConfig + build_streaming_callbacks()
  → AgentLoop.run(system_prompt, messages) → LoopResult
  → handle status (completed/max_steps/error/interrupted)
  → conversation.extend(new_messages)
  → render markdown panel
```

## Key Design Decisions

- **loop_engine is headless**: never writes to stdout. UX injected via LoopCallbacks.
- **terminal_ux is pure presentation**: all ANSI output, spinner, tool formatting lives here.
- **agent.py owns LoopResult**: no more untyped 3-tuple. Direct access to status, steps, tokens.
- **Engine handles truncation**: `LoopConfig.truncate_large_inputs=True` — engine truncates large tool_call inputs internally.
- **Engine handles fallback**: `LoopConfig.fallback_on_max_steps=True` — engine makes one more streaming call when max_steps hit. Streams naturally via on_text_delta.
- **Subagent reuses AgentLoop** with silent (no-op) callbacks and `fallback_on_max_steps=False`.
