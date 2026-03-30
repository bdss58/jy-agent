# JY Agent 🤖

A personal AI assistant powered by Claude, with self-evolution capabilities, MCP integration, and persistent memory.

## Features

- **Tool-use Loop**: Streaming Claude API with concurrent tool execution
- **MCP Integration**: Chrome DevTools, and any MCP-compatible server
- **Self-Evolution**: Agent can rewrite its own modules via `evolve_self`
- **Persistent Memory**: Cross-session memory with topic files, user profile, session summaries
- **Agent Skills**: Procedural knowledge system (agentskills.io standard)
- **Web Fetch**: 5-tier anti-blocking cascade (curl_cffi → httpx → Jina → Chrome → error)
- **Rich CLI**: Markdown rendering, syntax highlighting, multi-line input

## Quick Start

```bash
# Clone and enter
cd ~/jy-agent

# Create venv & install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run
jy-agent
# or: python -m jyagent
```

## Project Structure

```
jy-agent/
├── jyagent/              # Main package
│   ├── __main__.py       # Entry point
│   ├── agent.py          # Main loop, command dispatch, session lifecycle
│   ├── planner.py        # Streaming tool-use loop with Claude
│   ├── tools.py          # Core tools + auto-discovery
│   ├── registry.py       # Tool registry
│   ├── cli.py            # Rich + prompt_toolkit UI
│   ├── self_memory.py    # Persistent memory system
│   ├── mcp_client.py     # MCP SDK async client
│   ├── mcp_manager.py    # MCP server lifecycle
│   ├── skills.py         # Agent Skills engine
│   ├── evolver.py        # Self-evolution engine
│   ├── validator.py      # AST-based code validator
│   ├── evolution_strategy.py  # Evolution prompt templates
│   ├── tool_web_fetch.py # 5-tier web fetching
│   ├── tool_edit_file.py # Smart file editing
│   ├── tool_glob_grep.py # File search
│   └── tool_mcp.py       # MCP connection management
├── skills/               # Agent skill definitions
├── data/memory/          # Persistent memory data
├── .mcp.json             # MCP server configuration
├── .env                  # Environment variables
└── pyproject.toml        # Python package config
```

## Origin

Evolved from [ai-agent-boot](https://github.com/xxx/ai-agent-boot) — a bootstrapping experiment where a single Claude API call generates an entire agent. After dozens of iterations, the agent graduated to this standalone project.
