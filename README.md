# JY Agent 🤖

A personal AI assistant with Anthropic + OpenAI runtime support, self-evolution capabilities, MCP integration, and persistent memory.

## Features

- **Multi-Provider Runtime**: Anthropic Messages and OpenAI Responses with provider-neutral conversation state
- **Tool-use Loop**: Streaming tool execution with concurrent parallel-safe tools
- **MCP Integration**: Chrome DevTools, and any MCP-compatible server
- **Persistent Memory**: Cross-session memory with `MEMORY.md`, topic files, and user profile
- **Agent Skills**: Procedural knowledge system (agentskills.io standard)
- **Web Fetch**: 5-tier anti-blocking cascade (curl_cffi → httpx → Jina → Chrome → error)
- **Rich CLI**: Markdown rendering, syntax highlighting, multi-line input
- **Live Model Switching**: `/model <provider> <model>` swaps the active runtime for subsequent turns

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

## Origin

Evolved from [ai-agent-boot](https://github.com/xxx/ai-agent-boot) — a bootstrapping experiment where a single Claude API call generates an entire agent. After dozens of iterations, the agent graduated to this standalone project.
