# agents/ package — Declarative agent definitions loaded from Markdown files.
#
# Agent files live in data/agents/*.md and use YAML frontmatter for metadata.
# The body of the file becomes the agent's system prompt.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentDef:
    """Declarative agent definition."""
    name: str
    description: str = ""
    model: str = "default"
    tools: list[str] = field(default_factory=list)
    max_steps: int = 30
    system_prompt: str = ""

    def __repr__(self) -> str:
        tools_str = ", ".join(self.tools) if self.tools else "all"
        return (
            f"AgentDef(name={self.name!r}, model={self.model!r}, "
            f"max_steps={self.max_steps}, tools=[{tools_str}])"
        )


# ─── Default directory ───────────────────────────────────────────────────────

_DEFAULT_AGENTS_DIR = os.path.join("data", "agents")
_agents_cache: dict[str, AgentDef] | None = None


# ─── YAML frontmatter parser ────────────────────────────────────────────────

def _parse_yaml_block(yaml_text: str) -> dict:
    """Parse a YAML frontmatter block.

    Uses the ``yaml`` library if available, otherwise falls back to a
    simple line-by-line key-value parser that handles scalars and lists.
    """
    try:
        import yaml
        return yaml.safe_load(yaml_text) or {}
    except ImportError:
        pass

    # Simple fallback parser
    result: dict = {}
    current_key: str | None = None
    current_list: list[str] | None = None

    for line in yaml_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # List item under current key
        if stripped.startswith("- ") and current_key is not None and current_list is not None:
            current_list.append(stripped[2:].strip().strip('"').strip("'"))
            continue

        # Key: value
        if ":" in stripped:
            if current_key and current_list is not None:
                result[current_key] = current_list
                current_list = None
                current_key = None

            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()

            if not val:
                # Could be a list header
                current_key = key
                current_list = []
                continue

            # Scalar value
            val = val.strip('"').strip("'")
            # Try int
            try:
                result[key] = int(val)
            except ValueError:
                result[key] = val
            current_key = None
            current_list = None

    if current_key and current_list is not None:
        result[current_key] = current_list

    return result


def _parse_agent_file(path: str) -> AgentDef | None:
    """Parse a .md file with YAML frontmatter into an AgentDef.

    Expected format::

        ---
        name: explorer
        description: Fast read-only explorer
        model: fast
        max_steps: 15
        tools:
          - read_file
          - glob_files
        ---
        System prompt body here...
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return None

    content = content.strip()
    if not content.startswith("---"):
        return None

    # Split frontmatter from body
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None

    yaml_text = parts[1]
    body = parts[2].strip()

    meta = _parse_yaml_block(yaml_text)
    if not meta.get("name"):
        # Derive name from filename
        meta["name"] = os.path.splitext(os.path.basename(path))[0]

    tools = meta.get("tools", [])
    if isinstance(tools, str):
        tools = [t.strip() for t in tools.split(",") if t.strip()]

    return AgentDef(
        name=meta["name"],
        description=meta.get("description", ""),
        model=meta.get("model", "default"),
        tools=tools,
        max_steps=int(meta.get("max_steps", 30)),
        system_prompt=body,
    )


# ─── Public API ──────────────────────────────────────────────────────────────

def load_agents(agents_dir: str = "") -> dict[str, AgentDef]:
    """Load all agent definitions from the agents directory.

    Results are cached after the first call.  Pass *agents_dir* to override
    the default ``data/agents/`` location.
    """
    global _agents_cache
    if _agents_cache is not None and not agents_dir:
        return dict(_agents_cache)

    directory = agents_dir or _DEFAULT_AGENTS_DIR
    agents: dict[str, AgentDef] = {}

    if not os.path.isdir(directory):
        if not agents_dir:
            _agents_cache = agents
        return agents

    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(directory, fname)
        agent_def = _parse_agent_file(path)
        if agent_def is not None:
            agents[agent_def.name] = agent_def

    if not agents_dir:
        _agents_cache = agents
    return agents


def get_agent(name: str) -> AgentDef | None:
    """Look up an agent definition by name.  Returns None if not found."""
    agents = load_agents()
    return agents.get(name)


def list_agents() -> list[str]:
    """Return sorted list of available agent names."""
    return sorted(load_agents().keys())
