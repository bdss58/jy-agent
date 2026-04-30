# MCP subpackage — Model Context Protocol client + manager.
#
# Public API: import from ``jyagent.mcp`` directly.
# - ``MCPClient`` — sync wrapper around the official MCP SDK
# - ``MCPManager`` — server lifecycle + dynamic tool registration
# - ``get_manager`` / ``reset_manager`` — singleton accessors
# - ``load_config`` — read .mcp.json

from .client import MCPClient
from .manager import (
    MCPManager,
    get_manager,
    reset_manager,
    load_config,
)

__all__ = [
    "MCPClient",
    "MCPManager",
    "get_manager",
    "reset_manager",
    "load_config",
]
