# MCP subpackage — Model Context Protocol client + manager.
#
# Public API: import from ``jyagent.mcp`` directly.
# - ``MCPClient``                — sync wrapper around the official MCP SDK
# - ``MCPManager``               — server lifecycle + dynamic tool registration
# - ``get_manager``              — get-or-create the singleton manager
# - ``get_manager_if_exists``    — peek without lazy-creating (for shutdown hooks)
# - ``reset_manager``            — disconnect everything and drop the singleton
# - ``load_config``              — read .mcp.json

from .client import MCPClient
from .manager import (
    MCPManager,
    get_manager,
    get_manager_if_exists,
    reset_manager,
    load_config,
)

__all__ = [
    "MCPClient",
    "MCPManager",
    "get_manager",
    "get_manager_if_exists",
    "reset_manager",
    "load_config",
]
