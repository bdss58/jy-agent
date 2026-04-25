"""
MCP Management Tool — Lets the LLM manage MCP server connections.

This is a lightweight control tool. The actual MCP tools (navigate_page, click, etc.)
are registered DIRECTLY into the agent's tool registry by MCPManager when a server
is connected. The LLM calls them as native tools — no adapter layer.

This tool only handles meta-operations:
  - connect: Connect to an MCP server (triggers tool discovery + registration)
  - disconnect: Disconnect from an MCP server
  - status: Show status of all MCP servers
  - list_servers: List configured MCP servers
  - reconnect: Force reconnect to an MCP server (reloads .mcp.json config)
"""

import json
try:
    from ..mcp_manager import get_manager, reset_manager
except ImportError:
    from jyagent.mcp_manager import get_manager, reset_manager
try:
    from ..runtime.tools.result import ToolResult
except ImportError:
    from jyagent.runtime.tools.result import ToolResult


def mcp(action: str, server: str = "") -> ToolResult:
    """Manage MCP server connections.

    When you connect to an MCP server, its tools are automatically discovered
    and registered as native tools you can call directly. For example, connecting
    to 'chrome' registers tools like mcp__chrome__navigate_page, mcp__chrome__click, etc.

    Actions:
      connect      — Connect to an MCP server and register its tools
      disconnect   — Disconnect from an MCP server
      reconnect    — Force reconnect (reloads .mcp.json config, disconnect + connect)
      status       — Show connection status of all servers
      list_servers — List all configured MCP servers from .mcp.json
    """
    manager = get_manager()

    try:
        if action == "connect":
            if not server:
                # Connect to all configured servers
                results = []
                for name in manager.get_server_names():
                    try:
                        result = manager.connect(name)
                        results.append(f"  {name}: {result.get('status', '?')} ({result.get('tools_registered', 0)} tools)")
                    except Exception as e:
                        results.append(f"  {name}: ❌ {e}")
                if not results:
                    return ToolResult("No MCP servers configured. Create a .mcp.json file.", is_error=True)
                return ToolResult("Connected MCP servers:\n" + "\n".join(results))
            else:
                result = manager.connect(server)
                status = result.get("status", "?")
                tools = result.get("tools_registered", 0)
                info = result.get("server_info", {})
                proto = result.get("protocol_version", "?")
                msg = f"✅ MCP server '{server}': {status} ({tools} tools registered)"
                if info:
                    msg += f"\nServer: {info.get('name', '?')} v{info.get('version', '?')}"
                msg += f"\nProtocol: {proto}"
                return ToolResult(msg)

        elif action == "disconnect":
            if not server:
                manager.disconnect_all()
                return ToolResult("✅ All MCP servers disconnected")
            else:
                result = manager.disconnect(server)
                return ToolResult(f"✅ MCP server '{server}': {result.get('status', 'disconnected')}")

        elif action == "reconnect":
            if not server:
                return ToolResult("❌ Please specify which server to reconnect: server='chrome'", is_error=True)
            # Reload .mcp.json config before reconnecting so new args take effect
            manager.load_servers()
            manager.disconnect(server)
            result = manager.connect(server)
            status = result.get("status", "?")
            tools = result.get("tools_registered", 0)
            proto = result.get("protocol_version", "?")
            return ToolResult(f"✅ MCP server '{server}' reconnected (config reloaded): {status} ({tools} tools, protocol {proto})")

        elif action == "status":
            return ToolResult(manager.status())

        elif action == "list_servers":
            servers = manager.get_server_names()
            if not servers:
                return ToolResult("No MCP servers configured. Create a .mcp.json file.", is_error=True)
            lines = ["Configured MCP servers:"]
            for name in servers:
                connected = manager.is_connected(name)
                status = "✅" if connected else "⬚"
                lines.append(f"  {status} {name}")
            return ToolResult("\n".join(lines))

        else:
            return ToolResult(
                f"❌ Unknown action '{action}'. "
                f"Valid: connect, disconnect, reconnect, status, list_servers",
                is_error=True
            )

    except Exception as e:
        return ToolResult(f"❌ MCP error: {e}", is_error=True)


# ─── Tool schema for auto-discovery ──────────────────────────────────────────

TOOL_SCHEMA = {
    "name": "mcp",
    "description": (
        "Manage MCP (Model Context Protocol) server connections. "
        "Connect to servers to auto-discover and register their tools. "
        "For example, 'connect' to 'chrome' registers browser automation tools "
        "(navigate_page, click, take_snapshot, etc.) that you can then call directly."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action: connect, disconnect, reconnect, status, list_servers",
                "enum": ["connect", "disconnect", "reconnect", "status", "list_servers"],
            },
            "server": {
                "type": "string",
                "description": "MCP server name (e.g., 'chrome'). REQUIRED for 'reconnect'. Optional for 'connect'/'disconnect' (omit to target all). Not needed for 'status'/'list_servers'.",
            },
        },
        "required": ["action"],
    },
}
