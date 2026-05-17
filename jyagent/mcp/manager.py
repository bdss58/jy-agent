"""
MCP Manager — Manages MCP server lifecycle and dynamically registers tools.

Reads .mcp.json config, connects to servers, discovers tools via MCP SDK,
and registers them into the agent's ToolRegistry. Includes background
keepalive pings and tools/list_changed notification handling.
"""

import atexit
import json
import os
import threading
from typing import Optional
from .client import MCPClient
from .conversion import (
    extract_mcp_result as _extract_mcp_result,
    mcp_schema_to_agent_schema as _mcp_schema_to_agent_schema,
)
from ..runtime.tools.registry import get_registry
from ..runtime.tools.result import ToolResult

# Keepalive configuration
KEEPALIVE_INTERVAL_SECONDS = 60  # Ping every 60 seconds
KEEPALIVE_MAX_FAILURES = 3       # Mark dead after 3 consecutive failures


# ─── Config loading ───────────────────────────────────────────────────────────

_CONFIG_PATHS = [
    ".mcp.json",                    # Project root
    os.path.expanduser("~/.mcp.json"),  # Home directory
]


def _find_config() -> Optional[str]:
    """Find the first existing .mcp.json config file."""
    for path in _CONFIG_PATHS:
        if os.path.exists(path):
            return path
    return None


def load_config() -> dict:
    """Load MCP server configuration from .mcp.json."""
    config_path = _find_config()
    if not config_path:
        return {"mcpServers": {}}
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except (json.JSONDecodeError, IOError):
        return {"mcpServers": {}}


# ─── Per-server policy registry ──────────────────────────────────────────────
#
# Server-specific policy (pre-connect hooks, dead-error patterns, tool
# timeouts) lives in dedicated modules (e.g. ``mcp/chrome.py``); the manager
# only wires them by server name. Add new server-specific policies here.

from .chrome import CHROME_POLICY, ChromeBrowser  # noqa: E402

_SERVER_POLICIES: dict[str, dict] = {
    "chrome": CHROME_POLICY,
}



# ─── MCPManager ──────────────────────────────────────────────────────────────

class MCPManager:
    """Manages multiple MCP server connections and their tool registrations.
    
    Now backed by the official MCP SDK. Each server gets an MCPClient instance
    which wraps the SDK's async ClientSession with a sync interface.
    
    Improvements (2025-03):
    - Handles tools/list_changed notifications via MCPClient callback
    - Background keepalive thread pings connected servers every 60s
    """

    def __init__(self):
        self._clients: dict[str, MCPClient] = {}
        self._configs: dict[str, dict] = {}
        self._tool_to_server: dict[str, str] = {}  # agent_tool_name → server_name

        # Keepalive
        self._keepalive_thread: Optional[threading.Thread] = None
        self._keepalive_stop = threading.Event()
        self._ping_failures: dict[str, int] = {}  # server_name → consecutive failure count
        
        # Lock for thread-safe tool re-registration
        self._tools_lock = threading.Lock()

        # Browser-specific helper.  See ``mcp/chrome.py``.  All Chrome ops
        # (refcounted connection, page-lock serialisation, fetch_page) live
        # on this object; reach for ``manager._chrome`` directly when you
        # need them (e.g. ``tools/web_fetch.py``).
        self._chrome = ChromeBrowser(self)

    def load_servers(self, config: dict = None):
        """Load server configurations from config dict or .mcp.json."""
        if config is None:
            config = load_config()
        self._configs = config.get("mcpServers", {})

    def get_server_names(self) -> list[str]:
        """Return names of all configured MCP servers."""
        return list(self._configs.keys())

    def is_connected(self, server_name: str) -> bool:
        """Check if a specific server is connected."""
        client = self._clients.get(server_name)
        return client is not None and client.is_connected

    def connect(self, server_name: str) -> dict:
        """Connect to a specific MCP server and register its tools.
        
        Returns connection status dict.
        """
        if server_name not in self._configs:
            return {"status": "error", "message": f"Unknown MCP server: {server_name}"}

        if self.is_connected(server_name):
            return {"status": "already_connected", "name": server_name}

        server_config = dict(self._configs[server_name])

        # Apply pre-connect hook if registered for this server
        policy = _SERVER_POLICIES.get(server_name, {})
        hook = policy.get("pre_connect")
        if hook:
            server_config = hook(server_config)

        # Determine transport type
        transport = server_config.get("transport", "stdio")

        # Create client and connect
        client = MCPClient(name=server_name)
        
        # Register tools_changed callback before connecting
        client.set_on_tools_changed(self._on_tools_changed)

        if transport == "http":
            result = client.connect(
                command="",  # Not used for HTTP
                url=server_config["url"],
                transport="http",
                headers=server_config.get("headers"),
                init_timeout=server_config.get("init_timeout", 30),
            )
        else:
            result = client.connect(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=server_config.get("env"),
                init_timeout=server_config.get("init_timeout", 30),
                cwd=server_config.get("cwd"),
            )

        self._clients[server_name] = client
        self._ping_failures[server_name] = 0  # Reset failure counter

        # Discover and register tools
        tool_count = self._register_server_tools(server_name, client, server_config)
        result["tools_registered"] = tool_count

        # Start keepalive if not already running
        self._start_keepalive()

        return result

    def disconnect(self, server_name: str) -> dict:
        """Disconnect from a specific MCP server and unregister its tools."""
        # Unregister tools
        with self._tools_lock:
            self._unregister_server_tools(server_name)

        # Disconnect client
        client = self._clients.pop(server_name, None)
        self._ping_failures.pop(server_name, None)
        
        if client:
            result = client.disconnect()
        else:
            result = {"status": "not_connected", "name": server_name}
        
        # Stop keepalive if no more connected servers
        if not self._clients:
            self._stop_keepalive()
        
        return result

    def disconnect_all(self):
        """Disconnect from all MCP servers."""
        # Stop keepalive first
        self._stop_keepalive()
        
        for name in list(self._clients.keys()):
            self.disconnect(name)

    def _make_tool_name(self, server_name: str, mcp_tool_name: str,
                        use_prefix: bool = True) -> str:
        """Generate the agent-facing tool name.
        
        Convention: mcp__{server}__{tool} 
        This makes it clear to the LLM that it's an MCP tool,
        and avoids name collisions between different MCP servers.
        """
        if use_prefix:
            return f"mcp__{server_name}__{mcp_tool_name}"
        return mcp_tool_name

    def _register_server_tools(self, server_name: str, client: MCPClient,
                                server_config: dict) -> int:
        """Discover tools from MCP server and register them in the agent's registry."""
        registry = get_registry()
        use_prefix = server_config.get("prefix", True)
        
        try:
            mcp_tools = client.list_tools()
        except Exception:
            return 0

        count = 0
        for mcp_tool in mcp_tools:
            mcp_name = mcp_tool.get("name", "")
            if not mcp_name:
                continue

            agent_tool_name = self._make_tool_name(server_name, mcp_name, use_prefix)
            
            # Skip if already registered (e.g., from a previous connect)
            if agent_tool_name in self._tool_to_server:
                continue

            # Convert schema
            schema = _mcp_schema_to_agent_schema(mcp_tool, agent_tool_name)

            # Create a closure that routes calls to the correct MCP server
            # We need to capture server_name and mcp_name by value
            def make_tool_fn(sname, mname):
                def tool_fn(**kwargs):
                    return self.call_tool(sname, mname, kwargs)
                tool_fn.__name__ = agent_tool_name
                tool_fn.__doc__ = schema["description"]
                return tool_fn

            fn = make_tool_fn(server_name, mcp_name)

            registry.register(agent_tool_name, fn, schema)
            self._tool_to_server[agent_tool_name] = server_name
            count += 1

        return count

    def _unregister_server_tools(self, server_name: str):
        """Remove all tools registered by a specific MCP server."""
        registry = get_registry()
        to_remove = [
            name for name, sname in self._tool_to_server.items()
            if sname == server_name
        ]
        for tool_name in to_remove:
            registry.unregister(tool_name)
            self._tool_to_server.pop(tool_name, None)

    def call_tool(self, server_name: str, mcp_tool_name: str,
                  arguments: dict) -> ToolResult:
        """Execute an MCP tool call and return the result as a ToolResult.

        Public surface for callers outside ``MCPManager`` (notably
        ``mcp.chrome.ChromeBrowser``) that need to invoke an MCP tool
        without owning the connection. Handles:

          - Auto-connect if the server is not currently connected.
          - Auto-reconnect-and-retry if the call fails with a
            ``is_dead_server_error`` (e.g. Chrome's CDP pipe broke but
            stdio is alive — keepalive pings pass, tool calls fail).
          - Timeout escalation for long-running tools (Lighthouse,
            performance traces, screenshots).
        """
        client = self._clients.get(server_name)
        if client is None or not client.is_connected:
            # Try to auto-reconnect
            try:
                result = self.connect(server_name)
                if result.get("status") not in ("connected", "already_connected"):
                    return ToolResult(f"MCP server '{server_name}' not connected and auto-connect failed: {result}", is_error=True)
                client = self._clients.get(server_name)
            except Exception as e:
                return ToolResult(f"MCP server '{server_name}' not connected: {e}", is_error=True)

        # Determine timeout based on tool name (some tools need longer)
        timeout = self._get_tool_timeout(mcp_tool_name, server_name)

        try:
            result = client.call_tool(mcp_tool_name, arguments, timeout=timeout)
            return ToolResult(_extract_mcp_result(result))
        except Exception as e:
            error_msg = str(e)

            # Check if this is a "dead browser" error (Chrome process died but
            # MCP stdio pipe still alive — keepalive pings pass, tool calls fail).
            # If so, force-reconnect (disconnect + connect) and retry once.
            if self.is_dead_server_error(error_msg, server_name):
                try:
                    self.disconnect(server_name)
                    self.load_servers()  # Reload config in case it changed
                    reconnect_result = self.connect(server_name)
                    if reconnect_result.get("status") not in ("connected", "already_connected"):
                        return ToolResult(
                            f"Error calling {mcp_tool_name}: {error_msg} "
                            f"(auto-reconnect failed: {reconnect_result})",
                            is_error=True
                        )
                    client = self._clients.get(server_name)
                    # Retry the tool call once
                    result = client.call_tool(mcp_tool_name, arguments, timeout=timeout)
                    return ToolResult(_extract_mcp_result(result))
                except Exception as retry_err:
                    return ToolResult(
                        f"Error calling {mcp_tool_name}: {error_msg} "
                        f"(auto-reconnect retry also failed: {retry_err})",
                        is_error=True
                    )

            return ToolResult(f"Error calling {mcp_tool_name}: {error_msg}", is_error=True)

    # Generic transport-level dead-error patterns shared by every MCP
    # server. Server-specific patterns (Chrome's CDP errors, etc.) live
    # in the per-server policy module and are merged in at lookup time.
    _GENERIC_DEAD_ERROR_PATTERNS: tuple[str, ...] = (
        "protocol error",
        "connection refused",
        "broken pipe",
        "connection reset",
    )

    @classmethod
    def is_dead_server_error(cls, error_msg: str, server_name: str = "") -> bool:
        """Check if an error indicates the underlying server process is dead.

        Detects the silent failure mode where the MCP stdio pipe is alive
        (keepalive pings pass) but the server's backend (e.g., Chrome
        browser) has crashed or its internal connection (e.g., CDP pipe)
        has broken.

        Pattern source:
          - Generic transport errors are baked in (``broken pipe``,
            ``connection reset``, etc.).
          - Server-specific errors come from the per-server policy in
            ``_SERVER_POLICIES`` (e.g. ``CHROME_DEAD_ERROR_PATTERNS`` in
            ``mcp.chrome``). Pass ``server_name`` so we can look them up.
          - When ``server_name`` is empty, ALL server policies are
            unioned (legacy behaviour — keeps callers that don't know
            the server name working).
        """
        lower = error_msg.lower()
        for pattern in cls._GENERIC_DEAD_ERROR_PATTERNS:
            if pattern in lower:
                return True
        if server_name:
            policy = _SERVER_POLICIES.get(server_name, {})
            for pattern in policy.get("dead_error_patterns", ()):
                if pattern in lower:
                    return True
        else:
            for policy in _SERVER_POLICIES.values():
                for pattern in policy.get("dead_error_patterns", ()):
                    if pattern in lower:
                        return True
        return False

    # Default timeouts for MCP tool calls. Per-server policy can extend
    # the long-running pattern list (Chrome contributes Lighthouse,
    # screenshots, performance traces — see ``CHROME_POLICY``).
    _DEFAULT_TOOL_TIMEOUT = 60.0
    _LONG_RUNNING_TOOL_TIMEOUT = 120.0

    def _get_tool_timeout(self, tool_name: str, server_name: str = "") -> float:
        """Pick a per-call timeout based on the tool name + server policy.

        Returns ``_LONG_RUNNING_TOOL_TIMEOUT`` (120s) when ``tool_name``
        matches any long-running substring contributed by the server's
        policy (e.g. Chrome's ``lighthouse`` / ``screenshot`` / ``trace`` /
        ``snapshot`` / ``memory``). Otherwise returns the default (60s).

        Without a known ``server_name``, every server policy is unioned —
        same fallback shape as ``is_dead_server_error``.
        """
        lower = tool_name.lower()
        if server_name:
            policies = [_SERVER_POLICIES.get(server_name, {})]
        else:
            policies = list(_SERVER_POLICIES.values())
        for policy in policies:
            for pattern in policy.get("long_running_tool_patterns", ()):
                if pattern in lower:
                    return self._LONG_RUNNING_TOOL_TIMEOUT
        return self._DEFAULT_TOOL_TIMEOUT

    # ─── tools/list_changed notification handler ──────────────────────────

    def _on_tools_changed(self, server_name: str):
        """Called by MCPClient when a tools/list_changed notification is received.
        
        Runs on the MCP client's event loop thread, so we need to be thread-safe.
        Re-registers tools by unregistering old ones and fetching the new list.
        """
        client = self._clients.get(server_name)
        if client is None or not client.is_connected:
            return
        
        server_config = self._configs.get(server_name, {})
        
        with self._tools_lock:
            # Unregister old tools
            self._unregister_server_tools(server_name)
            # Re-register with fresh tool list (cache was already invalidated by MCPClient)
            self._register_server_tools(server_name, client, server_config)

    # ─── Background keepalive ─────────────────────────────────────────────

    def _start_keepalive(self):
        """Start the background keepalive thread if not already running."""
        if self._keepalive_thread is not None and self._keepalive_thread.is_alive():
            return
        
        self._keepalive_stop.clear()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop,
            daemon=True,
            name="mcp-keepalive",
        )
        self._keepalive_thread.start()

    def _stop_keepalive(self):
        """Stop the background keepalive thread."""
        self._keepalive_stop.set()
        if self._keepalive_thread is not None:
            self._keepalive_thread.join(timeout=5)
            self._keepalive_thread = None

    def _keepalive_loop(self):
        """Background loop that pings all connected servers periodically.
        
        Detects dead servers early so the agent gets a clean error on the next
        tool call instead of a hung timeout. After KEEPALIVE_MAX_FAILURES consecutive
        failures, logs a warning (but does NOT auto-disconnect — let the tool call
        trigger reconnection instead, so the agent sees the error and can react).
        """
        while not self._keepalive_stop.wait(timeout=KEEPALIVE_INTERVAL_SECONDS):
            for server_name, client in list(self._clients.items()):
                if not client.is_connected:
                    continue
                
                try:
                    alive = client.send_ping()
                    if alive:
                        self._ping_failures[server_name] = 0
                    else:
                        failures = self._ping_failures.get(server_name, 0) + 1
                        self._ping_failures[server_name] = failures
                except Exception:
                    failures = self._ping_failures.get(server_name, 0) + 1
                    self._ping_failures[server_name] = failures

    # ─── Public API ───────────────────────────────────────────────────────────

    def status(self) -> str:
        """Return a human-readable status of all MCP servers."""
        if not self._configs:
            return "No MCP servers configured. Create a .mcp.json file."
        
        lines = ["MCP Servers:"]
        for name in sorted(self._configs.keys()):
            connected = self.is_connected(name)
            transport = self._configs[name].get("transport", "stdio")
            
            if connected:
                client = self._clients[name]
                tool_count = sum(1 for s in self._tool_to_server.values() if s == name)
                proto = client._protocol_version or "?"
                failures = self._ping_failures.get(name, 0)
                health = "🟢" if failures == 0 else ("🟡" if failures < KEEPALIVE_MAX_FAILURES else "🔴")
                lines.append(
                    f"  • {name}: ✅ connected {health} ({tool_count} tools, "
                    f"protocol {proto}, transport: {transport})"
                )
            else:
                lines.append(f"  • {name}: ⬚ not connected (transport: {transport})")
        
        # Show keepalive status
        keepalive_status = "running" if (self._keepalive_thread and self._keepalive_thread.is_alive()) else "stopped"
        lines.append(f"\n  Keepalive: {keepalive_status} (interval: {KEEPALIVE_INTERVAL_SECONDS}s)")
        
        return "\n".join(lines)


# ─── Singleton ────────────────────────────────────────────────────────────────

_manager: Optional[MCPManager] = None


def get_manager() -> MCPManager:
    """Get or create the singleton MCPManager (lazy, idempotent)."""
    global _manager
    if _manager is None:
        _manager = MCPManager()
        _manager.load_servers()
    return _manager


def get_manager_if_exists() -> Optional[MCPManager]:
    """Return the singleton if it has been created, else ``None``.

    Useful for shutdown hooks that want to disconnect MCP servers ONLY if
    they were ever connected — calling ``get_manager()`` would lazily
    spawn an empty manager just to ask whether one exists. Replaces the
    old ``from .mcp.manager import _manager`` private peek that
    ``agent.py`` used at exit (the singleton was previously the only way
    to do this without triggering creation).
    """
    return _manager


def reset_manager():
    """Reset the singleton (disconnects all servers)."""
    global _manager
    if _manager:
        _manager.disconnect_all()
    _manager = None


def _atexit_cleanup():
    """Safety net: disconnect all MCP servers on process exit.

    Prevents Chrome and other MCP server processes from lingering as zombies
    when the agent exits (whether cleanly or via uncaught exception).
    """
    global _manager
    if _manager and _manager._clients:
        try:
            _manager.disconnect_all()
        except Exception:
            pass


atexit.register(_atexit_cleanup)
