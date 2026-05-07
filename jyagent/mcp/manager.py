"""
MCP Manager — Manages MCP server lifecycle and dynamically registers tools.

Reads .mcp.json config, connects to servers, discovers tools via MCP SDK,
and registers them into the agent's ToolRegistry. Includes background
keepalive pings and tools/list_changed notification handling.
"""

import atexit
import json
import os
import re
import threading
from typing import Optional
from .client import MCPClient
from ..runtime.tools.registry import get_registry
from ..runtime.tools.result import ToolResult

# Module version marker — bump this to confirm reloads are taking effect
_MODULE_VERSION = "2.1.0"

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


# ─── Pre-connect hook registry ────────────────────────────────────────────────
#
# Browser-specific hooks live in ``mcp/chrome.py``; the manager only wires
# them by server name.  Add new server-specific hooks here.

from .chrome import chrome_pre_connect, ChromeBrowser  # noqa: E402

_PRE_CONNECT_HOOKS = {
    "chrome": chrome_pre_connect,
}


# ─── MCP Tool → Agent Tool schema conversion ─────────────────────────────────

def _mcp_schema_to_agent_schema(mcp_tool: dict, tool_name: str) -> dict:
    """Convert an MCP tool schema to the agent's tool schema format.
    
    MCP format:
        {"name": "navigate_page", "description": "...", "inputSchema": {"type": "object", "properties": {...}}}
    
    Agent format:
        {"name": "mcp__chrome__navigate_page", "description": "...", "input_schema": {"type": "object", "properties": {...}}}
    """
    input_schema = mcp_tool.get("inputSchema", {"type": "object", "properties": {}})
    
    # Deep copy to avoid mutating the original
    import copy
    input_schema = copy.deepcopy(input_schema)
    
    # Ensure required field exists
    if "required" not in input_schema:
        input_schema["required"] = []
    
    # Strip 'default' keys that Bedrock doesn't support
    if "properties" in input_schema:
        for prop_name, prop_def in input_schema["properties"].items():
            if isinstance(prop_def, dict):
                prop_def.pop("default", None)
                # Strip 'additionalProperties' from nested objects
                prop_def.pop("additionalProperties", None)
    
    # Strip top-level additionalProperties  
    input_schema.pop("additionalProperties", None)
    
    return {
        "name": tool_name,
        "description": mcp_tool.get("description", f"MCP tool: {mcp_tool.get('name', tool_name)}"),
        "input_schema": input_schema,
    }


def _extract_mcp_result(result: dict) -> str:
    """Extract readable text from an MCP tool call result."""
    content = result.get("content", [])
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "image":
                    texts.append(f"[Image: {item.get('mimeType', 'image')}]")
                elif "text" in item:
                    texts.append(item["text"])
                else:
                    texts.append(json.dumps(item, ensure_ascii=False))
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts) if texts else json.dumps(result, indent=2, ensure_ascii=False)
    elif isinstance(content, str):
        return content
    else:
        return json.dumps(result, indent=2, ensure_ascii=False)


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
        self._tool_to_mcp_name: dict[str, str] = {}  # agent_tool_name → original mcp tool name
        self._registered_tools: set[str] = set()
        
        # Keepalive
        self._keepalive_thread: Optional[threading.Thread] = None
        self._keepalive_stop = threading.Event()
        self._ping_failures: dict[str, int] = {}  # server_name → consecutive failure count
        
        # Lock for thread-safe tool re-registration
        self._tools_lock = threading.Lock()

        # Browser-specific helper.  See ``mcp/chrome.py``.  All Chrome ops
        # (refcounted connection, page-lock serialisation, fetch_page) are
        # delegated to this object.  Backwards-compat shims further below
        # forward the legacy ``_chrome_*`` / ``chrome_*`` names so existing
        # tests and callers (notably ``tools/web_fetch.py``) keep working.
        self._chrome = ChromeBrowser(self)

        # Module version for debugging reload issues
        self._module_version = _MODULE_VERSION

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
        
        # Apply pre-connect hook if exists
        hook = _PRE_CONNECT_HOOKS.get(server_name)
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
            if agent_tool_name in self._registered_tools:
                continue

            # Convert schema
            schema = _mcp_schema_to_agent_schema(mcp_tool, agent_tool_name)

            # Create a closure that routes calls to the correct MCP server
            # We need to capture server_name and mcp_name by value
            def make_tool_fn(sname, mname):
                def tool_fn(**kwargs):
                    return self._call_mcp_tool(sname, mname, kwargs)
                tool_fn.__name__ = agent_tool_name
                tool_fn.__doc__ = schema["description"]
                return tool_fn

            fn = make_tool_fn(server_name, mcp_name)

            registry.register(agent_tool_name, fn, schema)
            self._tool_to_server[agent_tool_name] = server_name
            self._tool_to_mcp_name[agent_tool_name] = mcp_name
            self._registered_tools.add(agent_tool_name)
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
            self._tool_to_mcp_name.pop(tool_name, None)
            self._registered_tools.discard(tool_name)

    def _call_mcp_tool(self, server_name: str, mcp_tool_name: str,
                       arguments: dict) -> ToolResult:
        """Execute an MCP tool call and return the result as a ToolResult.
        
        Auto-reconnect: If the server is not connected, or if the call fails with
        a "dead browser" error (e.g., Chrome's CDP pipe broke but stdio is alive),
        this method will attempt to reconnect and retry once.
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
        timeout = self._get_tool_timeout(mcp_tool_name)

        try:
            result = client.call_tool(mcp_tool_name, arguments, timeout=timeout)
            return ToolResult(_extract_mcp_result(result))
        except Exception as e:
            error_msg = str(e)
            
            # Check if this is a "dead browser" error (Chrome process died but
            # MCP stdio pipe still alive — keepalive pings pass, tool calls fail).
            # If so, force-reconnect (disconnect + connect) and retry once.
            if self._is_dead_server_error(error_msg):
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

    @staticmethod
    def _is_dead_server_error(error_msg: str) -> bool:
        """Check if an error indicates the underlying server process is dead.
        
        This detects the silent failure mode where the MCP stdio pipe is alive
        (keepalive pings pass) but the server's backend (e.g., Chrome browser)
        has crashed or its internal connection (e.g., CDP pipe) has broken.
        """
        lower = error_msg.lower()
        dead_patterns = [
            "target closed",
            "session closed",
            "browser disconnected",
            "browser has been closed",
            "browser was closed",
            "not connected to devtools",
            "protocol error",
            "connection refused",
            "no page available",
            "page has been closed",
            "execution context was destroyed",
            "inspected target navigated or closed",
            # Generic MCP-level connection errors
            "broken pipe",
            "connection reset",
        ]
        return any(pattern in lower for pattern in dead_patterns)

    def _get_tool_timeout(self, tool_name: str) -> float:
        """Get appropriate timeout for a tool based on its name.
        
        Some MCP tools (Lighthouse, performance traces, etc.) can take
        much longer than the default 30s.
        """
        long_running_patterns = [
            "lighthouse", "performance", "trace", "audit",
            "screenshot", "snapshot", "memory",
        ]
        for pattern in long_running_patterns:
            if pattern in tool_name.lower():
                return 120
        return 60  # Default: 60s (was 30s, increased for reliability)

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

    # ─── Chrome delegation + back-compat shims ────────────────────────────────
    #
    # All Chrome implementation lives in ``mcp/chrome.py``.  ``MCPManager``
    # exposes the previously-public attribute / method names below as thin
    # forwarders so legacy callers (notably ``tools/web_fetch.py`` and
    # ``tests/test_chrome_concurrency.py``) keep working unchanged.
    # New code should reach into ``self._chrome`` (a ``ChromeBrowser``) directly.

    # Public delegators (kept stable for ``tools/web_fetch.py``).
    def chrome_fetch_page(self, url: str, *, timeout: int = 30,
                          js_function: str = "") -> str:
        return self._chrome.fetch_page(
            url, timeout=timeout, js_function=js_function,
        )

    def chrome_ensure_connected(self) -> None:
        return self._chrome.ensure_connected()

    # Private back-compat shims used by ``tests/test_chrome_concurrency.py``.
    def _chrome_acquire(self) -> None:
        return self._chrome.acquire()

    def _chrome_release(self) -> None:
        return self._chrome.release()

    def _chrome_fetch_page_inner(self, url: str, *, timeout: int = 30,
                                 js_function: str = "") -> str:
        return self._chrome._fetch_page_inner(
            url, timeout=timeout, js_function=js_function,
        )

    @staticmethod
    def _extract_js_result(evaluate_output: str) -> str:
        # Preserve the static-method import path used by tests / external code.
        return ChromeBrowser._extract_js_result(evaluate_output)

    @staticmethod
    def _parse_chrome_page_id(text: str, selected: bool = False):
        return ChromeBrowser._parse_page_id(text, selected=selected)

    # Attribute-level back-compat: the page lock and refcount state lived on
    # ``MCPManager`` historically; tests both read and WRITE these
    # (``mgr._chrome_refcount = 2`` to seed concurrency tests), so we expose
    # them as properties that proxy to ``self._chrome``.
    @property
    def _chrome_page_lock(self):
        return self._chrome._page_lock

    @property
    def _chrome_refcount(self) -> int:
        return self._chrome._refcount

    @_chrome_refcount.setter
    def _chrome_refcount(self, value: int) -> None:
        self._chrome._refcount = value

    @property
    def _chrome_refcount_lock(self):
        return self._chrome._refcount_lock

    # ─── Public API ───────────────────────────────────────────────────────────

    def ping(self, server_name: str) -> bool:
        """Ping a server to check if it's alive."""
        client = self._clients.get(server_name)
        if client is None:
            return False
        return client.send_ping()

    def server_capabilities(self, server_name: str) -> Optional[dict]:
        """Get server capabilities for a connected server."""
        client = self._clients.get(server_name)
        if client is None:
            return None
        return client.get_server_capabilities()

    def status(self) -> str:
        """Return a human-readable status of all MCP servers."""
        if not self._configs:
            return "No MCP servers configured. Create a .mcp.json file."
        
        lines = [f"MCP Servers (module v{_MODULE_VERSION}):"]
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
    """Get or create the singleton MCPManager.
    
    If the module has been reloaded (version mismatch), automatically
    recreates the manager to pick up code changes.
    """
    global _manager
    if _manager is None:
        _manager = MCPManager()
        _manager.load_servers()
    elif getattr(_manager, '_module_version', None) != _MODULE_VERSION:
        # Module was reloaded — recreate manager to pick up code changes
        old_clients = dict(_manager._clients)
        old_configs = dict(_manager._configs)
        _manager._stop_keepalive()
        # Don't disconnect existing clients — just re-wrap them
        _manager = MCPManager()
        _manager._configs = old_configs
        _manager._clients = old_clients
        _manager._module_version = _MODULE_VERSION
        # Re-register tools for all connected servers
        for server_name, client in old_clients.items():
            if client.is_connected:
                server_config = old_configs.get(server_name, {})
                _manager._register_server_tools(server_name, client, server_config)
                _manager._ping_failures[server_name] = 0
        if old_clients:
            _manager._start_keepalive()
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
