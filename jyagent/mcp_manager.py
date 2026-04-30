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
from .mcp_client import MCPClient
from .runtime.tools.registry import get_registry
from .runtime.tools.result import ToolResult

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


# ─── Chrome-specific pre-connect hook ────────────────────────────────────────

def _chrome_pre_connect(server_config: dict) -> dict:
    """No-op hook: lets chrome-devtools-mcp launch its own managed Chrome instance."""
    return server_config


# Map of server names to pre-connect hooks
_PRE_CONNECT_HOOKS = {
    "chrome": _chrome_pre_connect,
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
        
        # Chrome concurrency: page-level lock + connection refcount.
        # Chrome MCP has a single "selected page" cursor — evaluate_script and
        # take_snapshot always run on the selected page and do NOT accept pageId.
        # The page lock serializes multi-step Chrome operations (new_page →
        # select_page → evaluate/snapshot → close_page) so concurrent callers
        # (e.g. parallel subagents using web_fetch Chrome tier) don't interleave.
        self._chrome_page_lock = threading.Lock()
        self._chrome_refcount = 0
        self._chrome_refcount_lock = threading.Lock()
        
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

    # ─── Chrome high-level helpers ────────────────────────────────────────────
    # These encapsulate common Chrome operations so both the web_fetch Chrome
    # tier and the LLM's interactive browser-automation share one codepath.

    def _chrome_acquire(self) -> None:
        """Increment Chrome connection refcount; connect on 0 → 1 transition.

        Must be paired with ``_chrome_release()`` in a ``try/finally`` block.
        Thread-safe — multiple callers can hold a reference simultaneously.
        """
        with self._chrome_refcount_lock:
            self._chrome_refcount += 1
            need_connect = (self._chrome_refcount == 1)
        if need_connect:
            self.chrome_ensure_connected()

    def _chrome_release(self) -> None:
        """Decrement Chrome connection refcount; disconnect on 1 → 0 transition.

        Only the last caller to release triggers the actual disconnect,
        preventing premature kills while other threads are still using Chrome.
        """
        with self._chrome_refcount_lock:
            self._chrome_refcount = max(0, self._chrome_refcount - 1)
            need_disconnect = (self._chrome_refcount == 0)
        if need_disconnect and self.is_connected("chrome"):
            try:
                self.disconnect("chrome")
            except Exception:
                pass

    def chrome_ensure_connected(self) -> None:
        """Ensure Chrome MCP is connected and healthy.

        - If not connected, connects it.
        - If connected, does a health check (list_pages). If the browser is
          dead (CDP pipe broken), force-reconnects.

        Raises RuntimeError if Chrome cannot be brought to a healthy state.
        """
        server = "chrome"

        if not self.is_connected(server):
            result = self.connect(server)
            if result.get("status") not in ("connected", "already_connected"):
                raise RuntimeError(f"Failed to connect Chrome: {result}")

        # Health check: list_pages detects dead browser even if stdio pipe lives
        health = self._call_mcp_tool(server, "list_pages", {})
        if health.is_error:
            error_msg = health.content or ""
            if self._is_dead_server_error(error_msg):
                self.disconnect(server)
                result = self.connect(server)
                if result.get("status") not in ("connected", "already_connected"):
                    raise RuntimeError(
                        f"Chrome is dead and reconnect failed: {result}")
                # Re-check after reconnect
                health = self._call_mcp_tool(server, "list_pages", {})
                if health.is_error:
                    raise RuntimeError(
                        f"Chrome still unhealthy after reconnect: {health.content}")
            else:
                raise RuntimeError(f"Chrome health check failed: {error_msg}")

    def chrome_fetch_page(self, url: str, *, timeout: int = 30,
                          js_function: str = "") -> str:
        """Open a URL in a temporary Chrome tab, extract content, and return text.

        This is the shared primitive used by:
        - web_fetch's Chrome tier (Tier 4) for headless page fetching
        - Any future code that needs "get me the text of this page via Chrome"

        Uses reference-counted Chrome connection management so concurrent
        callers (parallel subagents) don't prematurely kill the browser.
        All page operations are serialized via ``_chrome_page_lock``.

        Args:
            url: The URL to fetch.
            timeout: Navigation timeout in seconds.
            js_function: Optional JS function string to run via evaluate_script
                for content extraction. If empty or if JS returns too little
                content, falls back to take_snapshot.

        Returns:
            The text content of the page.

        Raises:
            RuntimeError: If any step fails (connection, navigation, extraction).
        """
        self._chrome_acquire()
        try:
            return self._chrome_fetch_page_inner(
                url, timeout=timeout, js_function=js_function,
            )
        finally:
            self._chrome_release()

    def _chrome_fetch_page_inner(self, url: str, *, timeout: int = 30,
                                 js_function: str = "") -> str:
        """Core fetch logic — runs under ``_chrome_page_lock``.

        Caller must have already called ``_chrome_acquire()`` (refcount).
        The page lock is acquired here so the entire new_page → select_page →
        evaluate/snapshot → close_page sequence is atomic w.r.t. other threads.

        Args:
            url: URL to navigate to.
            timeout: Navigation timeout in seconds.
            js_function: Optional JS function for evaluate_script extraction.
                If it returns fewer than 100 chars, falls back to take_snapshot.
        """
        import json as _json
        import time as _time

        server = "chrome"
        call = self._call_mcp_tool

        with self._chrome_page_lock:
            # Remember which page was selected so we can restore focus after
            pages_result = call(server, "list_pages", {})
            original_page_id = self._parse_chrome_page_id(
                pages_result.content, selected=True
            ) if not pages_result.is_error else None

            new_page_id = None
            try:
                # Open URL in a new tab
                np = call(server, "new_page", {
                    "url": url,
                    "timeout": timeout * 1000,
                })
                if np.is_error:
                    raise RuntimeError(f"new_page failed: {np.content}")

                # Parse the new page ID
                new_page_id = self._parse_chrome_page_id(np.content, selected=True)
                if new_page_id is not None and new_page_id == original_page_id:
                    new_page_id = None  # mis-parse guard

                # Explicitly select the new page — critical for correctness.
                # Even though new_page auto-selects, another thread could have
                # snuck in a select_page between our new_page and this call if
                # the lock were not held.  The select_page call is cheap and
                # makes the code safe regardless of lock granularity changes.
                if new_page_id is not None:
                    call(server, "select_page", {"pageId": new_page_id})

                # Brief wait for dynamic content to render
                _time.sleep(1.0)

                content = ""

                # Try JS extraction first if a function was provided
                if js_function:
                    # Re-select to be absolutely sure we're on the right page
                    if new_page_id is not None:
                        call(server, "select_page", {"pageId": new_page_id})
                    extract_result = call(server, "evaluate_script", {
                        "function": js_function,
                    })
                    if not extract_result.is_error:
                        content = self._extract_js_result(extract_result.content)

                # Fall back to take_snapshot if JS extraction yielded too little
                if len(content) < 100:
                    if new_page_id is not None:
                        call(server, "select_page", {"pageId": new_page_id})
                    snap = call(server, "take_snapshot", {})
                    if not snap.is_error:
                        snapshot_text = snap.content.strip()
                        if len(snapshot_text) >= len(content):
                            content = snapshot_text

                if not content or len(content.strip()) < 50:
                    raise RuntimeError(
                        f"Chrome returned too little content: {len(content)} chars"
                    )

                return content

            finally:
                # Close the tab we opened
                if new_page_id is not None:
                    try:
                        call(server, "close_page", {"pageId": new_page_id})
                    except Exception:
                        pass
                # Restore original tab focus (may fail if page was closed
                # externally, which is fine — best effort)
                if original_page_id is not None:
                    try:
                        call(server, "select_page", {"pageId": original_page_id})
                    except Exception:
                        pass

    @staticmethod
    def _extract_js_result(evaluate_output: str) -> str:
        """Extract the actual JS result from evaluate_script output.

        Output format: 'Script ran on page and returned:\\n```json\\n"actual text"\\n```'
        """
        import json as _json
        if not evaluate_output:
            return ""
        m = re.search(r'```(?:json)?\s*\n(.*?)\n```', evaluate_output, re.DOTALL)
        if m:
            raw = m.group(1).strip()
            try:
                decoded = _json.loads(raw)
                if isinstance(decoded, str):
                    return decoded
            except (ValueError, _json.JSONDecodeError):
                pass
            return raw
        text = evaluate_output
        for prefix in ("Script ran on page and returned:", "Result:"):
            if text.startswith(prefix):
                text = text[len(prefix):]
        return text.strip()

    @staticmethod
    def _parse_chrome_page_id(text: str, selected: bool = False) -> int | None:
        """Parse a page ID from Chrome MCP list_pages / new_page output.

        Format:
            ## Pages
            2: https://example.com
            3: https://google.com [selected]

        Args:
            text: Raw MCP tool output.
            selected: If True, return the ``[selected]`` page ID.
                      If False, return the highest (most recent) page ID.
        """
        best_id = None
        for line in text.splitlines():
            m = re.match(r'\s*(\d+):\s', line)
            if not m:
                continue
            page_id = int(m.group(1))
            if selected:
                if "selected" in line.lower():
                    return page_id
            else:
                if best_id is None or page_id > best_id:
                    best_id = page_id
        return best_id

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
