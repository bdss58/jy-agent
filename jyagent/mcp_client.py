"""
MCP Client — Sync wrapper around the official MCP Python SDK (v1.26.0+).

Architecture: Background thread runs asyncio event loop. A long-lived session task
holds transport + session context managers open while a call queue processes requests.

Features: tools/list_changed notification, subprocess stderr redirection to log file,
structured content extraction, all transports (stdio/HTTP/SSE).
"""

import asyncio
import io
import logging
import os
import signal
import threading
from concurrent.futures import Future as ConcurrentFuture
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Optional, Any, Callable

import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)


# ─── P5: PID-capturing stdio_client wrapper ──────────────────────────────────

@asynccontextmanager
async def _stdio_client_with_pid(params: StdioServerParameters, errlog, pid_callback):
    """Wrap stdio_client to capture the subprocess PID.

    The MCP library's stdio_client spawns a subprocess but doesn't expose its PID.
    We need the PID to force-kill the process group if graceful disconnect times out.

    Strategy: snapshot child PIDs before/after stdio_client enters, diff to find the new one.
    """
    import subprocess

    # Get our PID to find children
    my_pid = os.getpid()

    def _get_child_pids():
        """Get all child PIDs of current process using pgrep."""
        try:
            result = subprocess.run(
                ["pgrep", "-P", str(my_pid)],
                capture_output=True, text=True, timeout=3
            )
            return set(int(p) for p in result.stdout.strip().split('\n') if p.strip())
        except Exception:
            return set()

    pids_before = _get_child_pids()

    async with stdio_client(params, errlog=errlog) as (read, write):
        # Find the new child PID(s)
        pids_after = _get_child_pids()
        new_pids = pids_after - pids_before
        if new_pids:
            # Pick the lowest PID (the direct child, npm or the server)
            child_pid = min(new_pids)
            pid_callback(child_pid)
            logger.debug(f"MCP subprocess PID captured: {child_pid} (new PIDs: {new_pids})")

        yield read, write


# ─── P4: MCP subprocess stderr log file ──────────────────────────────────────

def _get_mcp_stderr_log(server_name: str) -> io.TextIOWrapper:
    """Get a file object for redirecting MCP subprocess stderr.
    
    Instead of letting MCP subprocess stderr go to sys.stderr (which corrupts
    the prompt_toolkit terminal), redirect it to a log file under data/logs/.
    
    This prevents messages like "Assertion failed" from the Chrome DevTools MCP
    Node.js process from appearing inside the CLI input prompt.
    """
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"mcp-{server_name}-stderr.log")
    return open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered


# ─── P1: Extended ClientSession with notification handling ────────────────────

class AgentClientSession(ClientSession):
    """Extended ClientSession that handles tools/list_changed notifications.
    
    The base SDK ClientSession._received_notification() only handles
    LoggingMessageNotification and ElicitCompleteNotification. We override
    it to also intercept ToolListChangedNotification so the agent can
    auto-refresh its tool registrations when the server's tools change.
    """

    def __init__(self, *args, tools_list_changed_callback: Callable | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._tools_list_changed_callback = tools_list_changed_callback

    async def _received_notification(self, notification: types.ServerNotification) -> None:
        """Handle notifications from the server, including tools/list_changed."""
        # Let the parent handle its known notifications first
        await super()._received_notification(notification)

        # P1: Handle tools/list_changed
        if isinstance(notification.root, types.ToolListChangedNotification):
            logger.info("Received tools/list_changed notification from server")
            if self._tools_list_changed_callback:
                try:
                    self._tools_list_changed_callback()
                except Exception as e:
                    logger.warning(f"tools_list_changed callback error: {e}")


class MCPClient:
    """Sync wrapper around the official MCP SDK's async ClientSession.

    Architecture:
    - A background thread runs an asyncio event loop.
    - connect() starts a long-lived _session_lifecycle task that holds all
      async context managers (transport + session) open.
    - That task processes queued calls (tool invocations, list_tools, ping, etc.)
      in the same async context, avoiding cancel scope issues.
    - disconnect() signals the lifecycle task to exit, properly closing all contexts.
    
    P1: Uses AgentClientSession (subclass of ClientSession) that intercepts
        tools/list_changed notifications to auto-invalidate tool cache.
    """

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self._session: Optional[AgentClientSession] = None
        self._server_info: Optional[types.Implementation] = None
        self._server_capabilities: Optional[types.ServerCapabilities] = None
        self._protocol_version: Optional[str] = None
        self._tools_cache: Optional[list[types.Tool]] = None

        # Background event loop
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        
        # Lifecycle management
        self._call_queue: Optional[asyncio.Queue] = None
        self._disconnect_event: Optional[asyncio.Event] = None
        self._lifecycle_task = None
        self._connected = False

        # P4: stderr log file for MCP subprocess
        self._stderr_log: Optional[io.TextIOWrapper] = None

        # P5: Track subprocess PID for force-kill on disconnect timeout
        self._subprocess_pid: Optional[int] = None

        # P1: Notification tracking
        self._tools_changed = False  # Set by notification callback
        self._on_tools_changed_callback: Optional[Callable] = None  # External callback (MCPManager)

    @property
    def is_connected(self) -> bool:
        return self._connected and self._session is not None

    @property
    def tools_changed(self) -> bool:
        """Check if tools have changed since last list_tools call."""
        return self._tools_changed

    def set_on_tools_changed(self, callback: Callable):
        """Register an external callback for when tools/list_changed fires.
        
        This is used by MCPManager to auto-refresh tool registrations.
        Callback signature: callback(server_name: str)
        """
        self._on_tools_changed_callback = callback

    def _handle_tools_list_changed(self):
        """Internal callback invoked by AgentClientSession on tools/list_changed.
        
        Runs on the MCP event loop thread. Must be thread-safe.
        """
        logger.info(f"MCP server '{self.name}': tools/list_changed — invalidating cache")
        self._tools_cache = None  # Invalidate cache
        self._tools_changed = True
        # Invoke external callback (MCPManager) if registered
        if self._on_tools_changed_callback:
            try:
                self._on_tools_changed_callback(self.name)
            except Exception as e:
                logger.warning(f"tools_changed callback error for '{self.name}': {e}")

    def _start_loop(self):
        """Start the background event loop thread."""
        if self._loop is not None and self._loop.is_running():
            return

        ready = threading.Event()

        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()

        self._loop_thread = threading.Thread(target=_run, daemon=True, name=f"mcp-{self.name}")
        self._loop_thread.start()
        ready.wait(timeout=10)

    def _stop_loop(self):
        """Stop the background event loop and thread."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread:
            self._loop_thread.join(timeout=5)
        self._loop = None
        self._loop_thread = None

    def connect(self, command: str = "", args: list[str] = None,
                env: dict[str, str] = None, startup_wait: float = 3,
                init_timeout: float = 30, cwd: str = None,
                url: str = None, transport: str = "stdio",
                headers: dict[str, str] = None) -> dict:
        """Connect to an MCP server.

        Supports stdio (default) and streamable_http transports.
        """
        if self.is_connected:
            return {"status": "already_connected", "name": self.name}

        self._start_loop()

        # Shared state for connect result
        connect_result: dict = {}
        connect_error: list = [None]
        connect_done = threading.Event()

        async def _session_lifecycle():
            """Long-lived task that holds the session open and processes calls."""
            import os
            import contextlib

            try:
                async with contextlib.AsyncExitStack() as exit_stack:
                    # Set up transport
                    if transport == "http" and url:
                        from mcp.client.streamable_http import streamable_http_client
                        read, write, _ = await exit_stack.enter_async_context(
                            streamable_http_client(url)
                        )
                    else:
                        process_env = None
                        if env:
                            resolved = {}
                            for key, value in env.items():
                                if isinstance(value, str) and value.startswith("$"):
                                    value = os.environ.get(value[1:], value)
                                resolved[key] = value
                            process_env = resolved

                        params = StdioServerParameters(
                            command=command, args=args or [],
                            env=process_env, cwd=cwd,
                        )
                        
                        # P4: Redirect MCP subprocess stderr to log file
                        # instead of sys.stderr (which corrupts prompt_toolkit terminal).
                        # This prevents "Assertion failed" and other subprocess stderr
                        # messages from appearing inside the CLI input prompt.
                        self._stderr_log = _get_mcp_stderr_log(self.name)
                        
                        read, write = await exit_stack.enter_async_context(
                            _stdio_client_with_pid(
                                params, errlog=self._stderr_log,
                                pid_callback=lambda pid: setattr(self, '_subprocess_pid', pid)
                            )
                        )

                    # P1: Create session with tools_list_changed notification handler
                    self._session = await exit_stack.enter_async_context(
                        AgentClientSession(
                            read, write,
                            read_timeout_seconds=timedelta(seconds=init_timeout),
                            client_info=types.Implementation(
                                name="self-assembling-agent", version="2.1"
                            ),
                            tools_list_changed_callback=self._handle_tools_list_changed,
                        )
                    )

                    # Initialize handshake
                    result = await self._session.initialize()

                    self._server_info = result.serverInfo
                    self._server_capabilities = result.capabilities
                    self._protocol_version = str(result.protocolVersion)
                    self._connected = True

                    connect_result.update({
                        "status": "connected",
                        "name": self.name,
                        "server_info": {
                            "name": result.serverInfo.name if result.serverInfo else "unknown",
                            "version": result.serverInfo.version if result.serverInfo else "unknown",
                        },
                        "protocol_version": self._protocol_version,
                        "capabilities": {
                            "tools": bool(result.capabilities and result.capabilities.tools),
                            "resources": bool(result.capabilities and result.capabilities.resources),
                            "prompts": bool(result.capabilities and result.capabilities.prompts),
                        },
                    })
                    connect_done.set()

                    # ── Main call-processing loop ──
                    # Processes queued calls until disconnect is signaled
                    await self._run_call_loop()

                # exit_stack closes here: session closed, process terminated per spec

            except Exception as e:
                if not connect_done.is_set():
                    connect_error[0] = e
                    connect_done.set()
                else:
                    logger.error(f"MCP session '{self.name}' error: {e}")
            finally:
                self._session = None
                self._connected = False
                # P5: Clear subprocess PID — if we reach here, the AsyncExitStack
                # already ran stdio_client's cleanup which kills the subprocess.
                self._subprocess_pid = None

        # Initialize queues
        self._call_queue = asyncio.Queue()
        self._disconnect_event = asyncio.Event()

        # Schedule the lifecycle task on the background loop
        self._lifecycle_task = asyncio.run_coroutine_threadsafe(
            _session_lifecycle(), self._loop
        )

        # Wait for connect to complete or fail
        if not connect_done.wait(timeout=init_timeout + 30):
            raise RuntimeError(f"Timeout connecting to MCP server '{self.name}'")

        if connect_error[0]:
            raise RuntimeError(
                f"Failed to connect MCP server '{self.name}': {connect_error[0]}"
            ) from connect_error[0]

        return connect_result

    async def _run_call_loop(self):
        """Process queued calls until disconnect is signaled.
        
        Runs inside _session_lifecycle, in the same async context as the session.
        """
        while True:
            # Wait for either a call or disconnect
            get_task = asyncio.ensure_future(self._call_queue.get())
            disconnect_task = asyncio.ensure_future(self._disconnect_event.wait())

            done, pending = await asyncio.wait(
                [get_task, disconnect_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for p in pending:
                p.cancel()

            # Check if disconnect was signaled
            if disconnect_task in done:
                return

            # Process the call
            if get_task in done:
                item = get_task.result()
                if item is None:
                    return  # Shutdown sentinel

                coro_factory, result_future = item
                try:
                    result = await coro_factory()
                    result_future.set_result(result)
                except Exception as e:
                    result_future.set_exception(e)

    def _submit_call(self, coro_factory: Callable, timeout: float = 60) -> Any:
        """Submit an async call to run within the session's task context.
        
        Bridge pattern: The caller's thread blocks on a ConcurrentFuture,
        while the coroutine runs on the MCP event loop thread. The asyncio.Future
        in the call queue propagates its result to the ConcurrentFuture on completion.
        
        coro_factory: A zero-arg callable that returns a coroutine.
        """
        if not self.is_connected:
            raise RuntimeError(f"MCP server '{self.name}' not connected")
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError(f"MCP client '{self.name}' event loop not running")

        # ConcurrentFuture for cross-thread result passing (thread-safe)
        result_future = ConcurrentFuture()
        loop = self._loop

        async def _enqueue():
            # asyncio.Future lives on the MCP event loop
            af = loop.create_future()

            # When the asyncio.Future completes, propagate to ConcurrentFuture
            def _on_done(f: asyncio.Future):
                try:
                    result_future.set_result(f.result())
                except asyncio.CancelledError:
                    result_future.cancel()
                except Exception as exc:
                    result_future.set_exception(exc)

            af.add_done_callback(_on_done)
            await self._call_queue.put((coro_factory, af))

        asyncio.run_coroutine_threadsafe(_enqueue(), loop).result(timeout=5)
        return result_future.result(timeout=timeout)

    def disconnect(self) -> dict:
        """Disconnect from the MCP server. Follows MCP spec shutdown sequence.

        P5: If graceful shutdown times out, force-kill the subprocess by process group.
        This prevents zombie MCP server processes from accumulating on reconnect.
        """
        if not self._connected and not self._lifecycle_task:
            return {"status": "not_connected", "name": self.name}

        # Capture PID before we start cleanup (it gets cleared below)
        subprocess_pid = self._subprocess_pid
        graceful_ok = False

        # Signal disconnect
        if self._loop and self._disconnect_event:
            self._loop.call_soon_threadsafe(self._disconnect_event.set)

        # Wait for lifecycle task to finish (context managers close properly)
        if self._lifecycle_task:
            try:
                self._lifecycle_task.result(timeout=15)
                graceful_ok = True
            except Exception as e:
                logger.debug(f"Lifecycle task ended for '{self.name}': {e}")
            self._lifecycle_task = None

        self._session = None
        self._connected = False
        self._tools_cache = None
        self._tools_changed = False
        self._server_info = None
        self._server_capabilities = None
        self._call_queue = None
        self._disconnect_event = None

        # P4: Close stderr log file
        if self._stderr_log:
            try:
                self._stderr_log.close()
            except Exception:
                pass
            self._stderr_log = None

        # Stop the background event loop
        self._stop_loop()

        # P5+: ALWAYS force-kill the subprocess tree on disconnect.
        # Even when graceful shutdown "succeeds" (lifecycle task completes),
        # the MCP server's child processes (e.g., Chrome launched with setsid())
        # may survive in their own process group. Graceful shutdown only closes
        # the stdio pipe and kills the direct MCP server process, but Chrome's
        # independent process group lives on — preventing reconnect due to
        # port conflicts or stale state.
        if subprocess_pid:
            self._force_kill_subprocess(subprocess_pid)

        self._subprocess_pid = None

        return {"status": "disconnected", "name": self.name}

    def _force_kill_subprocess(self, pid: int):
        """Force-kill a subprocess and all its descendants.

        P5: Last-resort cleanup when graceful disconnect times out.

        Challenge: The MCP server (npm → chrome-devtools-mcp) is in one process group,
        but Chrome (launched by the MCP server) does setsid() and runs in its own
        process group. So os.killpg() on the npm group doesn't kill Chrome.

        Strategy:
        1. Walk the process tree to find ALL descendants (using pgrep -P recursively)
        2. Kill Chrome's process group first (SIGTERM)
        3. Kill the MCP server's process group (SIGTERM)
        4. Escalate to SIGKILL if anything survives
        """
        import time
        import subprocess as sp

        def _get_descendants(root_pid: int) -> set:
            """Recursively find all descendant PIDs."""
            descendants = set()
            to_visit = [root_pid]
            while to_visit:
                parent = to_visit.pop()
                try:
                    result = sp.run(
                        ["pgrep", "-P", str(parent)],
                        capture_output=True, text=True, timeout=3
                    )
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            child_pid = int(line.strip())
                            if child_pid not in descendants:
                                descendants.add(child_pid)
                                to_visit.append(child_pid)
                except Exception:
                    pass
            return descendants

        def _kill_pgid(pgid: int, sig: int):
            """Send signal to a process group, ignoring errors."""
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError, OSError):
                pass

        try:
            logger.info(
                f"MCP '{self.name}': force-killing subprocess tree to ensure "
                f"no orphan processes (root PID {pid})"
            )

            # Collect all process group IDs in the tree
            all_descendants = _get_descendants(pid)
            all_pids = {pid} | all_descendants
            pgids = set()
            for p in all_pids:
                try:
                    pgids.add(os.getpgid(p))
                except (ProcessLookupError, OSError):
                    pass

            logger.debug(f"MCP '{self.name}': killing PGIDs {pgids} (PIDs: {all_pids})")

            # SIGTERM all process groups
            for pgid in pgids:
                _kill_pgid(pgid, signal.SIGTERM)

            # Wait briefly for termination
            for _ in range(20):  # 2 seconds max
                alive = False
                for p in all_pids:
                    try:
                        os.kill(p, 0)  # Check if still alive
                        alive = True
                        break
                    except ProcessLookupError:
                        pass
                    except OSError:
                        alive = True
                        break
                if not alive:
                    return
                time.sleep(0.1)

            # Still alive — SIGKILL all process groups
            for pgid in pgids:
                _kill_pgid(pgid, signal.SIGKILL)

            logger.warning(f"MCP '{self.name}': sent SIGKILL to process groups {pgids}")

        except Exception as e:
            logger.warning(f"MCP '{self.name}': force-kill failed for PID {pid}: {e}")

    # ─── Public API ──────────────────────────────────────────────────────────

    def list_tools(self, use_cache: bool = True) -> list[dict]:
        """List all tools. Supports pagination automatically.
        
        P1: If tools_changed flag is set (from notification), cache is already
        invalidated. Clear the flag after re-fetching.
        """
        if use_cache and self._tools_cache is not None and not self._tools_changed:
            return [self._tool_to_dict(t) for t in self._tools_cache]

        def _factory():
            async def _list():
                tools = []
                cursor = None
                while True:
                    result = await self._session.list_tools(
                        params=types.PaginatedRequestParams(cursor=cursor) if cursor else None
                    )
                    tools.extend(result.tools)
                    cursor = result.nextCursor
                    if not cursor:
                        break
                return tools
            return _list()

        self._tools_cache = self._submit_call(_factory, timeout=30)
        self._tools_changed = False  # P1: Reset flag after successful refresh
        return [self._tool_to_dict(t) for t in self._tools_cache]

    def call_tool(self, tool_name: str, arguments: dict = None,
                  timeout: float = 60) -> dict:
        """Call an MCP tool by name."""
        def _factory():
            return self._session.call_tool(
                tool_name,
                arguments or {},
                read_timeout_seconds=timedelta(seconds=timeout),
            )

        result = self._submit_call(_factory, timeout=timeout + 10)
        return self._call_result_to_dict(result)

    def send_ping(self) -> bool:
        """Ping the MCP server. Returns True if alive."""
        if not self.is_connected:
            return False
        try:
            def _factory():
                return self._session.send_ping()
            self._submit_call(_factory, timeout=10)
            return True
        except Exception:
            return False

    def list_resources(self) -> list[dict]:
        """List resources exposed by the server."""
        def _factory():
            async def _list():
                result = await self._session.list_resources()
                return [{"uri": str(r.uri), "name": r.name, "mimeType": r.mimeType}
                        for r in result.resources]
            return _list()

        return self._submit_call(_factory, timeout=30)

    def get_server_capabilities(self) -> Optional[dict]:
        """Return server capabilities from initialization."""
        if self._server_capabilities is None:
            return None
        return self._server_capabilities.model_dump(exclude_none=True)

    # ─── Conversion helpers ──────────────────────────────────────────────────

    @staticmethod
    def _tool_to_dict(tool: types.Tool) -> dict:
        """Convert Pydantic Tool → dict for agent compatibility."""
        return {
            "name": tool.name,
            "description": tool.description or f"MCP tool: {tool.name}",
            "inputSchema": tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}},
        }

    @staticmethod
    def _call_result_to_dict(result: types.CallToolResult) -> dict:
        """Convert CallToolResult → dict for agent compatibility.
        
        P3: Forward-compatible with SDK v2's structured_content field.
        """
        content = []
        for item in (result.content or []):
            if isinstance(item, types.TextContent):
                content.append({"type": "text", "text": item.text})
            elif isinstance(item, types.ImageContent):
                content.append({
                    "type": "image",
                    "data": item.data,
                    "mimeType": item.mimeType,
                })
            elif isinstance(item, types.EmbeddedResource):
                content.append({
                    "type": "resource",
                    "resource": item.resource.model_dump() if item.resource else {},
                })
            else:
                content.append(item.model_dump() if hasattr(item, 'model_dump') else {"type": "unknown"})

        result_dict = {
            "content": content,
            "isError": result.isError or False,
        }

        # P3: Forward-compat — extract structured_content if SDK v2 adds it
        if hasattr(result, 'structuredContent') and result.structuredContent:
            result_dict["structuredContent"] = result.structuredContent
        elif hasattr(result, 'structured_content') and result.structured_content:
            result_dict["structuredContent"] = result.structured_content

        return result_dict

    def __del__(self):
        try:
            if self._connected:
                self.disconnect()
        except Exception:
            pass

    def __repr__(self):
        status = "connected" if self.is_connected else "disconnected"
        return f"MCPClient(name={self.name!r}, status={status}, protocol={self._protocol_version})"
