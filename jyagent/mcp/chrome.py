"""Chrome-specific MCP helpers.

The generic ``MCPManager`` should not know how to drive a browser.  This
module owns everything Chrome-related that previously lived in
``mcp/manager.py``:

  * pre-connect hook (``chrome_pre_connect``) — preserved for the
    ``MCPManager._PRE_CONNECT_HOOKS`` dispatch table
  * ``ChromeBrowser`` class — refcounted connection management, page-level
    serialisation, and the high-level ``fetch_page`` operation that
    ``tools/web_fetch.py`` (Tier 4) calls

Extracted from ``mcp/manager.py`` (2026-05-06) as part of the boundary
cleanup that made ``MCPManager`` browser-agnostic.

The manager owns the MCP connection lifecycle (``connect``, ``disconnect``,
``is_connected``, ``_call_mcp_tool``, ``_is_dead_server_error``); the
``ChromeBrowser`` instance composes those primitives into refcounted +
page-locked browser operations.

Backwards compatibility: ``MCPManager`` exposes the previously-public
attribute / method names (``_chrome_acquire``, ``chrome_fetch_page``,
``_chrome_refcount``, etc.) as thin proxies/properties to its
``self._chrome`` instance — so existing call sites and tests keep working
without source changes.
"""
from __future__ import annotations

import re
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import MCPManager


__all__ = ["ChromeBrowser", "chrome_pre_connect"]


# ─── Pre-connect hook ────────────────────────────────────────────────────────

def chrome_pre_connect(server_config: dict) -> dict:
    """No-op hook: lets chrome-devtools-mcp launch its own managed Chrome instance.

    Registered in ``MCPManager._PRE_CONNECT_HOOKS`` keyed on the ``"chrome"``
    server name.  Future Chrome-specific connection setup (e.g. detecting an
    already-running Chrome and attaching to it via CDP) belongs here.
    """
    return server_config


# ─── ChromeBrowser ───────────────────────────────────────────────────────────

class ChromeBrowser:
    """Refcounted, page-locked browser helper bound to one ``MCPManager``.

    State:
      * ``_page_lock`` — serialises multi-step page operations.  Chrome MCP
        has a single "selected page" cursor (evaluate_script / take_snapshot
        always run on the selected page and do NOT accept ``pageId``), so
        concurrent callers must run their new_page → select → eval → close
        sequence under this lock.
      * ``_refcount`` / ``_refcount_lock`` — connection reference count.
        Only the 0→1 transition triggers ``ensure_connected()``; only the
        1→0 transition triggers ``disconnect("chrome")``.  Prevents
        parallel sub-agents from kicking each other's Chrome session.
    """

    def __init__(self, manager: "MCPManager") -> None:
        self._mgr = manager
        # Chrome concurrency: page-level lock + connection refcount.
        # Chrome MCP has a single "selected page" cursor — evaluate_script and
        # take_snapshot always run on the selected page and do NOT accept pageId.
        # The page lock serializes multi-step Chrome operations (new_page →
        # select_page → evaluate/snapshot → close_page) so concurrent callers
        # (e.g. parallel subagents using web_fetch Chrome tier) don't interleave.
        self._page_lock = threading.Lock()
        self._refcount = 0
        self._refcount_lock = threading.Lock()

    # ─── Chrome high-level helpers ────────────────────────────────────────────
    # These encapsulate common Chrome operations so both the web_fetch Chrome
    # tier and the LLM's interactive browser-automation share one codepath.

    def acquire(self) -> None:
        """Increment Chrome connection refcount; connect on 0 → 1 transition.

        Must be paired with ``_chrome_release()`` in a ``try/finally`` block.
        Thread-safe — multiple callers can hold a reference simultaneously.
        """
        with self._refcount_lock:
            self._refcount += 1
            need_connect = (self._refcount == 1)
        if need_connect:
            # Route through the manager's public method so external callers
            # (and tests) can patch ``mgr.chrome_ensure_connected`` and have
            # it intercept the refcount-driven connect path too.
            self._mgr.chrome_ensure_connected()

    def release(self) -> None:
        """Decrement Chrome connection refcount; disconnect on 1 → 0 transition.

        Only the last caller to release triggers the actual disconnect,
        preventing premature kills while other threads are still using Chrome.
        """
        with self._refcount_lock:
            self._refcount = max(0, self._refcount - 1)
            need_disconnect = (self._refcount == 0)
        if need_disconnect and self._mgr.is_connected("chrome"):
            try:
                self._mgr.disconnect("chrome")
            except Exception:
                pass

    def ensure_connected(self) -> None:
        """Ensure Chrome MCP is connected and healthy.

        - If not connected, connects it.
        - If connected, does a health check (list_pages). If the browser is
          dead (CDP pipe broken), force-reconnects.

        Raises RuntimeError if Chrome cannot be brought to a healthy state.
        """
        server = "chrome"

        if not self._mgr.is_connected(server):
            result = self._mgr.connect(server)
            if result.get("status") not in ("connected", "already_connected"):
                raise RuntimeError(f"Failed to connect Chrome: {result}")

        # Health check: list_pages detects dead browser even if stdio pipe lives
        health = self._mgr._call_mcp_tool(server, "list_pages", {})
        if health.is_error:
            error_msg = health.content or ""
            if self._mgr._is_dead_server_error(error_msg):
                self._mgr.disconnect(server)
                result = self._mgr.connect(server)
                if result.get("status") not in ("connected", "already_connected"):
                    raise RuntimeError(
                        f"Chrome is dead and reconnect failed: {result}")
                # Re-check after reconnect
                health = self._mgr._call_mcp_tool(server, "list_pages", {})
                if health.is_error:
                    raise RuntimeError(
                        f"Chrome still unhealthy after reconnect: {health.content}")
            else:
                raise RuntimeError(f"Chrome health check failed: {error_msg}")

    def fetch_page(self, url: str, *, timeout: int = 30,
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
        # Route refcount + inner-fetch through the manager's public/private
        # delegators so tests can patch ``mgr._chrome_acquire`` /
        # ``mgr._chrome_release`` / ``mgr._chrome_fetch_page_inner`` and have
        # the patches take effect (the historical test surface).
        self._mgr._chrome_acquire()
        try:
            return self._mgr._chrome_fetch_page_inner(
                url, timeout=timeout, js_function=js_function,
            )
        finally:
            self._mgr._chrome_release()

    def _fetch_page_inner(self, url: str, *, timeout: int = 30,
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
        call = self._mgr._call_mcp_tool

        with self._page_lock:
            # Remember which page was selected so we can restore focus after
            pages_result = call(server, "list_pages", {})
            original_page_id = self._parse_page_id(
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
                new_page_id = self._parse_page_id(np.content, selected=True)
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
    def _parse_page_id(text: str, selected: bool = False) -> int | None:
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
