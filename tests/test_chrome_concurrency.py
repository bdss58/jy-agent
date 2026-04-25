"""Tests for Chrome MCP concurrency fixes.

Verifies:
1. Reference-counted Chrome connection (_chrome_acquire / _chrome_release)
2. Page-level lock prevents interleaving
3. chrome_fetch_page delegates correctly and serializes operations
"""

import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_manager():
    """Create an MCPManager with mocked MCP client."""
    from jyagent.mcp_manager import MCPManager
    mgr = MCPManager()
    mgr.load_servers()
    return mgr


def _make_tool_result(content="", is_error=False):
    """Create a mock ToolResult."""
    from jyagent.runtime.tools.result import ToolResult
    return ToolResult(content, is_error=is_error)


# ─── Tests: Reference counting ─────────────────────────────────────────────

class TestChromeRefcount:
    """_chrome_acquire / _chrome_release reference counting."""

    def test_acquire_increments_refcount(self):
        mgr = _make_manager()
        mgr.chrome_ensure_connected = MagicMock()  # Don't actually connect

        mgr._chrome_acquire()
        assert mgr._chrome_refcount == 1
        mgr._chrome_acquire()
        assert mgr._chrome_refcount == 2

    def test_release_decrements_refcount(self):
        mgr = _make_manager()
        mgr.chrome_ensure_connected = MagicMock()
        mgr._chrome_refcount = 2

        mgr._chrome_release()
        assert mgr._chrome_refcount == 1
        mgr._chrome_release()
        assert mgr._chrome_refcount == 0

    def test_release_never_goes_negative(self):
        mgr = _make_manager()
        mgr._chrome_release()
        assert mgr._chrome_refcount == 0

    def test_connect_only_on_first_acquire(self):
        mgr = _make_manager()
        mgr.chrome_ensure_connected = MagicMock()

        mgr._chrome_acquire()
        mgr._chrome_acquire()
        mgr._chrome_acquire()

        # chrome_ensure_connected called only once (on 0→1 transition)
        assert mgr.chrome_ensure_connected.call_count == 1

    def test_disconnect_only_on_last_release(self):
        mgr = _make_manager()
        mgr.chrome_ensure_connected = MagicMock()
        mgr.disconnect = MagicMock()
        mgr.is_connected = MagicMock(return_value=True)

        mgr._chrome_acquire()
        mgr._chrome_acquire()

        mgr._chrome_release()  # 2→1, should NOT disconnect
        mgr.disconnect.assert_not_called()

        mgr._chrome_release()  # 1→0, should disconnect
        mgr.disconnect.assert_called_once_with("chrome")

    def test_no_disconnect_if_not_connected(self):
        mgr = _make_manager()
        mgr.chrome_ensure_connected = MagicMock()
        mgr.disconnect = MagicMock()
        mgr.is_connected = MagicMock(return_value=False)

        mgr._chrome_acquire()
        mgr._chrome_release()  # refcount 1→0, but not connected
        mgr.disconnect.assert_not_called()


class TestChromeRefcountThreadSafety:
    """Concurrent acquire/release must not corrupt the refcount."""

    def test_concurrent_acquire_release(self):
        mgr = _make_manager()
        mgr.chrome_ensure_connected = MagicMock()
        mgr.disconnect = MagicMock()
        mgr.is_connected = MagicMock(return_value=True)

        N = 20
        barrier = threading.Barrier(N)

        def worker():
            barrier.wait()
            mgr._chrome_acquire()
            time.sleep(0.01)
            mgr._chrome_release()

        threads = [threading.Thread(target=worker) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # After all workers done, refcount must be exactly 0
        assert mgr._chrome_refcount == 0


# ─── Tests: Page lock serialization ────────────────────────────────────────

class TestChromePageLock:
    """_chrome_page_lock prevents concurrent page operations."""

    def test_fetch_page_inner_holds_lock(self):
        """Verify that _chrome_fetch_page_inner acquires the page lock."""
        mgr = _make_manager()

        # Track when the lock is held
        lock_was_held = []

        def mock_call(server, tool, args):
            if tool == "list_pages":
                return _make_tool_result("## Pages\n1: about:blank [selected]")
            elif tool == "new_page":
                return _make_tool_result("## Pages\n1: about:blank\n2: http://example.com [selected]")
            elif tool == "select_page":
                return _make_tool_result("OK")
            elif tool == "evaluate_script":
                # Check if lock is held (it should be)
                lock_held = mgr._chrome_page_lock.locked()
                lock_was_held.append(lock_held)
                return _make_tool_result('Script ran on page and returned:\n```json\n"Hello World from the page content that is long enough to pass threshold"\n```')
            elif tool == "close_page":
                return _make_tool_result("OK")
            return _make_tool_result("OK")

        mgr._call_mcp_tool = mock_call

        result = mgr._chrome_fetch_page_inner(
            "http://example.com", timeout=10,
            js_function="() => document.body.innerText"
        )

        assert any(lock_was_held), "Lock should have been held during evaluate_script"
        assert "Hello World" in result

    def test_concurrent_fetches_are_serialized(self):
        """Two concurrent _chrome_fetch_page_inner calls must not overlap."""
        mgr = _make_manager()

        execution_log = []  # [(thread_name, event, time), ...]
        log_lock = threading.Lock()

        def mock_call(server, tool, args):
            thread = threading.current_thread().name
            if tool == "list_pages":
                return _make_tool_result("## Pages\n1: about:blank [selected]")
            elif tool == "new_page":
                with log_lock:
                    execution_log.append((thread, "new_page_start", time.monotonic()))
                return _make_tool_result(f"## Pages\n1: about:blank\n2: http://test.com [selected]")
            elif tool == "select_page":
                return _make_tool_result("OK")
            elif tool == "evaluate_script":
                # Simulate some work — enough that interleaving would be visible
                time.sleep(0.05)
                with log_lock:
                    execution_log.append((thread, "evaluate_done", time.monotonic()))
                long_content = "X" * 200  # enough to pass threshold
                return _make_tool_result(f'Script ran on page and returned:\n```json\n"{long_content}"\n```')
            elif tool in ("close_page", "take_snapshot"):
                return _make_tool_result("OK")
            return _make_tool_result("OK")

        mgr._call_mcp_tool = mock_call

        def fetch(name):
            threading.current_thread().name = name
            mgr._chrome_fetch_page_inner(
                "http://test.com", timeout=10,
                js_function="() => 'test'"
            )

        t1 = threading.Thread(target=fetch, args=("T1",))
        t2 = threading.Thread(target=fetch, args=("T2",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # Both should have completed
        assert len(execution_log) >= 2

        # Check serialization: T2's new_page must start after T1's evaluate_done
        # (or vice versa). They must NOT overlap.
        new_pages = [(name, t) for name, event, t in execution_log if event == "new_page_start"]
        evals = [(name, t) for name, event, t in execution_log if event == "evaluate_done"]

        if len(new_pages) == 2 and len(evals) == 2:
            # Sort by time
            np1, np2 = sorted(new_pages, key=lambda x: x[1])
            ev1, ev2 = sorted(evals, key=lambda x: x[1])

            # The second new_page must start after the first evaluate_done
            # (serialization via lock)
            assert np2[1] >= ev1[1], (
                f"Second new_page ({np2[0]} at {np2[1]:.4f}) started before "
                f"first evaluate_done ({ev1[0]} at {ev1[1]:.4f}) — lock not working!"
            )


# ─── Tests: chrome_fetch_page integration ──────────────────────────────────

class TestChromeFetchPage:
    """chrome_fetch_page uses refcount and delegates to inner."""

    def test_acquire_release_bracketing(self):
        """chrome_fetch_page calls _chrome_acquire before and _chrome_release after."""
        mgr = _make_manager()

        call_log = []

        original_acquire = mgr._chrome_acquire
        original_release = mgr._chrome_release

        def mock_acquire():
            call_log.append("acquire")
            mgr.chrome_ensure_connected = MagicMock()
            original_acquire()

        def mock_release():
            call_log.append("release")
            # Don't actually disconnect
            with mgr._chrome_refcount_lock:
                mgr._chrome_refcount = max(0, mgr._chrome_refcount - 1)

        mgr._chrome_acquire = mock_acquire
        mgr._chrome_release = mock_release

        # Mock inner to succeed
        mgr._chrome_fetch_page_inner = MagicMock(return_value="page content")

        result = mgr.chrome_fetch_page("http://example.com")
        assert result == "page content"
        assert call_log == ["acquire", "release"]

    def test_release_called_on_error(self):
        """chrome_fetch_page calls _chrome_release even if inner raises."""
        mgr = _make_manager()
        mgr.chrome_ensure_connected = MagicMock()
        release_called = []

        original_release = mgr._chrome_release
        def mock_release():
            release_called.append(True)
            with mgr._chrome_refcount_lock:
                mgr._chrome_refcount = max(0, mgr._chrome_refcount - 1)

        mgr._chrome_release = mock_release
        mgr._chrome_fetch_page_inner = MagicMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            mgr.chrome_fetch_page("http://example.com")

        assert len(release_called) == 1, "_chrome_release must be called on error"


# ─── Tests: web_fetch _fetch_chrome delegation ─────────────────────────────

class TestFetchChromeDelgation:
    """web_fetch._fetch_chrome delegates to MCPManager.chrome_fetch_page."""

    def test_delegates_to_manager(self):
        mock_mgr = MagicMock()
        mock_mgr.chrome_fetch_page.return_value = "page content here"

        with patch("jyagent.mcp_manager.get_manager", return_value=mock_mgr):
            from jyagent.tools.web_fetch import _fetch_chrome
            status, content = _fetch_chrome("http://example.com", timeout=15)

        assert status == 200
        assert content == "page content here"
        mock_mgr.chrome_fetch_page.assert_called_once()

        # Verify js_function was passed
        call_kwargs = mock_mgr.chrome_fetch_page.call_args
        assert call_kwargs.kwargs.get("timeout") == 15
        assert "js_function" in call_kwargs.kwargs

    def test_search_url_uses_default_js(self):
        """After Chrome-SERP removal, search URLs use the same default JS as any other page.

        SERP scraping is now handled exclusively by the `web_search` tool's
        multi-engine cascade, not by web_fetch+Chrome.
        """
        mock_mgr = MagicMock()
        mock_mgr.chrome_fetch_page.return_value = "search results"

        with patch("jyagent.mcp_manager.get_manager", return_value=mock_mgr):
            from jyagent.tools.web_fetch import _fetch_chrome, _CHROME_EXTRACT_JS
            _fetch_chrome("https://www.google.com/search?q=test")

        call_kwargs = mock_mgr.chrome_fetch_page.call_args
        assert call_kwargs.kwargs["js_function"] == _CHROME_EXTRACT_JS

    def test_non_search_url_uses_default_js(self):
        """Non-search URLs should use the default JS extractor."""
        mock_mgr = MagicMock()
        mock_mgr.chrome_fetch_page.return_value = "page content"

        with patch("jyagent.mcp_manager.get_manager", return_value=mock_mgr):
            from jyagent.tools.web_fetch import _fetch_chrome, _CHROME_EXTRACT_JS
            _fetch_chrome("https://example.com/article")

        call_kwargs = mock_mgr.chrome_fetch_page.call_args
        assert call_kwargs.kwargs["js_function"] == _CHROME_EXTRACT_JS
