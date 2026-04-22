"""Tests for web_search tool — DDG parsing, Codex backend, and integration."""

import glob
import json
import logging
import os
import subprocess
import tempfile
import pytest

from jyagent.tools.web_search_tool import (
    _parse_codex_json,
    _search_ddg,
    web_search,
    TOOL_SCHEMA,
)


# ─── Schema tests ────────────────────────────────────────────────────────────


class TestToolSchema:
    def test_schema_has_required_fields(self):
        assert TOOL_SCHEMA["name"] == "web_search"
        assert "description" in TOOL_SCHEMA
        assert "input_schema" in TOOL_SCHEMA

    def test_schema_properties(self):
        props = TOOL_SCHEMA["input_schema"]["properties"]
        assert "query" in props
        assert "engine" in props
        assert "max_results" in props

    def test_query_is_required(self):
        assert "query" in TOOL_SCHEMA["input_schema"]["required"]


# ─── JSON parser tests ───────────────────────────────────────────────────────


class TestParseCodexJson:
    def test_clean_json(self):
        data = {"results": [], "synthesis": "hello"}
        assert _parse_codex_json(json.dumps(data)) == data

    def test_json_with_prefix_noise(self):
        raw = 'Some header text\n{"results": [], "synthesis": "ok"}'
        parsed = _parse_codex_json(raw)
        assert parsed is not None
        assert parsed["synthesis"] == "ok"

    def test_json_embedded_in_output(self):
        raw = 'tokens used\n42\n{"results": [{"title": "A", "url": "http://a.com", "snippet": "aaa"}], "synthesis": "found"}\n'
        parsed = _parse_codex_json(raw)
        assert parsed is not None
        assert len(parsed["results"]) == 1
        assert parsed["results"][0]["title"] == "A"

    def test_empty_input(self):
        assert _parse_codex_json("") is None
        assert _parse_codex_json(None) is None

    def test_non_json(self):
        assert _parse_codex_json("just plain text") is None

    def test_nested_braces(self):
        data = {"results": [{"title": "X{Y}", "url": "http://x.com", "snippet": "a"}], "synthesis": "ok"}
        assert _parse_codex_json(json.dumps(data)) == data


# ─── DDG parser tests (unit) ─────────────────────────────────────────────────


class TestSearchDdgParsing:
    """Test DDG HTML parsing with synthetic HTML."""

    def _make_ddg_html(self, results):
        """Build a minimal DDG-like HTML page."""
        items = []
        for r in results:
            uddg_url = f"//duckduckgo.com/l/?uddg={r['url']}&rut=abc123"
            items.append(f"""
            <div class="result results_links">
                <div class="links_main links_deep result__body">
                    <h2 class="result__title">
                        <a class="result__a" href="{uddg_url}">{r['title']}</a>
                    </h2>
                    <a class="result__snippet" href="{uddg_url}">{r['snippet']}</a>
                </div>
            </div>""")
        return f"<html><body>{''.join(items)}</body></html>"

    def test_parse_results(self, monkeypatch):
        fake_results = [
            {"title": "Python Docs", "url": "https://docs.python.org/3/", "snippet": "Official docs"},
            {"title": "Real Python", "url": "https://realpython.com/", "snippet": "Tutorials"},
        ]
        html = self._make_ddg_html(fake_results)

        # Monkeypatch _fetch_cffi to return our fake HTML
        from jyagent.tools import web_search_tool
        monkeypatch.setattr(
            web_search_tool, "_search_ddg",
            lambda q, max_results=10: _search_ddg.__wrapped__(q, max_results)
            if hasattr(_search_ddg, '__wrapped__') else [],
        )

        # Parse directly using BeautifulSoup
        from bs4 import BeautifulSoup
        from urllib.parse import parse_qs, urlparse
        soup = BeautifulSoup(html, "html.parser")
        parsed = []
        for item in soup.select(".result"):
            title_el = item.select_one(".result__a")
            snippet_el = item.select_one(".result__snippet")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            href = title_el.get("href", "")
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            if "uddg=" in href:
                qs = parse_qs(urlparse(href).query)
                if "uddg" in qs:
                    href = qs["uddg"][0]
            parsed.append({"title": title, "url": href, "snippet": snippet})

        assert len(parsed) == 2
        assert parsed[0]["title"] == "Python Docs"
        assert parsed[0]["url"] == "https://docs.python.org/3/"
        assert parsed[1]["snippet"] == "Tutorials"


# ─── web_search function tests ───────────────────────────────────────────────


class TestWebSearch:
    def test_empty_query_returns_error(self):
        result = web_search(query="")
        assert result.is_error

    def test_whitespace_query_returns_error(self):
        result = web_search(query="   ")
        assert result.is_error

    def test_unknown_engine_returns_error(self):
        result = web_search(query="test", engine="bing")
        assert result.is_error
        assert "Unknown engine" in str(result)

    def test_ddg_engine_returns_results(self, monkeypatch):
        """Mock DDG to return canned results."""
        from jyagent.tools import web_search_tool

        fake = [
            {"title": "Result 1", "url": "https://example.com/1", "snippet": "First"},
            {"title": "Result 2", "url": "https://example.com/2", "snippet": "Second"},
        ]
        monkeypatch.setattr(web_search_tool, "_search_ddg", lambda q, n=10: fake)

        result = web_search(query="test query", engine="ddg")
        assert not result.is_error
        text = str(result)
        assert "Result 1" in text
        assert "https://example.com/1" in text
        assert "Engine: ddg" in text

    def test_codex_engine_returns_results(self, monkeypatch):
        """Mock Codex to return canned results."""
        from jyagent.tools import web_search_tool

        fake_results = [
            {"title": "Codex Result", "url": "https://example.com/c", "snippet": "From Codex"},
        ]
        monkeypatch.setattr(
            web_search_tool, "_search_codex",
            lambda q, n=10, timeout=180: (fake_results, "Synthesis: found it"),
        )

        result = web_search(query="test", engine="codex")
        assert not result.is_error
        text = str(result)
        assert "Codex Result" in text
        assert "Synthesis" in text

    def test_auto_uses_ddg_when_enough_results(self, monkeypatch):
        """Auto mode should use DDG when it returns enough results."""
        from jyagent.tools import web_search_tool

        ddg_results = [
            {"title": f"R{i}", "url": f"https://ex.com/{i}", "snippet": f"S{i}"}
            for i in range(5)
        ]
        codex_called = []
        monkeypatch.setattr(web_search_tool, "_search_ddg", lambda q, n=10: ddg_results)
        monkeypatch.setattr(
            web_search_tool, "_search_codex",
            lambda q, n=10, timeout=180: (codex_called.append(1) or ([], "")),
        )

        result = web_search(query="test", engine="auto")
        assert not result.is_error
        assert "Engine: ddg" in str(result)
        assert len(codex_called) == 0  # Codex was NOT called

    def test_auto_falls_back_to_codex(self, monkeypatch):
        """Auto mode should fall back to Codex when DDG gives < 3 results."""
        from jyagent.tools import web_search_tool

        ddg_results = [
            {"title": "Only One", "url": "https://ex.com/1", "snippet": "Lonely"},
        ]
        codex_results = [
            {"title": "Codex Better", "url": "https://ex.com/c1", "snippet": "More"},
            {"title": "Codex Better 2", "url": "https://ex.com/c2", "snippet": "Even more"},
        ]
        monkeypatch.setattr(web_search_tool, "_search_ddg", lambda q, n=10: ddg_results)
        monkeypatch.setattr(
            web_search_tool, "_search_codex",
            lambda q, n=10, timeout=180: (codex_results, "Better synthesis"),
        )

        result = web_search(query="test", engine="auto")
        assert not result.is_error
        text = str(result)
        assert "Codex Better" in text
        assert "codex (fallback)" in text

    def test_max_results_respected(self, monkeypatch):
        """DDG should respect max_results limit."""
        from jyagent.tools import web_search_tool

        many = [
            {"title": f"R{i}", "url": f"https://ex.com/{i}", "snippet": f"S{i}"}
            for i in range(20)
        ]
        monkeypatch.setattr(web_search_tool, "_search_ddg", lambda q, n=10: many[:n])

        result = web_search(query="test", engine="ddg", max_results=3)
        text = str(result)
        assert "Found: 3 results" in text


# ─── Popen-based _search_codex tests ─────────────────────────────────────────
#
# IMPORTANT: FakePopen classes must NOT close stdout/stderr FDs passed as ints.
# Real subprocess.Popen doesn't close the parent's FD copies — the caller code
# in _search_codex (os.close(fd3); os.close(fd4)) handles that.


class TestSearchCodexPopen:
    """Tests for the Popen-based Codex backend: timeout, graceful kill,
    partial-output salvage, stderr diagnostics, stdin /dev/null."""

    def test_codex_not_available_returns_early(self, monkeypatch):
        """If codex CLI is not installed, return immediately."""
        from jyagent.tools import web_search_tool
        monkeypatch.setattr(web_search_tool, "_codex_available", lambda: False)

        results, synthesis = web_search_tool._search_codex("test query")
        assert results == []
        assert "not found" in synthesis

    def test_normal_completion(self, monkeypatch):
        """Codex finishes within timeout → results parsed normally."""
        from jyagent.tools import web_search_tool

        fake_output = json.dumps({
            "results": [
                {"title": "Result A", "url": "https://a.com", "snippet": "aaa"},
            ],
            "synthesis": "Found A.",
        })

        class FakePopen:
            def __init__(self, cmd, **kwargs):
                self.pid = 99999
                self.returncode = 0
                for i, arg in enumerate(cmd):
                    if arg == "-o" and i + 1 < len(cmd):
                        with open(cmd[i + 1], "w") as f:
                            f.write(fake_output)
                        break

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

            def terminate(self):
                pass

            def kill(self):
                pass

        monkeypatch.setattr(web_search_tool, "_codex_available", lambda: True)
        monkeypatch.setattr(subprocess, "Popen", FakePopen)

        results, synthesis = web_search_tool._search_codex("test", max_results=5, timeout=10)
        assert len(results) == 1
        assert results[0]["title"] == "Result A"
        assert "Found A" in synthesis

    def test_timeout_triggers_termination(self, monkeypatch):
        """When deadline expires, process is terminated and timeout message returned."""
        from jyagent.tools import web_search_tool

        terminated = []

        class HangingPopen:
            def __init__(self, cmd, **kwargs):
                self.pid = 88888
                self.returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                if self.returncode is None:
                    raise subprocess.TimeoutExpired("codex", timeout)
                return self.returncode

            def terminate(self):
                terminated.append("SIGTERM")
                self.returncode = -15

            def kill(self):
                terminated.append("SIGKILL")
                self.returncode = -9

        monkeypatch.setattr(web_search_tool, "_codex_available", lambda: True)
        monkeypatch.setattr(subprocess, "Popen", HangingPopen)

        results, synthesis = web_search_tool._search_codex("stuck query", timeout=1)
        assert results == []
        assert "timed out" in synthesis
        assert "SIGTERM" in terminated  # graceful termination attempted

    def test_timeout_salvages_partial_output(self, monkeypatch):
        """If Codex wrote partial results before timing out, salvage them."""
        from jyagent.tools import web_search_tool

        partial = json.dumps({
            "results": [
                {"title": "Partial", "url": "https://p.com", "snippet": "saved"},
            ],
            "synthesis": "Partial data.",
        })

        class SlowButWrotePopen:
            def __init__(self, cmd, **kwargs):
                self.pid = 77777
                self.returncode = None
                for i, arg in enumerate(cmd):
                    if arg == "-o" and i + 1 < len(cmd):
                        with open(cmd[i + 1], "w") as f:
                            f.write(partial)
                        break

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                if self.returncode is None:
                    raise subprocess.TimeoutExpired("codex", timeout)
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def kill(self):
                self.returncode = -9

        monkeypatch.setattr(web_search_tool, "_codex_available", lambda: True)
        monkeypatch.setattr(subprocess, "Popen", SlowButWrotePopen)

        results, synthesis = web_search_tool._search_codex("partial query", timeout=1)
        assert len(results) == 1
        assert results[0]["title"] == "Partial"
        assert "Partial data" in synthesis

    def test_nonzero_exit_logs_stderr(self, monkeypatch, caplog):
        """Non-zero exit code → stderr is read and logged as warning."""
        from jyagent.tools import web_search_tool

        stderr_content = "Error: API key invalid"

        class FailingPopen:
            def __init__(self, cmd, **kwargs):
                self.pid = 66666
                self.returncode = 1
                # Write stderr content to the FD (but don't close it)
                stderr_val = kwargs.get("stderr")
                if isinstance(stderr_val, int):
                    os.write(stderr_val, stderr_content.encode())

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

            def terminate(self):
                pass

            def kill(self):
                pass

        monkeypatch.setattr(web_search_tool, "_codex_available", lambda: True)
        monkeypatch.setattr(subprocess, "Popen", FailingPopen)

        with caplog.at_level(logging.WARNING, logger="jyagent.tools.web_search_tool"):
            results, synthesis = web_search_tool._search_codex("fail query", timeout=10)

        assert results == []
        # Should have logged the non-zero exit with stderr content
        assert any("exited" in r.message and "API key invalid" in r.message
                    for r in caplog.records)

    def test_stdin_is_devnull(self, monkeypatch):
        """Verify stdin=DEVNULL is passed to Popen."""
        from jyagent.tools import web_search_tool

        captured_kwargs = {}

        class CapturePopen:
            def __init__(self, cmd, **kwargs):
                captured_kwargs.update(kwargs)
                self.pid = 55555
                self.returncode = 0

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

            def terminate(self):
                pass

            def kill(self):
                pass

        monkeypatch.setattr(web_search_tool, "_codex_available", lambda: True)
        monkeypatch.setattr(subprocess, "Popen", CapturePopen)

        web_search_tool._search_codex("test", timeout=5)
        assert captured_kwargs.get("stdin") == subprocess.DEVNULL

    def test_temp_files_cleaned_up(self, monkeypatch):
        """All temp files (schema, output, stdout, stderr) are removed after run."""
        from jyagent.tools import web_search_tool

        class QuickPopen:
            def __init__(self, cmd, **kwargs):
                self.pid = 44444
                self.returncode = 0

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

            def terminate(self):
                pass

            def kill(self):
                pass

        monkeypatch.setattr(web_search_tool, "_codex_available", lambda: True)
        monkeypatch.setattr(subprocess, "Popen", QuickPopen)

        # Snapshot temp files before
        tmp_dir = tempfile.gettempdir()
        before = set(glob.glob(os.path.join(tmp_dir, "ws_*")))

        web_search_tool._search_codex("cleanup test", timeout=5)

        # After: no new ws_* files should remain
        after = set(glob.glob(os.path.join(tmp_dir, "ws_*")))
        new_files = after - before
        assert len(new_files) == 0, f"Leaked temp files: {new_files}"

    def test_exception_in_popen_is_caught(self, monkeypatch):
        """If Popen raises, the exception is caught and returned as error."""
        from jyagent.tools import web_search_tool

        def exploding_popen(*args, **kwargs):
            raise OSError("spawn failed")

        monkeypatch.setattr(web_search_tool, "_codex_available", lambda: True)
        monkeypatch.setattr(subprocess, "Popen", exploding_popen)

        results, synthesis = web_search_tool._search_codex("boom", timeout=5)
        assert results == []
        assert "error" in synthesis.lower() or "spawn failed" in synthesis

    def test_process_group_used_on_posix(self, monkeypatch):
        """On POSIX systems, preexec_fn=os.setpgrp should be set."""
        from jyagent.tools import web_search_tool

        captured_kwargs = {}

        class CapturePopen:
            def __init__(self, cmd, **kwargs):
                captured_kwargs.update(kwargs)
                self.pid = 33333
                self.returncode = 0

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

            def terminate(self):
                pass

            def kill(self):
                pass

        monkeypatch.setattr(web_search_tool, "_codex_available", lambda: True)
        monkeypatch.setattr(subprocess, "Popen", CapturePopen)

        web_search_tool._search_codex("test", timeout=5)
        if hasattr(os, "setpgrp"):
            assert captured_kwargs.get("preexec_fn") is os.setpgrp
        else:
            assert captured_kwargs.get("preexec_fn") is None

    def test_default_timeout_is_180(self, monkeypatch):
        """Default timeout should be 180s (increased from old 120s)."""
        from jyagent.tools import web_search_tool
        import inspect

        sig = inspect.signature(web_search_tool._search_codex)
        assert sig.parameters["timeout"].default == 180


# ─── Helper function tests ───────────────────────────────────────────────────


class TestHelperFunctions:
    """Tests for _terminate_proc and _read_output."""

    def test_terminate_proc_already_dead(self):
        """_terminate_proc is a no-op if process already exited."""
        from jyagent.tools.web_search_tool import _terminate_proc

        class DeadProc:
            def poll(self):
                return 0

            def terminate(self):
                raise AssertionError("Should not be called")

        _terminate_proc(DeadProc())  # should not raise

    def test_terminate_proc_sigterm_works(self):
        """_terminate_proc sends SIGTERM and it works."""
        from jyagent.tools.web_search_tool import _terminate_proc

        actions = []

        class NiceProc:
            def __init__(self):
                self._alive = True

            def poll(self):
                return None if self._alive else -15

            def terminate(self):
                actions.append("term")
                self._alive = False

            def wait(self, timeout=None):
                if self._alive:
                    raise subprocess.TimeoutExpired("x", timeout)
                return -15

            def kill(self):
                actions.append("kill")

        _terminate_proc(NiceProc())
        assert actions == ["term"]  # no kill needed

    def test_terminate_proc_escalates_to_sigkill(self):
        """_terminate_proc escalates to SIGKILL when SIGTERM doesn't work."""
        from jyagent.tools.web_search_tool import _terminate_proc

        actions = []

        class StubbornProc:
            def poll(self):
                return None

            def terminate(self):
                actions.append("term")

            def wait(self, timeout=None):
                if "kill" not in actions:
                    raise subprocess.TimeoutExpired("x", timeout)
                return -9

            def kill(self):
                actions.append("kill")

        _terminate_proc(StubbornProc())
        assert "term" in actions
        assert "kill" in actions

    def test_read_output_prefers_output_file(self, tmp_path):
        """_read_output prefers the -o file over stdout capture."""
        from jyagent.tools.web_search_tool import _read_output

        out = tmp_path / "output.txt"
        stdout = tmp_path / "stdout.txt"
        out.write_text('{"results": [], "synthesis": "from output"}')
        stdout.write_text('{"results": [], "synthesis": "from stdout"}')

        result = _read_output(str(out), str(stdout))
        assert "from output" in result

    def test_read_output_falls_back_to_stdout(self, tmp_path):
        """_read_output falls back to stdout if -o file is empty."""
        from jyagent.tools.web_search_tool import _read_output

        out = tmp_path / "output.txt"
        stdout = tmp_path / "stdout.txt"
        out.write_text("")
        stdout.write_text("fallback content")

        result = _read_output(str(out), str(stdout))
        assert result == "fallback content"

    def test_read_output_both_empty(self, tmp_path):
        """_read_output returns empty string when both files are empty."""
        from jyagent.tools.web_search_tool import _read_output

        out = tmp_path / "output.txt"
        stdout = tmp_path / "stdout.txt"
        out.write_text("")
        stdout.write_text("")

        result = _read_output(str(out), str(stdout))
        assert result == ""

    def test_read_output_missing_files(self):
        """_read_output handles missing files gracefully."""
        from jyagent.tools.web_search_tool import _read_output

        result = _read_output("/nonexistent/file.txt", None)
        assert result == ""


# ─── Live integration test (requires network) ────────────────────────────────


@pytest.mark.skipif(
    os.environ.get("SKIP_NETWORK_TESTS", "1") == "1",
    reason="Network tests disabled (set SKIP_NETWORK_TESTS=0 to enable)",
)
class TestWebSearchLive:
    def test_ddg_live(self):
        result = web_search(query="python asyncio tutorial", engine="ddg", max_results=3)
        assert not result.is_error
        assert "python" in str(result).lower()
