"""Tests for web_search tool — DDG-backed search and parsing."""

import os

import pytest

from jyagent.tools.web_search_tool import (
    TOOL_SCHEMA,
    _search_ddg,
    web_search,
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
        assert "max_results" in props
        # engine param has been removed (DDG-only backend)
        assert "engine" not in props

    def test_query_is_required(self):
        assert "query" in TOOL_SCHEMA["input_schema"]["required"]


# ─── DDG parser tests (unit) ─────────────────────────────────────────────────


class TestSearchDdgParsing:
    """Test DDG HTML parsing with synthetic HTML via mocked HTTP fetcher."""

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

    def test_parse_results_with_uddg_unwrap(self, monkeypatch):
        """DDG wraps URLs in /l/?uddg=…; parser must unwrap them."""
        fake_results = [
            {"title": "Python Docs", "url": "https://docs.python.org/3/",
             "snippet": "Official docs"},
            {"title": "Real Python", "url": "https://realpython.com/",
             "snippet": "Tutorials"},
        ]
        html = self._make_ddg_html(fake_results)

        # Mock the HTTP fetcher used by _search_ddg.
        # Note: jyagent.tools.__init__ re-exports web_fetch (the function),
        # which shadows the submodule attribute — fetch via sys.modules.
        import sys
        import jyagent.tools.web_fetch  # noqa: F401  (load into sys.modules)
        web_fetch_mod = sys.modules["jyagent.tools.web_fetch"]
        monkeypatch.setattr(
            web_fetch_mod, "_fetch_cffi", lambda url, **kw: (200, html),
        )

        results = _search_ddg("anything", max_results=10)

        assert len(results) == 2
        assert results[0]["title"] == "Python Docs"
        assert results[0]["url"] == "https://docs.python.org/3/"
        assert results[1]["snippet"] == "Tutorials"

    def test_max_results_caps_output(self, monkeypatch):
        fake_results = [
            {"title": f"R{i}", "url": f"https://example.com/{i}",
             "snippet": f"snippet {i}"}
            for i in range(10)
        ]
        html = self._make_ddg_html(fake_results)

        import sys
        import jyagent.tools.web_fetch  # noqa: F401
        web_fetch_mod = sys.modules["jyagent.tools.web_fetch"]
        monkeypatch.setattr(
            web_fetch_mod, "_fetch_cffi", lambda url, **kw: (200, html),
        )

        results = _search_ddg("q", max_results=3)
        assert len(results) == 3

    def test_empty_html_returns_empty(self, monkeypatch):
        import sys
        import jyagent.tools.web_fetch  # noqa: F401
        web_fetch_mod = sys.modules["jyagent.tools.web_fetch"]
        monkeypatch.setattr(
            web_fetch_mod, "_fetch_cffi", lambda url, **kw: (200, ""),
        )
        monkeypatch.setattr(
            web_fetch_mod, "_fetch_httpx", lambda url, **kw: (200, ""),
        )
        assert _search_ddg("q") == []


# ─── web_search function tests ───────────────────────────────────────────────


class TestWebSearch:
    def test_empty_query_returns_error(self):
        result = web_search(query="")
        assert result.is_error

    def test_whitespace_query_returns_error(self):
        result = web_search(query="   ")
        assert result.is_error

    def test_returns_formatted_results(self, monkeypatch):
        """Mock DDG layer to return canned results."""
        from jyagent.tools import web_search_tool

        fake = [
            {"title": "Result 1", "url": "https://example.com/1",
             "snippet": "First"},
            {"title": "Result 2", "url": "https://example.com/2",
             "snippet": "Second"},
        ]
        monkeypatch.setattr(
            web_search_tool, "_search_ddg", lambda q, max_results=10: fake,
        )

        result = web_search(query="test query")
        assert not result.is_error
        text = str(result)
        assert "Result 1" in text
        assert "https://example.com/1" in text
        assert "Engine: ddg" in text

    def test_no_results_returns_error(self, monkeypatch):
        from jyagent.tools import web_search_tool
        monkeypatch.setattr(
            web_search_tool, "_search_ddg", lambda q, max_results=10: [],
        )
        result = web_search(query="anything")
        assert result.is_error


# ─── Live integration test (requires network) ────────────────────────────────


@pytest.mark.skipif(
    os.environ.get("SKIP_NETWORK_TESTS", "1") == "1",
    reason="Network tests disabled (set SKIP_NETWORK_TESTS=0 to enable)",
)
class TestWebSearchLive:
    def test_ddg_live(self):
        result = web_search(query="python asyncio tutorial", max_results=3)
        assert not result.is_error
        assert "python" in str(result).lower()
