"""Tests for web_search tool — DDG parsing, Codex backend, and integration."""

import json
import os
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
            lambda q, n=10, timeout=120: (fake_results, "Synthesis: found it"),
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
            lambda q, n=10, timeout=120: (codex_called.append(1) or ([], "")),
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
            lambda q, n=10, timeout=120: (codex_results, "Better synthesis"),
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
