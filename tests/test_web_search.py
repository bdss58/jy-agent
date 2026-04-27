"""Tests for web_search tool — SearxNG → DDG cascade."""

import os

import pytest

from jyagent.tools.web_search_tool import (
    TOOL_SCHEMA,
    _cascade,
    _search_ddg,
    _search_searxng,
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
        # No engine override param — controlled via WEB_SEARCH_ENGINE env
        assert "engine" not in props

    def test_query_is_required(self):
        assert "query" in TOOL_SCHEMA["input_schema"]["required"]


# ─── DDG parser tests (unit, mocked HTML) ────────────────────────────────────


def _ddg_html(results):
    items = []
    for r in results:
        uddg_url = f"//duckduckgo.com/l/?uddg={r['url']}&rut=abc"
        items.append(f"""
        <div class="result results_links">
            <h2 class="result__title">
                <a class="result__a" href="{uddg_url}">{r['title']}</a>
            </h2>
            <a class="result__snippet" href="{uddg_url}">{r['snippet']}</a>
        </div>""")
    return f"<html><body>{''.join(items)}</body></html>"


class TestSearchDdgParsing:
    def test_parse_results_with_uddg_unwrap(self, monkeypatch):
        fake = [
            {"title": "Python Docs", "url": "https://docs.python.org/3/", "snippet": "Official"},
            {"title": "Real Python", "url": "https://realpython.com/", "snippet": "Tutorials"},
        ]
        html = _ddg_html(fake)
        import jyagent.tools.web_fetch as wf
        monkeypatch.setattr(wf, "_fetch_cffi", lambda url, **kw: (200, html))

        results = _search_ddg("anything", max_results=10)
        assert len(results) == 2
        assert results[0]["title"] == "Python Docs"
        assert results[0]["url"] == "https://docs.python.org/3/"
        assert results[1]["snippet"] == "Tutorials"

    def test_max_results_caps_output(self, monkeypatch):
        fake = [
            {"title": f"R{i}", "url": f"https://example.com/{i}", "snippet": f"s{i}"}
            for i in range(10)
        ]
        html = _ddg_html(fake)
        import jyagent.tools.web_fetch as wf
        monkeypatch.setattr(wf, "_fetch_cffi", lambda url, **kw: (200, html))

        assert len(_search_ddg("q", max_results=3)) == 3

    def test_empty_html_returns_empty(self, monkeypatch):
        import jyagent.tools.web_fetch as wf
        monkeypatch.setattr(wf, "_fetch_cffi", lambda url, **kw: (200, ""))
        monkeypatch.setattr(wf, "_fetch_httpx", lambda url, **kw: (200, ""))
        assert _search_ddg("q", max_results=10) == []


# ─── SearxNG tests (env-gated) ────────────────────────────────────────────────


class TestSearchSearxng:
    def test_disabled_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        assert _search_searxng("q", max_results=5) == []

    def test_parses_json_response(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://test-instance:8888")

        class FakeResp:
            status_code = 200
            text = ""
            def json(self):
                return {"results": [
                    {"title": "T1", "url": "https://a.com/", "content": "snip1"},
                    {"title": "T2", "url": "https://b.com/", "content": "snip2"},
                ]}

        class FakeClient:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def get(self, url, params=None): return FakeResp()

        import httpx
        monkeypatch.setattr(httpx, "Client", FakeClient)

        results = _search_searxng("q", max_results=5)
        assert len(results) == 2
        assert results[0]["url"] == "https://a.com/"
        assert results[0]["snippet"] == "snip1"


# ─── Cascade tests ───────────────────────────────────────────────────────────


class TestCascade:
    def test_first_engine_wins_searxng(self, monkeypatch):
        # Cascade order: searxng → ddg. With SEARXNG_URL set, searxng wins.
        from jyagent.tools import web_search_tool as m

        fake = [{"title": "T", "url": "https://a/", "snippet": ""} for _ in range(5)]
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:9999")
        monkeypatch.setitem(m._ENGINES, "searxng", lambda q, n: fake)
        # Downstream engines MUST NOT be called once searxng wins
        monkeypatch.setitem(m._ENGINES, "ddg", lambda q, n: pytest.fail("ddg called"))

        name, results, errs = _cascade("q", 5)
        assert name == "searxng"
        assert len(results) == 5

    def test_falls_through_to_ddg_when_searxng_underfills(self, monkeypatch):
        # searxng underfills → ddg wins.
        from jyagent.tools import web_search_tool as m

        monkeypatch.setenv("SEARXNG_URL", "http://localhost:9999")
        monkeypatch.setitem(m._ENGINES, "searxng", lambda q, n: [])
        monkeypatch.setitem(m._ENGINES, "ddg", lambda q, n: [
            {"title": "T", "url": "https://d/", "snippet": ""} for _ in range(5)
        ])

        name, results, errs = _cascade("q", 5)
        assert name == "ddg"
        assert len(results) == 5

    def test_searxng_skipped_when_env_unset(self, monkeypatch):
        # No SEARXNG_URL → searxng is silently skipped, ddg is the only hop.
        from jyagent.tools import web_search_tool as m

        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.setitem(m._ENGINES, "searxng", lambda q, n: pytest.fail("searxng called"))
        monkeypatch.setitem(m._ENGINES, "ddg", lambda q, n: [
            {"title": "D", "url": "https://d/", "snippet": ""} for _ in range(5)
        ])

        name, results, errs = _cascade("q", 5)
        assert name == "ddg"

    def test_force_override_via_env(self, monkeypatch):
        from jyagent.tools import web_search_tool as m

        monkeypatch.setenv("WEB_SEARCH_ENGINE", "ddg")
        monkeypatch.setenv("SEARXNG_URL", "http://localhost:9999")  # would normally win
        monkeypatch.setitem(m._ENGINES, "searxng", lambda q, n: pytest.fail("searxng called"))
        monkeypatch.setitem(m._ENGINES, "ddg", lambda q, n: [
            {"title": "D", "url": "https://d/", "snippet": ""} for _ in range(5)
        ])

        name, results, errs = _cascade("q", 5)
        assert name == "ddg"

    def test_engine_exception_recorded_and_continues(self, monkeypatch):
        # searxng throws → cascade records error and continues to ddg.
        from jyagent.tools import web_search_tool as m

        monkeypatch.setenv("SEARXNG_URL", "http://localhost:9999")

        def boom(q, n): raise RuntimeError("kaboom")
        monkeypatch.setitem(m._ENGINES, "searxng", boom)
        monkeypatch.setitem(m._ENGINES, "ddg", lambda q, n: [
            {"title": "D", "url": "https://d/", "snippet": ""} for _ in range(5)
        ])

        name, results, errs = _cascade("q", 5)
        assert name == "ddg"
        assert any("searxng" in e and "kaboom" in e for e in errs)


# ─── web_search() integration ─────────────────────────────────────────────────


class TestWebSearch:
    def test_empty_query_returns_error(self):
        assert web_search(query="").is_error

    def test_whitespace_query_returns_error(self):
        assert web_search(query="   ").is_error

    def test_returns_formatted_results(self, monkeypatch):
        from jyagent.tools import web_search_tool as m

        # Patch ddg so it wins outright with ≥ _MIN_RESULTS_TO_STOP;
        # otherwise cascade would fall through to live network.
        fake = [
            {"title": f"Result {i}", "url": f"https://example.com/{i}", "snippet": f"S{i}"}
            for i in range(1, 5)
        ]
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.setitem(m._ENGINES, "ddg", lambda q, n: fake)

        result = web_search(query="test query")
        assert not result.is_error
        text = str(result)
        assert "Result 1" in text
        assert "https://example.com/1" in text
        assert "Engine: ddg" in text

    def test_all_engines_empty_returns_error(self, monkeypatch):
        from jyagent.tools import web_search_tool as m

        monkeypatch.delenv("SEARXNG_URL", raising=False)
        for eng in ("searxng", "ddg"):
            monkeypatch.setitem(m._ENGINES, eng, lambda q, n: [])

        result = web_search(query="anything")
        assert result.is_error


# ─── Live integration test (network, opt-in) ─────────────────────────────────


@pytest.mark.skipif(
    os.environ.get("SKIP_NETWORK_TESTS", "1") == "1",
    reason="Network tests disabled (set SKIP_NETWORK_TESTS=0 to enable)",
)
class TestWebSearchLive:
    def test_cascade_live(self):
        result = web_search(query="python asyncio tutorial", max_results=3)
        assert not result.is_error
        assert "python" in str(result).lower()
