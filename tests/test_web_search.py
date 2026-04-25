"""Tests for web_search tool — multi-engine cascade (DDG → Brave → Mojeek)
plus opt-in SearxNG via env."""

import os

import pytest

from jyagent.tools.web_search_tool import (
    TOOL_SCHEMA,
    _cascade,
    _search_brave,
    _search_ddg,
    _search_mojeek,
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


# ─── Brave parser tests (unit, mocked HTML) ──────────────────────────────────


class TestSearchBraveParsing:
    def test_parses_a_l1_anchor(self, monkeypatch):
        # Synthetic HTML matching the observed Brave structure (a.l1 + .snippet)
        html = """<html><body><div id="results">
          <div class="snippet svelte-xyz">
            <div class="result-wrapper">
              <a class="l1 svelte-abc" href="https://example.com/page1">Example Page One</a>
              <p class="desc">A snippet describing example page one with more than forty characters of useful content.</p>
            </div>
          </div>
          <div class="snippet">
            <div class="result-wrapper">
              <a class="l1" href="https://other.com/page2">Other Page Two</a>
            </div>
          </div>
        </div></body></html>""" + ("x" * 2500)  # pad to clear min_len
        import jyagent.tools.web_fetch as wf
        monkeypatch.setattr(wf, "_fetch_cffi", lambda url, **kw: (200, html))

        results = _search_brave("anything", max_results=10)
        assert len(results) == 2
        assert results[0]["url"] == "https://example.com/page1"
        assert results[0]["title"] == "Example Page One"
        assert "example page one" in results[0]["snippet"].lower()
        assert results[1]["url"] == "https://other.com/page2"

    def test_skips_non_http_anchors(self, monkeypatch):
        html = """<html><body>
          <div class="snippet"><a class="l1" href="#fragment">Skip me</a></div>
          <div class="snippet"><a class="l1" href="https://ok.com/">Keep me</a></div>
        </body></html>""" + ("x" * 2500)
        import jyagent.tools.web_fetch as wf
        monkeypatch.setattr(wf, "_fetch_cffi", lambda url, **kw: (200, html))

        results = _search_brave("q", max_results=10)
        assert len(results) == 1
        assert results[0]["url"] == "https://ok.com/"

    def test_empty_returns_empty(self, monkeypatch):
        import jyagent.tools.web_fetch as wf
        monkeypatch.setattr(wf, "_fetch_cffi", lambda url, **kw: (200, ""))
        monkeypatch.setattr(wf, "_fetch_httpx", lambda url, **kw: (200, ""))
        assert _search_brave("q", max_results=10) == []


# ─── Mojeek parser tests (unit, mocked HTML) ─────────────────────────────────


class TestSearchMojeekParsing:
    def test_parses_results_standard(self, monkeypatch):
        html = """<html><body>
          <ul class="results-standard">
            <li class="r1">
              <h2><a class="title" href="https://docs.python.org/3/">Python 3 Docs</a></h2>
              <a class="ob" href="https://docs.python.org/3/">https://docs.python.org/3/</a>
              <p class="s">Python is an easy to learn, powerful programming language.</p>
            </li>
            <li class="r2 clu-result">
              <h2><a class="title" href="https://realpython.com/">Real Python</a></h2>
              <p class="s">Tutorials.</p>
            </li>
          </ul>
        </body></html>""" + ("x" * 1600)
        import jyagent.tools.web_fetch as wf
        monkeypatch.setattr(wf, "_fetch_cffi", lambda url, **kw: (200, html))

        results = _search_mojeek("anything", max_results=10)
        assert len(results) == 2
        assert results[0]["title"] == "Python 3 Docs"
        assert results[0]["url"] == "https://docs.python.org/3/"
        assert "powerful programming" in results[0]["snippet"]
        assert results[1]["title"] == "Real Python"


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
    def test_first_engine_wins(self, monkeypatch):
        from jyagent.tools import web_search_tool as m

        fake_ddg = [{"title": "T", "url": "https://a/", "snippet": ""} for _ in range(5)]
        monkeypatch.setenv("SEARXNG_URL", "")  # ensure searxng skipped
        monkeypatch.setitem(m._ENGINES, "ddg", lambda q, n: fake_ddg)
        # Brave/mojeek MUST NOT be called
        monkeypatch.setitem(m._ENGINES, "brave", lambda q, n: pytest.fail("brave called"))
        monkeypatch.setitem(m._ENGINES, "mojeek", lambda q, n: pytest.fail("mojeek called"))

        name, results, errs = _cascade("q", 5)
        assert name == "ddg"
        assert len(results) == 5

    def test_falls_through_to_next_when_underfilled(self, monkeypatch):
        from jyagent.tools import web_search_tool as m

        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.setitem(m._ENGINES, "ddg", lambda q, n: [])  # rate-limited
        monkeypatch.setitem(m._ENGINES, "brave", lambda q, n: [
            {"title": "T", "url": "https://b/", "snippet": ""} for _ in range(5)
        ])
        monkeypatch.setitem(m._ENGINES, "mojeek", lambda q, n: pytest.fail("should not reach"))

        name, results, errs = _cascade("q", 5)
        assert name == "brave"
        assert len(results) == 5
        assert any("ddg" in e for e in errs)

    def test_force_override_via_env(self, monkeypatch):
        from jyagent.tools import web_search_tool as m

        monkeypatch.setenv("WEB_SEARCH_ENGINE", "mojeek")
        monkeypatch.setitem(m._ENGINES, "ddg", lambda q, n: pytest.fail("ddg called"))
        monkeypatch.setitem(m._ENGINES, "brave", lambda q, n: pytest.fail("brave called"))
        monkeypatch.setitem(m._ENGINES, "mojeek", lambda q, n: [
            {"title": "M", "url": "https://m/", "snippet": ""} for _ in range(5)
        ])

        name, results, errs = _cascade("q", 5)
        assert name == "mojeek"

    def test_engine_exception_recorded_and_continues(self, monkeypatch):
        from jyagent.tools import web_search_tool as m

        monkeypatch.delenv("SEARXNG_URL", raising=False)

        def boom(q, n): raise RuntimeError("kaboom")
        monkeypatch.setitem(m._ENGINES, "ddg", boom)
        monkeypatch.setitem(m._ENGINES, "brave", lambda q, n: [
            {"title": "B", "url": "https://b/", "snippet": ""} for _ in range(5)
        ])

        name, results, errs = _cascade("q", 5)
        assert name == "brave"
        assert any("ddg" in e and "kaboom" in e for e in errs)


# ─── web_search() integration ─────────────────────────────────────────────────


class TestWebSearch:
    def test_empty_query_returns_error(self):
        assert web_search(query="").is_error

    def test_whitespace_query_returns_error(self):
        assert web_search(query="   ").is_error

    def test_returns_formatted_results(self, monkeypatch):
        from jyagent.tools import web_search_tool as m

        # ≥ _MIN_RESULTS_TO_STOP so DDG wins outright; otherwise cascade
        # would fall through to live engines.
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
        for eng in ("ddg", "brave", "mojeek"):
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
