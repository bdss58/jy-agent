"""
web_search — Native tool for searching the web with structured results.

Unlike web_fetch (which fetches a specific URL), web_search takes a query
and returns structured results (title, URL, snippet). The agent handles
iteration / synthesis: run web_search → pick URLs → web_fetch each.

Engine cascade (first non-empty wins):

  1. SearxNG      — if SEARXNG_URL env is set. Aggregates Google+Bing+Brave+
                    DDG+70 others via one self-hosted JSON endpoint.
                    No API key, best quality.
  2. DuckDuckGo   — HTML endpoint. Zero-config default.
  3. Brave Search — HTML endpoint. Independent index of ~8B docs.
  4. Mojeek       — HTML endpoint. Independent crawler, tiny but unique.

Each engine has its own parser, but they all return the same
{title, url, snippet} dict shape so the caller never needs to care
which engine served the result.

Environment variables:
  SEARXNG_URL       Base URL of a SearxNG instance, e.g. http://localhost:8888
                    (no trailing slash). When set, SearxNG goes first.
  WEB_SEARCH_ENGINE Force a single engine: "searxng" | "ddg" | "brave" |
                    "mojeek". Default: cascade.
"""

import logging
import os
from typing import Callable
from urllib.parse import parse_qs, quote_plus, urlparse

from ..toolresult import ToolResult

log = logging.getLogger(__name__)


# ─── HTTP helper (shared across engines) ────────────────────────────────────


def _http_get(url: str, *, min_len: int = 500) -> str | None:
    """Fetch a URL using web_fetch's cffi→httpx cascade. Returns HTML body or None."""
    from .web_fetch import _fetch_cffi, _fetch_httpx

    for fetcher in (_fetch_cffi, _fetch_httpx):
        try:
            status, body = fetcher(url)
            if status == 200 and body and len(body) > min_len:
                return body
        except Exception as exc:
            log.debug("fetch %s via %s failed: %s", url, fetcher.__name__, exc)
            continue
    return None


# ─── Engine 1: SearxNG (JSON, best quality, opt-in via env) ──────────────────


def _search_searxng(query: str, max_results: int) -> list[dict]:
    """Query a self-hosted SearxNG instance via its JSON API.

    Aggregates Google/Bing/Brave/DDG + many more. Zero API key.
    Enabled only when SEARXNG_URL env is set.
    """
    base = os.environ.get("SEARXNG_URL", "").rstrip("/")
    if not base:
        return []

    import httpx

    url = f"{base}/search"
    params = {
        "q": query,
        "format": "json",
        "safesearch": "0",
        # general category; SearxNG aggregates across its configured engines
        "categories": "general",
    }
    try:
        # SearxNG is typically self-hosted on a trusted network; TLS verify
        # is skipped to tolerate self-signed certs on internal deployments.
        with httpx.Client(timeout=15, verify=False) as client:
            resp = client.get(url, params=params)
            if resp.status_code != 200:
                log.debug("searxng HTTP %s: %s", resp.status_code, resp.text[:200])
                return []
            data = resp.json()
    except Exception as exc:
        log.debug("searxng query failed: %s", exc)
        return []

    results: list[dict] = []
    for item in data.get("results", [])[:max_results]:
        title = (item.get("title") or "").strip()
        href = (item.get("url") or "").strip()
        snippet = (item.get("content") or "").strip()
        if title and href:
            results.append({"title": title, "url": href, "snippet": snippet})
    return results


# ─── Engine 2: DuckDuckGo (HTML, default free engine) ────────────────────────


def _search_ddg(query: str, max_results: int) -> list[dict]:
    """Search DuckDuckGo's HTML endpoint."""
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    html = _http_get(url, min_len=500)
    if not html:
        return []

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    results: list[dict] = []
    for item in soup.select(".result"):
        title_el = item.select_one(".result__a")
        snippet_el = item.select_one(".result__snippet")
        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        href = title_el.get("href", "")
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        # DDG wraps URLs in //duckduckgo.com/l/?uddg=ENCODED_URL
        if "uddg=" in href:
            qs = parse_qs(urlparse(href).query)
            if "uddg" in qs:
                href = qs["uddg"][0]
        if title and href:
            results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


# ─── Engine 3: Brave Search (HTML, independent ~8B-doc index) ────────────────


def _search_brave(query: str, max_results: int) -> list[dict]:
    """Scrape Brave Search HTML SERPs.

    Brave runs an independent crawler, so results complement DDG/Bing.
    No API key needed for the HTML endpoint, but Brave rate-limits
    aggressively (HTTP 429) — expect this engine to miss on sustained use,
    which is why it sits behind DDG in the cascade.

    Selectors observed 2026-04 (Svelte-based SPA): `a.l1` is the stable
    title anchor inside `div.snippet > div.result-wrapper`. Class names
    like `svelte-<hash>` rotate and must NOT be relied on.
    """
    url = f"https://search.brave.com/search?q={quote_plus(query)}&source=web"
    html = _http_get(url, min_len=2000)
    if not html:
        return []

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    results: list[dict] = []
    seen_urls: set[str] = set()

    for a in soup.select("a.l1"):
        href = (a.get("href") or "").strip()
        if not href.startswith("http") or href in seen_urls:
            continue

        # Title is mixed with the display-URL chip; split on the first
        # occurrence of the actual page title heuristically, or just take
        # the whole text — DDG does the same and agents handle it fine.
        title = a.get_text(" ", strip=True)

        # Snippet: look inside the enclosing result-wrapper / snippet block
        wrap = a.find_parent("div", class_="snippet") or a.find_parent("div", class_="result-wrapper")
        snippet = ""
        if wrap:
            # Any <div> / <p> with medium-length non-title text is likely the description
            for el in wrap.find_all(["p", "div"]):
                txt = el.get_text(" ", strip=True)
                if 40 <= len(txt) <= 400 and not txt.startswith(title[:20]):
                    snippet = txt
                    break

        seen_urls.add(href)
        results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= max_results:
            break

    return results


# ─── Engine 4: Mojeek (HTML, independent crawler, minimal JS) ────────────────


def _search_mojeek(query: str, max_results: int) -> list[dict]:
    """Scrape Mojeek — independent UK-based crawler. Small index but unique.

    Mojeek's HTML is static, parser-friendly, and not rate-limited the way
    Brave is — making it the most reliable third fallback.

    Selectors observed 2026-04: `ul.results-standard > li` with
    `h2 a.title` for the title (real anchor), `a.ob` for the URL chip,
    `p.s` for the snippet.
    """
    url = f"https://www.mojeek.com/search?q={quote_plus(query)}"
    html = _http_get(url, min_len=1500)
    if not html:
        return []

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    results: list[dict] = []

    for li in soup.select("ul.results-standard > li"):
        # Title anchor — h2 a.title is the real titled link
        title_a = li.select_one("h2 a.title") or li.select_one("h2 a")
        if not title_a:
            continue
        href = (title_a.get("href") or "").strip()
        title = title_a.get_text(" ", strip=True)
        if not (title and href.startswith("http")):
            continue

        snip_el = li.select_one("p.s")
        snippet = snip_el.get_text(" ", strip=True) if snip_el else ""

        results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= max_results:
            break

    return results


# ─── Cascade orchestrator ───────────────────────────────────────────────────

# Registry: name → callable. Order matters when no override is set.
_ENGINES: dict[str, Callable[[str, int], list[dict]]] = {
    "searxng": _search_searxng,
    "ddg": _search_ddg,
    "brave": _search_brave,
    "mojeek": _search_mojeek,
}

# Minimum results an engine must return before we stop cascading.
# If an engine returns fewer, we try the next one.
_MIN_RESULTS_TO_STOP = 3


def _cascade(query: str, max_results: int) -> tuple[str, list[dict], list[str]]:
    """Run engines in priority order until one returns enough results.

    Returns (winning_engine_name, results, errors).
    """
    # Force-override via env?
    forced = os.environ.get("WEB_SEARCH_ENGINE", "").strip().lower()
    if forced and forced in _ENGINES:
        engines = [(forced, _ENGINES[forced])]
    else:
        # SearxNG first (only active if SEARXNG_URL is set; otherwise returns [])
        engines = [
            ("searxng", _ENGINES["searxng"]),
            ("ddg", _ENGINES["ddg"]),
            ("brave", _ENGINES["brave"]),
            ("mojeek", _ENGINES["mojeek"]),
        ]

    errors: list[str] = []
    best_name, best_results = "", []

    for name, fn in engines:
        # Skip SearxNG silently when not configured
        if name == "searxng" and not os.environ.get("SEARXNG_URL"):
            continue
        try:
            results = fn(query, max_results)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue

        if len(results) >= _MIN_RESULTS_TO_STOP:
            return name, results, errors

        # Keep the best partial result in case every engine underperforms
        if len(results) > len(best_results):
            best_name, best_results = name, results
        errors.append(f"{name}: only {len(results)} results")

    return best_name, best_results, errors


# ─── Main function ───────────────────────────────────────────────────────────


def web_search(query: str, max_results: int = 10) -> ToolResult:
    """Search the web and return structured results.

    Cascade: SearxNG (if SEARXNG_URL set) → DuckDuckGo → Brave → Mojeek.
    First engine returning ≥3 results wins.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default 10)

    Returns:
        Structured search results with titles, URLs, and snippets.
    """
    if not query or not query.strip():
        return ToolResult("Error: query parameter is required", is_error=True)

    query = query.strip()
    engine, results, errors = _cascade(query, max_results)

    if not results:
        err_summary = "; ".join(errors) if errors else "no engine returned results"
        return ToolResult(
            f"All search engines failed for query: {query}\n{err_summary}\n\n"
            f"Try rephrasing, or set SEARXNG_URL to use a self-hosted meta-search.",
            is_error=True,
        )

    # ── Format output ────────────────────────────────────────────────────
    lines: list[str] = [f"Search results for: {query}"]
    status = f"Engine: {engine} | Found: {len(results)} results"
    if errors:
        status += f" | Tried: {len(errors) + 1} engines"
    lines.append(status)
    lines.append("")

    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', 'Untitled')}")
        lines.append(f"   URL: {r.get('url', '')}")
        snippet = r.get("snippet", "")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")

    return ToolResult("\n".join(lines))


# ─── Tool schema (for registration) ─────────────────────────────────────────

TOOL_SCHEMA = {
    "name": "web_search",
    "description": (
        "Search the web and return structured results (title, URL, snippet). "
        "Cascades through SearxNG (if SEARXNG_URL env set) → DuckDuckGo → "
        "Brave Search → Mojeek until one returns enough results. No API key "
        "required. Use this for finding information, news, comparisons, "
        "fact-checking. For fetching a known URL, use web_fetch instead. "
        "For comprehensive multi-source research, iterate: web_search → "
        "pick top URLs → web_fetch each."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 10)",
            },
        },
        "required": ["query"],
    },
}
