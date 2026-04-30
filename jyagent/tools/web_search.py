"""
web_search — Native tool for searching the web with structured results.

Unlike web_fetch (which fetches a specific URL), web_search takes a query
and returns structured results (title, URL, snippet). The agent handles
iteration / synthesis: run web_search → pick URLs → web_fetch each.

Engine cascade (first non-empty wins):

  1. SearxNG    — if SEARXNG_URL env is set. Aggregates Google + Bing +
                  Brave + DDG + ~70 others via one self-hosted JSON
                  endpoint. No API key, best quality.
  2. DuckDuckGo — HTML endpoint. Default fallback, no auth.

Brave (HTML scrape) and Mojeek were removed 2026-04-26. Brave now serves
a PoW captcha to scrapers; even curl_cffi Chrome impersonation returns
CURLE_WRITE_ERROR. Mojeek aggressively IP-blocks all available exit
ranges (datacenter + Cloudflare WARP), serving a 340-byte
"403 / automated queries" page regardless of TLS impersonation. Run a
SearxNG instance for richer aggregation — it queries Brave/Mojeek/etc.
on your behalf from a residential-style request flow.

Each engine has its own parser, but they all return the same
{title, url, snippet} dict shape so the caller never needs to care
which engine served the result.

Environment variables:
  SEARXNG_URL       Base URL of a SearxNG instance, e.g.
                    http://localhost:8888 (no trailing slash).
                    When set, SearxNG goes first.
  WEB_SEARCH_ENGINE Force a single engine: "searxng" | "ddg".
                    Default: cascade.
"""

import logging
import os
from typing import Callable
from urllib.parse import parse_qs, quote_plus, urlparse

from ..runtime.tools.result import ToolResult

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


# ─── Cascade orchestrator ───────────────────────────────────────────────────

# Registry: name → callable. Order matters when no override is set.
_ENGINES: dict[str, Callable[[str, int], list[dict]]] = {
    "searxng": _search_searxng,
    "ddg": _search_ddg,
}

# Minimum results an engine must return before we stop cascading.
# If an engine returns fewer, we try the next one.
_MIN_RESULTS_TO_STOP = 3


def _cascade(
    query: str,
    max_results: int,
    cancel_event: "threading.Event | None" = None,
) -> tuple[str, list[dict], list[str]]:
    """Run engines in priority order until one returns enough results.

    Returns (winning_engine_name, results, errors).

    ``cancel_event`` (optional) is checked between engine attempts so a
    cancelled search returns within the current engine's HTTP timeout
    rather than running the full cascade.
    """
    # Force-override via env?
    forced = os.environ.get("WEB_SEARCH_ENGINE", "").strip().lower()
    if forced and forced in _ENGINES:
        engines = [(forced, _ENGINES[forced])]
    else:
        # SearxNG first (only active if SEARXNG_URL is set; otherwise returns []).
        # DDG is the universal fallback — no auth, no env required.
        engines = [
            ("searxng", _ENGINES["searxng"]),
            ("ddg", _ENGINES["ddg"]),
        ]

    errors: list[str] = []
    best_name, best_results = "", []

    for name, fn in engines:
        # Cooperative-cancel check between engines.  An in-flight HTTP
        # call cannot be interrupted, but skipping the next engine on
        # cancel is enough to make Ctrl-C feel responsive.
        if cancel_event is not None and cancel_event.is_set():
            errors.append(f"{name}: skipped (cancel signal)")
            break
        # Skip SearxNG silently when not configured.
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


def web_search(
    query: str,
    max_results: int = 10,
    _cancel_event: "threading.Event | None" = None,
) -> ToolResult:
    """Search the web and return structured results.

    Cascade: SearxNG (if SEARXNG_URL set) → DuckDuckGo. First engine
    returning ≥3 results wins.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default 10)
        _cancel_event: cooperative-cancel hook (runtime-injected; see
            ``runtime/loop/tool_executor.py``).  Checked between engine
            attempts so a cancelled search returns promptly instead of
            running the full cascade.

    Returns:
        Structured search results with titles, URLs, and snippets.
    """
    if not query or not query.strip():
        return ToolResult("Error: query parameter is required", is_error=True)

    query = query.strip()

    # Pre-cancellation short-circuit.
    if _cancel_event is not None and _cancel_event.is_set():
        return ToolResult(
            "Cancelled: web_search aborted on cancel signal "
            "(no engine queried).",
            is_error=True,
        )

    engine, results, errors = _cascade(query, max_results, cancel_event=_cancel_event)

    if not results:
        # Distinguish "all engines failed naturally" from "cancelled" so
        # the model can plan accordingly.
        if _cancel_event is not None and _cancel_event.is_set():
            err_summary = "; ".join(errors) if errors else "cancelled before any engine returned"
            return ToolResult(
                f"Cancelled: web_search aborted for query: {query}\n"
                f"{err_summary}",
                is_error=True,
            )
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
        "Cascades through SearxNG (if SEARXNG_URL set) → DuckDuckGo until "
        "one returns enough results. No API key required for the DDG "
        "fallback. Use this for finding information, news, comparisons, "
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
