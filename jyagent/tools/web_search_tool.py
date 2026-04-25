"""
web_search — Native tool for searching the web with structured results.

Unlike web_fetch (which fetches a specific URL), web_search takes a query
and returns structured search results with titles, URLs, and snippets.

Backend: DuckDuckGo HTML endpoint (fast, free, no auth, no dependencies
beyond the existing web_fetch HTTP cascade + BeautifulSoup).

The agent itself handles iteration / synthesis: run web_search to get
URLs, then web_fetch the promising ones. This is cheaper and more
controllable than delegating to a sub-agent search engine.
"""

import logging
from urllib.parse import parse_qs, quote_plus, urlparse

from ..toolresult import ToolResult

log = logging.getLogger(__name__)


# ─── DuckDuckGo backend ─────────────────────────────────────────────────────


def _search_ddg(query: str, max_results: int = 10) -> list[dict]:
    """Search DuckDuckGo HTML endpoint and parse structured results."""
    from .web_fetch import _fetch_cffi, _fetch_httpx

    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"

    html = None
    for fetcher in (_fetch_cffi, _fetch_httpx):
        try:
            status, body = fetcher(url)
            if status == 200 and len(body) > 500:
                html = body
                break
        except Exception:
            continue

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

        # DDG wraps URLs in a redirect: //duckduckgo.com/l/?uddg=ENCODED_URL
        if "uddg=" in href:
            qs = parse_qs(urlparse(href).query)
            if "uddg" in qs:
                href = qs["uddg"][0]

        if title and href:
            results.append({"title": title, "url": href, "snippet": snippet})

        if len(results) >= max_results:
            break

    return results


# ─── Main function ───────────────────────────────────────────────────────────


def web_search(
    query: str,
    max_results: int = 10,
) -> ToolResult:
    """Search the web and return structured results.

    Unlike web_fetch (which fetches a specific URL), web_search takes a query
    and returns DuckDuckGo search results with titles, URLs, and snippets.

    For deeper investigation: take the URLs from these results and call
    web_fetch on the most promising ones.

    Args:
        query: Search query string
        max_results: Maximum number of results (default 10)

    Returns:
        Structured search results with titles, URLs, and snippets.
    """
    if not query or not query.strip():
        return ToolResult("Error: query parameter is required", is_error=True)

    query = query.strip()
    results = _search_ddg(query, max_results)

    if not results:
        return ToolResult(
            f"DuckDuckGo returned no results for: {query}",
            is_error=True,
        )

    # ── Format output ────────────────────────────────────────────────────
    lines: list[str] = [f"Search results for: {query}"]
    lines.append(f"Engine: ddg | Found: {len(results)} results")
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
        "Search the web (DuckDuckGo) and return structured results "
        "(title, URL, snippet). Use this for finding information, news, "
        "comparisons, fact-checking. For fetching a known URL, use "
        "web_fetch instead. For comprehensive multi-source research, "
        "iterate: web_search → pick top URLs → web_fetch each."
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
