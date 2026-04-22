"""
web_search — Native tool for searching the web with structured results.

Unlike web_fetch (which fetches a specific URL), web_search takes a query
and returns structured search results with titles, URLs, and snippets.

Engines:
  - ddg:   DuckDuckGo HTML search (fast, reliable, no Chrome needed)
  - codex: Codex CLI with native web_search tool (best quality, includes
           multi-source synthesis, slower and more expensive)
  - auto:  DuckDuckGo first; falls back to Codex if results are insufficient
"""

import json
import logging
import os
import signal
import subprocess
import tempfile
import time
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


# ─── Codex backend ───────────────────────────────────────────────────────────

_CODEX_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "snippet": {"type": "string"},
                },
                "required": ["title", "url", "snippet"],
                "additionalProperties": False,
            },
        },
        "synthesis": {
            "type": "string",
            "description": "Brief synthesis of key findings across all results",
        },
    },
    "required": ["results", "synthesis"],
    "additionalProperties": False,
}


def _codex_available() -> bool:
    """Check if codex CLI is installed and reachable."""
    try:
        r = subprocess.run(
            ["codex", "--version"], capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _parse_codex_json(text: str) -> dict | None:
    """Try to parse JSON from Codex output, handling surrounding noise."""
    # Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    if not text:
        return None

    # Scan lines for a JSON object
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    # Try to extract the largest {...} substring
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    return None


_POLL_INTERVAL = 2.0   # seconds between liveness checks
_KILL_GRACE = 5        # seconds to wait after SIGTERM before SIGKILL


def _terminate_proc(proc: subprocess.Popen) -> None:
    """Gracefully terminate a Popen process (SIGTERM → SIGKILL)."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=_KILL_GRACE)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def _read_output(output_path: str | None, stdout_path: str | None) -> str:
    """Read Codex output, preferring the -o file, falling back to stdout capture."""
    for path in (output_path, stdout_path):
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    text = f.read().strip()
                if text:
                    return text
            except OSError:
                continue
    return ""


def _search_codex(
    query: str, max_results: int = 10, timeout: int = 180,
) -> tuple[list[dict], str]:
    """Search via Codex CLI with native web_search tool.

    Uses ``Popen`` + poll loop instead of blocking ``subprocess.run()`` so that:
    - The process can be gracefully terminated (SIGTERM → SIGKILL) on timeout.
    - Partial output can be salvaged if Codex wrote results before the deadline.
    - stdin is redirected from ``/dev/null`` to prevent interactive hangs.
    - stderr is captured for diagnostics.

    Returns ``(results_list, synthesis_text)``.
    Both may be empty on failure.
    """
    if not _codex_available():
        return [], "[codex CLI not found]"

    schema_path: str | None = None
    output_path: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    proc: subprocess.Popen | None = None

    try:
        # ── Temp files ───────────────────────────────────────────────────
        fd, schema_path = tempfile.mkstemp(suffix=".json", prefix="ws_schema_")
        with os.fdopen(fd, "w") as f:
            json.dump(_CODEX_SCHEMA, f)

        fd2, output_path = tempfile.mkstemp(suffix=".txt", prefix="ws_out_")
        os.close(fd2)

        # Capture stdout/stderr to files (avoids PIPE-buffer deadlocks)
        fd3, stdout_path = tempfile.mkstemp(suffix=".txt", prefix="ws_stdout_")
        fd4, stderr_path = tempfile.mkstemp(suffix=".txt", prefix="ws_stderr_")

        prompt = (
            f"Search the web for: {query}\n"
            f"Return up to {max_results} results. "
            "For each result include title, URL, and a one-sentence snippet.\n"
            "Also provide a brief synthesis of the key findings."
        )

        cmd = [
            "codex", "exec",
            "--skip-git-repo-check",
            "--sandbox", "read-only",
            "-c", "tools_web_search=true",
            "--output-schema", schema_path,
            "-o", output_path,
            prompt,
        ]

        log.debug("web_search codex cmd: %s", " ".join(cmd))

        # ── Launch ───────────────────────────────────────────────────────
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=fd3,
            stderr=fd4,
            # Start a new process group so we can signal the whole tree
            preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None,
        )
        # Close our copies of the FDs — the child owns them now
        os.close(fd3)
        os.close(fd4)

        # ── Poll loop ────────────────────────────────────────────────────
        deadline = time.monotonic() + timeout
        timed_out = False

        while proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                proc.wait(timeout=min(_POLL_INTERVAL, max(0.1, remaining)))
            except subprocess.TimeoutExpired:
                pass

        # ── Handle timeout ───────────────────────────────────────────────
        if timed_out:
            elapsed = int(time.monotonic() - (deadline - timeout))
            log.warning(
                "Codex web_search timed out after %ds — terminating (pid %d)",
                elapsed, proc.pid,
            )
            _terminate_proc(proc)

            # Try to salvage partial output
            raw = _read_output(output_path, stdout_path)
            if raw:
                data = _parse_codex_json(raw)
                if data and data.get("results"):
                    results = data["results"][:max_results]
                    synthesis = data.get("synthesis", "")
                    log.info(
                        "Salvaged %d results from timed-out Codex search",
                        len(results),
                    )
                    return results, synthesis

            return [], f"[Codex search timed out after {elapsed}s]"

        # ── Process finished — read output ───────────────────────────────
        if proc.returncode != 0:
            stderr_text = ""
            if stderr_path and os.path.exists(stderr_path):
                try:
                    with open(stderr_path) as f:
                        stderr_text = f.read().strip()[:500]
                except OSError:
                    pass
            log.warning(
                "Codex web_search exited %d: %s", proc.returncode, stderr_text,
            )

        raw = _read_output(output_path, stdout_path)

        if not raw:
            log.warning("Codex web_search returned empty output")
            return [], ""

        data = _parse_codex_json(raw)
        if data is None:
            log.warning("Codex web_search: JSON parse failed, returning raw text")
            return [], raw

        results = data.get("results", [])
        synthesis = data.get("synthesis", "")
        return results[:max_results], synthesis

    except Exception as e:
        log.warning("Codex web_search error: %s", e)
        return [], f"[Codex search error: {e}]"
    finally:
        # Kill process if still alive (safety net)
        if proc is not None and proc.poll() is None:
            _terminate_proc(proc)
        # Clean up temp files
        for p in (schema_path, output_path, stdout_path, stderr_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


# ─── Main function ───────────────────────────────────────────────────────────


def web_search(
    query: str,
    engine: str = "auto",
    max_results: int = 10,
) -> ToolResult:
    """Search the web and return structured results.

    Unlike web_fetch (which fetches a specific URL), web_search takes a query
    and returns search results with titles, URLs, and snippets.

    Engines:
      - auto:  DuckDuckGo first, falls back to Codex if < 3 results
      - ddg:   DuckDuckGo only (fast, reliable, no dependencies)
      - codex: Codex CLI with native web_search (best quality, includes
               multi-source synthesis, slower/more expensive ~40K-80K tokens)

    Args:
        query: Search query string
        engine: Search engine — auto, ddg, codex (default "auto")
        max_results: Maximum number of results (default 10)

    Returns:
        Structured search results with titles, URLs, snippets, and optional
        synthesis (Codex engine only).
    """
    if not query or not query.strip():
        return ToolResult("Error: query parameter is required", is_error=True)

    query = query.strip()
    results: list[dict] = []
    synthesis = ""
    engine_used = engine

    if engine == "codex":
        results, synthesis = _search_codex(query, max_results)
        if not results and not synthesis:
            return ToolResult(
                "Codex web search returned no results. Try engine='ddg'.",
                is_error=True,
            )
        engine_used = "codex"

    elif engine == "ddg":
        results = _search_ddg(query, max_results)
        if not results:
            return ToolResult(
                f"DuckDuckGo returned no results for: {query}",
                is_error=True,
            )
        engine_used = "ddg"

    elif engine == "auto":
        # Phase 1: fast DuckDuckGo
        results = _search_ddg(query, max_results)
        engine_used = "ddg"

        # Phase 2: Codex fallback when DDG gives poor results
        if len(results) < 3:
            log.info(
                "DDG returned only %d results; falling back to Codex",
                len(results),
            )
            codex_results, synthesis = _search_codex(query, max_results)
            if codex_results:
                results = codex_results
                engine_used = "codex (fallback)"
            elif synthesis:
                engine_used = "codex (fallback, synthesis only)"
    else:
        return ToolResult(
            f"Error: Unknown engine '{engine}'. Choose: auto, ddg, codex",
            is_error=True,
        )

    if not results and not synthesis:
        return ToolResult(f"No results found for: {query}", is_error=True)

    # ── Format output ────────────────────────────────────────────────────
    lines: list[str] = [f"Search results for: {query}"]
    lines.append(f"Engine: {engine_used} | Found: {len(results)} results")
    lines.append("")

    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', 'Untitled')}")
        lines.append(f"   URL: {r.get('url', '')}")
        snippet = r.get("snippet", "")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")

    if synthesis:
        lines.append("--- Synthesis ---")
        lines.append(synthesis)

    return ToolResult("\n".join(lines))


# ─── Tool schema (for registration) ─────────────────────────────────────────

TOOL_SCHEMA = {
    "name": "web_search",
    "description": (
        "Search the web and return structured results. "
        "Use this for finding information, news, comparisons, fact-checking. "
        "For fetching a known URL, use web_fetch instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string",
            },
            "engine": {
                "type": "string",
                "description": (
                    "Search engine: auto (DDG→Codex fallback), "
                    "ddg (fast/free), codex (best quality, expensive). "
                    "Default: auto"
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 10)",
            },
        },
        "required": ["query"],
    },
}
