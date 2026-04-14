# remediation.py — Error remediation messages for tool results.
#
# When a tool call fails with a known error pattern, append a targeted
# remediation hint so the LLM can self-correct without an extra round-trip.
#
# Inspired by OpenAI's harness engineering insight:
#   "custom linter error messages double as remediation instructions
#    injected into agent context."

from __future__ import annotations

import re
from typing import Callable

from .toolresult import ToolResult


# ─── Pattern registry ────────────────────────────────────────────────────────
#
# Each entry: (compiled_regex, remediation_message_template)
# Template can use {match} for the regex match object and {tool} for tool name.

_PATTERNS: list[tuple[re.Pattern, str | Callable]] = []


def _p(pattern: str, remedy: str | Callable, flags: int = 0) -> None:
    """Register a pattern → remedy pair."""
    _PATTERNS.append((re.compile(pattern, flags), remedy))


# ── File/path errors ─────────────────────────────────────────────────────────

_p(
    r"(?:FileNotFoundError|No such file or directory|File not found)[:\s]*(.+)",
    "[REMEDIATION: Check the file path. Use glob_files('**/*pattern*') to search "
    "for the file, or list_directory() to see what's in the target directory.]",
)

_p(
    r"IsADirectoryError|Is a directory",
    "[REMEDIATION: The path is a directory, not a file. Use list_directory() to "
    "see its contents, or adjust the path to target a specific file.]",
)

_p(
    r"PermissionError|Permission denied",
    "[REMEDIATION: Permission denied. Check if the file is read-only or owned "
    "by another user. Try a different path or use run_shell('ls -la <path>') "
    "to inspect permissions.]",
)

# ── Edit/write errors ────────────────────────────────────────────────────────

_p(
    r"old_text not found in file|No match found for replacement",
    "[REMEDIATION: The exact text to replace was not found. Use "
    "read_file(path, line_numbers=True) to see the current file content "
    "and copy the exact text including whitespace and indentation.]",
    re.IGNORECASE,
)

_p(
    r"SyntaxError.+\.py",
    "[REMEDIATION: Python syntax error detected. Review the generated code "
    "carefully — check for missing colons, unmatched brackets, incorrect "
    "indentation, or unterminated strings.]",
)

# ── Network/fetch errors ─────────────────────────────────────────────────────

_p(
    r"(?:ConnectionError|ConnectError|Connection refused|ECONNREFUSED)",
    "[REMEDIATION: Connection failed. The server may be down or the URL "
    "incorrect. Verify the URL, check if the service is running, or try "
    "an alternative source.]",
)

_p(
    r"(?:ssl|certificate|CERTIFICATE_VERIFY_FAILED)",
    "[REMEDIATION: SSL/TLS certificate error. This environment has known "
    "CA cert issues. If using web_fetch, it handles this automatically. "
    "For run_shell with curl, add --insecure flag.]",
    re.IGNORECASE,
)

_p(
    r"(?:TimeoutError|timed? ?out|Read timed out|408|504)",
    "[REMEDIATION: Request timed out. The operation took too long. "
    "Try again with a longer timeout, or break the task into smaller "
    "pieces. For run_shell, increase the timeout parameter.]",
    re.IGNORECASE,
)

_p(
    r"(?:403 Forbidden|401 Unauthorized|Access Denied)",
    "[REMEDIATION: Access denied. The target requires authentication or "
    "blocks automated access. Try a different source, use web_fetch "
    "which has anti-blocking cascading, or check authentication.]",
    re.IGNORECASE,
)

_p(
    r"404 Not Found|Page not found|returned 404",
    "[REMEDIATION: Page not found (404). The URL may be wrong or the "
    "content moved. Verify the URL, search for the correct page, or "
    "try web_fetch with a search query instead.]",
    re.IGNORECASE,
)

# ── Shell/command errors ─────────────────────────────────────────────────────

_p(
    r"command not found|No such file or directory.+/(?:bin|usr)",
    "[REMEDIATION: Command not found. Check the command name, or use "
    "run_shell('which <cmd>') to verify it's installed. You may need "
    "to install it first or use the full path.]",
)

_p(
    r"(?:ModuleNotFoundError|ImportError): No module named '(\w+)'",
    "[REMEDIATION: Python module not found. Install it with "
    "run_shell('pip install <module>') or check if you're using "
    "the correct virtual environment.]",
)

# ── JSON/parsing errors ──────────────────────────────────────────────────────

_p(
    r"(?:JSONDecodeError|json\.decoder\.JSONDecodeError|Expecting value)",
    "[REMEDIATION: Invalid JSON. The input or response is not valid JSON. "
    "Check for trailing commas, unquoted keys, or truncated content. "
    "Use run_shell to inspect the raw content if needed.]",
)

# ── MCP errors ───────────────────────────────────────────────────────────────

_p(
    r"MCP.+not connected|No MCP connection|MCP server.+not running",
    "[REMEDIATION: MCP server not connected. Use mcp(action='connect') "
    "or mcp(action='connect', server='<name>') to establish the connection "
    "first, then retry the operation.]",
    re.IGNORECASE,
)


# ─── Public API ──────────────────────────────────────────────────────────────

def enrich_error(result: ToolResult, tool_name: str = "") -> ToolResult:
    """If *result* is an error matching a known pattern, return a new
    ToolResult with a remediation hint appended.  Otherwise return as-is.

    Non-error results are never modified.
    """
    if not result.is_error:
        return result

    content = result.content
    for pattern, remedy in _PATTERNS:
        if pattern.search(content):
            if callable(remedy):
                hint = remedy(pattern.search(content), tool_name)
            else:
                hint = remedy
            # Don't append if remediation already present (idempotent)
            if hint in content:
                return result
            return ToolResult(f"{content}\n\n{hint}", is_error=True)

    return result
