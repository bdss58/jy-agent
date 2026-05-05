"""Tool display metadata — icon + arg formatter for terminal rendering.

Lives in the *tools* package (not *ui*) because it's part of each tool's
public-facing surface, not part of the renderer.  ``ui/terminal.py`` should
ask the tools package what to show; it must NOT embed a per-tool switch
table — that drifts every time a new tool is added.

To register a custom display for a new tool, add an entry to
``TOOL_DISPLAY`` below.  Tools without an entry render with the default
icon and a generic stringified-input preview (capped at 120 chars).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# ─── Default formatters ──────────────────────────────────────────────────────


def _truncate(s: str, max_chars: int = 120) -> str:
    return s if len(s) <= max_chars else s[: max_chars - 3] + "..."


def _generic_args(_name: str, tool_input: dict) -> str:
    if not tool_input:
        return ""
    return _truncate(str(tool_input))


def _shell_args(_name: str, tool_input: dict) -> str:
    cmd = tool_input.get("command", "")
    return f"$ {_truncate(cmd, 120)}"


def _file_args(_name: str, tool_input: dict) -> str:
    path = tool_input.get("path", "")
    extras: list[str] = []
    if tool_input.get("operation"):
        extras.append(str(tool_input["operation"]))
    if tool_input.get("insert_at_line"):
        extras.append(f"L{tool_input['insert_at_line']}")
    if tool_input.get("dry_run"):
        extras.append("dry-run")
    suffix = f" ({', '.join(extras)})" if extras else ""
    return f"{path}{suffix}"


def _list_dir_args(_name: str, tool_input: dict) -> str:
    return tool_input.get("path", ".") or "."


def _pattern_args(_name: str, tool_input: dict) -> str:
    pattern = tool_input.get("pattern", "")
    path = tool_input.get("path", "")
    return f"'{pattern}'" + (f" in {path}" if path else "")


def _url_args(_name: str, tool_input: dict) -> str:
    return _truncate(tool_input.get("url", ""), 100)


def _action_name_args(_name: str, tool_input: dict) -> str:
    action = tool_input.get("action", "")
    name = tool_input.get("name", "")
    return action + (f" {name}" if name else "")


def _mcp_args(_name: str, tool_input: dict) -> str:
    action = tool_input.get("action", "")
    server = tool_input.get("server", "")
    return action + (f" {server}" if server else "")


# ─── Display registry ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolDisplay:
    """How a tool call is shown in the terminal.

    icon         — short emoji / glyph rendered before the tool name.
    format_args  — callable ``(name, input) -> str`` returning a one-line
                   argument summary.  Receives the tool name so a single
                   formatter can serve a small family (e.g. read/write/edit).
    """
    icon: str = "🔧"
    format_args: Callable[[str, dict], str] = field(default=_generic_args)


_DEFAULT = ToolDisplay()


TOOL_DISPLAY: dict[str, ToolDisplay] = {
    "run_shell":       ToolDisplay(icon="⚡",  format_args=_shell_args),
    "read_file":       ToolDisplay(icon="📄",  format_args=_file_args),
    "write_file":      ToolDisplay(icon="📝",  format_args=_file_args),
    "edit_file":       ToolDisplay(icon="✏️",  format_args=_file_args),
    "list_directory":  ToolDisplay(icon="📁",  format_args=_list_dir_args),
    "glob_files":      ToolDisplay(icon="🔍",  format_args=_pattern_args),
    "grep_files":      ToolDisplay(icon="🔎",  format_args=_pattern_args),
    "web_fetch":       ToolDisplay(icon="🌐",  format_args=_url_args),
    "manage_memory":   ToolDisplay(icon="🧠",  format_args=_action_name_args),
    "manage_skills":   ToolDisplay(icon="📦",  format_args=_action_name_args),
    "mcp":             ToolDisplay(icon="🔌",  format_args=_mcp_args),
}


def get_display(tool_name: str) -> ToolDisplay:
    """Return the ToolDisplay for ``tool_name`` (default if unregistered)."""
    return TOOL_DISPLAY.get(tool_name, _DEFAULT)


def get_icon(tool_name: str) -> str:
    return get_display(tool_name).icon


def format_tool_args(tool_name: str, tool_input: dict) -> str:
    """Format tool arguments for terminal display.

    Returns a one-line summary suitable for appending after the icon+name.
    Empty string if the tool has no input.
    """
    if not tool_input:
        return ""
    return get_display(tool_name).format_args(tool_name, tool_input)


__all__ = [
    "ToolDisplay",
    "TOOL_DISPLAY",
    "get_display",
    "get_icon",
    "format_tool_args",
]
