"""Tool display + blast-radius metadata.

Lives in the *tools* package (not *ui*) because it's part of each tool's
public-facing surface, not part of the renderer.  ``ui/terminal.py`` should
ask the tools package what to show; it must NOT embed a per-tool switch
table — that drifts every time a new tool is added.

This module now hosts TWO related per-tool registries:

1. **ToolDisplay** (icon + arg formatter) — for terminal rendering.
2. **BlastRadius** (verb + target + irreversibility + taint markers) —
   for the approval gate (``on_tool_pre_execute``) and the upcoming
   per-tool sandbox policy decisions.

Both registries are deliberately co-located: they share the same
per-tool lookup key, the same "add an entry when you add a tool"
discipline, and the same "no per-tool switch table in ui/" goal.
Keeping them in one file means a new tool is registered once.

Security note: BlastRadius is a HINT shown to the user, not a hard
boundary.  Sandboxing of the actual operation (filesystem deny-lists,
``sandbox-exec`` for shell, etc.) is Tier 3+ in the sandboxing roadmap
(see ``docs/design/2026-05-sandboxing-tier1-tier2.md``).  The
``destructive_shell`` regex below is BEST-EFFORT — Cursor's
CVE-2026-22708 demonstrated that shell-builtin-based bypass of any
syntactic allowlist is trivially possible.  This module's role is to
nudge the user toward attention, not to enforce policy.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlparse

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


_DEFAULT_DISPLAY = ToolDisplay()


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
    return TOOL_DISPLAY.get(tool_name, _DEFAULT_DISPLAY)


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


# ─── Blast-radius registry ───────────────────────────────────────────────────


@dataclass(frozen=True)
class BlastRadius:
    """Per-call security tag shown to the user before execution.

    Fields:
        verb            — short uppercase action category. One of:
                          NONE | READ | WRITE | EXEC | FETCH | SEND |
                          DELETE | SPAWN | NETWORK.
        target          — short human-readable target (path, host, contact,
                          command head).  Capped at ~60 chars.
        irreversible    — True if the side-effect cannot be undone by the
                          agent (shell with ``rm``/``mv``/``curl|sh``/
                          ``git push``, WeChat send, memory ``forget``,
                          skill ``delete``).  Approval gate FORCES a prompt
                          when this is True, even in allow-all mode.
        taint_source    — True if this call IMPORTS untrusted content into
                          the agent's context (``web_fetch``, ``web_search``).
                          When such a call returns, the runtime sets the
                          step on ``RunState.tainted_steps``.
        taint_sensitive — True if this call CONSUMES context that could be
                          poisoned by upstream taint (``run_shell``,
                          ``write_file``, ``edit_file``, ``manage_memory``
                          write, ``manage_skills`` write).  Approval gate
                          FORCES a prompt when this is True AND any taint
                          is active in the session, even in allow-all mode.
    """
    verb: str = "NONE"
    target: str = ""
    irreversible: bool = False
    taint_source: bool = False
    taint_sensitive: bool = False


_DEFAULT_BLAST = BlastRadius()

# ─── Per-tool blast-radius callables ─────────────────────────────────────────


# Best-effort regex of "destructive" shell verbs.  See module docstring:
# this is a HINT not a hard boundary.  We err on the side of marking too
# many commands irreversible (extra prompt) rather than missing one.
_DESTRUCTIVE_SHELL = re.compile(
    r"(?:^|[\s|;&`(])\s*"
    r"(?:rm\s+(?:-[a-zA-Z]*[rf]|-r|-f)|"        # rm -r / -f / -rf etc.
    r"mv\s+[^\s]+\s+/|"                         # mv X /…
    r"shred|dd\s+if=|mkfs|fdisk|diskutil\s+erase|"
    r"git\s+push(?:\s+(?:-f|--force))?|"
    r":(?:>|>>)\s*/|"                           # truncating redirect to root
    r"curl[^|]*\|\s*(?:bash|sh|zsh)|"
    r"wget[^|]*\|\s*(?:bash|sh|zsh)|"
    r"sudo\s+|"
    r"kill(?:all)?\s+|launchctl\s+(?:unload|remove)|"
    r"defaults\s+delete|networksetup\s+|"
    r"npm\s+(?:publish|unpublish)|pip\s+install|brew\s+(?:install|uninstall|cleanup))",
    re.IGNORECASE,
)

# Paths that we always treat as sensitive (write or read).
_SENSITIVE_PATH_GLOBS = (
    re.compile(r"(?:^|/)\.ssh(?:/|$)"),
    re.compile(r"(?:^|/)\.aws(?:/|$)"),
    re.compile(r"(?:^|/)\.gnupg(?:/|$)"),
    re.compile(r"(?:^|/)\.netrc(?:$|\.)"),
    re.compile(r"\.zsh_history|\.bash_history|\.python_history"),
    re.compile(r"(?:^|/)\.zshrc|\.bashrc|\.zprofile|\.bash_profile|\.profile(?:$|\.)"),
    re.compile(r"\.git/(?:hooks/|config$)"),       # CVE-2026-26268 lesson
    re.compile(r"\.gitconfig(?:$|\.)"),
)


def _sensitive_path(path: str) -> bool:
    if not path:
        return False
    expanded = os.path.expanduser(path)
    return any(p.search(expanded) for p in _SENSITIVE_PATH_GLOBS)


def _short(s: str, n: int = 60) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


# Read-only / informational
def _br_none(_name: str, _input: dict) -> BlastRadius:
    return BlastRadius()


def _br_read(_name: str, tool_input: dict) -> BlastRadius:
    path = tool_input.get("path", "")
    return BlastRadius(verb="READ", target=_short(path))


def _br_list(_name: str, tool_input: dict) -> BlastRadius:
    return BlastRadius(verb="READ", target=_short(tool_input.get("path", ".") or "."))


def _br_glob(_name: str, tool_input: dict) -> BlastRadius:
    return BlastRadius(verb="READ", target=_short(tool_input.get("pattern", "")))


def _br_grep(_name: str, tool_input: dict) -> BlastRadius:
    target = tool_input.get("pattern", "")
    path = tool_input.get("path", "")
    return BlastRadius(verb="READ", target=_short(f"'{target}'" + (f" in {path}" if path else "")))


def _br_write(_name: str, tool_input: dict) -> BlastRadius:
    path = tool_input.get("path", "")
    dry_run = bool(tool_input.get("dry_run"))
    return BlastRadius(
        verb="WRITE",
        target=_short(path),
        irreversible=(not dry_run) and _sensitive_path(path),
        taint_sensitive=True,
    )


def _br_shell(_name: str, tool_input: dict) -> BlastRadius:
    cmd = tool_input.get("command", "") or ""
    destructive = bool(_DESTRUCTIVE_SHELL.search(cmd))
    return BlastRadius(
        verb="EXEC",
        target=_short(cmd, 60),
        irreversible=destructive,
        taint_sensitive=True,
    )


def _br_background(_name: str, tool_input: dict) -> BlastRadius:
    cmd = tool_input.get("command", "") or ""
    return BlastRadius(
        verb="EXEC",
        target=_short(f"bg: {cmd}", 60),
        irreversible=bool(_DESTRUCTIVE_SHELL.search(cmd)),
        taint_sensitive=True,
    )


def _br_web_fetch(_name: str, tool_input: dict) -> BlastRadius:
    url = tool_input.get("url", "") or ""
    host = ""
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        host = ""
    return BlastRadius(verb="FETCH", target=_short(host or url), taint_source=True)


def _br_web_search(_name: str, tool_input: dict) -> BlastRadius:
    query = tool_input.get("query", "") or ""
    return BlastRadius(verb="FETCH", target=_short(query), taint_source=True)


def _br_memory(_name: str, tool_input: dict) -> BlastRadius:
    action = (tool_input.get("action") or "").lower()
    # Pure-read actions on memory are not sensitive.
    if action in ("show", "search", "consolidate", "goal"):
        return BlastRadius(verb="READ", target=_short(f"memory {action}"))
    # "topic" with read:* is read; other topic ops are writes.
    if action == "topic":
        text = (tool_input.get("text") or "").lstrip()
        cmd = text.split(":", 1)[0].lower() if ":" in text else text.lower()
        if cmd in ("read", "list", "sections"):
            return BlastRadius(verb="READ", target=_short(f"topic {text}"))
        return BlastRadius(
            verb="WRITE", target=_short(f"topic {text}"),
            irreversible=True, taint_sensitive=True,
        )
    if action in ("forget",):
        return BlastRadius(
            verb="DELETE", target=_short(tool_input.get("text", "")),
            irreversible=True, taint_sensitive=True,
        )
    # remember | journal | other writes.
    return BlastRadius(
        verb="WRITE", target=_short(f"{action}: {tool_input.get('text', '')}"),
        irreversible=True, taint_sensitive=True,
    )


def _br_skills(_name: str, tool_input: dict) -> BlastRadius:
    action = (tool_input.get("action") or "").lower()
    sk_name = tool_input.get("name", "") or ""
    if action in ("create", "delete"):
        return BlastRadius(
            verb=("DELETE" if action == "delete" else "WRITE"),
            target=_short(f"skill {sk_name}"),
            irreversible=True, taint_sensitive=True,
        )
    # load / pin / list / info / read / deactivate / reload are non-mutating
    return BlastRadius(verb="READ", target=_short(f"{action} {sk_name}"))


def _br_mcp(_name: str, tool_input: dict) -> BlastRadius:
    action = (tool_input.get("action") or "").lower()
    server = tool_input.get("server", "") or ""
    if action in ("connect", "disconnect", "reconnect"):
        # Connection lifecycle — mutates external state.
        return BlastRadius(verb="NETWORK", target=_short(f"{action} {server}"))
    return BlastRadius(verb="READ", target=_short(f"{action} {server}"))


def _br_dispatch_agent(_name: str, tool_input: dict) -> BlastRadius:
    task = tool_input.get("task", "") or ""
    # Sub-agent spawn is itself reversible (the parent can ignore the
    # result), but it consumes parent context that may be tainted, and a
    # child agent inherits the full toolset.  Mark taint_sensitive so the
    # gate prompts when active taint exists.
    return BlastRadius(verb="SPAWN", target=_short(task, 60), taint_sensitive=True)


# Registry — per-tool blast-radius callables.  Tools not listed fall back
# to the registry's mutating-flag (in runtime/tools/registry.py) for
# approval routing; ``blast_radius()`` returns BlastRadius() (verb=NONE).
TOOL_BLAST_RADIUS: dict[str, Callable[[str, dict], BlastRadius]] = {
    # Read-only
    "read_file":       _br_read,
    "list_directory":  _br_list,
    "glob_files":      _br_glob,
    "grep_files":      _br_grep,
    # Write
    "write_file":      _br_write,
    "edit_file":       _br_write,
    # Exec
    "run_shell":       _br_shell,
    "run_background":  _br_background,
    "check_background": _br_none,
    # Network
    "web_fetch":       _br_web_fetch,
    "web_search":      _br_web_search,
    # Agent-internal
    "manage_memory":   _br_memory,
    "manage_skills":   _br_skills,
    "mcp":             _br_mcp,
    "dispatch_agent":  _br_dispatch_agent,
    "check_agent":     _br_none,
}


def blast_radius(tool_name: str, tool_input: dict) -> BlastRadius:
    """Return the BlastRadius tag for a tool call.

    Unregistered tools return ``BlastRadius()`` (verb=NONE) so the
    approval gate falls back to its mutating-flag heuristic.
    """
    fn = TOOL_BLAST_RADIUS.get(tool_name)
    if fn is None:
        return _DEFAULT_BLAST
    try:
        return fn(tool_name, tool_input or {})
    except Exception:
        # Never let a metadata bug crash the loop — fall back to safe default.
        return _DEFAULT_BLAST


__all__ = [
    "ToolDisplay",
    "TOOL_DISPLAY",
    "get_display",
    "get_icon",
    "format_tool_args",
    "BlastRadius",
    "TOOL_BLAST_RADIUS",
    "blast_radius",
]
