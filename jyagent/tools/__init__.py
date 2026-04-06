# tools/ package — Core tool implementations, schemas, and registration.
#
# This package replaces the monolithic tools.py (900 lines) with focused modules:
#   core.py        — run_shell, read_file, write_file, list_directory, edit_file
#   search.py      — glob_files, grep_files
#   facades.py     — manage_memory, manage_skills (thin wrappers)
#   schemas.py     — CORE_TOOLS JSON schema definitions

from ..registry import get_registry

# Re-export all tool functions for backward compatibility
from .core import run_shell, read_file, write_file, list_directory, edit_file
from .search import glob_files, grep_files
from .facades import manage_memory, manage_skills
from .schemas import CORE_TOOLS
from .web_fetch import web_fetch, TOOL_SCHEMA as WEB_FETCH_SCHEMA
from .mcp_tool import mcp, TOOL_SCHEMA as MCP_SCHEMA
from .subagent import dispatch_agent, set_client as set_subagent_client, TOOL_SCHEMA as SUBAGENT_SCHEMA

# Re-export constants from config (backward compat)
from ..config import SKIP_DIRS, BINARY_EXTS

# ─── Register core tools ──────────────────────────────────────────────────────

_TOOL_FN_MAP = {
    "run_shell": run_shell,
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "edit_file": edit_file,
    "glob_files": glob_files,
    "grep_files": grep_files,
    "manage_memory": manage_memory,
    "manage_skills": manage_skills,
    "web_fetch": web_fetch,
    "mcp": mcp,
    "dispatch_agent": dispatch_agent,
}

_TOOL_METADATA = {
    "read_file":       {"parallel_safe": True},
    "list_directory":  {"parallel_safe": True},
    "glob_files":      {"parallel_safe": True},
    "grep_files":      {"parallel_safe": True},
    "run_shell":       {"parallel_safe": False, "timeout_hint": "from_input"},
    "write_file":      {"parallel_safe": False, "large_input_keys": {"content"}},
    "edit_file":       {"parallel_safe": False, "large_input_keys": {"new_text", "old_text"}},
    "manage_memory":   {"parallel_safe": False},
    "manage_skills":   {"parallel_safe": False},
    "web_fetch":       {"parallel_safe": False, "timeout_hint": 180},
    "mcp":             {"parallel_safe": False, "timeout_hint": 180},
    "dispatch_agent":  {"parallel_safe": True, "timeout_hint": 300, "large_input_keys": {"context"}},
}

_registry = get_registry()
for tool_def in CORE_TOOLS + [WEB_FETCH_SCHEMA, MCP_SCHEMA, SUBAGENT_SCHEMA]:
    fn = _TOOL_FN_MAP.get(tool_def["name"])
    if fn:
        meta = _TOOL_METADATA.get(tool_def["name"], {})
        timeout = meta.get("timeout_hint")
        # "from_input" is a sentinel — run_shell reads timeout from its own input
        timeout_hint = timeout if isinstance(timeout, int) else None
        large_keys = meta.get("large_input_keys")
        _registry.register(
            tool_def["name"], fn, tool_def,
            parallel_safe=meta.get("parallel_safe", False),
            timeout_hint=timeout_hint,
            large_input_keys=large_keys,
        )
