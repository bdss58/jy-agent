# tools/ package — Core tool implementations, schemas, and registration.
#
# This package replaces the monolithic tools.py (900 lines) with focused modules:
#   core.py        — run_shell, read_file, write_file, list_directory, edit_file
#   search.py      — glob_files, grep_files
#   facades.py     — manage_memory, manage_skills (thin wrappers)
#   schemas.py     — CORE_TOOLS JSON schema definitions

from ..registry import get_registry

# Re-export all tool functions for backward compatibility
from .core import run_shell, read_file, write_file, list_directory, edit_file, run_background, check_background
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
    "run_background": run_background,
    "check_background": check_background,
}

_TOOL_METADATA = {
    "read_file":       {"parallel_safe": True, "compaction_priority": "persistent"},
    "list_directory":  {"parallel_safe": True, "compaction_priority": "ephemeral"},
    "glob_files":      {"parallel_safe": True, "compaction_priority": "ephemeral"},
    "grep_files":      {"parallel_safe": True, "compaction_priority": "ephemeral"},
    "run_shell":       {"parallel_safe": False, "timeout_hint": "from_input", "compaction_priority": "ephemeral"},
    "write_file":      {"parallel_safe": False, "large_input_keys": {"content"}},
    "edit_file":       {"parallel_safe": False, "large_input_keys": {"new_text", "old_text"}},
    "manage_memory":   {"parallel_safe": False},
    "manage_skills":   {"parallel_safe": False},
    "web_fetch":       {"parallel_safe": False, "timeout_hint": 180, "compaction_priority": "persistent"},
    "mcp":             {"parallel_safe": False, "timeout_hint": 180},
    "dispatch_agent":  {"parallel_safe": True, "timeout_hint": 300, "large_input_keys": {"context"}},
    "run_background":  {"parallel_safe": False},
    "check_background": {"parallel_safe": True, "compaction_priority": "ephemeral", "dedup_exempt": True},
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
        compaction_priority = meta.get("compaction_priority")
        _registry.register(
            tool_def["name"], fn, tool_def,
            parallel_safe=meta.get("parallel_safe", False),
            timeout_hint=timeout_hint,
            large_input_keys=large_keys,
            compaction_priority=compaction_priority,
            dedup_exempt=meta.get("dedup_exempt", False),
        )
