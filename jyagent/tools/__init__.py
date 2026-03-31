# tools/ package — Core tool implementations, schemas, registration, and auto-discovery.
#
# This package replaces the monolithic tools.py (900 lines) with focused modules:
#   core.py        — run_shell, read_file, write_file, list_directory
#   agent_tools.py — evolve_self, add_tool
#   facades.py     — manage_memory, manage_skills (thin wrappers)
#   schemas.py     — CORE_TOOLS JSON schema definitions
#   discovery.py   — auto_discover_tools (loads tool_*.py at import time)
#   utils.py       — strip_unsupported_schema_keys, shared helpers

from ..registry import get_registry

# Re-export all tool functions for backward compatibility
from .core import run_shell, read_file, write_file, list_directory
from .agent_tools import evolve_self, add_tool
from .facades import manage_memory, manage_skills
from .schemas import CORE_TOOLS
from .discovery import auto_discover_tools

# Re-export constants from config (backward compat for tool_glob_grep etc.)
from ..config import SKIP_DIRS, BINARY_EXTS
from .utils import strip_unsupported_schema_keys

# ─── Register core tools ──────────────────────────────────────────────────────

_TOOL_FN_MAP = {
    "run_shell": run_shell,
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "evolve_self": evolve_self,
    "add_tool": add_tool,
    "manage_memory": manage_memory,
    "manage_skills": manage_skills,
}

_registry = get_registry()
for tool_def in CORE_TOOLS:
    fn = _TOOL_FN_MAP.get(tool_def["name"])
    if fn:
        _registry.register(tool_def["name"], fn, tool_def)

# ─── Auto-discover tool_*.py files ────────────────────────────────────────────
auto_discover_tools()
