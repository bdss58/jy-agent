# tools/ package — Core tool implementations, schemas, and registration.

from ..runtime.tools.registry import get_registry

# Bring tool function impls into module scope so ``_TOOL_FN_MAP`` below
# can wire them up at registry-init time.
from .core import run_shell, read_file, write_file, list_directory, edit_file
from .background import run_background, check_background
from .search import glob_files, grep_files
from .facades import manage_memory, manage_skills
from .schemas import CORE_TOOLS
# NOTE: alias as `web_fetch_fn` so the `.web_fetch` submodule is NOT shadowed
# at `jyagent.tools.web_fetch`. Tests and downstream code can then do
#     import jyagent.tools.web_fetch as web_fetch_mod
# and patch internals naturally. (Same pattern as `web_search_fn` below.)
from .web_fetch import web_fetch as web_fetch_fn, TOOL_SCHEMA as WEB_FETCH_SCHEMA
from .mcp_tool import mcp, TOOL_SCHEMA as MCP_SCHEMA
from .subagent import (
    dispatch_agent, check_agent,
    TOOL_SCHEMA as SUBAGENT_SCHEMA,
    CHECK_AGENT_SCHEMA,
)
from .web_search import web_search as web_search_fn, TOOL_SCHEMA as WEB_SEARCH_SCHEMA

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
    "web_fetch": web_fetch_fn,
    "mcp": mcp,
    "dispatch_agent": dispatch_agent,
    "check_agent": check_agent,
    "run_background": run_background,
    "check_background": check_background,
    "web_search": web_search_fn,
}

_TOOL_METADATA = {
    "read_file":       {"parallel_safe": True, "compaction_priority": "persistent"},
    "list_directory":  {"parallel_safe": True, "compaction_priority": "ephemeral"},
    "glob_files":      {"parallel_safe": True, "compaction_priority": "ephemeral"},
    "grep_files":      {"parallel_safe": True, "compaction_priority": "ephemeral"},
    "run_shell":       {"parallel_safe": False, "timeout_hint": "from_input", "compaction_priority": "ephemeral", "mutating": True},
    "write_file":      {"parallel_safe": False, "large_input_keys": {"content"}, "mutating": True},
    "edit_file":       {"parallel_safe": False, "large_input_keys": {"new_text", "old_text"}, "mutating": True},
    "manage_memory":   {"parallel_safe": False, "mutating": True},
    "manage_skills":   {"parallel_safe": False, "mutating": True},
    "web_fetch":       {"parallel_safe": False, "timeout_hint": 180, "compaction_priority": "persistent"},
    "mcp":             {"parallel_safe": False, "timeout_hint": 180, "mutating": True},
    # dispatch_agent is now serial.  Sub-agents
    # are coarse-grained (each one runs an entire AgentLoop on the shared
    # tool-dispatch pool); serialising at the top level avoids cross-pool
    # reentrancy under high parallel-tool-call fan-outs.  The bg path still
    # works fine because dispatch_agent returns immediately after the grace
    # period for ``background=True`` — it doesn't rely on the parent batch's
    # parallel_safe flag for liveness.
    "dispatch_agent":  {"parallel_safe": False, "timeout_hint": 300, "large_input_keys": {"context"}, "mutating": True},
    "check_agent":     {"parallel_safe": True, "compaction_priority": "ephemeral"},
    "run_background":  {"parallel_safe": False, "mutating": True},
    "check_background": {"parallel_safe": True, "compaction_priority": "ephemeral", "timeout_hint": 360, "mutating": True},
    "web_search":      {"parallel_safe": True, "timeout_hint": 180, "compaction_priority": "persistent"},
}

# NOTE on the ``mutating`` flag:
#   Flagged tools have externally-observable side effects (filesystem writes,
#   shell commands, sub-process spawns, sub-agent dispatches, MCP calls) that
#   the dispatch loop cannot cancel when the tool times out — the daemon
#   thread carrying the side effect keeps running past the timeout report.
#   The loop engine uses this flag to (a) log a loud WARNING, (b) rewrite the
#   ToolResult error text so the model knows to re-verify state before
#   retrying, and (c) accumulate the tool name in
#   ``LoopResult.partial_side_effects`` for outer layers to reconcile.
#   Read-only / query tools (read_file, list_directory, grep_files, glob_files,
#   web_search, web_fetch, check_agent, manage_memory, manage_skills) default
#   to mutating=False because a timed-out read is idempotent — retrying is
#   always safe.
#
#   ``check_background`` is FLAGGED mutating
#   because the ``action="kill"`` branch SIGTERM/SIGKILLs the target
#   process group — that side effect cannot be undone if the call times
#   out mid-kill.  ``timeout_hint=360`` gives 60 s slack on top of the
#   schema-documented ``wait_timeout_seconds`` cap of 300 s.

_registry = get_registry()
for tool_def in CORE_TOOLS + [WEB_FETCH_SCHEMA, MCP_SCHEMA, SUBAGENT_SCHEMA, CHECK_AGENT_SCHEMA, WEB_SEARCH_SCHEMA]:
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
            mutating=meta.get("mutating", False),
        )
