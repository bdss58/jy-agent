"""jyagent.runtime — the in-tree agent runtime (loop engine, tools, stats, skills).

Public API surface (will be filled in across phases 2-4):
    AgentLoop, LoopConfig, LoopResult, LoopCallbacks   (loop.engine / loop.config / loop.callbacks)
    get_registry, ToolResult                           (tools.registry / tools.result)
    get_stats, SessionStats                            (stats)
"""
from .tools import get_registry, ToolResult
from .stats import get_stats, SessionStats

__all__ = [
    "get_registry",
    "ToolResult",
    "get_stats",
    "SessionStats",
]
