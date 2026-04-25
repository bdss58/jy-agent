"""jyagent.runtime — in-tree agent runtime (loop engine, tools, stats).

The runtime owns the agentic execution loop, tool dispatch/registry, and
session statistics.  Skill management (prompt-context routing, LLM-based
auto-activation) lives in ``jyagent.skills`` — it is a higher-level
application concern that consumes the runtime, not part of it.
Refactor #3 (2026-04-25, Codex review Part 3 #3 #4) lifted ``SkillManager``
out of this package precisely because the loop engine has zero
dependency on it.
"""
from .loop import AgentLoop, LoopConfig, LoopResult, LoopCallbacks
from .loop.llm_client import LLMClient
from .loop.llm_types import LLMOptions, ModelSpec
from .tools import get_registry, ToolResult
from .stats import get_stats, SessionStats

__all__ = [
    "AgentLoop",
    "LoopConfig",
    "LoopResult",
    "LoopCallbacks",
    "LLMClient",
    "LLMOptions",
    "ModelSpec",
    "get_registry",
    "ToolResult",
    "get_stats",
    "SessionStats",
]
