"""jyagent.runtime — in-tree agent runtime (loop engine, tools, stats).

The runtime owns the agentic execution loop, tool dispatch/registry, and
session statistics.  Skill management (prompt-context routing, LLM-based
auto-activation) lives in ``jyagent.skills`` — it is a higher-level
application concern that consumes the runtime, not part of it.
``SkillManager`` lives outside this package precisely because the loop engine
has zero dependency on it.

Lazy public API.
~~~~~~~~~~~~~~~~

Cheap symbols (configs, callbacks, types, tools, stats) are imported
eagerly because their leaf modules have zero transitive dependency on
the engine.  The engine itself (``AgentLoop`` and its private helpers)
is loaded lazily via PEP-562 ``__getattr__`` — so callers that only
need ``LoopConfig`` / ``LoopCallbacks`` / ``LLMOptions`` / ``ModelSpec``
to construct a config don't pay the engine import cost (4 sub-modules,
a dispatch thread pool, atexit hooks).

Behaviour:

* ``from jyagent.runtime import AgentLoop`` — works, lazy-loads.
* ``import jyagent.runtime; jyagent.runtime.AgentLoop`` — works, lazy-loads.
* ``from jyagent.runtime import LoopConfig`` — eager, no engine load.
* ``import jyagent.runtime; sys.modules['jyagent.runtime.loop.engine']``
  is NOT in ``sys.modules`` until something touches ``AgentLoop``.

The first lazy access triggers a normal ``from .loop import AgentLoop``,
which loads ``loop/__init__.py`` → ``engine.py`` and caches the result
on this module so subsequent attribute lookups are O(1).
"""
from .loop.callbacks import LoopCallbacks
from .loop.config import LoopConfig, LoopResult
from .loop.llm_client import LLMClient
from .loop.llm_types import LLMOptions, ModelSpec
from .tools import get_registry, ToolResult
from .stats import get_stats, SessionStats

__all__ = [
    "AgentLoop",       # lazy
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


# ─── PEP 562 lazy attribute loader ──────────────────────────────────────────
#
# Maps public-API names served lazily to their canonical source module.
# A miss raises AttributeError per PEP 562 contract.  The loaded value is
# cached on this module by Python's normal `from X import Y` machinery on
# subsequent accesses (CPython resolves `module.Y` from `module.__dict__`
# before falling back to `__getattr__`), so the lazy path runs at most once
# per name.

_LAZY_ATTRS = {
    "AgentLoop": ("jyagent.runtime.loop", "AgentLoop"),
}


def __getattr__(name: str):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module_name, attr = target
    value = getattr(importlib.import_module(module_name), attr)
    # Cache on this module so future accesses don't hit __getattr__.
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(__all__) | set(globals()))
