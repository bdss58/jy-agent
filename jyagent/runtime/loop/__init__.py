"""Agent loop runtime: engine, callbacks, config, and supporting helpers.

Eager re-exports of the small public surface.  The previous PEP-562 lazy
loader was deleted in the 2026-05 simplification pass — for a personal
CLI the cold-import cost saved (a few ms) was not worth the
``__getattr__`` indirection and the test scaffolding that came with it.
"""
from .callbacks import LoopCallbacks
from .config import LoopConfig, LoopResult, build_default_loop_config
from .engine import AgentLoop

__all__ = [
    "AgentLoop",
    "LoopConfig",
    "LoopResult",
    "build_default_loop_config",
    "LoopCallbacks",
]
