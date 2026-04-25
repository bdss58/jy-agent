"""Backward-compat shim — module moved to ``jyagent.runtime.loop.engine``.

Re-exports the *live* module so AgentLoop / LoopConfig / LoopResult /
LoopCallbacks / private helpers (_compact_messages, _CostTracker,
_StuckLoopDetector, ToolCallRequest, …) all resolve to the same objects
as ``jyagent.runtime.loop.engine``.
"""
import warnings as _warnings

from jyagent.runtime.loop.engine import *  # noqa: F401,F403
from jyagent.runtime.loop.engine import __dict__ as _new_dict

globals().update({k: v for k, v in _new_dict.items() if not k.startswith("__")})

_warnings.warn(
    "jyagent.loop_engine has moved to jyagent.runtime.loop.engine; update your imports.",
    DeprecationWarning,
    stacklevel=2,
)
