"""Tool execution stack — back-compat facade for the three leaf modules.

The implementation was split in 2026-05 into three focused modules:

* :mod:`jyagent.runtime.loop.tool_pool` — shared dispatch executor
  (``tool_dispatch_executor``, ``tool_dispatch_lock``,
  ``tool_dispatch_cap``, ``get_tool_dispatch_executor``).
* :mod:`jyagent.runtime.loop.tool_invocation` — single-call invocation
  (``execute_tool``, ``execute_tool_with_timeout``,
  ``_accepts_cancel_event``).
* :mod:`jyagent.runtime.loop.tool_dispatch` — fan-out across parallel-
  safe / mutating sub-batches (``execute_tools``).

This file remains as a stable import target for callers that already
say ``from jyagent.runtime.loop.tool_executor import execute_tools`` (or
``from jyagent.runtime.loop import tool_executor as te`` followed by
``te.execute_tool_with_timeout(...)``).  Every public name from the
three leaf modules is re-exported here under the original spelling.

Live-state passthrough (PEP 562)
--------------------------------
The mutable pool state — ``tool_dispatch_executor``,
``tool_dispatch_lock``, ``tool_dispatch_cap`` — lives in
``tool_pool``.  This module's ``__getattr__`` resolves those names
against ``tool_pool`` on every read, so legacy access patterns like
``te.tool_dispatch_executor`` always return the LIVE pool, not a
snapshot taken at import time.

WRITES are different.  A direct assignment ``te.tool_dispatch_executor
= X`` creates a real attribute on this module and shadows the
``__getattr__`` passthrough — subsequent reads return the static
shadow rather than the canonical state in ``tool_pool``.  Test setup
that needs to mutate pool state MUST target ``tool_pool`` directly:

.. code-block:: python

    from jyagent.runtime.loop import tool_pool
    tool_pool.tool_dispatch_executor = saved
    tool_pool.tool_dispatch_cap = saved_cap

The few in-tree tests that previously assigned through this facade
were updated when the split landed.
"""

from __future__ import annotations

# Re-export every public name from the three leaf modules so existing
# imports keep working unchanged.  Function and helper names are
# normal ``from`` imports (object identity is shared with the leaf
# modules — patches via ``setattr(tool_executor, "execute_tools", X)``
# only affect the facade's local binding, NOT the canonical home).
from .tool_invocation import (
    _accepts_cancel_event,
    execute_tool,
    execute_tool_with_timeout,
)
from .tool_dispatch import execute_tools
from .tool_pool import get_tool_dispatch_executor


# PEP-562 passthrough: read access to the mutable pool-state names
# always returns the current value from ``tool_pool``.  We deliberately
# do NOT cache these names at module level — that would defeat the
# whole point of the passthrough (a captured snapshot can be left
# pointing at a shut-down pool after growth).
_PASSTHROUGH = frozenset({
    "tool_dispatch_executor",
    "tool_dispatch_lock",
    "tool_dispatch_cap",
})


def __getattr__(name: str):
    if name in _PASSTHROUGH:
        from . import tool_pool
        return getattr(tool_pool, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _PASSTHROUGH)


__all__ = [
    # Tool invocation
    "execute_tool",
    "execute_tool_with_timeout",
    # Dispatch fan-out
    "execute_tools",
    # Pool lifecycle
    "get_tool_dispatch_executor",
    "tool_dispatch_executor",
    "tool_dispatch_lock",
    "tool_dispatch_cap",
]
