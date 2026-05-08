"""Shared tool-dispatch pool — module state + lazy-grow helper.

Owns three pieces of process-wide mutable state:

* ``tool_dispatch_executor`` — the ``ThreadPoolExecutor`` that
  ``execute_tools`` fans parallel-safe sub-batches onto.  Lazy: ``None``
  until first use.
* ``tool_dispatch_lock`` — guards growth (recreate-and-replace).
* ``tool_dispatch_cap`` — current ``max_workers`` of the live pool;
  monotonic across the process lifetime (the helper grows but never
  shrinks).

The state lives at module scope rather than inside a class because
there is exactly one shared pool per process.  Multiple ``AgentLoop``
instances in the same process — including parent + sub-agent stacks —
share the pool to avoid accumulating ThreadPoolExecutor objects and
``atexit`` handlers across turns.

Why a single shared pool (vs. one per loop)?
  * Pools cost OS threads and an ``atexit.shutdown`` registration each.
    A long-running session that spawns many loops would leak both.
  * Per-loop concurrency caps are honoured at a finer layer: each
    ``execute_tools`` call uses a per-call ``BoundedSemaphore`` sized
    from ``LoopConfig.max_tool_workers``, so the shared pool can stay
    hot at a high width without violating any individual loop's
    preferences.

Why grow but never shrink?
  * Shrinking would require shutting down workers that may still be
    holding in-flight dispatches.  Cheaper to leave the pool sized at
    the high-water mark — idle workers cost ~0 CPU.

Read access via the helper, NOT a captured snapshot
---------------------------------------------------
Callers MUST use ``get_tool_dispatch_executor()`` (or, in the
``AgentLoop`` case, the ``_executor`` property which calls into here)
rather than capture ``tool_dispatch_executor`` at import time.  Growth
calls ``old.shutdown(wait=False)`` and replaces the module-level
binding; a captured snapshot would point at the dead pool and the next
``.submit`` would raise ``RuntimeError: cannot schedule new futures
after shutdown``.  The 2026-05 ``AgentLoop._executor`` fix removed the
last such snapshot in production code.

Module layout
-------------
This module was extracted from ``runtime/loop/tool_executor.py`` in
2026-05 alongside ``tool_invocation.py`` (``execute_tool`` /
``execute_tool_with_timeout``) and ``tool_dispatch.py``
(``execute_tools``).  ``tool_executor.py`` is now a thin facade that
re-exports the three leaf modules' public names; existing imports keep
working unchanged.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import logging
import threading


_logger = logging.getLogger(__name__)


# ── Module state (mutable; canonical home) ────────────────────────────────
#
# Reads should go through ``get_tool_dispatch_executor`` so the caller
# always sees the live pool.  ``tool_executor`` (the back-compat facade)
# uses PEP-562 ``__getattr__`` to mirror these names so legacy reads via
# ``tool_executor.tool_dispatch_executor`` still observe the live value
# without snapshotting.
#
# Writes (e.g. test setup / teardown) MUST target this module — assigning
# through the facade creates a static attribute that shadows the
# ``__getattr__`` passthrough and breaks subsequent live-pool reads.

tool_dispatch_executor: concurrent.futures.ThreadPoolExecutor | None = None
tool_dispatch_lock = threading.Lock()
tool_dispatch_cap = 0


def get_tool_dispatch_executor(
    min_workers: int = 8,
) -> concurrent.futures.ThreadPoolExecutor:
    """Return the shared dispatch executor, growing it if needed.

    The historical eagerly-created executor was hard-capped at 8
    workers, so ``LoopConfig.max_tool_workers > 8`` was silently
    honoured at the body-permit layer but starved at dispatch.  This
    helper lazy-creates and grows the pool to the largest
    ``max_tool_workers`` ever requested across all live ``AgentLoop``
    instances in the process.

    Growth recreates the pool: the old one's ``shutdown(wait=False)``
    lets in-flight dispatches finish in their own threads (we never
    block on them here), but no new tasks are accepted on it.
    Concurrent callers are serialised by ``tool_dispatch_lock``.

    Floor: returns at least an 8-worker pool even when ``min_workers <
    8`` so an under-configured ``LoopConfig`` still has reasonable
    parallelism for the common parallel-safe-batch path.
    """
    global tool_dispatch_executor, tool_dispatch_cap
    target = max(int(min_workers), 8)
    # Fast path: existing executor already big enough.
    if tool_dispatch_executor is not None and tool_dispatch_cap >= target:
        return tool_dispatch_executor
    with tool_dispatch_lock:
        if tool_dispatch_executor is not None and tool_dispatch_cap >= target:
            return tool_dispatch_executor
        if tool_dispatch_executor is not None:
            _logger.info(
                "expanding tool dispatch pool: %d -> %d workers",
                tool_dispatch_cap, target,
            )
            old = tool_dispatch_executor
            try:
                atexit.unregister(old.shutdown)
            except Exception:  # noqa: BLE001 — atexit unregister is best-effort
                pass
            old.shutdown(wait=False)
        tool_dispatch_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=target,
            thread_name_prefix="jyagent-tool-dispatch",
        )
        tool_dispatch_cap = target
        atexit.register(tool_dispatch_executor.shutdown, wait=False)
        return tool_dispatch_executor


__all__ = [
    "tool_dispatch_executor",
    "tool_dispatch_lock",
    "tool_dispatch_cap",
    "get_tool_dispatch_executor",
]
