"""Tool dispatch — fan-out across parallel-safe / mutating sub-batches.

Owns the order-and-concurrency policy for executing a list of
``ToolCallRequest`` blocks.  Reads ``parallel_safe`` and ``mutating``
flags from the immutable per-step ``ToolBatch`` snapshot, so a
concurrent registry mutation cannot flip a tool's classification
mid-partition.

Algorithm
---------
1. **Fast paths** — single tool or ``concurrent_mode=False`` falls back
   to a simple sequential loop, eliminating thread creation overhead.
2. **All-sequential check** — if no tool in the batch is parallel-safe,
   skip the partition machinery and run sequentially.
3. **Partition + dispatch** — walk the blocks, gathering contiguous
   parallel-safe runs and submitting each as one batch onto the shared
   dispatch pool (capped per-call by a ``BoundedSemaphore`` sized from
   ``LoopConfig.max_tool_workers``).  Mutating tools run sequentially
   between parallel batches as ordering barriers.

Result ordering
---------------
Results are always returned in the original block order.  Parallel
sub-batches use index-keyed result slots so out-of-order completion
(``as_completed``) cannot scramble them.

Body invocations are NOT pooled
-------------------------------
``execute_tools`` uses the shared pool only for **dispatch** — submitting
work and waiting for futures.  The actual tool body runs inside
``execute_tool_with_timeout``, which spawns a daemon thread per
invocation rather than a pool worker.  Rationale (P0 fix, 2026-04):
Python threads are not cancellable, so a timed-out body in a shared
pool permanently consumes a worker slot; daemon threads sidestep this.
The shared pool stays hot at a high width without leaking workers on
timeout.

Module layout
-------------
Extracted from ``runtime/loop/tool_executor.py`` in 2026-05 alongside
``tool_pool.py`` (shared executor lifecycle) and ``tool_invocation.py``
(``execute_tool`` / ``execute_tool_with_timeout``).  ``tool_executor.py``
is a thin facade re-exporting these names; existing imports keep
working unchanged.
"""

from __future__ import annotations

import collections
import concurrent.futures
import threading
import time
import traceback
from typing import MutableSequence

from ..tools.registry import ToolBatch
from ..tools.result import ToolResult
from .llm_types import ToolCallRequest
from .tool_invocation import execute_tool_with_timeout
from .tool_pool import get_tool_dispatch_executor


def execute_tools(
    blocks: list[ToolCallRequest],
    batch: ToolBatch,
    concurrent_mode: bool,
    max_workers: int,
    timeout: int,
    executor: concurrent.futures.ThreadPoolExecutor | None = None,
    partial_side_effects: "MutableSequence[str] | collections.deque[str] | None" = None,
    cancel_event: threading.Event | None = None,
) -> list[tuple[ToolCallRequest, ToolResult]]:
    """Execute tool calls with selective parallelisation.

    Parallel-safe tools run concurrently; state-mutating tools run sequentially
    as barriers between parallel batches.  Results are always in original order.

    ``max_workers`` caps how many tool bodies may execute concurrently across
    a parallel sub-batch.  A per-call ``BoundedSemaphore`` enforces this cap
    on top of the shared dispatch pool so the shared pool can stay hot (with
    a larger worker count) without violating per-loop concurrency preferences.
    Sequential paths don't acquire permits — they're serial by construction.

    All ``parallel_safe`` decisions read from the immutable ``batch`` — a
    concurrent registry mutation cannot flip a tool's flag mid-partition.

    ``partial_side_effects`` (optional) is an accumulator the caller owns —
    every mutating-tool timeout appends its name so ``AgentLoop`` can surface
    it on ``LoopResult.partial_side_effects``.  Non-mutating timeouts and
    successful calls never touch it.
    ``None`` disables the accumulator (for ad-hoc callers that don't care).

    The accumulator must accept both ``list[str]`` (legacy callers) and
    ``collections.deque[str]`` (the
    AgentLoop default — chosen for free-threaded-Python forward-compat
    where ``list.append`` is no longer atomic but ``deque.append`` is).
    Both share the ``.append(...)`` interface; that's all the timeout
    branch in ``execute_tool_with_timeout`` calls.

    ``cancel_event`` (optional) is threaded down to every tool invocation
    so cooperating tools (those declaring ``_cancel_event`` in their
    signature) can abort cleanly, AND so the outer wait loop in
    ``execute_tool_with_timeout`` can short-circuit on cancel for
    non-cooperating tools.  ``None`` disables both behaviours.
    """
    if not blocks:
        return []

    def _timed(*args, **kwargs) -> ToolResult:
        """Wrap ``execute_tool_with_timeout`` with wall-clock timing.

        Stamps ``ToolResult.duration_ms`` on the returned result so the UI can
        surface per-call timing in ``on_tool_end`` without re-measuring (a
        UI-level measurement is meaningless for parallel batches whose
        ``on_tool_end`` callbacks fire serially in submission order after the
        whole batch has completed).  Kept as a nested closure so the 4
        dispatch sites below don't each need to thread a timing block.
        """
        _t0 = time.perf_counter()
        _r = execute_tool_with_timeout(*args, **kwargs)
        try:
            _r.duration_ms = (time.perf_counter() - _t0) * 1000.0
        except AttributeError:
            # Defensive: a custom ToolResult subclass without the slot
            # should not break dispatch — timing is best-effort.
            pass
        return _r

    # Fast path: single tool or concurrency disabled
    if len(blocks) <= 1 or not concurrent_mode:
        results = []
        for block in blocks:
            result = _timed(
                block.name, block.input, batch, timeout,
                partial_side_effects=partial_side_effects,
                cancel_event=cancel_event,
            )
            results.append((block, result))
        return results

    # Check if any tool is parallel-safe
    if not any(batch.is_parallel_safe(b.name) for b in blocks):
        results = []
        for block in blocks:
            result = _timed(
                block.name, block.input, batch, timeout,
                partial_side_effects=partial_side_effects,
                cancel_event=cancel_event,
            )
            results.append((block, result))
        return results

    # Per-batch concurrency cap (honours cfg.max_tool_workers).  Only applied
    # on the parallel path — sequential calls are already serialised.
    body_permits = threading.BoundedSemaphore(max(1, max_workers))

    # Partition into contiguous groups
    results_arr: list[tuple[ToolCallRequest, ToolResult] | None] = [None] * len(blocks)
    i = 0
    while i < len(blocks):
        if batch.is_parallel_safe(blocks[i].name):
            parallel_batch = []
            while i < len(blocks) and batch.is_parallel_safe(blocks[i].name):
                parallel_batch.append((i, blocks[i]))
                i += 1

            # The module-level `tool_dispatch_executor` is now lazy.  In
            # production, `executor` is always set
            # (`AgentLoop.__init__` passes `loop._executor`), but direct
            # callers (`tool_executor.execute_tools()` with `executor=None`)
            # need us to materialise the pool here.
            # `get_tool_dispatch_executor` is idempotent and grows in
            # place, so this is cheap.
            pool = executor or get_tool_dispatch_executor(max_workers)
            futures = {
                pool.submit(
                    _timed,
                    block.name, block.input, batch, timeout,
                    body_permits=body_permits,
                    partial_side_effects=partial_side_effects,
                    cancel_event=cancel_event,
                ): (idx, block)
                for idx, block in parallel_batch
            }
            for future in concurrent.futures.as_completed(futures):
                idx, block = futures[future]
                try:
                    results_arr[idx] = (block, future.result())
                except Exception as exc:
                    error_detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    results_arr[idx] = (block, ToolResult(
                        f"Error calling tool {block.name}: {exc}\n{error_detail}",
                        is_error=True,
                    ))
        else:
            block = blocks[i]
            result = _timed(
                block.name, block.input, batch, timeout,
                partial_side_effects=partial_side_effects,
                cancel_event=cancel_event,
            )
            results_arr[i] = (block, result)
            i += 1

    # Guard: fill any slots that are still None (e.g. executor.submit() itself failed)
    return [
        r if r is not None else (blocks[i], ToolResult("Internal dispatch error", is_error=True))
        for i, r in enumerate(results_arr)
    ]


__all__ = ["execute_tools"]
