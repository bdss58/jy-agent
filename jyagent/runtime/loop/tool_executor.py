"""Tool execution stack — validated body invocation + timeout + dispatch.

Extracted from ``engine.py`` as Phase 2 of the 5-phase engine split plan
(C4 follow-up to the codex review 2026-04-25 — see
``data/memory/topics/runtime-c1-c4-deferrals.md``).

The goal of the split is to reduce ``engine.py`` (still >2100 lines) into
five owned components:

    cost.py        ← Phase 1 (landed)
    tool_executor  ← this module (Phase 2)
    llm_runner     (Phase 3)
    compaction     (Phase 4)
    LoopController (Phase 5 — what remains in engine.py)

Phase 2 owns the tool-side of the loop — everything a tool body touches
from the moment a ToolCallRequest is dequeued to the moment a ToolResult
comes back:

    * The shared dispatch pool (``tool_dispatch_executor``) and its
      lazy-grow helper (``get_tool_dispatch_executor``).  Per-run widths
      come from ``LoopConfig.max_tool_workers``; growth recreates the
      pool under a lock (A2 fix, codex review 2026-04-25).
    * ``execute_tool`` — validated single-call invocation, always
      returns a ToolResult (never raises, except KeyboardInterrupt).
    * ``execute_tool_with_timeout`` — daemon-thread wrapper that will
      never leak a pool slot on timeout (P0 fix, 2026-04).  Flags
      mutating-tool timeouts for the caller (A1 fix, codex review
      2026-04-25) and tolerates non-coercible ``timeout`` inputs (B3).
    * ``execute_tools`` — fan-out with parallel-safe / mutating
      partitioning.  Reads parallel-safe + mutating flags from the
      immutable per-step ``ToolBatch`` snapshot, so a concurrent
      registration cannot flip them mid-batch (Codex Part 1 #4/#11).

Engine.py keeps an alias import block that re-exports these under their
legacy underscore-prefixed names plus a PEP-562 ``__getattr__`` for the
pool module-state (``_tool_dispatch_executor``, ``_tool_dispatch_cap``,
``_tool_dispatch_lock``, ``_tool_executor``) — those mutate in-place
when the pool grows, so back-compat readers MUST see the live value.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import logging
import threading
import traceback
from typing import TYPE_CHECKING

from ..tools.registry import ToolBatch
from ..tools.result import ToolResult
from ..tools.validation import validate_tool_input
from .remediation import enrich_error

if TYPE_CHECKING:
    # ToolCallRequest is a dataclass defined in engine.py.  Only used
    # for type hints in ``execute_tools`` — keep the import lazy to
    # avoid a circular engine↔tool_executor load at module import.
    from .engine import ToolCallRequest


_logger = logging.getLogger(__name__)


# ─── Shared dispatch executor ────────────────────────────────────────────────
# `execute_tools()` fans out a parallel-safe batch onto this shared pool.
# Order is preserved via index-keyed result slots, so out-of-order completion
# (`as_completed`) doesn't scramble results.  Per-call concurrency is capped
# by a BoundedSemaphore sized from LoopConfig.max_tool_workers — the shared
# pool stays hot while each batch still honours its configured width.
#
# Tool *bodies* (inside `execute_tool_with_timeout`) do NOT use a pool —
# they run in daemon threads so a timed-out body holds no pool slot and dies
# with the process.  Python futures aren't cancellable, so pooling bodies
# would permanently leak workers every time a tool timed out.

tool_dispatch_executor: concurrent.futures.ThreadPoolExecutor | None = None
tool_dispatch_lock = threading.Lock()
tool_dispatch_cap = 0


def get_tool_dispatch_executor(
    min_workers: int = 8,
) -> concurrent.futures.ThreadPoolExecutor:
    """Return the shared dispatch executor, growing it if needed.

    A2 fix (codex review 2026-04-25): the eagerly-created executor was
    hard-capped at 8 workers, so ``LoopConfig.max_tool_workers > 8`` was
    silently honoured at the body-permit layer but starved at dispatch.
    This helper lazy-creates and grows the pool to the largest
    ``max_tool_workers`` ever requested across all live ``AgentLoop``
    instances in the process.

    Growth recreates the pool: the old one's ``shutdown(wait=False)`` lets
    in-flight dispatches finish in their own threads (we never block on
    them here), but no new tasks are accepted on it.  Concurrent callers
    are serialised by ``tool_dispatch_lock``.
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


# P3-2 (Codex review 2026-04-25 Part 3 #2): the pool is now LAZY.
# It used to be eagerly initialised here at module import — but that meant
# `import jyagent.runtime` (which transitively imports this module) spun up
# a background thread pool and registered an `atexit.shutdown` hook even
# for callers that never run an `AgentLoop` (CLI subcommands, unit tests
# that only touch tool registration, doc generation, etc.).
#
# All in-process callers go through `get_tool_dispatch_executor(...)`:
#   * `AgentLoop.__init__` (engine.py) — calls `_get_tool_dispatch_executor(
#     config.max_tool_workers)` so the pool is sized correctly on first use.
#   * `execute_tools(executor=...)` — called from step.py with `loop._executor`
#     pre-set, so the fallback `pool = executor or tool_dispatch_executor`
#     branch never reaches a None pool in production.
#
# Tests that previously did `from .engine import _tool_dispatch_executor` at
# module-import (snapshotting the eagerly-created pool) MUST switch to live
# attribute access (`engine._tool_dispatch_executor`) — the engine PEP-562
# `__getattr__` shim already returns the live value.


# ─── Tool invocation ─────────────────────────────────────────────────────────


def execute_tool(
    name: str,
    tool_input: dict,
    batch: ToolBatch,
) -> ToolResult:
    """Execute a single tool call with validation.  Always returns ToolResult.

    All tool resolution (function lookup, schema for validation) goes
    through the per-step ``batch`` snapshot — no live registry reads
    here, so a concurrent ``register()``/``unregister()`` cannot pair
    a function with a different schema mid-batch (Codex Part 1 #4).
    """
    fn = batch.get_function(name)
    if fn is None:
        return enrich_error(ToolResult(
            f"Error: Unknown tool '{name}'. Available: {batch.list_tools()[:20]}",
            is_error=True,
        ), name)

    tool_schema = batch.get_schema(name)
    validation_error = validate_tool_input(name, tool_input, fn, tool_schema)
    if validation_error:
        return enrich_error(ToolResult(validation_error, is_error=True), name)

    try:
        if tool_input is None:
            tool_input = {}
        raw = fn(**tool_input)
        if isinstance(raw, ToolResult):
            return enrich_error(raw, name)
        return ToolResult(str(raw))
    except KeyboardInterrupt:
        raise
    except Exception as e:
        error_detail = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return enrich_error(ToolResult(
            f"Error calling tool {name}: {e}\n{error_detail}",
            is_error=True,
        ), name)


def execute_tools(
    blocks: list["ToolCallRequest"],
    batch: ToolBatch,
    concurrent_mode: bool,
    max_workers: int,
    timeout: int,
    executor: concurrent.futures.ThreadPoolExecutor | None = None,
    partial_side_effects: list[str] | None = None,
) -> list[tuple["ToolCallRequest", ToolResult]]:
    """Execute tool calls with selective parallelisation.

    Parallel-safe tools run concurrently; state-mutating tools run sequentially
    as barriers between parallel batches.  Results are always in original order.

    ``max_workers`` caps how many tool bodies may execute concurrently across
    a parallel sub-batch.  A per-call ``BoundedSemaphore`` enforces this cap
    on top of the shared dispatch pool so the shared pool can stay hot (with
    a larger worker count) without violating per-loop concurrency preferences.
    Sequential paths don't acquire permits — they're serial by construction.

    All ``parallel_safe`` decisions read from the immutable ``batch`` — a
    concurrent registry mutation cannot flip a tool's flag mid-partition
    (Codex Part 1 #11).

    ``partial_side_effects`` (optional) is an accumulator list the caller owns
    — every mutating-tool timeout appends its name to this list so
    ``AgentLoop`` can surface it on ``LoopResult.partial_side_effects`` (A1
    fix, codex review 2026-04-25).  Non-mutating timeouts and successful
    calls never touch it.  ``None`` disables the accumulator (for ad-hoc
    callers that don't care).
    """
    if not blocks:
        return []

    # Fast path: single tool or concurrency disabled
    if len(blocks) <= 1 or not concurrent_mode:
        results = []
        for block in blocks:
            result = execute_tool_with_timeout(
                block.name, block.input, batch, timeout,
                partial_side_effects=partial_side_effects,
            )
            results.append((block, result))
        return results

    # Check if any tool is parallel-safe
    if not any(batch.is_parallel_safe(b.name) for b in blocks):
        results = []
        for block in blocks:
            result = execute_tool_with_timeout(
                block.name, block.input, batch, timeout,
                partial_side_effects=partial_side_effects,
            )
            results.append((block, result))
        return results

    # Per-batch concurrency cap (honours cfg.max_tool_workers).  Only applied
    # on the parallel path — sequential calls are already serialised.
    body_permits = threading.BoundedSemaphore(max(1, max_workers))

    # Partition into contiguous groups
    results_arr: list[tuple["ToolCallRequest", ToolResult] | None] = [None] * len(blocks)
    i = 0
    while i < len(blocks):
        if batch.is_parallel_safe(blocks[i].name):
            parallel_batch = []
            while i < len(blocks) and batch.is_parallel_safe(blocks[i].name):
                parallel_batch.append((i, blocks[i]))
                i += 1

            # P3-2 (2026-04-27): the module-level `tool_dispatch_executor`
            # is now lazy.  In production, `executor` is always set
            # (`AgentLoop.__init__` passes `loop._executor`), but direct
            # callers (`tool_executor.execute_tools()` with `executor=None`,
            # or the back-compat `engine._execute_tools()` shim) need us to
            # materialise the pool here.  `get_tool_dispatch_executor` is
            # idempotent and grows in place, so this is cheap.
            pool = executor or get_tool_dispatch_executor(max_workers)
            futures = {
                pool.submit(
                    execute_tool_with_timeout,
                    block.name, block.input, batch, timeout,
                    body_permits=body_permits,
                    partial_side_effects=partial_side_effects,
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
            result = execute_tool_with_timeout(
                block.name, block.input, batch, timeout,
                partial_side_effects=partial_side_effects,
            )
            results_arr[i] = (block, result)
            i += 1

    # Guard: fill any slots that are still None (e.g. executor.submit() itself failed)
    return [
        r if r is not None else (blocks[i], ToolResult("Internal dispatch error", is_error=True))
        for i, r in enumerate(results_arr)
    ]


def execute_tool_with_timeout(
    name: str,
    tool_input: dict,
    batch: ToolBatch,
    default_timeout: int,
    executor: concurrent.futures.ThreadPoolExecutor | None = None,
    body_permits: threading.BoundedSemaphore | None = None,
    partial_side_effects: list[str] | None = None,
) -> ToolResult:
    """Execute a tool body with a timeout.

    Uses a daemon thread per invocation rather than a shared pool.  Rationale
    (P0 fix, 2026-04): Python threads are not cancellable — `future.cancel()`
    on a running thread is a no-op, so a timed-out tool running in a shared
    pool permanently consumes a worker slot.  Under enough timeouts the pool
    starves and every subsequent tool call blocks waiting for a slot that
    never frees.

    Daemon threads sidestep this:
      * A timed-out thread keeps running but holds no pool slot.
      * Daemon status guarantees it cannot block process exit.
      * Thread creation overhead is ~0.1 ms — negligible next to any LLM call.

    The ``executor`` parameter is kept for backwards compatibility (callers
    that passed an explicit pool) but is now ignored.  Dispatch-level
    parallelism still uses ``tool_dispatch_executor``; only the inner
    timeout wrapper changed.

    ``body_permits`` (optional) caps concurrent live tool bodies to honour
    ``LoopConfig.max_tool_workers`` independent of the dispatch-pool width.
    Released as soon as ``done.wait(timeout)`` returns, so a tool that
    times out does not hold a permit while its daemon thread continues in
    the background.

    The per-step ``batch`` snapshot supplies the timeout hint, function,
    and schema — no live registry reads, so a concurrent registration
    cannot change the effective timeout mid-call.

    ``partial_side_effects`` (optional, A1 fix — codex review 2026-04-25):
    when the tool is flagged mutating and times out, its name is appended
    to this list and a WARNING is logged on the module ``_logger``.  The
    returned ToolResult also gets a stronger error message that tells the
    model the operation *may have completed in the background*, so the
    follow-up plan should verify state rather than blindly retrying.
    Non-mutating timeouts retain their historical "consider smaller steps"
    hint — a read-only tool's timeout is safe to retry verbatim.  Passing
    ``None`` disables the accumulator (the warning + error-text rewrite
    still fire).
    """
    timeout = default_timeout
    hint = batch.get_timeout_hint(name)
    if hint is not None:
        timeout = max(timeout, hint)

    # run_shell manages its own timeout — give extra slack
    if name == "run_shell":
        user_timeout = (tool_input or {}).get("timeout", 60)
        # B3 fix (codex review 2026-04-25): a non-coercible timeout
        # (e.g. the model hallucinated ``"30s"`` or a list) used to raise
        # ``TypeError``/``ValueError`` here BEFORE _execute_tool could
        # surface the error through its normal ToolResult path, crashing
        # the step with an uncaught exception.  Treat coercion failure as
        # "use the default" — _execute_tool's own schema validation will
        # still reject the bad input cleanly and the model gets an
        # actionable error ToolResult instead of a loop crash.
        try:
            timeout = max(timeout, int(user_timeout) + 10)
        except (TypeError, ValueError):
            pass

    result_holder: list[ToolResult | None] = [None]
    exc_holder: list[BaseException | None] = [None]
    done = threading.Event()

    def _run_body() -> None:
        try:
            result_holder[0] = execute_tool(name, tool_input, batch)
        except BaseException as e:  # noqa: BLE001 — we need to propagate everything
            exc_holder[0] = e
        finally:
            done.set()

    if body_permits is not None:
        body_permits.acquire()
    try:
        t = threading.Thread(
            target=_run_body,
            name=f"jyagent-tool-body:{name}",
            daemon=True,
        )
        t.start()
        timed_out = not done.wait(timeout)
    finally:
        if body_permits is not None:
            body_permits.release()

    if timed_out:
        # Timeout — the daemon thread continues running but holds no pool
        # slot, so there's nothing to leak in terms of worker capacity.
        # However, for MUTATING tools (run_shell, edit_file, write_file,
        # run_background, mcp, dispatch_agent) the *side effect* is still
        # in flight in the background thread and may partially or fully
        # complete after we've told the model "timeout, try something
        # else".  A1 fix (codex review 2026-04-25): classify the timeout,
        # log loudly, rewrite the error text so the model knows to verify
        # state, and accumulate the name for LoopResult.partial_side_effects
        # so outer layers can reconcile.  A future PR will tackle the full
        # subprocess-isolation / hard-kill story for shell-class tools.
        if batch.is_mutating(name):
            _logger.warning(
                "mutating tool '%s' timed out after %ds — "
                "side effects may have occurred and are now untracked",
                name, timeout,
            )
            if partial_side_effects is not None:
                partial_side_effects.append(name)
            return ToolResult(
                f"Error: Tool '{name}' timed out after {timeout}s. "
                f"NOTE: This is a mutating tool — the operation may have "
                f"partially or fully completed in the background. The agent "
                f"should verify state before retrying.",
                is_error=True,
            )
        # Non-mutating (read-only / queryable) timeout: safe to retry
        # verbatim, so keep the historical hint.
        return ToolResult(
            f"Error: Tool '{name}' timed out after {timeout}s. "
            f"Consider breaking the operation into smaller steps.",
            is_error=True,
        )

    if exc_holder[0] is not None:
        # KeyboardInterrupt in the worker is rare (main thread gets SIGINT)
        # but propagate anyway.  _execute_tool normally catches exceptions
        # and returns an error ToolResult, so reaching this branch implies
        # something pathological.
        if isinstance(exc_holder[0], KeyboardInterrupt):
            raise exc_holder[0]
        return enrich_error(
            ToolResult(
                f"Error: Tool '{name}' raised an uncaught exception: "
                f"{type(exc_holder[0]).__name__}: {exc_holder[0]}",
                is_error=True,
            ),
            name,
        )

    result = result_holder[0]
    if result is None:
        return ToolResult(
            f"Error: Tool '{name}' returned no result (worker finished "
            f"without producing output)",
            is_error=True,
        )
    return result


__all__ = [
    "execute_tool",
    "execute_tool_with_timeout",
    "execute_tools",
    "get_tool_dispatch_executor",
    "tool_dispatch_executor",
    "tool_dispatch_lock",
    "tool_dispatch_cap",
]
