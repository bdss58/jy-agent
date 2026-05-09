"""Tool execution stack — validated body invocation + timeout + dispatch.

Extracted from ``engine.py`` to keep the loop controller focused on
orchestration.  This module owns the tool-side of the loop — everything a
tool body touches from the moment a ToolCallRequest is dequeued to the moment
a ToolResult comes back:

    * The shared dispatch pool (``tool_dispatch_executor``) and its
      lazy-grow helper (``get_tool_dispatch_executor``).  Per-run widths
      come from ``LoopConfig.max_tool_workers``; growth recreates the
      pool under a lock.
    * ``execute_tool`` — validated single-call invocation, always
      returns a ToolResult (never raises, except KeyboardInterrupt).
    * ``execute_tool_with_timeout`` — daemon-thread wrapper that will
      never leak a pool slot on timeout (P0 fix, 2026-04).  Flags
      mutating-tool timeouts for the caller and tolerates non-coercible
      ``timeout`` inputs.
    * ``execute_tools`` — fan-out with parallel-safe / mutating
      partitioning.  Reads parallel-safe + mutating flags from the
      immutable per-step ``ToolBatch`` snapshot, so a concurrent
      registration cannot flip them mid-batch.

Pool module-state (``tool_dispatch_executor``, ``_tool_dispatch_cap``,
``_tool_dispatch_lock``) lives at module scope here and mutates in-place
when ``get_tool_dispatch_executor`` grows the pool — read it via
``get_tool_dispatch_executor()`` rather than snapshotting at import time
so callers always see the live value.
"""

from __future__ import annotations

import atexit
import collections
import concurrent.futures
import copy
import functools
import inspect
import logging
import threading
import time
import traceback
from typing import MutableSequence

from ..tools.registry import ToolBatch
from ..tools.result import ToolResult
from ..tools.validation import validate_tool_input
from .llm_types import ToolCallRequest
from .remediation import enrich_error


_logger = logging.getLogger(__name__)


# ─── Cooperative cancellation ────────────────────────────────────────────────
# Tools opt into cooperative cancellation by declaring a ``_cancel_event``
# keyword parameter (``threading.Event | None``).  At dispatch time the
# runtime introspects the tool's signature once and, if the parameter
# exists, threads the active ``AgentLoop._cancel_event`` into the call.
#
# This is purely additive: tools without the parameter are called
# unchanged (back-compat).  Tools that opt in should poll
# ``_cancel_event.is_set()`` in their inner loops (subprocess polls,
# HTTP retries, large-file scans) and abort with a ToolResult or by
# raising ``KeyboardInterrupt``.  The outer ``done.wait()`` loop in
# ``execute_tool_with_timeout`` also short-circuits on ``cancel_event``
# so non-cooperating tools see Ctrl-C latency bounded to the polling
# interval — the kwarg is for tools that want clean teardown of in-
# flight side effects (open subprocesses, half-uploaded files, …).
#
# The introspection cache is keyed on the function object itself;
# lambdas and bound methods are handled transparently because
# ``inspect.signature`` accepts both.  Callables whose signature cannot
# be resolved (C extensions, ``builtin_function_or_method``, exotic
# ``__call__`` shapes) return False and are called the legacy way.


@functools.lru_cache(maxsize=512)
def _accepts_cancel_event(fn: object) -> bool:
    """Return True iff ``fn`` declares a ``_cancel_event`` parameter."""
    try:
        sig = inspect.signature(fn)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return "_cancel_event" in sig.parameters


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

    The eagerly-created executor was hard-capped at 8 workers, so
    ``LoopConfig.max_tool_workers > 8`` was silently honoured at the
    body-permit layer but starved at dispatch.  This helper lazy-creates and
    grows the pool to the largest
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


# The pool is now LAZY.
# It used to be eagerly initialised here at module import — but that meant
# `import jyagent.runtime` (which transitively imports this module) spun up
# a background thread pool and registered an `atexit.shutdown` hook even
# for callers that never run an `AgentLoop` (CLI subcommands, unit tests
# that only touch tool registration, doc generation, etc.).
#
# All in-process callers go through `get_tool_dispatch_executor(...)`:
#   * `AgentLoop.__init__` (engine.py) — calls `get_tool_dispatch_executor(
#     config.max_tool_workers)` to warm the pool to the configured size.
#     The return value is intentionally discarded; readers go through
#     ``AgentLoop._executor`` (a property) which re-resolves on every
#     access so a later grow-and-replace cannot leave the loop holding
#     a shut-down pool.
#   * `execute_tools(executor=...)` — called from step.py with
#     `loop._executor` (the property), which always returns the live
#     pool.  The `pool = executor or tool_dispatch_executor` fallback
#     branch never reaches a None pool in production.
#
# Tests that previously did `from .engine import _tool_dispatch_executor` at
# module-import (snapshotting the eagerly-created pool) MUST switch to
# `get_tool_dispatch_executor()` here so they see the live (possibly
# resized) pool rather than a stale captured reference.


# ─── Tool invocation ─────────────────────────────────────────────────────────


def execute_tool(
    name: str,
    tool_input: dict,
    batch: ToolBatch,
    cancel_event: threading.Event | None = None,
) -> ToolResult:
    """Execute a single tool call with validation.  Always returns ToolResult.

    All tool resolution (function lookup, schema for validation) goes
    through the per-step ``batch`` snapshot — no live registry reads
    here, so a concurrent ``register()``/``unregister()`` cannot pair
    a function with a different schema mid-batch.

    ``cancel_event`` (optional): when the tool body declares a
    ``_cancel_event`` keyword parameter, this event is threaded in so
    the tool can cooperatively abort.  Tools that don't opt in are
    called unchanged (back-compat).
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
        # Deep-copy before invocation so a tool that mutates a nested
        # structure it receives (e.g. ``paths.append(...)``,
        # ``data["new_key"] = ...``) cannot corrupt the original
        # ``ToolCallRequest.input`` — that dict is shared by reference
        # with the persisted transcript, and a mutation here would
        # silently drift the recorded call from the call the model
        # actually issued.  Latent-bug fix 2026-05.
        #
        # The previous shallow ``dict(tool_input)`` only protected the
        # top-level keys (and only on the cancel-injection branch) —
        # nested mutable values (list, dict, set) still aliased the
        # transcript.  ``deepcopy`` closes both gaps.
        #
        # Cost: O(n) in input size.  Strings and primitives are
        # refcount-only (O(1)); lists/dicts are O(n).  Typical tool
        # inputs are <1 KB, so this is sub-millisecond in the common
        # case.  Deliberately unconditional — gating on "does this
        # tool mutate its inputs" would require a per-tool flag we
        # can't reliably populate for external / MCP tools.
        safe_input = copy.deepcopy(tool_input)
        # Inject the cancel event only when the tool has opted in by
        # declaring ``_cancel_event`` in its signature.  We pass the
        # engine's event even when it's None (not set yet) so tools can
        # treat it uniformly — polling ``event.is_set()`` on a never-set
        # event is always False, which is the right semantic.
        #
        # Passed as an explicit kwarg (not merged into ``safe_input``)
        # so it never appears as a regular parameter to the tool —
        # threading.Event is not JSON-serialisable, and a stray
        # ``_cancel_event`` key in a kwargs-spread would corrupt any
        # downstream serialisation of the call arguments.
        if cancel_event is not None and _accepts_cancel_event(fn):
            raw = fn(_cancel_event=cancel_event, **safe_input)
        else:
            raw = fn(**safe_input)
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


def execute_tool_with_timeout(
    name: str,
    tool_input: dict,
    batch: ToolBatch,
    default_timeout: int,
    body_permits: threading.BoundedSemaphore | None = None,
    partial_side_effects: "MutableSequence[str] | collections.deque[str] | None" = None,
    cancel_event: threading.Event | None = None,
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

    ``body_permits`` (optional) caps concurrent live tool bodies to honour
    ``LoopConfig.max_tool_workers`` independent of the dispatch-pool width.
    Released as soon as the wait loop returns, so a tool that times out
    does not hold a permit while its daemon thread continues in the
    background.

    The per-step ``batch`` snapshot supplies the timeout hint, function,
    and schema — no live registry reads, so a concurrent registration
    cannot change the effective timeout mid-call.

    ``partial_side_effects`` (optional):
    when the tool is flagged mutating and times out, its name is appended
    to this list and a WARNING is logged on the module ``_logger``.  The
    returned ToolResult also gets a stronger error message that tells the
    model the operation *may have completed in the background*, so the
    follow-up plan should verify state rather than blindly retrying.
    Non-mutating timeouts retain their historical "consider smaller steps"
    hint — a read-only tool's timeout is safe to retry verbatim.  Passing
    ``None`` disables the accumulator (the warning + error-text rewrite
    still fire).

    ``cancel_event`` (optional): a ``threading.Event`` the engine sets on
    Ctrl-C / programmatic cancel.  Two effects:
      * Threaded down to ``execute_tool`` which injects it into the tool
        body's call kwargs *iff* the tool declares ``_cancel_event`` in
        its signature (cooperative-cancel opt-in).
      * The outer wait loop polls it so a cancelled run returns promptly
        even when the tool body doesn't cooperate — it returns a
        cancellation-flagged ToolResult and lets the daemon thread continue
        in the background like any other timeout.
    """
    timeout = default_timeout
    hint = batch.get_timeout_hint(name)
    if hint is not None:
        timeout = max(timeout, hint)

    # run_shell manages its own timeout — give extra slack
    if name == "run_shell":
        user_timeout = (tool_input or {}).get("timeout", 60)
        # A non-coercible timeout
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
            result_holder[0] = execute_tool(
                name, tool_input, batch, cancel_event=cancel_event,
            )
        except BaseException as e:  # noqa: BLE001 — we need to propagate everything
            exc_holder[0] = e
        finally:
            done.set()

    # Polling interval for the cancel-event short-circuit.  Small enough
    # that Ctrl-C feels responsive (≤500ms perceived latency); large
    # enough that a tool that finishes quickly doesn't pay an extra
    # syscall round-trip per body invocation.
    _CANCEL_POLL_INTERVAL = 0.5

    cancelled = False
    if body_permits is not None:
        body_permits.acquire()
    try:
        t = threading.Thread(
            target=_run_body,
            name=f"jyagent-tool-body:{name}",
            daemon=True,
        )
        t.start()
        if cancel_event is None:
            timed_out = not done.wait(timeout)
        else:
            # Polling loop: wait for either the body to finish, the
            # cancel event to fire, or the timeout to elapse.  We
            # short-circuit on cancellation so a Ctrl-C during a slow
            # non-cooperating tool returns within ``_CANCEL_POLL_INTERVAL``
            # rather than after the full ``timeout`` budget.
            import time as _time  # local — keeps module-level imports tidy
            deadline = _time.monotonic() + timeout
            while True:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    timed_out = not done.is_set()
                    break
                if done.wait(min(_CANCEL_POLL_INTERVAL, remaining)):
                    timed_out = False
                    break
                if cancel_event.is_set():
                    cancelled = True
                    timed_out = False
                    break
    finally:
        # The permit is released as soon as the wait loop returns, which
        # on timeout is BEFORE the daemon body thread has actually
        # finished.  This is intentional — alternative #1 (hold the
        # permit until the daemon completes) would leak a slot
        # permanently for any infinite-loop tool body, draining
        # ``LoopConfig.max_tool_workers`` after enough timeouts and
        # eventually starving the dispatch loop.  Alternative #2 (kill
        # the daemon) is impossible — Python threads are not cancellable.
        # The trade-off we accept here:
        #   - A leaked-thread-from-timeout no longer counts against
        #     ``max_tool_workers`` (good — the dispatch loop stays healthy).
        #   - The leaked daemon's CPU / memory keeps running in the
        #     background until it finishes naturally or the process exits
        #     (acceptable — daemon=True guarantees no clean-up will block).
        # If a future contributor "fixes" this by holding the permit longer,
        # the LoopConfig.max_tool_workers semantic becomes "max EVER-LIVE
        # tool bodies" instead of "max CURRENTLY-RUNNING bodies that the
        # dispatch loop is aware of", which is the wrong invariant.
        # Document loudly here so the trade-off is visible at the call site.
        if body_permits is not None:
            body_permits.release()

    if cancelled:
        # Cooperative cancel — the daemon body may still be running.
        # Tools that opted into ``_cancel_event`` should be teardown-
        # complete by the time they observe the event; tools that did
        # not opt in keep running and will leak like any other timed-out
        # body.  Mutating tools get the same partial-side-effects hint
        # so outer layers can reconcile state.
        if batch.is_mutating(name):
            if partial_side_effects is not None:
                partial_side_effects.append(name)
            return ToolResult(
                f"Cancelled: Tool '{name}' aborted on cancel signal. "
                f"NOTE: This is a mutating tool — the operation may have "
                f"partially or fully completed in the background. The agent "
                f"should verify state before retrying.",
                is_error=True,
            )
        return ToolResult(
            f"Cancelled: Tool '{name}' aborted on cancel signal.",
            is_error=True,
        )

    if timed_out:
        # Timeout — the daemon thread continues running but holds no pool
        # slot, so there's nothing to leak in terms of worker capacity.
        # However, for MUTATING tools (run_shell, edit_file, write_file,
        # run_background, mcp, dispatch_agent) the *side effect* is still
        # in flight in the background thread and may partially or fully
        # complete after we've told the model "timeout, try something
        # else".  Classify the timeout,
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
