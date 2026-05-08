"""Single-tool invocation runtime — validation, cancel propagation, timeout.

Owns everything between "a ToolCallRequest is dequeued" and "a ToolResult
comes back" for ONE tool body.  Two layered entry points:

* :func:`execute_tool` — synchronous, validated single-call invocation.
  Reads function + schema from the immutable per-step ``ToolBatch``
  snapshot (no live registry), so a concurrent ``register()``/
  ``unregister()`` cannot pair a function with a different schema
  mid-batch.  Always returns a ``ToolResult``; the only exception
  propagated is ``KeyboardInterrupt``.

* :func:`execute_tool_with_timeout` — daemon-thread wrapper that
  enforces a wall-clock deadline.  Daemon (not pool worker) because
  Python threads are not cancellable: a ``future.cancel()`` on a
  running thread is a no-op, and a timed-out body in a shared pool
  permanently consumes a worker slot.  Daemon threads sidestep this —
  a timed-out body keeps running but holds no slot, and ``daemon=True``
  guarantees it cannot block process exit.

Cooperative cancel
------------------
Tools opt in to cooperative cancellation by declaring a
``_cancel_event`` keyword parameter (``threading.Event | None``).
At dispatch time the runtime introspects the tool's signature once
(via :func:`_accepts_cancel_event`, ``lru_cache``-d) and threads the
active engine cancel event into the call iff the parameter exists.
Tools without it are called unchanged (back-compat).  The
``execute_tool_with_timeout`` outer wait loop also short-circuits on
the cancel event so non-cooperating tools see Ctrl-C latency bounded
to ``_CANCEL_POLL_INTERVAL`` rather than the full timeout budget.

Mutating-tool timeouts
----------------------
For tools flagged ``mutating`` in the ``ToolBatch`` (run_shell,
edit_file, write_file, run_background, mcp, dispatch_agent), a timeout
or cancel does NOT mean the side effect is rolled back — the daemon
body keeps running in the background and may complete the operation
asynchronously.  This module surfaces that hazard via two channels:

  1. The returned ``ToolResult`` text rewrites the error to "the
     operation may have partially or fully completed in the
     background; the agent should verify state before retrying".
  2. The ``partial_side_effects`` accumulator (passed by the caller)
     records the tool name so ``AgentLoop.run()`` can snapshot it
     onto ``LoopResult.partial_side_effects`` for outer-layer
     reconciliation.

Module layout
-------------
Extracted from ``runtime/loop/tool_executor.py`` in 2026-05 alongside
``tool_pool.py`` (shared executor lifecycle) and ``tool_dispatch.py``
(``execute_tools`` fan-out).  ``tool_executor.py`` is a thin facade
re-exporting these names; existing imports keep working unchanged.
"""

from __future__ import annotations

import collections
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
from .remediation import enrich_error


_logger = logging.getLogger(__name__)


# ── Cooperative-cancel signature introspection ───────────────────────────


@functools.lru_cache(maxsize=512)
def _accepts_cancel_event(fn: object) -> bool:
    """Return True iff ``fn`` declares a ``_cancel_event`` parameter.

    The cache is keyed on the function object itself; lambdas and
    bound methods are handled transparently because
    ``inspect.signature`` accepts both.  Callables whose signature
    cannot be resolved (C extensions, ``builtin_function_or_method``,
    exotic ``__call__`` shapes) return False and are called the
    legacy way (no kwarg injection).
    """
    try:
        sig = inspect.signature(fn)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return "_cancel_event" in sig.parameters


# ── Tool invocation: validated single call ───────────────────────────────


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
        # Inject the cancel event only when the tool has opted in by
        # declaring ``_cancel_event`` in its signature.  We pass the
        # engine's event even when it's None (not set yet) so tools can
        # treat it uniformly — polling ``event.is_set()`` on a never-set
        # event is always False, which is the right semantic.
        if cancel_event is not None and _accepts_cancel_event(fn):
            # Don't mutate the caller's dict — the ToolCallRequest.input
            # is shared back into the persisted transcript and the
            # runtime-injected kwarg must not leak there.
            call_kwargs = dict(tool_input)
            call_kwargs["_cancel_event"] = cancel_event
            raw = fn(**call_kwargs)
        else:
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


# ── Tool invocation: timeout + cancel-aware wait ─────────────────────────


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
    "_accepts_cancel_event",
    "execute_tool",
    "execute_tool_with_timeout",
]
