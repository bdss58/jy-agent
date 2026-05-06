"""Terminal-path helpers for the agent loop.

Extracted from ``engine.py`` so that ``step.py`` (and any future
sub-module that needs to terminate a run) can import these without
reaching back into ``engine.py``.  Before this split, ``step.py``
imported ``_finalize_run`` and ``_is_truncated`` from ``.engine`` —
that closed a circular conceptual dependency (engine owns step, step
reaches back into engine for terminal helpers).

Three helpers live here:

  * ``is_truncated`` — pure predicate over ``(stop_reason, tool_calls)``;
    classifies a provider response as a mid-tool-call truncation that
    should trigger token-budget rescaling and a retry.

  * ``strip_dangling_verification`` — idempotent in-place cleaner for
    the trailing ``[VERIFICATION]`` user message injected by
    ``verification.py``.  Run on every terminal path so an early exit
    does not leak an unanswered self-check prompt into persisted
    sessions.

  * ``finalize_run`` — the canonical ``_run_impl`` exit path.  Funnels
    every terminal ``LoopResult`` construction through one place:
    strips dangling verification, finishes + flushes the trace span
    (best-effort), and returns a populated ``LoopResult``.

These were ``_``-prefixed in ``engine.py`` because they were treated as
module-private.  Now that they are shared across two modules, the
public names drop the underscore — but the engine and step modules
both alias the old names for back-compat so test imports keep working
unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .config import LoopResult
from .llm_types import ToolCallRequest

if TYPE_CHECKING:
    from ...llm.types import Message

_logger = logging.getLogger(__name__)


def is_truncated(stop_reason: str, tool_calls: list[ToolCallRequest]) -> bool:
    """Detect if a response was truncated while emitting tool calls."""
    return stop_reason == "length" and bool(tool_calls)


def strip_dangling_verification(messages: "list[Message]") -> None:
    """Remove a trailing unanswered ``[VERIFICATION]`` user message in-place.

    The verification gate appends a user prompt asking the model to self-
    check before returning.  If the loop exits before the model replies
    (max_steps, KeyboardInterrupt, uncaught exception), that unanswered user
    message would leak into the persisted session and poison the next turn.

    Idempotent: safe to call on every terminal path regardless of whether a
    verification was actually injected.  This is why the canonical exit
    helper ``finalize_run`` calls it unconditionally — gating on a
    ``verification_injected`` flag is a micro-optimization that historically
    led to bugs (cleanup forgotten on new exit paths).
    """
    if not messages:
        return
    tail = messages[-1]
    if not isinstance(tail, dict):
        return
    if tail.get("role") != "user":
        return
    tail_content = tail.get("content", "")
    if isinstance(tail_content, str) and tail_content.startswith("[VERIFICATION]"):
        messages.pop()


def finalize_run(
    *,
    status: str,
    text: str,
    final_text: str,
    messages: "list[Message]",
    steps: int,
    total_input_tokens: int,
    total_output_tokens: int,
    tool_calls_count: int,
    total_cache_creation_tokens: int = 0,
    total_cache_read_tokens: int = 0,
    api_calls: int = 0,
    error: str | None = None,
    trace: Any = None,
    trace_status: str | None = None,
    trace_total_steps: int | None = None,
    trace_total_cost_usd: float | None = None,
) -> LoopResult:
    """Centralized exit path for ``_run_impl``.

    Every ``return LoopResult(...)`` in the loop must funnel through here so
    that:

      1. Dangling ``[VERIFICATION]`` user messages are *always* stripped
         (idempotent — see ``strip_dangling_verification``).  Historically
         this was open-coded at every exit, and three exit paths
         (``cost_limit``, repeated truncation, cooperative cancellation)
         were missed, leaking unanswered prompts into persisted sessions.

      2. Trace finish + flush happens uniformly, eliminating exit paths
         that emitted a ``LoopResult`` but never closed the trace span.

    The ``trace_*`` overrides exist for cases where the trace status string
    or step count differs from the ``LoopResult`` (currently only
    ``max_steps`` uses ``trace_total_steps=cfg.max_steps`` while reporting
    ``steps=cfg.max_steps`` — both happen to match, but the override keeps
    the seam explicit for future use).

    Keyword-only by design: every field is named at the call site so that
    a careless ``LoopResult(*args)`` style cannot regress the contract.
    """
    strip_dangling_verification(messages)
    if trace is not None:
        finish_kwargs: dict = {
            "status": trace_status or status,
            "total_steps": trace_total_steps if trace_total_steps is not None else steps,
        }
        if trace_total_cost_usd is not None:
            finish_kwargs["total_cost_usd"] = trace_total_cost_usd
        # Tracing must never fail-close a successful run.  Disk-full /
        # read-only fs / permission errors here used to bubble up and
        # discard the entire LoopResult.  Log + swallow so observability
        # stays non-fatal.
        try:
            trace.finish(**finish_kwargs)
            trace.flush()
        except Exception as trace_err:  # noqa: BLE001 — observability is best-effort
            _logger.warning(
                "trace finalize failed (non-fatal): %s: %s",
                type(trace_err).__name__,
                trace_err,
            )
    return LoopResult(
        status=status,
        text=text,
        final_text=final_text,
        messages=messages,
        steps=steps,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        tool_calls_count=tool_calls_count,
        total_cache_creation_tokens=total_cache_creation_tokens,
        total_cache_read_tokens=total_cache_read_tokens,
        api_calls=api_calls,
        error=error,
    )


__all__ = [
    "finalize_run",
    "is_truncated",
    "strip_dangling_verification",
]
