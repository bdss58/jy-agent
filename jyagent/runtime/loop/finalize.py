"""Terminal-path helpers for the agent loop.

Extracted from ``engine.py`` so that ``step.py`` (and any future
sub-module that needs to terminate a run) can import these without
reaching back into ``engine.py``.  Before this split, ``step.py``
imported helpers from ``.engine`` — that closed a circular conceptual
dependency (engine owns step, step reaches back into engine for
terminal helpers).

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

These used to carry ``_``-prefixed names in ``engine.py`` while they
were module-private.  Both the extraction and the subsequent rename
landed in one branch — no alias bridge, no dual-name surface.  Engine
and step both import the public names directly.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from .config import LoopResult
from .llm_types import ToolCallRequest

if TYPE_CHECKING:
    from ...llm.types import Message

_logger = logging.getLogger(__name__)


def is_truncated(stop_reason: str, tool_calls: list[ToolCallRequest]) -> bool:
    """Detect if a response was truncated while emitting tool calls."""
    return stop_reason == "length" and bool(tool_calls)


# ── Prose-shaped tool call detector (Bug A — see fix plan 2026-05) ──────────
#
# The agent loop only continues when the assistant response carries a
# structured tool_use block (or stop_reason == "tool_use").  When the model
# instead WRITES what looks like a tool call into its assistant text — e.g.
# ``[Tool call: run_shell]{"cmd": "ls"}`` or a fenced ```tool_use``` block —
# there are no tool_call_blocks and the loop terminates the turn cleanly.
# From the user's perspective the agent "suddenly stopped mid-task".
#
# This predicate is a high-precision detector for that specific class of
# malformed-attempt prose.  The patterns are deliberately narrow:
#
#   1. ``[Tool call: NAME]`` or ``[Tool: NAME]`` at the start of a line.
#      This is the literal Claude-Code transcript-render syntax that the
#      model occasionally falls back to when it's narrating a tool use
#      instead of invoking one.
#   2. A fenced code block whose info-string is ``tool_use`` / ``tool_call``.
#      Same intent, different rendering.
#
# Anchoring matters.  Discussing tool calls in prose ("I'll run the
# `run_shell` tool to ...") must NOT trigger; only line-leading invocation-
# shaped strings do.  A false positive here means re-prompting a turn that
# was actually fine — annoying but not silently broken.  A false negative
# means the original Bug A reappears — which is what we already had.
#
# Returning True does NOT terminate or alter state; the caller decides the
# remediation (currently: inject a corrective user message and continue the
# loop).  Keeping this as a pure predicate keeps it trivially testable.
_PROSE_TOOL_CALL_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Line-leading "[Tool call: NAME]" / "[Tool: NAME]" / "[tool_use: NAME]".
    # Tolerates surrounding brackets, optional space, common case variants.
    re.compile(r"(?im)^\s*\[\s*tool(?:[ _-]?call|[ _-]?use)?\s*:\s*[A-Za-z_][\w-]*\s*\]"),
    # Fenced code block tagged tool_use / tool_call (info string after ```).
    re.compile(r"(?im)^\s*```\s*tool[_-]?(?:use|call)\b"),
)


def looks_like_prose_tool_call(text: str) -> bool:
    """Detect when assistant prose appears to be a malformed tool invocation.

    Returns True iff ``text`` contains a high-confidence pattern that the
    model TRIED to emit a tool call but did so as prose instead of a real
    structured tool_use block.  Used by ``run_step`` to decide whether to
    inject a corrective re-prompt rather than terminating the turn.

    Conservative by design — see module-level comment for the patterns
    matched and the false-positive / false-negative trade-off.  Any future
    pattern additions belong in ``_PROSE_TOOL_CALL_PATTERNS``.
    """
    if not text:
        return False
    return any(p.search(text) for p in _PROSE_TOOL_CALL_PATTERNS)


# Sentinel marker (parallel to ``_VERIFICATION_MARKER`` in verification.py)
# that prefixes the corrective user message injected when a prose tool call
# is detected.  Used to (a) detect repeats / break out if the model keeps
# doing it and (b) strip a dangling unanswered correction at terminal
# cleanup time.  Public so ``step.py`` can compose the message and the
# strip helper can recognize it.
PROSE_TOOL_CALL_MARKER = "[MALFORMED_TOOL_CALL]"


def build_prose_tool_call_correction() -> str:
    """Build the corrective user message injected when Bug A fires.

    The message tells the model exactly what went wrong and what to do
    next.  Kept terse — verbose prompts here just consume tokens that the
    next attempt needs.  Returns a string starting with
    ``PROSE_TOOL_CALL_MARKER`` so the gate can detect repeated firings.
    """
    return (
        f"{PROSE_TOOL_CALL_MARKER} Your previous message contained text that "
        "looks like a tool call (e.g. `[Tool call: ...]` or a ```tool_use``` "
        "block) but no actual structured tool invocation was emitted, so the "
        "tool did NOT run.  If you intended to call a tool, invoke it via "
        "the real function-calling channel now.  If you intended plain prose, "
        "continue without the pseudo-syntax."
    )


def strip_dangling_verification(messages: "list[Message]") -> None:
    """Remove a trailing unanswered ``[VERIFICATION]`` user message in-place.

    The verification gate appends a user prompt asking the model to self-
    check before returning.  If the loop exits before the model replies
    (max_steps, KeyboardInterrupt, uncaught exception), that unanswered user
    message would leak into the persisted session and poison the next turn.

    Also strips a trailing unanswered ``[MALFORMED_TOOL_CALL]`` user
    message — same failure mode (injected by the prose-tool-call gate and
    never answered because the run terminated before the next step),
    same fix (drop it from the persisted tail).

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
    if isinstance(tail_content, str) and (
        tail_content.startswith("[VERIFICATION]")
        or tail_content.startswith(PROSE_TOOL_CALL_MARKER)
    ):
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
    "build_prose_tool_call_correction",
    "finalize_run",
    "is_truncated",
    "looks_like_prose_tool_call",
    "PROSE_TOOL_CALL_MARKER",
    "strip_dangling_verification",
]
