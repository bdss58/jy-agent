"""Per-step tool-round semantics.

Owns the helpers that handle TOOL EXECUTION and stuck-loop detection:

  * ``_execute_tool_round`` — dispatch + denial-result synthesis + cancel
  * ``_check_stuck_loop``   — repeat-call detection feeding off the round's
                              denial results

These two MUST stay together: the denial-result synthesis in
``_execute_tool_round`` intentionally feeds the stuck-loop signal that
``_check_stuck_loop`` reads.

Extracted from ``runtime/loop/step.py`` (2026-05-06).  Re-exported from
``step.py`` so existing ``from jyagent.runtime.loop.step import
_execute_tool_round`` callers (notably tests using
``inspect.getsource``) keep working unchanged.

Layering: imports from ``step_state`` only (no other helper module).
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..tools.registry import ToolBatch
from .step_state import RunState, StepBreak, StepOutcome, StepTerminate

if TYPE_CHECKING:
    from .run_context import RunContext


def _execute_tool_round(
    loop: "RunContext",
    state: RunState,
    step_batch: ToolBatch,
    tool_call_blocks: list,
) -> tuple["StepOutcome | None", list]:
    """Fire ``on_tool_start`` for every call, check the before-tools cancel
    gate, dispatch the batch, fire ``on_tool_end``, append ``tool_result``
    messages, and trace each call.

    Returns ``(outcome, tool_results_tuples)``:
      - If cooperative cancellation fires BEFORE dispatch, returns
        ``(StepBreak("cancelled"), [])`` after firing a matching
        ``on_tool_end`` + appending a ``Cancelled`` tool_result for every
        already-fired ``on_tool_start`` (so UI spinner counts stay
        balanced).
      - Otherwise returns ``(None, tool_results_tuples)`` — the caller
        runs stuck-loop detection on the tuples and continues.
    """
    from .compaction import truncate_result as _truncate_result
    from .tool_executor import execute_tools as _execute_tools
    from ..tools.result import ToolResult

    cfg = loop._config
    messages = state.messages
    step = state.step
    trace = state.trace

    # Fire on_tool_batch for multi-tool batches.
    if len(tool_call_blocks) > 1:
        loop._fire("on_tool_batch", len(tool_call_blocks))

    # Fire on_tool_start for all tool calls BEFORE execution.
    for block in tool_call_blocks:
        loop._fire("on_tool_start", block.name, block.input)

    # Approval gate — fires ``on_tool_pre_execute`` per block; callbacks
    # returning ``"deny"`` cause the engine to synthesize a denial
    # ToolResult *in place of* the real call, keeping tool_start / tool_end
    # pairs balanced for UIs that count them.  We split the blocks into
    # two lists here so the executor only sees the allowed ones, then emit
    # results in the ORIGINAL ``tool_call_blocks`` order (Bug-fix 2026-05:
    # the previous denied-first-then-allowed ordering broke positional
    # consumers and made ``/history`` unreadable).
    denied_blocks: list = []
    allowed_blocks: list = []
    for block in tool_call_blocks:
        decision = loop._fire_with_return(
            "on_tool_pre_execute", block.name, block.input, default=None,
        )
        if decision == "deny":
            denied_blocks.append(block)
        else:
            allowed_blocks.append(block)

    # Cooperative cancellation check (before tools).  Fire on_tool_end for
    # each pending call so callbacks see the matching close event for the
    # on_tool_start fired above (without this, UIs that count starts vs.
    # ends — spinners, progress bars — leak resources on cancel).
    if loop._is_cancelled():
        for block in tool_call_blocks:
            messages.append({
                "role": "tool_result",
                "tool_call_id": block.id,
                "tool_name": block.name,
                "content": "Cancelled",
                "is_error": True,
            })
            loop._fire("on_tool_end", block.name, "Cancelled", True, None)
        return StepBreak(reason="cancelled"), []

    # Phase 1: dispatch the allowed blocks (if any).
    executed_results: list = []
    if allowed_blocks:
        tools_t0 = time.perf_counter()
        executed_results = _execute_tools(
            allowed_blocks,
            step_batch,
            cfg.concurrent_tools,
            cfg.max_tool_workers,
            cfg.tool_timeout,
            executor=loop._executor,
            partial_side_effects=loop._partial_side_effects,
            cancel_event=loop._cancel_event,
        )
        # Batch duration is measured but not recorded — per-call durations are
        # traced at dispatch time inside execute_tool_with_timeout, and keeping
        # the clock read here preserves the option to add a batch-level span
        # later without reshaping the call site. Intentional no-op.
        _ = (time.perf_counter() - tools_t0) * 1000

    # Phase 2: synthesize denial results so stuck-loop detection sees them.
    # Bug-fix 2026-05 (codex review): without this, repeatedly-denied tool
    # calls bypassed _check_stuck_loop entirely (it iterates over the
    # returned tuples), so a model that kept retrying the same denied call
    # could loop forever.  By including denial results — which all share
    # the same fixed content — the dedup detector trips on the second or
    # third repeat just like a normal stuck loop.
    #
    # Latent-bug fix 2026-05: key the denial set + result map by Python
    # object identity (``id(block)``) rather than by the provider-assigned
    # ``block.id`` string.  A malformed provider that emits two tool_call
    # blocks with the SAME ``id`` used to silently collapse them into a
    # single entry here, so Phase 3 would look up one result for two
    # distinct blocks — stuck-loop detection saw only one call, the
    # transcript emitted only one tool_result, and positional pairing
    # broke downstream (``/history`` rendering, subagent arg reconstruction).
    # Python object ids are unique per ToolCallRequest by construction, so
    # the dict keys cannot collide even if ``block.id`` does.
    denial_msg = "Denied by user (on_tool_pre_execute gate)."
    denied_oids = {id(b) for b in denied_blocks}
    results_by_oid = {id(b): r for (b, r) in executed_results}

    # Phase 3: emit messages + fire on_tool_end + trace, IN ORIGINAL
    # tool_call_blocks ORDER.  Returns a ``tool_results_tuples`` list that
    # mirrors the input order so downstream consumers (stuck-loop detection,
    # /history rendering, positional pairing) see what they expect.
    tool_results_tuples: list = []
    for block in tool_call_blocks:
        if id(block) in denied_oids:
            result = ToolResult(content=denial_msg, is_error=True)
            # Stamp duration_ms=0 for parity with executed tools (the
            # ``on_tool_end`` callback and trace pipeline both read this).
            try:
                result.duration_ms = 0.0  # type: ignore[attr-defined]
            except Exception:
                pass
        else:
            result = results_by_oid[id(block)]

        state.tool_calls_count += 1
        content_str = _truncate_result(
            result.content, cfg.max_tool_result_chars, result.is_error,
        )
        # ``duration_ms`` is stamped by ``tool_executor._timed`` at each
        # dispatch site (sequential + parallel).  It will be ``None`` only if
        # a custom ToolResult subclass rejected the attribute write or the
        # caller bypassed ``execute_tools`` entirely (e.g. the cancel-shim
        # path that synthesises a Cancelled ToolResult above).  The UI
        # callback tolerates ``None``.
        duration_ms = getattr(result, "duration_ms", None)
        loop._fire(
            "on_tool_end", block.name, content_str, result.is_error, duration_ms,
        )

        if trace:
            trace.add_span(
                step=step,
                event_type="tool_call",
                tool_name=block.name,
                tool_args=block.input,
                success=not result.is_error,
                error=content_str[:200] if result.is_error else None,
            )

        messages.append({
            "role": "tool_result",
            "tool_call_id": block.id,
            "tool_name": block.name,
            "content": content_str,
            "is_error": result.is_error,
        })
        tool_results_tuples.append((block, result))

    return None, tool_results_tuples
def _check_stuck_loop(
    loop: "RunContext",
    state: RunState,
    tool_results_tuples: list,
) -> "StepOutcome | None":
    """Response-aware stuck-loop detection.

    Run AFTER execution so we can compare responses: a tool is only
    "stuck" when the same (tool, args) returns the same response
    repeatedly.  Polling tools (e.g. check_background) naturally return
    different responses (elapsed_seconds etc.) and are never flagged.

    Two correctness rules (P0 fixes, 2026-04):
      1. Hash the RAW tool output, not the UI-truncated string.  Two
         different long outputs that happen to share a common prefix up
         to ``max_tool_result_chars`` would collide on the truncated
         string and look "stuck".
      2. Deduplicate (name, args) keys *within a single batch*.  A
         legitimate parallel fanout of e.g. 3 identical ``read_file``
         calls in one step is not a stuck loop — it's the model doing
         simultaneous reads.  Without this, such a batch alone can hit
         ``threshold=3`` in a single step.

    Returns ``StepTerminate(...)`` when a stuck-loop pattern is detected,
    else ``None``.
    """
    from .finalize import finalize_run
    from .stuck_loop import StuckLoopDetector

    step = state.step
    trace = state.trace
    cost_tracker = state.cost_tracker
    stuck_detector = state.stuck_detector
    messages = state.messages

    stuck_feedback = None
    seen_batch_keys: set[str] = set()
    for block, result in tool_results_tuples:
        batch_key = StuckLoopDetector._make_key(
            block.name, block.input if isinstance(block.input, dict) else {},
        )
        if batch_key in seen_batch_keys:
            continue
        seen_batch_keys.add(batch_key)
        feedback = stuck_detector.record(
            block.name,
            block.input,
            result.content,  # raw content — not the truncated display string
        )
        if feedback and not stuck_feedback:
            stuck_feedback = feedback

    if not stuck_feedback:
        return None

    loop._fire("on_warning", stuck_feedback)
    cost = cost_tracker.cost if cost_tracker else 0.0
    if trace:
        trace.add_span(step=step, event_type="dedup_break", success=False,
                       error=stuck_feedback)
    return StepTerminate(finalize_run(
        status="dedup_break",
        text=state.all_text or "",
        final_text=state.final_text,
        messages=messages,
        steps=step + 1,
        total_input_tokens=state.total_input_tokens,
        total_output_tokens=state.total_output_tokens,
        tool_calls_count=state.tool_calls_count,
        total_cache_creation_tokens=state.total_cache_creation_tokens,
        total_cache_read_tokens=state.total_cache_read_tokens,
        api_calls=state.api_calls,
        error=stuck_feedback,
        trace=trace,
        trace_total_cost_usd=cost or 0.0,
    ))
