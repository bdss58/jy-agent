"""Per-step bookkeeping (post-LLM-call effects).

Owns the helpers that run AFTER each LLM call to advance run-scoped state:

  * ``_record_llm_usage_and_cost``  — usage accounting + cost-budget enforcement
  * ``_append_assistant_message``   — assistant turn materialisation
  * ``_maybe_reflect``              — mid-loop reflection injection
  * ``_maybe_checkpoint``           — periodic checkpoint trigger

Extracted from ``runtime/loop/step.py`` (2026-05-06).  Re-exported from
``step.py`` so existing ``from jyagent.runtime.loop.step import
_record_llm_usage_and_cost`` callers (notably tests using
``inspect.getsource`` for source-text invariants) keep working unchanged.

Named ``step_bookkeeping`` rather than ``step_metrics`` because these
helpers do more than metrics: they also mutate the assistant message,
inject reflection prompts, and drive checkpoint cadence.

Layering: imports from ``step_state`` only.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..tools.registry import ToolBatch
from .step_state import RunState, StepOutcome, StepTerminate

if TYPE_CHECKING:
    from .run_context import RunContext


def _record_llm_usage_and_cost(
    loop: "RunContext",
    state: RunState,
    final_message: dict,
    llm_dur_ms: float,
) -> StepOutcome | None:
    """Fire runtime warnings / ``on_usage`` / trace span / cost recording
    and enforce the cost budget.  Returns ``StepTerminate(...)`` when the
    budget is exhausted, else ``None``.

    Accumulates ``state.total_input_tokens`` / ``state.total_output_tokens``.
    """
    from .finalize import finalize_run

    cfg = loop._config
    step = state.step
    trace = state.trace
    cost_tracker = state.cost_tracker
    effective_spec = state.effective_spec
    messages = state.messages

    # Fire runtime warnings surfaced by the provider adapter.
    for warning in final_message.get("llm_warnings", []):
        loop._fire("on_warning", warning)

    # Accumulate usage.
    usage = final_message.get("usage", {})
    state.total_input_tokens += usage.get("input_tokens", 0)
    state.total_output_tokens += usage.get("output_tokens", 0)
    # Cache-token accumulators (see RunState comments).  Anthropic reports
    # both fields; OpenAI currently only reports cache_read.  Missing keys
    # default to 0 — never raise on a provider that doesn't emit them.
    state.total_cache_creation_tokens += usage.get("cache_creation_input_tokens", 0) or 0
    state.total_cache_read_tokens += usage.get("cache_read_input_tokens", 0) or 0
    # Count any call that produced a usage record as one API call.  Empty
    # usage dicts (e.g. when the provider adapter fabricates a blank
    # message on a hard cancel) are NOT counted so the meter doesn't
    # drift upward on no-op turns.
    if usage:
        state.api_calls += 1
    loop._fire("on_usage", usage)

    # Trace LLM call.
    if trace:
        trace.add_span(
            step=step,
            event_type="llm_call",
            duration_ms=llm_dur_ms,
            tokens_in=usage.get("input_tokens"),
            tokens_out=usage.get("output_tokens"),
        )

    # Cost budget check.
    if cost_tracker is None:
        return None

    # Use the effective spec so sub-agent model overrides are billed at
    # the right rate (P0 fix).
    cost_tracker.record(usage, effective_spec.provider, effective_spec.model)

    # One-shot warning if any call lacked pricing data.  The budget still
    # enforces on the priced subtotal (lower bound) — silent "None ⇒ skip"
    # would disable the gate.
    if cost_tracker.has_unpriced_usage and not state.unpriced_warned:
        state.unpriced_warned = True
        loop._fire(
            "on_warning",
            f"Cost budget using lower bound: "
            f"{cost_tracker.unpriced_calls} call(s) had no pricing data "
            f"({effective_spec.provider}/{effective_spec.model}).",
        )

    current_cost = cost_tracker.cost
    if current_cost >= cfg.max_cost_usd:
        loop._fire(
            "on_warning",
            f"Cost budget exceeded: ${current_cost:.4f} >= ${cfg.max_cost_usd:.4f}",
        )
        if trace:
            trace.add_span(step=step, event_type="cost_check", success=False,
                           error=f"budget ${cfg.max_cost_usd} exceeded")
        return StepTerminate(finalize_run(
            status="cost_limit",
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
            error=f"Cost budget exceeded: ${current_cost:.4f} >= ${cfg.max_cost_usd:.4f}",
            trace=trace,
            trace_total_cost_usd=current_cost,
        ))

    return None
def _append_assistant_message(
    loop: "RunContext",
    state: RunState,
    step_batch: ToolBatch,
    final_message: dict,
) -> None:
    """Apply large-input truncation, invoke the ``on_assistant_message``
    callback if set, and append to ``state.messages`` in-place.

    Used by BOTH the no-tool completion path and the tool-use success path.
    The verification-gate branch intentionally does NOT use this helper —
    that path appends the raw message because the loop will continue and
    the caller's transform should fire on the FINAL assistant reply, not
    on intermediate tool-use turns.

    Callback semantics: if the callback returns ``None`` the original
    message is kept; any other return value replaces it.  Previously the
    two call sites had slightly different semantics (``cb() or final``
    vs ``if transformed is not None``) — unified here on the stricter
    ``is not None`` check so a callback that legitimately returns an
    empty-but-truthy placeholder doesn't get silently overwritten.
    """
    from .compaction import truncate_tool_call_blocks as _truncate_tool_call_blocks

    cfg = loop._config

    if cfg.truncate_large_inputs:
        content = final_message.get("content", [])
        final_message = dict(final_message)
        final_message["content"] = _truncate_tool_call_blocks(content, step_batch)

    cb_am = loop._callbacks.on_assistant_message
    if cb_am is not None:
        transformed = cb_am(final_message)
        if transformed is not None:
            final_message = transformed

    state.messages.append(final_message)
def _maybe_reflect(
    loop: "RunContext",
    state: RunState,
    tool_results_tuples: list,
) -> None:
    """After meaningful work boundaries (every-N cadence or sub-agent
    return), append a short progress-check user message so the next LLM
    call re-grounds on the task.  Avoids drift on long-horizon rollouts.
    No-op unless reflection is enabled via LoopConfig.
    """
    if state.reflection_module is None:
        return

    cfg = loop._config
    step = state.step
    trace = state.trace
    messages = state.messages

    reflection = state.reflection_module
    batch_names = [b.name for b, _ in tool_results_tuples]
    inject, reason = reflection.should_reflect(
        reflect_every_n=cfg.reflect_every_n_tool_calls,
        reflect_after_subagent=cfg.reflect_after_subagent,
        tool_calls_total=state.tool_calls_count,
        tool_calls_at_last_reflection=state.last_reflection_count,
        batch_tool_names=batch_names,
        messages=messages,
    )
    if not inject:
        return

    prompt = reflection.build_reflection_prompt(reason, state.tool_calls_count)
    messages.append({"role": "user", "content": prompt})
    state.last_reflection_count = state.tool_calls_count
    loop._fire("on_reflection", reason)
    if trace:
        trace.add_span(step=step, event_type="reflection")
def _maybe_checkpoint(loop: "RunContext", state: RunState) -> None:
    """Persist a checkpoint at the configured cadence.  No-op when
    ``checkpoint_dir`` is unset or we're not on a cadence boundary.
    """
    cfg = loop._config
    step = state.step

    if (
        cfg.checkpoint_every_n_steps > 0
        and cfg.checkpoint_dir
        and (step + 1) % cfg.checkpoint_every_n_steps == 0
    ):
        loop._write_checkpoint(
            step=step,
            messages=state.messages,
            total_input_tokens=state.total_input_tokens,
            total_output_tokens=state.total_output_tokens,
            tool_calls_count=state.tool_calls_count,
            total_cache_creation_tokens=state.total_cache_creation_tokens,
            total_cache_read_tokens=state.total_cache_read_tokens,
            api_calls=state.api_calls,
            status="in_progress",
        )
