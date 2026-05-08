# step.py — One iteration of the agent loop.
#
# The per-step body of
# ``AgentLoop._run_impl`` (LLM call → tool dispatch → reflection → checkpoint
# → cancel checks) lives here as a free function ``run_step(loop, state)``.
#
# The engine still owns:
#   - the for-step counter
#   - the post-loop terminal handlers (cancelled-exit, max_steps fallback,
#     max_steps exit)
#   - the outer try/except that wraps the loop with KeyboardInterrupt /
#     Exception → finalized LoopResult conversion.
#
# This file owns the per-step semantics:
#   - tools_batch refresh + write_todos overlay
#   - context compaction
#   - phase-aware tool_choice shaping
#   - LLM call dispatch (delegated to ``loop._call_llm_with_retry``, which
    #     keeps the subclass-override contract)
#   - cost-budget enforcement
#   - completion / verification-gate path (no tool calls)
#   - truncation detection + retry
#   - tool execution
#   - response-aware stuck-loop detection
#   - mid-loop reflection injection
#   - periodic checkpoint
#
# Cross-iteration state is threaded through the mutable ``RunState``
# dataclass — passed by reference (no copy). Per-iteration locals
# (step_text, tool_call_blocks, opts, etc.) stay as locals to ``run_step``.

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .run_context import RunContext


# ─── Run-scoped mutable state + outcome types ────────────────────────────────
#
# ``RunState`` and the ``StepContinue`` / ``StepTerminate`` / ``StepBreak``
# outcome union live in ``step_state.py`` (extracted 2026-05-06 so helper
# modules can import them without risking cycles back through this file).
# ``run_step`` below uses all four in its signature / return statements,
# so this is a normal load-bearing import, not a re-export shim.
#
# Side effect: tests that introspect with ``inspect.getsource(step.RunState
# .prepare_for_run)`` or import ``RunState`` from ``step`` keep working
# because the names are reachable on this module.

from .step_state import (
    RunState,
    StepContinue,
    StepTerminate,
    StepBreak,
    StepOutcome,
)


# ─── The per-step body ───────────────────────────────────────────────────────


def run_step(loop: "RunContext", state: RunState) -> StepOutcome:
    """Execute one iteration of the agent loop.

    Thin orchestrator. Each phase is extracted into a private helper
    below so ``run_step`` remains a top-to-bottom narrative of the step's
    control flow. Helpers take ``(loop, state, ...)`` and either mutate
    state / messages in place or return a ``StepOutcome`` when they need
    to terminate the step early.

    Behavioural branch points deliberately stay inline here for reviewer
    visibility: the no-tool verification gate + completion, the
    truncation retry, and the final ``StepContinue()`` at the bottom.
    Codex's 2026-04 review explicitly recommended against pulling those
    into helpers — they are the places where the step's semantics diverge.

    Returns one of:
      - ``StepContinue()``           — run the next iteration
      - ``StepTerminate(result)``    — engine returns ``result`` immediately
      - ``StepBreak("cancelled")``   — engine runs the cancelled-exit handler

    The engine owns the for-step counter, the post-loop terminal handlers
    (cancelled-exit / max_steps fallback / max_steps exit), and the outer
    try/except that converts KeyboardInterrupt / unhandled Exception into
    a finalized LoopResult.

    The subclass-override contract (tests subclassing AgentLoop and
    overriding _call_streaming / _call_complete / _call_llm_with_retry)
    is preserved automatically — every LLM-related call here goes through
    ``loop.<method>``, resolved by Python's normal attribute lookup.
    """
    # Lazy imports break the engine→step→engine cycle without polluting
    # module-load order. Cost: one dict lookup per step (negligible).
    from .finalize import finalize_run, is_truncated
    from .llm_runner import extract_text as _extract_text
    from .compaction import truncate_tool_call_blocks as _truncate_tool_call_blocks
    from .verification import should_verify, build_verification_prompt

    cfg = loop._config
    messages = state.messages
    step = state.step
    trace = state.trace
    cost_tracker = state.cost_tracker

    loop._fire("on_step_progress", step, cfg.max_steps)

    # ── Cooperative cancellation check (top of loop) ─────
    if loop._is_cancelled():
        return StepBreak(reason="cancelled")

    # ── 1. Tools batch refresh + write_todos overlay ─────
    step_batch = _prepare_step_batch(loop, state)

    # ── 2. Context compaction + build context dict ───────
    context = _compact_and_build_context(loop, state, step_batch)

    # ── 3. LLMOptions + phase-policy shaping ─────────────
    opts = _build_step_options(loop, state)

    # ── 4. LLM call (subclass-overridable via loop._call_llm_with_retry) ─
    llm_t0 = time.perf_counter()
    step_text, tool_call_blocks, stop_reason, final_message = loop._call_llm_with_retry(
        context, opts, step,
    )
    llm_dur_ms = (time.perf_counter() - llm_t0) * 1000

    # ── 5. Runtime warnings + usage + trace + cost check (may terminate) ─
    outcome = _record_llm_usage_and_cost(loop, state, final_message, llm_dur_ms)
    if outcome is not None:
        return outcome

    # NOTE: text accumulation happens AFTER the cost check so that a budget-
    # terminated step does not include its just-generated text in the result.
    # Do not reorder.
    state.all_text += step_text
    state.final_text = step_text

    # ── 6. No-tool path: verification gate OR terminal completion ─────────
    #
    # Kept inline per codex's review — this is a behavioural branch point,
    # not a helper candidate.
    if not tool_call_blocks:
        # Pre-completion verification gate: if we mutated files and haven't
        # verified yet, inject a self-check prompt and loop once more instead
        # of returning.
        #
        # Boundary guard (P0 fix): never inject on the final allowed step —
        # the follow-up model reply has no iteration left to run, and the
        # dangling `[VERIFICATION]` user message would otherwise leak into
        # the persisted session and poison the next turn.
        if (
            not state.verification_injected
            and should_verify(
                messages,
                state.tool_calls_count,
                since_index=state.turn_start_idx,
                batch=step_batch,
            )
            and step + 1 < cfg.max_steps
        ):
            state.verification_injected = True
            if trace:
                trace.add_span(step=step, event_type="verification")
            # Append the assistant's response, then inject verification.
            # NOTE: the verification gate intentionally does NOT run the
            # standard assistant-message truncation / on_assistant_message
            # callback — the message is not final from the caller's point
            # of view (the loop will continue).
            messages.append(final_message)
            messages.append({
                "role": "user",
                "content": build_verification_prompt(messages),
            })
            return StepContinue()

        if not step_text:
            state.final_text = _extract_text(final_message)
            state.all_text = state.final_text or state.all_text

        _append_assistant_message(loop, state, step_batch, final_message)
        result_text = state.all_text if state.all_text else "I processed your request but had no text response to return."

        cost = cost_tracker.cost if cost_tracker else 0.0
        return StepTerminate(finalize_run(
            status="completed",
            text=result_text,
            final_text=state.final_text,
            messages=messages,
            steps=step + 1,
            total_input_tokens=state.total_input_tokens,
            total_output_tokens=state.total_output_tokens,
            tool_calls_count=state.tool_calls_count,
            total_cache_creation_tokens=state.total_cache_creation_tokens,
            total_cache_read_tokens=state.total_cache_read_tokens,
            api_calls=state.api_calls,
            trace=trace,
            trace_total_cost_usd=cost or 0.0,
        ))

    # ── 7. Truncation detection → scale up and retry step (may terminate) ─
    #
    # Kept inline per codex's review — another behavioural branch point.
    if cfg.auto_scale_on_truncation and is_truncated(stop_reason, tool_call_blocks):
        state.consecutive_truncations += 1
        if state.consecutive_truncations > state.max_truncation_retries:
            cost = cost_tracker.cost if cost_tracker else 0.0
            return StepTerminate(finalize_run(
                status="error",
                text=state.all_text or "",
                final_text="",
                messages=messages,
                steps=step + 1,
                total_input_tokens=state.total_input_tokens,
                total_output_tokens=state.total_output_tokens,
                tool_calls_count=state.tool_calls_count,
                total_cache_creation_tokens=state.total_cache_creation_tokens,
                total_cache_read_tokens=state.total_cache_read_tokens,
                api_calls=state.api_calls,
                error=f"Repeated truncation ({state.consecutive_truncations}x) — model output exceeds capacity",
                trace=trace,
                trace_total_cost_usd=cost or 0.0,
            ))
        loop._fire("on_truncation")
        # Also fire the unified stream-retry hook so UIs that already handle
        # transient-error duplication can use the same visual treatment for
        # truncation-recovery replays.
        loop._fire("on_stream_retry", "truncation", step_text or "")
        state.current_max_tokens = min(
            state.current_max_tokens * cfg.token_scale_factor,
            cfg.max_tokens_cap,
        )
        # Remove the partial step text — the next attempt will regenerate it.
        state.all_text = state.all_text[: -len(step_text)] if step_text else state.all_text
        return StepContinue()

    # ── 8. Successful tool-use step ───────────────────────────────────────
    state.consecutive_truncations = 0

    _append_assistant_message(loop, state, step_batch, final_message)

    # ── 9. Tool execution round (on_tool_start + before-tools cancel shim
    # + dispatch + on_tool_end + append tool_result messages) ─────────────
    outcome, tool_results_tuples = _execute_tool_round(
        loop, state, step_batch, tool_call_blocks,
    )
    if outcome is not None:
        return outcome

    # ── 10. Response-aware stuck-loop detection (may terminate) ──────────
    outcome = _check_stuck_loop(loop, state, tool_results_tuples)
    if outcome is not None:
        return outcome

    # ── Cooperative cancellation check (after tools) ──────
    if loop._is_cancelled():
        return StepBreak(reason="cancelled")

    # ── 11. Mid-loop reflection + 12. Periodic checkpoint ─
    _maybe_reflect(loop, state, tool_results_tuples)
    _maybe_checkpoint(loop, state)

    return StepContinue()


# ─── Per-step helpers (extracted to leaf modules) ────────────────────────────
#
# The 9 helpers below were extracted from this file (2026-05-06) into
# 3 leaf modules grouped by life-cycle position:
#
#   step_setup.py        — pre-LLM-call setup (batch, context, options)
#   step_tools.py        — tool-round semantics (execute, stuck-loop)
#   step_bookkeeping.py  — post-LLM-call effects (cost, message, reflect, checkpoint)
#
# ``run_step`` (the only function defined in this file now) calls each
# helper directly, so the imports below are normal load-bearing imports.
#
# Side effect: tests that import ``from jyagent.runtime.loop.step import
# _<helper>`` or use ``inspect.getsource(step._<helper>)`` keep working
# because the function objects are reachable on this module — inspect
# follows ``__module__`` to read the source bytes from the leaf module
# where the helper is defined.
#
# Layering invariant: the leaf modules import from ``step_state`` (RunState +
# outcome types) but NEVER from ``step.py``.  ``run_step`` is the only thing
# that lives here now; everything else is composition.

from .step_setup import (
    _prepare_step_batch,
    _compact_and_build_context,
    _build_step_options,
)
from .step_tools import (
    _execute_tool_round,
    _check_stuck_loop,
)
from .step_bookkeeping import (
    _record_llm_usage_and_cost,
    _append_assistant_message,
    _maybe_reflect,
    _maybe_checkpoint,
)
