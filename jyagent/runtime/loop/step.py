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
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING, Union
import collections

from .llm_types import LLMOptions
from ..tools.registry import get_registry, ToolBatch

if TYPE_CHECKING:
    from .config import LoopResult
    from .cost import CostTracker
    from .run_context import RunContext


# ─── Run-scoped mutable state ────────────────────────────────────────────────


@dataclass
class RunState:
    """Mutable per-run state threaded through every ``run_step`` call.

    Built once at the top of ``AgentLoop._run_impl`` via
    ``RunState.prepare_for_run()``, mutated in place by ``run_step`` on
    every iteration.

    Field categories:
      - **Conversation**: ``system_prompt``, ``messages``, ``turn_start_idx``.
      - **Step bookkeeping**: ``step`` (set by engine before each call).
      - **Token / cost accumulators**.
      - **Tool / reflection counters**.
      - **Truncation retry state**.
      - **One-shot flags**: ``verification_injected``, ``unpriced_warned``.
      - **Heavy collaborators**: ``cost_tracker``, ``stuck_detector``,
        ``tools_batch``, ``trace``, ``effective_spec``. Built in setup;
        their object identity must remain stable for the entire run.
      - **Optional collaborators**: ``write_todos_fn`` (closure when todos
        enabled), ``reflection_module`` (lazily imported when reflection
        is enabled).
      - **Latest per-step artefacts**: ``last_step_batch`` (the engine's
        max_steps fallback path reads it for ``_truncate_tool_call_blocks``).

    Anti-patterns:
      - Do NOT copy ``loop._partial_side_effects`` or ``loop._todos`` into
        RunState fields. Those live on the AgentLoop instance because
        their lifecycle is cross-turn (re-used across run() calls), not
        per-run. Tool execution and the write_todos closure already reach
        for them via ``loop.``.
      - Do NOT replace ``cost_tracker`` / ``stuck_detector`` / ``tools_batch``
        with new objects mid-run. Mutate in place — ``cost_tracker.record``,
        ``stuck_detector.record``, ``tools_batch = registry.freeze()``
        (assigning a new ToolBatch is fine; the field is reassignable but
        the engine never reads its previous value across iterations).
    """

    # Conversation (mutated in place by ``run_step``)
    system_prompt: str
    messages: list
    turn_start_idx: int

    # Step bookkeeping (engine writes ``step`` before each ``run_step`` call)
    step: int = 0

    # Token / cost accumulators
    current_max_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    # Cache-token totals (Anthropic prompt cache).  OpenAI reports only
    # cache_read; both are accumulated here so LoopResult can surface
    # them to sub-agent stats without the parent losing visibility.
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    # Discrete LLM API calls made during the run.  Used by the sub-agent
    # accounting path so parent stats.api_calls reflects the child's
    # actual call count instead of the legacy "1 per dispatch".
    api_calls: int = 0

    # Tool / reflection counters
    tool_calls_count: int = 0
    last_reflection_count: int = 0

    # Truncation retry state
    consecutive_truncations: int = 0
    max_truncation_retries: int = 3

    # One-shot flags
    verification_injected: bool = False
    unpriced_warned: bool = False

    # Text accumulators
    all_text: str = ""
    final_text: str = ""

    # Heavy collaborators (built in setup; never replaced after)
    cost_tracker: "CostTracker | None" = None
    stuck_detector: Any = None        # StuckLoopDetector instance
    tools_batch: ToolBatch = field(default_factory=ToolBatch.empty)
    trace: Any = None                 # tracer or None
    effective_spec: Any = None        # ModelSpec

    # Optional collaborators (None when feature is disabled)
    write_todos_fn: Any = None
    reflection_module: Any = None

    # Latest per-step artefact — read by engine's max_steps fallback path
    last_step_batch: ToolBatch = field(default_factory=ToolBatch.empty)

    # ── Constructor ────────────────────────────────────────────────────────

    @classmethod
    def prepare_for_run(
        cls,
        loop: "RunContext",
        system_prompt: str,
        messages: list,
        initial_todos: list | None = None,
    ) -> "RunState":
        """Run-setup factory: prepares the AgentLoop instance for a fresh
        run and constructs the per-run mutable state.

        Renamed from ``from_loop``: the old name read like a pure factory but
        the method intentionally mutates ``loop._partial_side_effects``,
        ``loop._run_id``, and ``loop._todos``. The new name signals the
        side-effect contract.

        Side effects on ``loop`` (intentional, not just construction):
          - ``loop._partial_side_effects = []`` — reset mutating-timeout
            accumulator so back-to-back turns on the same instance don't
            carry stale names forward.
          - ``loop._run_id`` set when checkpointing is enabled and not
            already preset.
          - ``loop._todos`` seeded from ``initial_todos`` (validated via
            ``normalize_todo``); falls back to ``[]`` on TypeError with
            a warning.

        Side-effect-free construction:
          - tracer started against the effective model spec (provider/
            model picked from ``loop._model_spec`` if set, else
            ``loop._runtime_owner.model_spec``).
          - cost_tracker built only when ``cfg.max_cost_usd`` is set.
          - stuck_detector built unconditionally (cheap; the dedup
            threshold gates its actual blocking).
          - write_todos closure bound when todos are enabled — the
            closure reads/writes ``loop._todos`` via accessor functions
            so later mutation by the engine remains visible.
          - reflection module lazily imported only when at least one
            reflection knob is on.

        Why these mutations live here and not on the loop's __init__:
        ``run()`` is reentrant within a single instance (turn-by-turn or
        sub-agent re-use) but ``__init__`` only fires once. Per-run reset
        must happen at the top of every run, and the cleanest place is
        the run-state constructor.
        """
        cfg = loop._config

        # Reset accumulator. MUST happen at every run, not just init.
        # Use ``deque`` (not list) for free-
        # threaded-Python forward-compat — ``deque.append`` is documented
        # thread-safe for single-element ops, while ``list.append`` is
        # only atomic on the stock GIL build (PEP 703 / 3.13t / 3.14t
        # remove that guarantee).  ``list(deque(...))`` later in run()
        # converts to the historical list type for the LoopResult.
        loop._partial_side_effects = collections.deque()

        turn_start_idx = len(messages)

        # Lazy reflection import — opt-in by config.
        if cfg.reflect_every_n_tool_calls > 0 or cfg.reflect_after_subagent:
            from . import reflection as _reflection
            reflection_module = _reflection
        else:
            reflection_module = None

        # Ensure a run id when checkpointing is enabled (CLI may have
        # preset one via set_run_id() — preserve that).
        if cfg.checkpoint_dir and not loop._run_id:
            from .checkpoint import new_run_id
            loop._run_id = new_run_id()

        # Seed todos + bind write_todos closure if enabled.
        write_todos_fn = None
        if cfg.todos_enabled:
            from .todos import build_write_todos_tool, normalize_todo
            if initial_todos:
                try:
                    loop._todos = [normalize_todo(t) for t in initial_todos]
                except TypeError as e:
                    loop._fire("on_warning", f"ignoring invalid initial_todos: {e}")
                    loop._todos = []
            elif initial_todos is None:
                # Preserve any cross-turn todos
                # the caller already set on the loop instance — e.g. an outer
                # session that wants to chain multiple ``run()`` calls without
                # restating the plan each time.  Callers that want a fresh
                # start can pass ``initial_todos=[]`` explicitly.
                pass
            else:  # explicit empty list → caller wants the store cleared
                loop._todos = []

            # Closure reads/writes loop._todos via accessors so engine
            # mutations remain visible to the write_todos tool.
            def _get_store() -> list:
                return loop._todos

            def _set_store(new_list: list) -> None:
                loop._todos = new_list

            write_todos_fn = build_write_todos_tool(_get_store, _set_store)

        # Effective model spec for tracing + cost accounting (sub-agent
        # tier override wins over owner default).
        effective_spec = loop._model_spec or loop._runtime_owner.model_spec

        # Tracer + cost tracker + stuck detector. Late imports kept local
        # to avoid pulling these modules during ``import engine`` when a
        # caller never runs the loop (e.g. test collection).
        from .tracing import get_tracer
        from .cost import CostTracker
        from .stuck_loop import StuckLoopDetector

        trace = get_tracer()
        if trace:
            trace.start(effective_spec.provider, effective_spec.model)
        cost_tracker = CostTracker() if cfg.max_cost_usd is not None else None
        stuck_detector = StuckLoopDetector(cfg.dedup_threshold)

        return cls(
            system_prompt=system_prompt,
            messages=messages,
            turn_start_idx=turn_start_idx,
            current_max_tokens=cfg.initial_max_tokens,
            cost_tracker=cost_tracker,
            stuck_detector=stuck_detector,
            tools_batch=ToolBatch.empty(),
            trace=trace,
            effective_spec=effective_spec,
            write_todos_fn=write_todos_fn,
            reflection_module=reflection_module,
        )


# ─── Tagged-union step outcome ───────────────────────────────────────────────


@dataclass(frozen=True)
class StepContinue:
    """Step finished cleanly; outer for-loop continues to next iteration."""
    pass


@dataclass(frozen=True)
class StepTerminate:
    """Step produced a terminal LoopResult; outer loop returns it."""
    result: "LoopResult"


@dataclass(frozen=True)
class StepBreak:
    """Step requests outer-loop break (cooperative cancellation).

    The engine's post-loop cancelled-exit handler runs after this — it
    funnels through ``finalize_run`` with status='interrupted'.
    """
    reason: Literal["cancelled"]


StepOutcome = Union[StepContinue, StepTerminate, StepBreak]


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


# ─── Per-step helpers ────────────────────────────────────────────────────────
#
# All helpers are module-level (not nested in ``run_step``) so they:
#   - stay importable for whitebox tests (though the existing test suite
#     mostly tests through ``run_step`` end-to-end);
#   - can be grep'd independently;
#   - don't pay a closure-capture cost per-step.
#
# Each helper redoes the lazy imports it needs.  The dict-lookup cost for
# already-imported modules is sub-microsecond and called at most once per
# step, so there is no reason to push them to module top at the cost of
# re-introducing the engine→step import cycle.


def _prepare_step_batch(loop: "RunContext", state: RunState) -> ToolBatch:
    """Refresh ``state.tools_batch`` and apply the ``write_todos`` overlay.

    When ``_tool_source`` is provided (e.g. MCP integration that builds
    tool sets dynamically per turn), its (schemas, functions) supersede
    the registry's, but we still freeze the registry to inherit metadata
    (parallel_safe, timeout hints, large_input_keys, compaction_priority)
    for any tool whose name happens to be registered too.  This preserves
    the historical "tool_source funcs + registry metadata" behaviour but
    now atomically.

    Stashes the final per-step batch on ``state.last_step_batch`` so the
    engine's max_steps fallback path can read it for
    ``_truncate_tool_call_blocks``.
    """
    cfg = loop._config
    registry = get_registry()
    step = state.step

    if loop._tool_source is not None:
        src_schemas, src_functions = loop._tool_source()
        state.tools_batch = ToolBatch.from_source(
            src_schemas, src_functions, base=registry.freeze(),
        )
    elif step == 0 or registry.version != state.tools_batch.version:
        # Re-freeze only when the registry has changed.  The version read is
        # locked (defense-in-depth), and even if a stale-by-one read causes
        # us to skip a freeze, the next step will catch up — at most one step
        # uses slightly-stale metadata, never inconsistent metadata.
        state.tools_batch = registry.freeze()

    # Overlay the per-loop write_todos tool on top of the registry snapshot
    # when todos are enabled.  Closure-scoped injection (recommended by the
    # design doc — avoids ContextVar propagation issues with our daemon-
    # thread tool executor).
    if cfg.todos_enabled:
        from .todos import WRITE_TODOS_SCHEMA
        step_batch = state.tools_batch.with_overlay(
            functions={"write_todos": state.write_todos_fn},
            schemas=[WRITE_TODOS_SCHEMA],
            # write_todos must NOT be parallel-safe — it would then run
            # concurrently with itself in a batch and the replace-all
            # semantics would silently drop one of the writes.
        )
    else:
        step_batch = state.tools_batch

    state.last_step_batch = step_batch
    return step_batch


def _compact_and_build_context(
    loop: "RunContext",
    state: RunState,
    step_batch: ToolBatch,
) -> dict:
    """Apply context compaction (in-place mutation of ``state.messages``)
    and return the context dict for the LLM adapter.

    Todos are injected as a ``<system-reminder>`` text block appended to
    the tail user message — NOT persisted into ``messages``, so compaction
    never touches them.  The base ``system_prompt`` stays untouched to
    preserve Anthropic prefix caching.
    """
    from .compaction import compact_messages as _compact_messages

    cfg = loop._config
    messages = state.messages

    if cfg.compact_messages:
        before_len = len(messages)
        messages_maybe = _compact_messages(
            messages, cfg.max_working_tokens, cfg.compact_tool_result_chars,
            step_batch,
        )
        if messages_maybe is not messages:
            after_len = len(messages_maybe)
            messages[:] = messages_maybe
            loop._fire("on_compaction", before_len, after_len)

    if cfg.todos_enabled and loop._todos:
        from .todos import inject_todos_into_messages
        context_messages = inject_todos_into_messages(messages, loop._todos)
    else:
        context_messages = messages

    context: dict[str, Any] = {
        "system_prompt": state.system_prompt,
        "messages": context_messages,
    }
    if step_batch.schemas:
        context["tools"] = list(step_batch.schemas)
    return context


def _build_step_options(loop: "RunContext", state: RunState) -> LLMOptions:
    """Build this step's ``LLMOptions``, applying phase-policy ``tool_choice``
    shaping if the loop config provides a policy.

    Phase policy is consulted once per step.  Returning a ``PhaseDirective``
    with ``tool_choice=None`` is informational only (engine fires
    ``on_phase_enter`` for observability but leaves ``tool_choice``
    unchanged).  A non-None tool_choice rebuilds ``opts`` so the runtime
    adapter sees the override.
    """
    from .llm_runner import build_runtime_options as _build_runtime_options

    cfg = loop._config
    step = state.step
    trace = state.trace

    opts = _build_runtime_options(
        loop._runtime_owner,
        state.current_max_tokens,
        model_spec=loop._model_spec,
        metadata={"component": "loop_engine", "step": step + 1},
        session_id=getattr(loop, "_session_id", ""),
    )

    if cfg.phase_policy is None:
        return opts

    try:
        directive = cfg.phase_policy(step, cfg.max_steps, state.tool_calls_count)
    except Exception as e:
        directive = None
        loop._fire("on_warning", f"phase_policy raised: {e}")

    if directive is None:
        return opts

    loop._fire("on_phase_enter", directive.phase)
    if trace:
        trace.add_span(step=step, event_type="phase", tool_name=directive.phase)

    if directive.tool_choice is None:
        return opts

    return LLMOptions(
        max_output_tokens=opts.max_output_tokens,
        timeout=opts.timeout,
        reasoning=opts.reasoning,
        metadata={**(opts.metadata or {}), "phase": directive.phase},
        tool_choice=directive.tool_choice,
    )


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
    denial_msg = "Denied by user (on_tool_pre_execute gate)."
    denied_ids = {b.id for b in denied_blocks}
    results_by_id = {b.id: r for (b, r) in executed_results}

    # Phase 3: emit messages + fire on_tool_end + trace, IN ORIGINAL
    # tool_call_blocks ORDER.  Returns a ``tool_results_tuples`` list that
    # mirrors the input order so downstream consumers (stuck-loop detection,
    # /history rendering, positional pairing) see what they expect.
    tool_results_tuples: list = []
    for block in tool_call_blocks:
        if block.id in denied_ids:
            result = ToolResult(content=denial_msg, is_error=True)
            # Stamp duration_ms=0 for parity with executed tools (the
            # ``on_tool_end`` callback and trace pipeline both read this).
            try:
                result.duration_ms = 0.0  # type: ignore[attr-defined]
            except Exception:
                pass
        else:
            result = results_by_id[block.id]

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

