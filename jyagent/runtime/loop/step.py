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
    from .engine import AgentLoop
    from .config import LoopResult
    from .cost import CostTracker


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
        loop: "AgentLoop",
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
    funnels through ``_finalize_run`` with status='interrupted'.
    """
    reason: Literal["cancelled"]


StepOutcome = Union[StepContinue, StepTerminate, StepBreak]


# ─── The per-step body ───────────────────────────────────────────────────────


def run_step(loop: "AgentLoop", state: RunState) -> StepOutcome:
    """Execute one iteration of the agent loop.

    Owns: tools_batch refresh, compaction, LLM call dispatch, tool execution,
    stuck-loop detection, reflection, checkpoint, and the inline early-return
    paths (cost_limit, completed-no-tools, repeated truncation, dedup_break).

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
    from .engine import _finalize_run, _is_truncated
    from .stuck_loop import StuckLoopDetector
    from .llm_runner import (
        extract_text as _extract_text,
        build_runtime_options as _build_runtime_options,
    )
    from .compaction import (
        compact_messages as _compact_messages,
        truncate_result as _truncate_result,
        truncate_tool_call_blocks as _truncate_tool_call_blocks,
    )
    from .tool_executor import execute_tools as _execute_tools
    from .verification import should_verify, build_verification_prompt

    cfg = loop._config
    registry = get_registry()
    messages = state.messages
    system_prompt = state.system_prompt
    step = state.step
    trace = state.trace
    cost_tracker = state.cost_tracker
    stuck_detector = state.stuck_detector
    effective_spec = state.effective_spec

    loop._fire("on_step_progress", step, cfg.max_steps)

    # ── Cooperative cancellation check (top of loop) ─────
    if loop._is_cancelled():
        return StepBreak(reason="cancelled")

    # Refresh tool batch each step.
    #
    # When ``_tool_source`` is provided (e.g. MCP integration that
    # builds tool sets dynamically per turn), its (schemas,
    # functions) supersede the registry's, but we still freeze
    # the registry to inherit metadata (parallel_safe, timeout
    # hints, large_input_keys, compaction_priority) for any
    # tool whose name happens to be registered too.  This
    # preserves the historical "tool_source funcs + registry
    # metadata" behaviour but now atomically.
    if loop._tool_source is not None:
        src_schemas, src_functions = loop._tool_source()
        # Deep-copy schemas + wrap
        # all maps in MappingProxyType views, so the tool_source path
        # has the same immutability guarantees as ToolRegistry.freeze().
        # Metadata classification (parallel_safe, mutating, timeout
        # hints, large_input_keys, compaction_priority) is inherited
        # from the registry freeze for any tool whose name happens to
        # be registered too.
        state.tools_batch = ToolBatch.from_source(
            src_schemas, src_functions, base=registry.freeze(),
        )
    elif step == 0 or registry.version != state.tools_batch.version:
        # Re-freeze only when the registry has changed.  The
        # version read is locked (defense-in-depth), and even
        # if a stale-by-one read causes us to skip a freeze,
        # the next step will catch up — at most one step uses
        # slightly-stale metadata, never inconsistent metadata.
        state.tools_batch = registry.freeze()

    # Overlay the per-loop write_todos tool on top of the
    # registry snapshot when todos are enabled.  This is the
    # closure-scoped injection point recommended by the design
    # design (avoids ContextVar propagation issues with our
    # daemon-thread tool executor).
    if cfg.todos_enabled:
        from .todos import WRITE_TODOS_SCHEMA
        step_batch = state.tools_batch.with_overlay(
            functions={"write_todos": state.write_todos_fn},
            schemas=[WRITE_TODOS_SCHEMA],
            # write_todos must NOT be parallel-safe — it would
            # then run concurrently with itself in a batch and
            # the replace-all semantics would silently drop
            # one of the writes.
        )
    else:
        step_batch = state.tools_batch
    state.last_step_batch = step_batch

    tool_schemas = list(step_batch.schemas)

    # Context compaction
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

    # Build context dict.  Todos are injected as a
    # <system-reminder> text block appended to the tail user
    # message — NOT persisted into `messages`, so compaction
    # never touches them.  The base system_prompt stays
    # untouched to preserve Anthropic prefix caching.
    if cfg.todos_enabled and loop._todos:
        from .todos import inject_todos_into_messages
        context_messages = inject_todos_into_messages(messages, loop._todos)
    else:
        context_messages = messages

    context: dict[str, Any] = {
        "system_prompt": system_prompt,
        "messages": context_messages,
    }
    if tool_schemas:
        context["tools"] = tool_schemas

    # LLM call with retry
    opts = _build_runtime_options(
        loop._runtime_owner,
        state.current_max_tokens,
        model_spec=loop._model_spec,
        metadata={"component": "loop_engine", "step": step + 1},
        session_id=getattr(loop, "_session_id", ""),
    )

    # Phase-aware tool_choice shaping (see jyagent/phases.py).
    # The policy is consulted once per step.  Returning a
    # PhaseDirective with `tool_choice=None` is informational
    # only (engine fires on_phase_enter for observability but
    # leaves tool_choice unchanged).  A non-None tool_choice
    # rebuilds `opts` so the runtime adapter sees the override.
    if cfg.phase_policy is not None:
        try:
            directive = cfg.phase_policy(step, cfg.max_steps, state.tool_calls_count)
        except Exception as e:
            directive = None
            loop._fire("on_warning", f"phase_policy raised: {e}")
        if directive is not None:
            loop._fire("on_phase_enter", directive.phase)
            if trace:
                trace.add_span(
                    step=step, event_type="phase",
                    tool_name=directive.phase,
                )
            if directive.tool_choice is not None:
                opts = LLMOptions(
                    max_output_tokens=opts.max_output_tokens,
                    timeout=opts.timeout,
                    reasoning=opts.reasoning,
                    metadata={**(opts.metadata or {}), "phase": directive.phase},
                    tool_choice=directive.tool_choice,
                )

    llm_t0 = time.perf_counter()
    step_text, tool_call_blocks, stop_reason, final_message = loop._call_llm_with_retry(
        context, opts, step,
    )
    llm_dur_ms = (time.perf_counter() - llm_t0) * 1000

    # Fire runtime warnings
    for warning in final_message.get("llm_warnings", []):
        loop._fire("on_warning", warning)

    # Accumulate usage
    usage = final_message.get("usage", {})
    state.total_input_tokens += usage.get("input_tokens", 0)
    state.total_output_tokens += usage.get("output_tokens", 0)
    loop._fire("on_usage", usage)

    # ── Trace LLM call ────────────────────────────────────
    if trace:
        trace.add_span(
            step=step,
            event_type="llm_call",
            duration_ms=llm_dur_ms,
            tokens_in=usage.get("input_tokens"),
            tokens_out=usage.get("output_tokens"),
        )

    # ── Cost budget check ─────────────────────────────────
    if cost_tracker is not None:
        # Use the effective spec so sub-agent model overrides are
        # billed at the right rate (P0 fix).
        cost_tracker.record(
            usage,
            effective_spec.provider,
            effective_spec.model,
        )
        # One-shot warning if any call lacked pricing data.  The
        # budget still enforces on the priced subtotal (lower
        # bound) — silent "None ⇒ skip" would disable the gate.
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
            return StepTerminate(_finalize_run(
                status="cost_limit",
                text=state.all_text or "",
                final_text=state.final_text,
                messages=messages,
                steps=step + 1,
                total_input_tokens=state.total_input_tokens,
                total_output_tokens=state.total_output_tokens,
                tool_calls_count=state.tool_calls_count,
                error=f"Cost budget exceeded: ${current_cost:.4f} >= ${cfg.max_cost_usd:.4f}",
                trace=trace,
                trace_total_cost_usd=current_cost,
            ))

    state.all_text += step_text
    state.final_text = step_text

    # No tool calls → done (or verification gate)
    if not tool_call_blocks:
        # ── Pre-completion verification gate ───────────────
        # If we mutated files and haven't verified yet, inject a
        # self-check prompt and loop once more instead of returning.
        #
        # Boundary guard (P0 fix): never inject on the final allowed
        # step — the follow-up model reply has no iteration left to
        # run, and the dangling `[VERIFICATION]` user message would
        # otherwise leak into the persisted session and poison the
        # next turn.
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
            # Append the assistant's response, then inject verification
            messages.append(final_message)
            messages.append({
                "role": "user",
                "content": build_verification_prompt(messages),
            })
            return StepContinue()

        if not step_text:
            state.final_text = _extract_text(final_message)
            state.all_text = state.final_text or state.all_text

        # Apply truncation if enabled
        if cfg.truncate_large_inputs:
            content = final_message.get("content", [])
            final_message = dict(final_message)
            final_message["content"] = _truncate_tool_call_blocks(content, step_batch)

        # Allow caller to transform before append
        cb_am = loop._callbacks.on_assistant_message
        if cb_am is not None:
            final_message = cb_am(final_message) or final_message
        messages.append(final_message)
        result_text = state.all_text if state.all_text else "I processed your request but had no text response to return."

        cost = cost_tracker.cost if cost_tracker else 0.0
        return StepTerminate(_finalize_run(
            status="completed",
            text=result_text,
            final_text=state.final_text,
            messages=messages,
            steps=step + 1,
            total_input_tokens=state.total_input_tokens,
            total_output_tokens=state.total_output_tokens,
            tool_calls_count=state.tool_calls_count,
            trace=trace,
            trace_total_cost_usd=cost or 0.0,
        ))

    # Truncation detection → scale up and retry step
    if cfg.auto_scale_on_truncation and _is_truncated(stop_reason, tool_call_blocks):
        state.consecutive_truncations += 1
        if state.consecutive_truncations > state.max_truncation_retries:
            cost = cost_tracker.cost if cost_tracker else 0.0
            return StepTerminate(_finalize_run(
                status="error",
                text=state.all_text or "",
                final_text="",
                messages=messages,
                steps=step + 1,
                total_input_tokens=state.total_input_tokens,
                total_output_tokens=state.total_output_tokens,
                tool_calls_count=state.tool_calls_count,
                error=f"Repeated truncation ({state.consecutive_truncations}x) — model output exceeds capacity",
                trace=trace,
                trace_total_cost_usd=cost or 0.0,
            ))
        loop._fire("on_truncation")
        # Also fire the unified stream-retry hook so UIs that
        # already handle transient-error duplication can use the
        # same visual treatment for truncation-recovery replays.
        loop._fire("on_stream_retry", "truncation", step_text or "")
        state.current_max_tokens = min(
            state.current_max_tokens * cfg.token_scale_factor,
            cfg.max_tokens_cap,
        )
        # Remove the partial step text
        state.all_text = state.all_text[: -len(step_text)] if step_text else state.all_text
        return StepContinue()

    # Successful step — reset truncation counter
    state.consecutive_truncations = 0

    # Append assistant message (allow caller to transform)
    if cfg.truncate_large_inputs:
        content = final_message.get("content", [])
        final_message = dict(final_message)
        final_message["content"] = _truncate_tool_call_blocks(content, step_batch)

    cb_am = loop._callbacks.on_assistant_message
    if cb_am is not None:
        transformed = cb_am(final_message)
        if transformed is not None:
            final_message = transformed
    messages.append(final_message)

    # Fire on_tool_batch for multi-tool batches
    if len(tool_call_blocks) > 1:
        loop._fire("on_tool_batch", len(tool_call_blocks))

    # Fire on_tool_start for all tool calls BEFORE execution
    for block in tool_call_blocks:
        loop._fire("on_tool_start", block.name, block.input)

    # Execute tools
    # ── Cooperative cancellation check (before tools) ────
    if loop._is_cancelled():
        # Return error results for all pending tool calls.
        # Fire on_tool_end for each so callbacks see the matching
        # close event for the on_tool_start fired above
        # (without this, UIs that count starts vs. ends — e.g.
        # spinners, progress bars — leak resources on cancel).
        for block in tool_call_blocks:
            messages.append({
                "role": "tool_result",
                "tool_call_id": block.id,
                "tool_name": block.name,
                "content": "Cancelled",
                "is_error": True,
            })
            loop._fire("on_tool_end", block.name, "Cancelled", True)
        return StepBreak(reason="cancelled")

    tools_t0 = time.perf_counter()
    tool_results_tuples = _execute_tools(
        tool_call_blocks,
        step_batch,
        cfg.concurrent_tools,
        cfg.max_tool_workers,
        cfg.tool_timeout,
        executor=loop._executor,
        partial_side_effects=loop._partial_side_effects,
        cancel_event=loop._cancel_event,
    )
    tools_dur_ms = (time.perf_counter() - tools_t0) * 1000

    for block, result in tool_results_tuples:
        state.tool_calls_count += 1
        content_str = _truncate_result(result.content, cfg.max_tool_result_chars, result.is_error)
        loop._fire("on_tool_end", block.name, content_str, result.is_error)

        # ── Trace tool call ────────────────────────────────
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

    # ── Response-aware stuck-loop detection ────────────────
    # Check AFTER execution so we can compare responses.  A tool
    # is only "stuck" when the same (tool, args) returns the same
    # response repeatedly — polling tools like check_background
    # naturally return different responses (elapsed_seconds etc.)
    # and are never flagged.
    #
    # Two correctness rules (P0 fixes, 2026-04):
    #   1. Hash the RAW tool output, not the UI-truncated string.
    #      Two different long outputs that happen to share a
    #      common prefix up to max_tool_result_chars would
    #      collide on the truncated string and look "stuck".
    #   2. Deduplicate (name, args) keys *within a single batch*.
    #      A legitimate parallel fanout of e.g. 3 identical
    #      read_file calls in one step is not a stuck loop — it's
    #      the model doing simultaneous reads.  Without this, such
    #      a batch alone can hit threshold=3 in a single step.
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
    if stuck_feedback:
        loop._fire("on_warning", stuck_feedback)
        cost = cost_tracker.cost if cost_tracker else 0.0
        if trace:
            trace.add_span(step=step, event_type="dedup_break", success=False, error=stuck_feedback)
        return StepTerminate(_finalize_run(
            status="dedup_break",
            text=state.all_text or "",
            final_text=state.final_text,
            messages=messages,
            steps=step + 1,
            total_input_tokens=state.total_input_tokens,
            total_output_tokens=state.total_output_tokens,
            tool_calls_count=state.tool_calls_count,
            error=stuck_feedback,
            trace=trace,
            trace_total_cost_usd=cost or 0.0,
        ))

    # ── Cooperative cancellation check (after tools) ─────
    if loop._is_cancelled():
        return StepBreak(reason="cancelled")

    # ── Mid-loop reflection / critic step ─────────────────
    # After meaningful work boundaries (every-N cadence or
    # sub-agent return), append a short progress-check user
    # message so the next LLM call re-grounds on the task.
    # Avoids drift on long-horizon rollouts.
    if state.reflection_module is not None:
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
        if inject:
            prompt = reflection.build_reflection_prompt(
                reason, state.tool_calls_count,
            )
            messages.append({"role": "user", "content": prompt})
            state.last_reflection_count = state.tool_calls_count
            loop._fire("on_reflection", reason)
            if trace:
                trace.add_span(step=step, event_type="reflection")

    # ── Periodic checkpoint ──────────────────────────────
    # At the end of each step, if a cadence is configured and
    # we're on the boundary, persist state so crashes can
    # resume from here.  No-op when checkpoint_dir is None.
    if (
        cfg.checkpoint_every_n_steps > 0
        and cfg.checkpoint_dir
        and (step + 1) % cfg.checkpoint_every_n_steps == 0
    ):
        loop._write_checkpoint(
            step=step,
            messages=messages,
            total_input_tokens=state.total_input_tokens,
            total_output_tokens=state.total_output_tokens,
            tool_calls_count=state.tool_calls_count,
            status="in_progress",
        )

    return StepContinue()
