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

import collections
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, Union, runtime_checkable

from .llm_types import LLMOptions
from .tool_dispatch import execute_tools
from ..tools.registry import ToolBatch, get_registry
from ..tools.result import ToolResult

if TYPE_CHECKING:
    from .callbacks import LoopCallbacks
    from .config import LoopConfig, LoopResult
    from .llm_client import LLMClient
    from .llm_types import ModelSpec, ToolCallRequest


# ─── Run-context contract ────────────────────────────────────────────────────

ToolSource = Callable[[], tuple[list[dict], dict[str, Callable]]]


@runtime_checkable
class RunContext(Protocol):
    """Structural contract for objects that drive ``run_step``."""

    _config: "LoopConfig"
    _callbacks: "LoopCallbacks"
    _runtime_owner: "LLMClient"
    _model_spec: "ModelSpec | None"
    _tool_source: "ToolSource | None"
    _executor: ThreadPoolExecutor
    _cancel_event: "threading.Event | None"
    _partial_side_effects: "collections.deque[str]"
    _run_id: str
    _todos: list

    def _fire(self, event_name: str, *args: Any) -> None:
        ...

    def _fire_with_return(self, event_name: str, *args: Any) -> Any:
        ...

    def _is_cancelled(self) -> bool:
        ...

    def _call_llm_with_retry(
        self,
        context: dict,
        options: "LLMOptions",
        step: int,
    ) -> "tuple[str, list[ToolCallRequest], str, dict]":
        ...

    def _write_checkpoint(
        self,
        *,
        step: "int | str",
        messages: list,
        total_input_tokens: int,
        total_output_tokens: int,
        tool_calls_count: int,
        status: str,
        total_cache_creation_tokens: int = 0,
        total_cache_read_tokens: int = 0,
        api_calls: int = 0,
        error: "str | None" = None,
    ) -> None:
        ...


# ─── Run-scoped mutable state + outcome types ────────────────────────────────


@dataclass
class RunState:
    """Mutable per-run state threaded through every ``run_step`` call."""

    system_prompt: str
    messages: list
    turn_start_idx: int
    step: int = 0
    current_max_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    api_calls: int = 0
    tool_calls_count: int = 0
    last_reflection_count: int = 0
    consecutive_truncations: int = 0
    max_truncation_retries: int = 3
    last_verification_idx: int | None = None

    # Prose-shaped tool-call recovery counter (Bug A — see finalize.py).
    # Incremented each time the no-tool branch detects assistant text that
    # looks like a malformed tool invocation and injects a corrective
    # ``[MALFORMED_TOOL_CALL]`` user message instead of terminating the
    # turn.  The gate only fires while this counter is below
    # ``max_prose_tool_call_corrections`` — past that, the loop accepts
    # the turn as terminal rather than re-prompting forever (e.g. against
    # a model that genuinely insists on emitting pseudo-syntax).
    prose_tool_call_corrections: int = 0
    max_prose_tool_call_corrections: int = 2

    unpriced_warned: bool = False
    all_text: str = ""
    final_text: str = ""
    cost_tracker: "Any | None" = None
    stuck_detector: Any = None
    tools_batch: ToolBatch = field(default_factory=ToolBatch.empty)
    trace: Any = None
    effective_spec: Any = None
    write_todos_fn: Any = None
    reflection_module: Any = None
    last_step_batch: ToolBatch = field(default_factory=ToolBatch.empty)

    @classmethod
    def prepare_for_run(
        cls,
        loop: "RunContext",
        system_prompt: str,
        messages: list,
        initial_todos: list | None = None,
    ) -> "RunState":
        cfg = loop._config

        loop._partial_side_effects = collections.deque()
        turn_start_idx = len(messages)

        if cfg.reflect_every_n_tool_calls > 0 or cfg.reflect_after_subagent:
            from . import reflection as _reflection
            reflection_module = _reflection
        else:
            reflection_module = None

        if cfg.checkpoint_dir and not loop._run_id:
            from .checkpoint import new_run_id
            loop._run_id = new_run_id()

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
                pass
            else:
                loop._todos = []

            def _get_store() -> list:
                return loop._todos

            def _set_store(new_list: list) -> None:
                loop._todos = new_list

            write_todos_fn = build_write_todos_tool(_get_store, _set_store)

        effective_spec = loop._model_spec or loop._runtime_owner.model_spec

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


@dataclass(frozen=True)
class StepContinue:
    pass


@dataclass(frozen=True)
class StepTerminate:
    result: "LoopResult"


@dataclass(frozen=True)
class StepBreak:
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
    from .finalize import (
        build_prose_tool_call_correction,
        finalize_run,
        is_truncated,
        looks_like_prose_tool_call,
    )
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
        # ── Prose-shaped tool-call gate (Bug A — see finalize.py) ─────────
        #
        # If the assistant text looks like a malformed tool invocation
        # (e.g. ``[Tool call: run_shell]{...}`` or a ```tool_use``` fence)
        # but no real structured tool_use block was emitted, the model
        # most likely TRIED to call a tool and failed.  Terminating the
        # turn here would silently swallow the user's request — the
        # exact failure mode that motivated this gate.
        #
        # Remediation: append the assistant message (so the conversation
        # records what was attempted), inject a corrective user message
        # explaining the failure, and continue the loop so the model
        # can retry — either with a real tool call or by switching to
        # plain prose.
        #
        # Capped retries (``max_prose_tool_call_corrections``) prevent
        # an infinite re-prompt loop if the model insists on the
        # pseudo-syntax — past the cap we accept the turn as terminal.
        # The cap is per-run, not per-step: once exceeded, subsequent
        # prose-shaped attempts in the same run terminate normally.
        if (
            looks_like_prose_tool_call(step_text)
            and state.prose_tool_call_corrections < state.max_prose_tool_call_corrections
            and step + 1 < cfg.max_steps
        ):
            if trace:
                trace.add_span(step=step, event_type="prose_tool_call_correction")
            # Persist the assistant's malformed attempt so the model sees
            # its own previous message + the correction in the next turn.
            messages.append(final_message)
            messages.append({
                "role": "user",
                "content": build_prose_tool_call_correction(),
            })
            state.prose_tool_call_corrections += 1
            return StepContinue()

        # Pre-completion verification gate: if NEW mutations have been
        # appended since the last verification (or since the turn started,
        # if no verification has fired yet), inject a self-check prompt
        # and loop once more instead of returning.
        #
        # Re-arming contract (latent-bug fix, 2026-05): verification can
        # fire MULTIPLE TIMES per run.  Previously a one-shot
        # ``verification_injected: bool`` flag locked the gate after the
        # first fire, so a "verify → model does more mutations → return"
        # pattern was never re-checked.  Now the gate scans messages from
        # ``max(turn_start_idx, last_verification_idx + 1)`` so each
        # verification only sees mutations newer than itself.  No
        # mutations newer than the last verification → gate stays closed
        # → loop terminates cleanly on the same step (idempotent — a
        # second call against the same suffix returns False).
        #
        # Boundary guard (P0 fix): never inject on the final allowed step —
        # the follow-up model reply has no iteration left to run, and the
        # dangling `[VERIFICATION]` user message would otherwise leak into
        # the persisted session and poison the next turn.
        verify_since = state.turn_start_idx
        if state.last_verification_idx is not None:
            verify_since = max(verify_since, state.last_verification_idx + 1)
        if (
            should_verify(
                messages,
                state.tool_calls_count,
                since_index=verify_since,
                batch=step_batch,
            )
            and step + 1 < cfg.max_steps
        ):
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
            # Record the index of the just-appended verification user
            # message so a subsequent gate evaluation only re-arms when
            # NEW mutations land at indices > this one.
            state.last_verification_idx = len(messages) - 1
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


# ─── Per-step setup helpers ─────────────────────────────────────────────────


def _prepare_step_batch(loop: "RunContext", state: RunState) -> ToolBatch:
    cfg = loop._config
    registry = get_registry()
    step = state.step

    if loop._tool_source is not None:
        src_schemas, src_functions = loop._tool_source()
        state.tools_batch = ToolBatch.from_source(
            src_schemas, src_functions, base=registry.freeze(),
        )
    elif step == 0 or registry.version != state.tools_batch.version:
        state.tools_batch = registry.freeze()

    if cfg.todos_enabled:
        from .todos import WRITE_TODOS_SCHEMA
        step_batch = state.tools_batch.with_overlay(
            functions={"write_todos": state.write_todos_fn},
            schemas=[WRITE_TODOS_SCHEMA],
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


# ─── Per-step tool-round semantics ──────────────────────────────────────────


def _execute_tool_round(
    loop: "RunContext",
    state: RunState,
    step_batch: ToolBatch,
    tool_call_blocks: list,
) -> tuple["StepOutcome | None", list]:
    from .compaction import truncate_result as _truncate_result
    from ..tools.result import ToolResult as _ToolResult

    cfg = loop._config
    messages = state.messages
    step = state.step
    trace = state.trace

    if len(tool_call_blocks) > 1:
        loop._fire("on_tool_batch", len(tool_call_blocks))

    for block in tool_call_blocks:
        loop._fire("on_tool_start", block.name, block.input)

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

    executed_results: list = []
    if allowed_blocks:
        tools_t0 = time.perf_counter()
        executed_results = execute_tools(
            allowed_blocks,
            step_batch,
            cfg.concurrent_tools,
            cfg.max_tool_workers,
            cfg.tool_timeout,
            executor=loop._executor,
            partial_side_effects=loop._partial_side_effects,
            cancel_event=loop._cancel_event,
        )
        _ = (time.perf_counter() - tools_t0) * 1000

    denial_msg = "Denied by user (on_tool_pre_execute gate)."
    denied_oids = {id(b) for b in denied_blocks}
    results_by_oid = {id(b): r for (b, r) in executed_results}

    tool_results_tuples: list = []
    for block in tool_call_blocks:
        if id(block) in denied_oids:
            result = _ToolResult(content=denial_msg, is_error=True)
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
    from .finalize import finalize_run
    from .stuck_loop import StuckLoopDetector

    step = state.step
    trace = state.trace
    cost_tracker = state.cost_tracker
    stuck_detector = state.stuck_detector

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
            result.content,
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
        messages=state.messages,
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


# ─── Per-step bookkeeping ───────────────────────────────────────────────────


def _record_llm_usage_and_cost(
    loop: "RunContext",
    state: RunState,
    final_message: dict,
    llm_dur_ms: float,
) -> StepOutcome | None:
    from .finalize import finalize_run

    cfg = loop._config
    step = state.step
    trace = state.trace
    cost_tracker = state.cost_tracker
    effective_spec = state.effective_spec
    messages = state.messages

    for warning in final_message.get("llm_warnings", []):
        loop._fire("on_warning", warning)

    usage = final_message.get("usage", {})
    state.total_input_tokens += usage.get("input_tokens", 0)
    state.total_output_tokens += usage.get("output_tokens", 0)
    state.total_cache_creation_tokens += usage.get("cache_creation_input_tokens", 0) or 0
    state.total_cache_read_tokens += usage.get("cache_read_input_tokens", 0) or 0
    if usage:
        state.api_calls += 1
    loop._fire("on_usage", usage)

    if trace:
        trace.add_span(
            step=step,
            event_type="llm_call",
            duration_ms=llm_dur_ms,
            tokens_in=usage.get("input_tokens"),
            tokens_out=usage.get("output_tokens"),
        )

    if cost_tracker is None:
        return None

    cost_tracker.record(usage, effective_spec.provider, effective_spec.model)

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
