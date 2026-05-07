"""Per-run mutable state + tagged-union step outcome.

Extracted from ``runtime/loop/step.py`` (2026-05-06) so the per-step body
in ``step.py`` can focus on the loop coordinator and helpers can import
``RunState`` / ``StepOutcome`` from a single low-level module without
risking import cycles back through ``step.py``.

The original module-level definitions are also re-exported from
``step.py`` for back-compat — every existing
``from jyagent.runtime.loop.step import RunState, StepBreak, ...`` import
keeps working unchanged.

Layering: helper modules (step_setup, step_tools, step_bookkeeping)
import from this module, never from ``step.py``.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING, Union

from .llm_types import LLMOptions
from ..tools.registry import ToolBatch

if TYPE_CHECKING:
    from .config import LoopResult
    from .cost import CostTracker
    from .run_context import RunContext



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

