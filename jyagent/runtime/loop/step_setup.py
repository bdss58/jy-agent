"""Per-step setup helpers.

Owns the helpers that run BEFORE each LLM call:

  * ``_prepare_step_batch``       — refresh tools_batch + write_todos overlay
  * ``_compact_and_build_context`` — context compaction + final messages slice
  * ``_build_step_options``        — phase-aware LLMOptions construction

Extracted from ``runtime/loop/step.py`` (2026-05-06) so the per-step
coordinator can stay readable.  Re-exported from ``step.py`` so existing
``from jyagent.runtime.loop.step import _build_step_options`` callers
(notably ``tests/test_phases.py`` which uses ``inspect.getsource``) keep
working unchanged.

Layering: imports from ``step_state`` (RunState) and the leaf utility
modules (``compaction``, ``llm_runner``, ``todos``, ``verification``);
NEVER imports from ``step.py`` itself.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .llm_types import LLMOptions
from ..tools.registry import get_registry, ToolBatch
from .step_state import RunState

if TYPE_CHECKING:
    from .run_context import RunContext


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
