# loop_engine.py — Reusable agentic tool-use loop engine.
#
# Shared algorithm for both planner (streaming, full-featured) and sub-agent
# (non-streaming, silent).  Callers configure behaviour via LoopConfig and
# LoopCallbacks; the engine never writes to stdout directly.

from __future__ import annotations

import collections
import logging
import random
import threading
from typing import Any, Callable

# Behavioural dependency: the runtime engine consumes an `LLMClient`
# (Protocol).  Concrete provider classes such as `jyagent.llm.LLMOwner`
# satisfy the Protocol structurally — no inheritance required.
from .llm_client import LLMClient

# Value-type dependency: `LLMOptions` and `ModelSpec` are bag-of-fields
# dataclasses the engine constructs (in `_build_runtime_options`) and
# threads through sub-agent tier swaps.  They live under the runtime
# package itself (`runtime.loop.llm_types`) — provider packages
# re-export from `jyagent.llm.types` for backward compat.  After this
# move, the runtime has **zero** runtime-import of `jyagent.llm`.
from .llm_types import LLMOptions, ModelSpec, ToolCallRequest
from ...config import get_reasoning_config_for_provider, STREAM_TIMEOUT, MAX_TOOL_USE_INPUT_CHARS
from ..tools.registry import get_registry, ToolBatch
from ..tools.result import ToolResult
from ..tools.validation import validate_tool_input
from ...memory.conversation import estimate_conversation_tokens
from .remediation import enrich_error
from .tracing import get_tracer
from .verification import should_verify, build_verification_prompt
from .callbacks import LoopCallbacks  # re-exported for back-compat
from .config import LoopConfig, LoopResult  # re-exported for back-compat
from ._thread_helpers import LoopThreadHelper  # cancel/_fire helpers


_logger = logging.getLogger(__name__)


# ─── Core types ──────────────────────────────────────────────────────────────
# ``ToolCallRequest`` lives in ``runtime/loop/llm_types.py`` (sibling to
# ``LLMOptions`` and ``ModelSpec``).  The import above re-exports it here
# so legacy ``from jyagent.runtime.loop.engine import ToolCallRequest``
# imports (tests, out-of-tree callers) continue to work unchanged.  The
# canonical location is ``llm_types`` — new code should import from
# there.

# Type alias: returns (schemas_list, functions_dict)
ToolSource = Callable[[], tuple[list[dict], dict[str, Callable]]]


def _t_as_dict(t: Any) -> dict:
    """Best-effort TodoItem → dict.  Tolerates raw dicts already."""
    if isinstance(t, dict):
        return t
    try:
        from dataclasses import asdict
        return asdict(t)
    except Exception:
        return {"content": str(getattr(t, "content", t))}


# ─── Shared dispatch executor ────────────────────────────────────────────────
# The shared tool-dispatch pool, its lazy-grow helper, and the
# tool-execution helpers live in ``runtime/loop/tool_executor.py``.
# Engine imports the public names directly — the per-step body in
# ``runtime/loop/step.py::run_step`` calls ``tool_executor.execute_tools``
# directly, so any patching/monkeypatching for tests should target
# ``jyagent.runtime.loop.tool_executor`` (not ``engine``).
from .tool_executor import (  # noqa: E402
    execute_tool_with_timeout,
    get_tool_dispatch_executor,
)


# ─── Private helpers ─────────────────────────────────────────────────────────


# ─── Harness helpers ─────────────────────────────────────────────────────────

# Cost-accounting helpers live in runtime/loop/cost.py.
from .cost import CostTracker  # noqa: E402


# Stuck-loop detector lives in runtime/loop/stuck_loop.py.
from .stuck_loop import StuckLoopDetector  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────


# LLM extraction helpers live in runtime/loop/llm_runner.py.
from .llm_runner import (
    is_transient_error,
    build_runtime_options,
)


def _is_truncated(stop_reason: str, tool_calls: list[ToolCallRequest]) -> bool:
    """Detect if a response was truncated while emitting tool calls."""
    return stop_reason == "length" and bool(tool_calls)


def _strip_dangling_verification(messages: list) -> None:
    """Remove a trailing unanswered ``[VERIFICATION]`` user message in-place.

    The verification gate appends a user prompt asking the model to self-
    check before returning.  If the loop exits before the model replies
    (max_steps, KeyboardInterrupt, uncaught exception), that unanswered user
    message would leak into the persisted session and poison the next turn.

    Idempotent: safe to call on every terminal path regardless of whether a
    verification was actually injected.  This is why the canonical exit
    helper ``_finalize_run`` calls it unconditionally — gating on a
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


def _finalize_run(
    *,
    status: str,
    text: str,
    final_text: str,
    messages: list,
    steps: int,
    total_input_tokens: int,
    total_output_tokens: int,
    tool_calls_count: int,
    total_cache_creation_tokens: int = 0,
    total_cache_read_tokens: int = 0,
    api_calls: int = 0,
    error: str | None = None,
    trace=None,
    trace_status: str | None = None,
    trace_total_steps: int | None = None,
    trace_total_cost_usd: float | None = None,
) -> LoopResult:
    """Centralized exit path for ``_run_impl``.

    Every ``return LoopResult(...)`` in the loop must funnel through here so
    that:

      1. Dangling ``[VERIFICATION]`` user messages are *always* stripped
         (idempotent — see ``_strip_dangling_verification``).  Historically
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
    _strip_dangling_verification(messages)
    if trace is not None:
        finish_kwargs: dict = {
            "status": trace_status or status,
            "total_steps": trace_total_steps if trace_total_steps is not None else steps,
        }
        if trace_total_cost_usd is not None:
            finish_kwargs["total_cost_usd"] = trace_total_cost_usd
        # Tracing must never fail-close a
        # successful run.  Disk-full / read-only fs / permission errors here
        # used to bubble up and discard the entire LoopResult.  Log + swallow
        # so observability stays non-fatal.
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


# Compaction helpers live in runtime/loop/compaction.py.
from .compaction import (  # noqa: E402
    truncate_tool_call_blocks,
)


# ─── AgentLoop ───────────────────────────────────────────────────────────────

class AgentLoop(LoopThreadHelper):
    """Reusable agentic tool-use loop engine.

    Supports both streaming and non-streaming modes, concurrent tool execution,
    context compaction, truncation recovery, and transient-error retry.

    Inherits ``_is_cancelled`` / ``_cancellable_sleep`` / ``_fire`` from
    ``LoopThreadHelper``.  Default helper
    attribute names (``_cancel_event`` / ``_callbacks``) match the
    instance attributes set in ``__init__``, so no override is needed.
    """

    def __init__(
        self,
        runtime_owner: LLMClient,
        config: LoopConfig,
        callbacks: LoopCallbacks | None = None,
        tool_source: ToolSource | None = None,
        model_spec: ModelSpec | None = None,
        cancel_event: threading.Event | None = None,
        session_id: str | None = None,
    ):
        self._runtime_owner = runtime_owner
        self._config = config
        self._callbacks = callbacks or LoopCallbacks()
        self._tool_source = tool_source
        self._model_spec = model_spec  # override for sub-agent model tier
        self._cancel_event = cancel_event
        self._session_id = session_id or ""
        # Reuse the module-level shared executor to avoid accumulating
        # ThreadPoolExecutor objects and atexit handlers across turns and
        # sub-agent dispatches.  Ensure the pool is at least as
        # wide as the configured ``max_tool_workers`` (the historical
        # singleton was hard-capped at 8, silently throttling configs
        # that asked for more dispatch parallelism).
        self._executor = get_tool_dispatch_executor(config.max_tool_workers)
        # Task-plan scratchpad (see jyagent/todos.py).  Populated via the
        # `write_todos` tool and seeded optionally via run(initial_todos=...)
        # so outer layers can carry the plan across turns.
        self._todos: list = []
        # Accumulator for mutating-tool
        # timeouts.  Populated by ``_execute_tool_with_timeout`` via the
        # ``partial_side_effects=`` kwarg threaded through ``_execute_tools``;
        # snapshotted onto ``LoopResult.partial_side_effects`` in ``run()``.
        # Reset at the top of ``_run_impl`` so back-to-back .run() calls on
        # the same AgentLoop instance don't bleed state across turns.
        #
        # Backed by ``collections.deque``
        # because parallel-safe tool batches can fan out across multiple
        # daemon threads, each of which may hit a timeout simultaneously
        # and call ``.append(name)`` from its own worker.  Under PEP 703
        # free-threaded CPython (3.13t / 3.14t) ``list.append`` is no
        # longer atomic w.r.t. concurrent mutation; ``deque.append`` IS
        # documented thread-safe for single-element ops in the stdlib
        # (see CPython Issue #117 and the collections module docs).
        # ``list(deque(...))`` is a normal iteration — the snapshot
        # in ``run()`` still works unchanged.
        self._partial_side_effects: collections.deque[str] = collections.deque()
        # AgentLoop holds substantial per-run
        # state on the instance (_todos, _run_id, _partial_side_effects,
        # closures, etc).  Concurrent .run() calls on a single instance
        # would silently corrupt that state — they'd share the todo list,
        # the same checkpoint identity, and the same mutating-timeout
        # accumulator.  Enforce single-run ownership with an exclusive
        # threading.Lock acquired non-blockingly at the top of run() so the
        # second caller sees a clear RuntimeError instead of a silent race.
        # ``threading.Lock`` (not RLock) by design: even nested .run() from
        # the same thread is wrong (the inner call would clobber the outer
        # call's _partial_side_effects on entry).
        self._run_lock = threading.Lock()
        # Run id for checkpointing.  Fresh per AgentLoop; outer layers can
        # override via `set_run_id()` before calling run() to correlate
        # checkpoints with an external request/session.
        self._run_id: str = ""

    def set_run_id(self, run_id: str) -> None:
        """Override the run id used by checkpoint paths.  Must be called
        before ``run()``.  Empty string clears the run id (resets to ``''``)."""
        self._run_id = run_id or ""

    # ``_is_cancelled``, ``_cancellable_sleep``, and ``_fire`` are inherited
    # from ``LoopThreadHelper`` (see ``_thread_helpers.py``), shared with
    # LLMRunner where they were previously duplicated verbatim.

    def _write_checkpoint(
        self,
        *,
        step: int | str,
        messages: list,
        total_input_tokens: int,
        total_output_tokens: int,
        tool_calls_count: int,
        status: str,
        error: str | None = None,
    ) -> None:
        """Persist a LoopCheckpoint if checkpointing is enabled.

        ``step`` may be an int (regular step boundary) or ``"final"``
        (terminal exit).  Errors are logged via ``on_warning`` — never
        propagated, checkpointing must never break a run.
        """
        cfg = self._config
        if not cfg.checkpoint_dir:
            return
        from .checkpoint import (
            LoopCheckpoint,
            checkpoint_path,
            iso_utc_now,
        )
        effective_spec = self._model_spec or self._runtime_owner.model_spec
        try:
            cp = LoopCheckpoint(
                run_id=self._run_id,
                step=step if isinstance(step, int) else -1,
                saved_at=iso_utc_now(),
                messages=list(messages),
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                tool_calls_count=tool_calls_count,
                todos=[_t_as_dict(t) for t in self._todos] if cfg.todos_enabled else [],
                provider=effective_spec.provider,
                model=effective_spec.model,
                status=status,
                error=error,
            )
            path = checkpoint_path(cfg.checkpoint_dir, self._run_id, step)
            cp.save(path)
            self._fire(
                "on_checkpoint", path,
                step if isinstance(step, int) else -1,
            )
        except Exception as e:
            self._fire("on_warning", f"checkpoint write failed: {e}")

    # ── callback helpers ──────────────────────────────────────────────────
    # ``_fire`` is provided by ``LoopThreadHelper``.

    # ── public entry point ────────────────────────────────────────────────

    def run(
        self,
        system_prompt: str,
        messages: list,
        initial_todos: list | None = None,
    ) -> LoopResult:
        """Run the agentic tool-use loop.  *messages* is mutated in-place.

        Thin wrapper around ``_run_impl`` that attaches the final todos
        scratchpad and writes a terminal checkpoint (if enabled),
        regardless of which exit path fired.

        Raises ``RuntimeError`` if a previous ``run()`` on this instance
        is still in flight: AgentLoop
        owns per-run mutable state (_todos, _run_id, _partial_side_effects,
        closures over _todos) that concurrent or re-entrant runs would
        silently corrupt.
        """
        # Lazy-init: test utilities that build AgentLoop via __new__ to
        # skip __init__ don't set _run_lock.  Installing it on first call
        # is safe because the very first .run() caller has exclusive
        # access by construction.
        if not hasattr(self, "_run_lock"):
            self._run_lock = threading.Lock()
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError(
                "AgentLoop.run() is already in progress on this instance. "
                "AgentLoop is not reentrant; construct a new AgentLoop "
                "(or wait for the previous run to return) before calling "
                "run() again."
            )
        try:
            result = self._run_impl(system_prompt, messages, initial_todos)
            if self._config.todos_enabled:
                # Serialize to dict-form for easy JSON persistence by outer layers.
                from .todos import todo_to_dict
                result.todos = [todo_to_dict(t) for t in self._todos]
            # Mirror the todos pattern — snapshot
            # the mutating-timeout accumulator onto the result so every exit
            # path benefits without having to thread the list through every
            # _finalize_run() call site.  Copy defensively so a caller that
            # retains the returned list can't mutate the AgentLoop's internal
            # state on the next run.
            result.partial_side_effects = list(self._partial_side_effects)
            if self._config.checkpoint_dir:
                # Terminal ("final") checkpoint — includes status + error.
                self._write_checkpoint(
                    step="final",
                    messages=result.messages,
                    total_input_tokens=result.total_input_tokens,
                    total_output_tokens=result.total_output_tokens,
                    tool_calls_count=result.tool_calls_count,
                    status=result.status,
                    error=result.error,
                )
            return result
        finally:
            # Release the reentrance guard regardless of how _run_impl
            # exited (return, raise, KeyboardInterrupt) so a subsequent
            # run() on the same instance is not deadlocked.
            self._run_lock.release()

    def _run_impl(
        self,
        system_prompt: str,
        messages: list,
        initial_todos: list | None = None,
    ) -> LoopResult:
        """Core run loop.  Public entry point is ``run()`` which also
        snapshots the final todos onto the result.

        This method is a thin orchestrator: setup is in
        ``RunState.prepare_for_run()``, the per-step body is in
        ``runtime/loop/step.py::run_step``, and only the for-step counter,
        post-loop terminal handlers (cancelled-exit / max_steps fallback /
        max_steps exit), and the outer try/except live here.
        """
        from .step import RunState, run_step, StepContinue, StepTerminate, StepBreak

        cfg = self._config
        state = RunState.prepare_for_run(self, system_prompt, messages, initial_todos)
        # Aliases for the post-loop terminal handlers below.  ``trace`` is
        # threaded into 5 ``_finalize_run`` calls and ``cost_tracker`` into
        # 7 lexical sites, so the locals earn their keep on readability.
        # ``effective_spec`` is read inline from ``state`` at its single
        # use site.
        trace = state.trace
        cost_tracker = state.cost_tracker

        try:
            for step in range(cfg.max_steps):
                state.step = step
                outcome = run_step(self, state)
                if isinstance(outcome, StepTerminate):
                    return outcome.result
                if isinstance(outcome, StepBreak):
                    # Cooperative cancellation requested by ``run_step``
                    # (cancel checked at top of step or before/after tools).
                    # Fall through to the cancelled-exit handler below.
                    break
                # Defense-in-depth: every other ``run_step`` return must be
                # ``StepContinue``.  Any future tagged-union member would
                # silently fall through to the next iteration without this
                # check.
                assert isinstance(outcome, StepContinue), (
                    f"run_step returned unknown outcome type: {type(outcome).__name__}"
                )

            # ── Cooperative cancellation — early exit ────────────────
            if self._is_cancelled():
                return _finalize_run(
                    status="interrupted",
                    text=state.all_text or "",
                    final_text=state.final_text,
                    messages=messages,
                    steps=state.step + 1,
                    total_input_tokens=state.total_input_tokens,
                    total_output_tokens=state.total_output_tokens,
                    tool_calls_count=state.tool_calls_count,
                    total_cache_creation_tokens=state.total_cache_creation_tokens,
                    total_cache_read_tokens=state.total_cache_read_tokens,
                    api_calls=state.api_calls,
                    trace=trace,
                )

            # Max steps reached
            # Fallback always fires when enabled: reaching max_steps means the
            # loop never hit a no-tool terminal step, so the incidental text
            # accumulated from prior tool-use steps is NOT a real answer.
            # (Old condition `not final_text` was wrong — `final_text` is
            # written on every step including ones that also had tool calls.)
            #
            # Defense-in-depth: the canonical _finalize_run() path always
            # strips dangling [VERIFICATION] (idempotently), so we no longer
            # need a guarded pre-strip here.  The boundary guard at the
            # gate (step + 1 < cfg.max_steps) should already prevent the
            # leak, but _finalize_run cleans up unconditionally as belt-
            # and-suspenders.

            if cfg.fallback_on_max_steps:
                # Try one more call with a finalize directive.  Preserves the
                # Anthropic prompt cache by leaving ``system_prompt`` byte-
                # identical and injecting the directive as a tail user
                # message — see MEMORY.md (durable rule):
                #   "Mutating Anthropic system_prompt breaks prompt caching —
                #    inject dynamic context as a non-persisted tail message
                #    block instead."
                # The previous implementation concatenated the directive into
                # ``system_prompt``, which broke the cached prefix on this
                # terminal turn (~12× cost penalty on the cached portion).
                try:
                    finalize_directive = {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": (
                                "[SYSTEM: You have reached the maximum number "
                                "of tool-use steps. Please provide your best "
                                "answer now WITHOUT using any tools.]"
                            ),
                        }],
                    }
                    # Transient view — do NOT mutate ``messages`` until the
                    # call succeeds, so a fallback failure can fall through
                    # to the normal max_steps exit with clean history.
                    fallback_messages = messages + [finalize_directive]
                    fallback_context = {
                        "system_prompt": system_prompt,  # unchanged — cache stays warm
                        "messages": fallback_messages,
                    }

                    # Create fallback options with tool_choice=none
                    _base = build_runtime_options(
                        self._runtime_owner,
                        cfg.initial_max_tokens,
                        model_spec=self._model_spec,
                        metadata={"component": "loop_engine", "step": cfg.max_steps + 1, "fallback": True},
                        session_id=self._session_id,
                    )
                    fallback_opts = LLMOptions(
                        max_output_tokens=_base.max_output_tokens,
                        timeout=_base.timeout,
                        reasoning=_base.reasoning,
                        metadata=_base.metadata,
                        tool_choice={"type": "none"},
                    )

                    if cfg.streaming:
                        fallback_text, _, _, fallback_message = self._call_streaming(fallback_context, fallback_opts)
                    else:
                        fallback_text, _, _, fallback_message = self._call_complete(fallback_context, fallback_opts)

                    # Accumulate usage
                    usage = fallback_message.get("usage", {})
                    state.total_input_tokens += usage.get("input_tokens", 0)
                    state.total_output_tokens += usage.get("output_tokens", 0)
                    state.total_cache_creation_tokens += usage.get("cache_creation_input_tokens", 0) or 0
                    state.total_cache_read_tokens += usage.get("cache_read_input_tokens", 0) or 0
                    if usage:
                        state.api_calls += 1
                    self._fire("on_usage", usage)
                    # The fallback call's
                    # tokens were being added to the totals reported on
                    # LoopResult, but ``cost_tracker`` was never updated —
                    # so ``trace_total_cost_usd`` (read three lines below
                    # from ``cost_tracker.cost``) silently under-counted by
                    # whatever the fallback turn cost.  Record here using
                    # the same effective_spec the rest of the loop uses, so
                    # sub-agent tier overrides bill at the right rate.
                    if cost_tracker is not None:
                        cost_tracker.record(
                            usage,
                            state.effective_spec.provider,
                            state.effective_spec.model,
                        )
                        if cost_tracker.has_unpriced_usage and not state.unpriced_warned:
                            state.unpriced_warned = True
                            self._fire(
                                "on_warning",
                                "cost_tracker: at least one call lacked pricing data; "
                                "budget enforcement uses the priced subtotal only",
                            )

                    # Apply truncation if enabled — uses the last step_batch
                    # built by run_step (threaded via state.last_step_batch).
                    if cfg.truncate_large_inputs:
                        content = fallback_message.get("content", [])
                        fallback_message = dict(fallback_message)
                        fallback_message["content"] = truncate_tool_call_blocks(content, state.last_step_batch)

                    # Append fallback turn — directive first, then the
                    # assistant reply — so the persisted transcript stays
                    # symmetric (every assistant message answers a real
                    # preceding user message).
                    messages.append(finalize_directive)
                    messages.append(fallback_message)

                    # Return completed since we got a final answer.
                    # Note: previously this path skipped trace.finish() — the
                    # max_steps trace block below was unreachable on success.
                    cost = cost_tracker.cost if cost_tracker else 0.0
                    return _finalize_run(
                        status="completed",
                        text=fallback_text or state.all_text,
                        final_text=fallback_text,
                        messages=messages,
                        steps=cfg.max_steps,
                        total_input_tokens=state.total_input_tokens,
                        total_output_tokens=state.total_output_tokens,
                        tool_calls_count=state.tool_calls_count,
                        total_cache_creation_tokens=state.total_cache_creation_tokens,
                        total_cache_read_tokens=state.total_cache_read_tokens,
                        api_calls=state.api_calls,
                        trace=trace,
                        trace_total_cost_usd=cost or 0.0,
                    )
                except KeyboardInterrupt:
                    raise
                except Exception:
                    # If fallback fails, fall through to normal max_steps handling
                    pass

            # ── max_steps exit ─────────────────────────────────────────
            cost = cost_tracker.cost if cost_tracker else 0.0
            return _finalize_run(
                status="max_steps",
                text=state.all_text or "",
                final_text=state.final_text,
                messages=messages,
                steps=cfg.max_steps,
                total_input_tokens=state.total_input_tokens,
                total_output_tokens=state.total_output_tokens,
                tool_calls_count=state.tool_calls_count,
                total_cache_creation_tokens=state.total_cache_creation_tokens,
                total_cache_read_tokens=state.total_cache_read_tokens,
                api_calls=state.api_calls,
                trace=trace,
                trace_total_cost_usd=cost or 0.0,
            )

        except KeyboardInterrupt:
            return _finalize_run(
                status="interrupted",
                text=state.all_text + "\n\n[Interrupted by user]" if state.all_text else "[Interrupted by user]",
                final_text="",
                messages=messages,
                steps=state.step + 1,
                total_input_tokens=state.total_input_tokens,
                total_output_tokens=state.total_output_tokens,
                tool_calls_count=state.tool_calls_count,
                total_cache_creation_tokens=state.total_cache_creation_tokens,
                total_cache_read_tokens=state.total_cache_read_tokens,
                api_calls=state.api_calls,
                trace=trace,
            )
        except Exception as e:
            return _finalize_run(
                status="error",
                text=state.all_text or "",
                final_text="",
                messages=messages,
                steps=state.step + 1,
                total_input_tokens=state.total_input_tokens,
                total_output_tokens=state.total_output_tokens,
                tool_calls_count=state.tool_calls_count,
                total_cache_creation_tokens=state.total_cache_creation_tokens,
                total_cache_read_tokens=state.total_cache_read_tokens,
                api_calls=state.api_calls,
                error=str(e),
                trace=trace,
            )


    # ── LLM call + retry/fallback (delegated to LLMRunner) ──────────────
    #
    # The call machinery lives in
    # ``llm_runner.LLMRunner``.  These methods are kept as thin delegates
    # because internal code and tests call them by name (and tests
    # monkeypatch them on the instance).  The runner is created lazily on
    # first use so subclasses / tests that mutate ``self._runtime_owner``,
    # ``self._config``, ``self._callbacks``, ``self._cancel_event``, or
    # ``self._model_spec`` after ``__init__`` still see the new values —
    # one-shot build, then cached for the remainder of the instance's life.

    def _get_llm_runner(self):
        """Return (and memoise) the per-instance ``LLMRunner``.

        Built on first demand so post-__init__ swaps of runtime_owner /
        callbacks / cancel_event / model_spec are visible.  Once built, the
        runner is reused for the rest of the AgentLoop's lifetime.
        """
        from .llm_runner import LLMRunner
        runner = getattr(self, "_llm_runner_cached", None)
        if runner is None:
            runner = LLMRunner(
                runtime_owner=self._runtime_owner,
                config=self._config,
                callbacks=self._callbacks,
                cancel_event=self._cancel_event,
                model_spec=self._model_spec,
            )
            self._llm_runner_cached = runner
        return runner

    def _call_llm_with_retry(
        self,
        context: dict,
        options: LLMOptions,
        step: int,
    ) -> tuple[str, list[ToolCallRequest], str, dict]:
        """Call the LLM (streaming or complete) with transient-error retry.

        We keep the retry loop in AgentLoop (rather than routing
        straight to ``LLMRunner.call_with_retry``) so that it dispatches
        through ``self._call_streaming`` / ``self._call_complete``.  Several
        tests and internal diagnostics override those methods on a subclass
        or monkeypatch them on the instance to inject transient failures —
        that contract is preserved.

        Returns (step_text, tool_call_blocks, stop_reason, final_message).
        """
        cfg = self._config
        last_error: BaseException | None = None

        for attempt in range(cfg.retry_attempts + 1):
            try:
                if cfg.streaming:
                    return self._call_streaming(context, options)
                else:
                    return self._call_complete(context, options)
            except KeyboardInterrupt:
                raise
            except Exception as err:
                last_error = err
                transient = is_transient_error(err)
                should_retry = (
                    (transient or cfg.retry_on_all_errors)
                    and attempt < cfg.retry_attempts
                )
                if should_retry:
                    if self._is_cancelled():
                        raise
                    # Exponential backoff with "equal jitter" (AWS
                    # architecture recommendation) to avoid thundering-herd
                    # when multiple parallel sub-agents all retry a 529 at
                    # the same moment:
                    #   half the delay is deterministic exponential,
                    #   half is uniform random in [0, base * 2^attempt / 2].
                    base = cfg.retry_base_delay * (2 ** attempt)
                    delay = base / 2 + random.uniform(0, base / 2)
                    self._fire("on_retry", attempt + 1, err)
                    # Signal UI that any partial output from the failed
                    # attempt will be replayed on retry (visual
                    # de-duplication hook).  ``partial_stream_text`` is
                    # stashed by ``_call_streaming`` on the exception;
                    # missing for the non-streaming path.
                    partial_text = getattr(err, "partial_stream_text", "")
                    reason = "transient_error" if transient else "error"
                    self._fire("on_stream_retry", reason, partial_text)
                    # Cancel-aware backoff: wake immediately on Ctrl-C so we
                    # don't burn through a long retry window after cancel.
                    if self._cancellable_sleep(delay):
                        raise KeyboardInterrupt("cancelled during retry backoff")
                    continue
                raise

        # Should not reach here, but just in case:
        raise last_error  # type: ignore[misc]

    def _call_complete(
        self,
        context: dict,
        options: LLMOptions,
    ) -> tuple[str, list[ToolCallRequest], str, dict]:
        """Thin delegate → ``LLMRunner.call_complete``."""
        return self._get_llm_runner().call_complete(context, options)

    def _call_streaming(
        self,
        context: dict,
        options: LLMOptions,
    ) -> tuple[str, list[ToolCallRequest], str, dict]:
        """Thin delegate → ``LLMRunner.call_streaming``."""
        return self._get_llm_runner().call_streaming(context, options)
