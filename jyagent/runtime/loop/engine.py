# loop_engine.py — Reusable agentic tool-use loop engine.
#
# Shared algorithm for both planner (streaming, full-featured) and sub-agent
# (non-streaming, silent).  Callers configure behaviour via LoopConfig and
# LoopCallbacks; the engine never writes to stdout directly.

from __future__ import annotations

import atexit
import concurrent.futures
import logging
import hashlib
import json
import random
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
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
# move (Codex review 2026-04-25 Part 3 #5, follow-up commit), the
# runtime has **zero** runtime-import of `jyagent.llm`.
from .llm_types import LLMOptions, ModelSpec
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


_logger = logging.getLogger(__name__)


# ─── Core types ──────────────────────────────────────────────────────────────

@dataclass
class ToolCallRequest:
    id: str
    name: str
    input: dict


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
# C4 Phase 2 (codex review 2026-04-25): the shared tool-dispatch pool, its
# lazy-grow helper, and the ``_execute_tool*`` helpers moved to
# ``runtime/loop/tool_executor.py``.  Internal call sites still use the
# underscore-prefixed names via these aliases; tests that poke module
# globals (``_tool_dispatch_executor``, ``_tool_dispatch_cap``, etc.) also
# see them here via the PEP-562 ``__getattr__`` at the bottom of the file.
from .tool_executor import (  # noqa: E402
    execute_tool as _execute_tool,
    execute_tool_with_timeout as _execute_tool_with_timeout,
    execute_tools as _execute_tools,
    get_tool_dispatch_executor as _get_tool_dispatch_executor,
)
# The pool + lock + cap are MODULE STATE that ``get_tool_dispatch_executor``
# REBINDS when the pool grows.  A plain ``from .tool_executor import
# _tool_dispatch_executor`` would snapshot the pre-grow object and go stale.
# The engine-level back-compat names (``_tool_dispatch_executor``,
# ``_tool_dispatch_cap``, ``_tool_dispatch_lock``, ``_tool_executor``) are
# served by a module-level ``__getattr__`` (PEP 562) at the bottom of this
# file, which delegates to the live attribute on ``tool_executor``.


# ─── Private helpers ─────────────────────────────────────────────────────────


# ─── Harness helpers ─────────────────────────────────────────────────────────

# C4 Phase 1 (codex review 2026-04-25): extracted to runtime/loop/cost.py.
# Kept as a private alias so internal imports (`_CostTracker()`) continue
# to work without churn; phases 2-5 will similarly extract tool executor,
# LLM runner, compaction, and leave engine.py as just the LoopController.
from .cost import CostTracker as _CostTracker  # noqa: E402


class _StuckLoopDetector:
    """Detect stuck loops by tracking whether repeated calls yield new responses.

    Key insight: a loop is "stuck" only when the same tool call returns the
    same response **consecutively**.  Polling tools (``check_background``,
    ``take_snapshot``) naturally return changing responses (e.g. different
    ``elapsed_seconds``) — they are never flagged without any exemption metadata.

    Interleaved calls are also safe: if the agent alternates
    ``run_shell(A) → check_background → run_shell(A) → check_background``
    that's a polling pattern, not a stuck loop — even if ``run_shell(A)``
    returns the same result each time.  Only **truly consecutive** identical
    calls (``A → A → A``) trigger the detector.

    This replaces the old ``_DedupTracker`` which required a whitelist of
    ``dedup_exempt`` tools and a regex hack for ``sleep`` commands.

    Design:
        Track ``(tool_name, args_key) → (consecutive_identical_count, last_response_hash)``

        * If a **different** key was recorded since the last call to *this* key,
          the pattern is interleaved — reset the counter (not a stuck loop).
        * If the response hash differs from the last recorded one for the same
          ``(tool, args)`` key, the world is making progress — reset the counter.
        * If the response hash is identical, increment the counter.
        * At ``threshold``: return a feedback message so the engine can break.
    """

    def __init__(self, threshold: int = 3):
        # key → (consecutive_identical_count, last_response_hash)
        self._state: dict[str, tuple[int, str]] = {}
        self._threshold = threshold
        self._last_key: str | None = None

    @staticmethod
    def _make_key(name: str, args: dict) -> str:
        """Stable string key for a tool call."""
        try:
            args_str = json.dumps(args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            args_str = str(args)
        return f"{name}::{args_str}"

    @staticmethod
    def _hash_response(content: str) -> str:
        # Non-cryptographic: MD5 is fine for collision-detection here, and
        # `usedforsecurity=False` silences security-linter false positives.
        return hashlib.md5(
            content.encode(errors="replace"), usedforsecurity=False,
        ).hexdigest()

    def record(self, name: str, args: dict, response: str) -> str | None:
        """Record a single (tool, args, response) observation.

        Returns a feedback message when a stuck loop is detected (same tool
        called with identical arguments AND identical response ``threshold``
        times **truly consecutively**), or ``None`` if everything is fine.

        "Truly consecutive" means no other ``(tool, args)`` key was recorded
        in between.  Interleaved patterns like ``A → B → A → B → A`` never
        trigger — they represent polling, not a stuck loop.
        """
        key = self._make_key(name, args if isinstance(args, dict) else {})
        resp_hash = self._hash_response(response)

        prev_count, prev_hash = self._state.get(key, (0, ""))

        # If a different tool/args was called since our last record() call,
        # this is an interleaved pattern (e.g. polling).  Reset the counter
        # for this key so it starts fresh.
        if self._last_key is not None and self._last_key != key:
            prev_count, prev_hash = 0, ""

        self._last_key = key

        if prev_hash and resp_hash != prev_hash:
            # Response changed — progress is being made, reset.
            self._state[key] = (1, resp_hash)
            return None

        # Response identical (or first observation) — increment.
        new_count = prev_count + 1
        self._state[key] = (new_count, resp_hash)

        if new_count >= self._threshold:
            return (
                f"STUCK LOOP: Tool '{name}' was called {new_count} times with "
                f"identical arguments AND identical response.  The external "
                f"state is not changing.  Stop repeating this call and try a "
                f"different approach, or explain to the user why you're stuck."
            )
        return None


# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────


# C4 Phase 3 (codex review 2026-04-25): extraction helpers moved to
# runtime/loop/llm_runner.py.  Engine keeps back-compat aliases with the
# historical underscore-prefixed names so internal call sites (and tests
# that monkeypatch them) continue to work unchanged.
from .llm_runner import (
    extract_text as _extract_text,
    extract_tool_calls as _extract_tool_calls,
    is_transient_error as _is_transient_error,
    build_runtime_options as _build_runtime_options,
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
        # A3 (codex review 2026-04-25): tracing must never fail-close a
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
        error=error,
    )


# C4 Phase 4 (codex-review 2026-04-25): compaction helpers moved to
# runtime/loop/compaction.py.  Engine keeps underscore-prefixed back-compat
# aliases so tests and internal callers that import the historical names
# continue to work unchanged.
from .compaction import (  # noqa: E402
    compact_messages as _compact_messages,
    truncate_result as _truncate_result,
    truncate_tool_call_blocks as _truncate_tool_call_blocks,
)


# ─── AgentLoop ───────────────────────────────────────────────────────────────

class AgentLoop:
    """Reusable agentic tool-use loop engine.

    Supports both streaming and non-streaming modes, concurrent tool execution,
    context compaction, truncation recovery, and transient-error retry.
    """

    def __init__(
        self,
        runtime_owner: LLMClient,
        config: LoopConfig,
        callbacks: LoopCallbacks | None = None,
        tool_source: ToolSource | None = None,
        model_spec: ModelSpec | None = None,
        cancel_event: threading.Event | None = None,
    ):
        self._runtime_owner = runtime_owner
        self._config = config
        self._callbacks = callbacks or LoopCallbacks()
        self._tool_source = tool_source
        self._model_spec = model_spec  # override for sub-agent model tier
        self._cancel_event = cancel_event
        # Reuse the module-level shared executor to avoid accumulating
        # ThreadPoolExecutor objects and atexit handlers across turns and
        # sub-agent dispatches.  A2 fix: ensure the pool is at least as
        # wide as the configured ``max_tool_workers`` (the historical
        # singleton was hard-capped at 8, silently throttling configs
        # that asked for more dispatch parallelism).
        self._executor = _get_tool_dispatch_executor(config.max_tool_workers)
        # Task-plan scratchpad (see jyagent/todos.py).  Populated via the
        # `write_todos` tool and seeded optionally via run(initial_todos=...)
        # so outer layers can carry the plan across turns.
        self._todos: list = []
        # A1 fix (codex review 2026-04-25): accumulator for mutating-tool
        # timeouts.  Populated by ``_execute_tool_with_timeout`` via the
        # ``partial_side_effects=`` kwarg threaded through ``_execute_tools``;
        # snapshotted onto ``LoopResult.partial_side_effects`` in ``run()``.
        # Reset at the top of ``_run_impl`` so back-to-back .run() calls on
        # the same AgentLoop instance don't bleed state across turns.
        self._partial_side_effects: list[str] = []
        # C2 (codex review 2026-04-25): AgentLoop holds substantial per-run
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
        before ``run()``.  Empty string / None restores default."""
        self._run_id = run_id or ""

    def _is_cancelled(self) -> bool:
        """Check if external cancellation has been requested."""
        return self._cancel_event is not None and self._cancel_event.is_set()

    def _cancellable_sleep(self, seconds: float) -> bool:
        """Sleep that returns early if cancellation is signalled.

        Returns True if cancelled during the wait, False otherwise.  When no
        cancel_event is attached, falls back to a plain blocking sleep.
        """
        if self._cancel_event is None:
            time.sleep(seconds)
            return False
        # Event.wait returns True when set, False on timeout.
        return self._cancel_event.wait(seconds)

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

    # ── callback helpers (no-op when callback is None) ────────────────────

    def _fire(self, name: str, *args: Any) -> None:
        cb = getattr(self._callbacks, name, None)
        if cb is not None:
            try:
                cb(*args)
            except Exception:
                # Callbacks are for presentation — never abort the engine loop.
                print(f"[warning] callback {name!r} raised:", traceback.format_exc(), file=sys.stderr)

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
        is still in flight (C2 fix, codex review 2026-04-25): AgentLoop
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
            # A1 (codex review 2026-04-25): mirror the todos pattern — snapshot
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
            # C2: release the reentrance guard regardless of how _run_impl
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

        After C4 Phase 5, this method is a thin orchestrator: setup is in
        ``RunState.from_loop()``, the per-step body is in
        ``runtime/loop/step.py::run_step``, and only the for-step counter,
        post-loop terminal handlers (cancelled-exit / max_steps fallback /
        max_steps exit), and the outer try/except live here.
        """
        from .step import RunState, run_step, StepContinue, StepTerminate, StepBreak

        cfg = self._config
        state = RunState.from_loop(self, system_prompt, messages, initial_todos)
        # Aliases for the post-loop terminal handlers below — keeps the
        # diff minimal vs. pre-Phase-5 while still routing through state.
        trace = state.trace
        cost_tracker = state.cost_tracker
        effective_spec = state.effective_spec

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
                # Try one more streaming call with system instruction to avoid tools
                try:
                    fallback_context = {
                        "system_prompt": system_prompt + "\n\n[SYSTEM: You have reached the maximum number of tool-use steps. Please provide your best answer now WITHOUT using any tools.]",
                        "messages": messages,
                    }

                    # Create fallback options with tool_choice=none
                    _base = _build_runtime_options(
                        self._runtime_owner,
                        cfg.initial_max_tokens,
                        model_spec=self._model_spec,
                        metadata={"component": "loop_engine", "step": cfg.max_steps + 1, "fallback": True},
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
                    self._fire("on_usage", usage)
                    # B1 fix (codex review 2026-04-25): the fallback call's
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
                            effective_spec.provider,
                            effective_spec.model,
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
                        fallback_message["content"] = _truncate_tool_call_blocks(content, state.last_step_batch)

                    # Append fallback response
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
                error=str(e),
                trace=trace,
            )


    # ── LLM call + retry/fallback (delegated to LLMRunner) ──────────────
    #
    # C4 Phase 3 (codex review 2026-04-25): the call machinery lives in
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

        C4 Phase 3: we keep the retry loop in AgentLoop (rather than routing
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
                if _is_transient_error(err) and attempt < cfg.retry_attempts:
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
                    self._fire("on_stream_retry", "transient_error", partial_text)
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


# ─── PEP 562 back-compat shim for tool_executor module state ────────────────
#
# C4 Phase 2: the pool + lock + cap are MUTABLE module state that
# ``get_tool_dispatch_executor()`` rebinds inside ``tool_executor.py`` when
# the pool grows.  If engine.py imported them as values at module-import
# time, every post-grow read from ``loop_engine._tool_dispatch_executor``
# would see the pre-grow snapshot.  47 test references across 4 files
# read those names through the engine module path.
#
# PEP 562 (Python 3.7+) lets a module define ``__getattr__`` for lazy /
# forwarding attribute access.  Each lookup takes one import (cheap) plus
# one ``getattr`` and returns the live object from tool_executor.py.

_TOOL_EXECUTOR_PASSTHROUGH = {
    "_tool_dispatch_executor": "tool_dispatch_executor",
    "_tool_dispatch_cap":      "tool_dispatch_cap",
    "_tool_dispatch_lock":     "tool_dispatch_lock",
    "_tool_executor":          "tool_dispatch_executor",
}


def __getattr__(name: str):
    target = _TOOL_EXECUTOR_PASSTHROUGH.get(name)
    if target is not None:
        from . import tool_executor as _te
        return getattr(_te, target)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
