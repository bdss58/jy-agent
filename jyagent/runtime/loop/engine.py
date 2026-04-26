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
        snapshots the final todos onto the result."""
        cfg = self._config
        all_text = ""
        final_text = ""
        current_max_tokens = cfg.initial_max_tokens
        total_input_tokens = 0
        total_output_tokens = 0
        tool_calls_count = 0
        last_reflection_count = 0  # tool_calls_count at last reflection injection
        # A1 (codex review 2026-04-25): reset the mutating-timeout
        # accumulator at the top of every run so back-to-back turns on the
        # same AgentLoop instance don't carry stale names forward.
        self._partial_side_effects = []
        # Boundary between prior-turn history and this-turn appends.
        # Passed to ``should_verify`` so a replayed historical mutation
        # cannot re-arm the verification gate on a non-mutating new turn
        # (Codex review 2026-04-25 Part 2 #5).
        turn_start_idx = len(messages)
        registry = get_registry()
        step = 0
        consecutive_truncations = 0  # cap truncation recovery retries
        max_truncation_retries = 3
        verification_injected = False  # only verify once per run

        # Lazy import of the reflection module so test imports of
        # loop_engine stay cheap and reflection is opt-in by config.
        if cfg.reflect_every_n_tool_calls > 0 or cfg.reflect_after_subagent:
            from . import reflection  # noqa: F401 — referenced below
        else:
            reflection = None  # type: ignore[assignment]

        # Ensure a run id is set when checkpointing is enabled (outer
        # layers may have preset one via set_run_id).
        if cfg.checkpoint_dir and not self._run_id:
            from .checkpoint import new_run_id
            self._run_id = new_run_id()

        # ── Seed todos scratchpad ─────────────────────────────────────
        # Lazy import to keep the dependency optional.
        if cfg.todos_enabled:
            from .todos import (
                WRITE_TODOS_SCHEMA,
                build_write_todos_tool,
                inject_todos_into_messages,
                normalize_todo,
            )
            if initial_todos:
                try:
                    self._todos = [normalize_todo(t) for t in initial_todos]
                except TypeError as e:
                    self._fire("on_warning", f"ignoring invalid initial_todos: {e}")
                    self._todos = []
            else:
                self._todos = []

            # Per-loop write_todos tool closing over self._todos.
            def _get_store() -> list:
                return self._todos

            def _set_store(new_list: list) -> None:
                self._todos = new_list

            _write_todos_fn = build_write_todos_tool(_get_store, _set_store)

        # ── Harness trackers ──────────────────────────────────────────
        # Effective model spec — sub-agent override wins over owner default.
        # Used for tracing and cost accounting so sub-agents on a different
        # tier are billed against the correct pricing.
        effective_spec = self._model_spec or self._runtime_owner.model_spec

        trace = get_tracer()
        if trace:
            trace.start(effective_spec.provider, effective_spec.model)
        cost_tracker = _CostTracker() if cfg.max_cost_usd is not None else None
        unpriced_warned = False  # one-shot flag for cost_tracker.has_unpriced_usage
        stuck_detector = _StuckLoopDetector(cfg.dedup_threshold)

        # Resolve tools.  ``tools_batch`` is the immutable per-step snapshot
        # consumed by every dispatch/compaction helper.  Built once per step
        # via ``ToolRegistry.freeze()`` (or from ``_tool_source()`` when
        # provided), so concurrent registry mutations cannot race against
        # in-flight metadata reads (Codex Part 1 #4, #11, #12).
        tools_batch: ToolBatch = ToolBatch.empty()

        try:
            for step in range(cfg.max_steps):
                self._fire("on_step_progress", step, cfg.max_steps)

                # ── Cooperative cancellation check (top of loop) ─────
                if self._is_cancelled():
                    break

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
                if self._tool_source is not None:
                    src_schemas, src_functions = self._tool_source()
                    reg_batch = registry.freeze()
                    src_schema_map = {
                        s.get("name"): s for s in src_schemas if s.get("name")
                    }
                    tools_batch = ToolBatch(
                        version=reg_batch.version,
                        schemas=tuple(src_schemas),
                        schema_map=src_schema_map,
                        functions=dict(src_functions),
                        parallel_safe=reg_batch.parallel_safe,
                        timeout_hints=reg_batch.timeout_hints,
                        large_input_keys=reg_batch.large_input_keys,
                        compaction_priority=reg_batch.compaction_priority,
                        # Inherit mutating classification from the registry
                        # freeze — tool_source functions that happen to share
                        # a registered name pick up the registered metadata;
                        # purely dynamic names (e.g. MCP tools that
                        # auto-registered via the real register() path)
                        # bring their own.  A1 fix (codex review 2026-04-25).
                        mutating=reg_batch.mutating,
                    )
                elif step == 0 or registry.version != tools_batch.version:
                    # Re-freeze only when the registry has changed.  The
                    # version read is locked (defense-in-depth), and even
                    # if a stale-by-one read causes us to skip a freeze,
                    # the next step will catch up — at most one step uses
                    # slightly-stale metadata, never inconsistent metadata.
                    tools_batch = registry.freeze()

                # Overlay the per-loop write_todos tool on top of the
                # registry snapshot when todos are enabled.  This is the
                # closure-scoped injection point recommended by the design
                # review (avoids ContextVar propagation issues with our
                # daemon-thread tool executor).
                if cfg.todos_enabled:
                    step_batch = tools_batch.with_overlay(
                        functions={"write_todos": _write_todos_fn},
                        schemas=[WRITE_TODOS_SCHEMA],
                        # write_todos must NOT be parallel-safe — it would
                        # then run concurrently with itself in a batch and
                        # the replace-all semantics would silently drop
                        # one of the writes (Codex Part 2 #4).
                    )
                else:
                    step_batch = tools_batch

                tool_schemas = list(step_batch.schemas)
                tool_functions = step_batch.functions

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
                        self._fire("on_compaction", before_len, after_len)

                # Build context dict.  Todos are injected as a
                # <system-reminder> text block appended to the tail user
                # message — NOT persisted into `messages`, so compaction
                # never touches them.  The base system_prompt stays
                # untouched to preserve Anthropic prefix caching.
                if cfg.todos_enabled and self._todos:
                    context_messages = inject_todos_into_messages(messages, self._todos)
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
                    self._runtime_owner,
                    current_max_tokens,
                    model_spec=self._model_spec,
                    metadata={"component": "loop_engine", "step": step + 1},
                )

                # Phase-aware tool_choice shaping (see jyagent/phases.py).
                # The policy is consulted once per step.  Returning a
                # PhaseDirective with `tool_choice=None` is informational
                # only (engine fires on_phase_enter for observability but
                # leaves tool_choice unchanged).  A non-None tool_choice
                # rebuilds `opts` so the runtime adapter sees the override.
                if cfg.phase_policy is not None:
                    try:
                        directive = cfg.phase_policy(step, cfg.max_steps, tool_calls_count)
                    except Exception as e:
                        directive = None
                        self._fire("on_warning", f"phase_policy raised: {e}")
                    if directive is not None:
                        self._fire("on_phase_enter", directive.phase)
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
                step_text, tool_call_blocks, stop_reason, final_message = self._call_llm_with_retry(
                    context, opts, step,
                )
                llm_dur_ms = (time.perf_counter() - llm_t0) * 1000

                # Fire runtime warnings
                for warning in final_message.get("llm_warnings", []):
                    self._fire("on_warning", warning)

                # Accumulate usage
                usage = final_message.get("usage", {})
                total_input_tokens += usage.get("input_tokens", 0)
                total_output_tokens += usage.get("output_tokens", 0)
                self._fire("on_usage", usage)

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
                    if cost_tracker.has_unpriced_usage and not unpriced_warned:
                        unpriced_warned = True
                        self._fire(
                            "on_warning",
                            f"Cost budget using lower bound: "
                            f"{cost_tracker.unpriced_calls} call(s) had no pricing data "
                            f"({effective_spec.provider}/{effective_spec.model}).",
                        )
                    current_cost = cost_tracker.cost
                    if current_cost >= cfg.max_cost_usd:
                        self._fire(
                            "on_warning",
                            f"Cost budget exceeded: ${current_cost:.4f} >= ${cfg.max_cost_usd:.4f}",
                        )
                        if trace:
                            trace.add_span(step=step, event_type="cost_check", success=False,
                                           error=f"budget ${cfg.max_cost_usd} exceeded")
                        return _finalize_run(
                            status="cost_limit",
                            text=all_text or "",
                            final_text=final_text,
                            messages=messages,
                            steps=step + 1,
                            total_input_tokens=total_input_tokens,
                            total_output_tokens=total_output_tokens,
                            tool_calls_count=tool_calls_count,
                            error=f"Cost budget exceeded: ${current_cost:.4f} >= ${cfg.max_cost_usd:.4f}",
                            trace=trace,
                            trace_total_cost_usd=current_cost,
                        )

                all_text += step_text
                final_text = step_text

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
                        not verification_injected
                        and should_verify(messages, tool_calls_count, since_index=turn_start_idx)
                        and step + 1 < cfg.max_steps
                    ):
                        verification_injected = True
                        if trace:
                            trace.add_span(step=step, event_type="verification")
                        # Append the assistant's response, then inject verification
                        messages.append(final_message)
                        messages.append({
                            "role": "user",
                            "content": build_verification_prompt(messages),
                        })
                        continue

                    if not step_text:
                        final_text = _extract_text(final_message)
                        all_text = final_text or all_text

                    # Apply truncation if enabled
                    if cfg.truncate_large_inputs:
                        content = final_message.get("content", [])
                        final_message = dict(final_message)
                        final_message["content"] = _truncate_tool_call_blocks(content, step_batch)

                    # Allow caller to transform before append
                    cb_am = self._callbacks.on_assistant_message
                    if cb_am is not None:
                        final_message = cb_am(final_message) or final_message
                    messages.append(final_message)
                    result_text = all_text if all_text else "I processed your request but had no text response to return."

                    cost = cost_tracker.cost if cost_tracker else 0.0
                    return _finalize_run(
                        status="completed",
                        text=result_text,
                        final_text=final_text,
                        messages=messages,
                        steps=step + 1,
                        total_input_tokens=total_input_tokens,
                        total_output_tokens=total_output_tokens,
                        tool_calls_count=tool_calls_count,
                        trace=trace,
                        trace_total_cost_usd=cost or 0.0,
                    )

                # Truncation detection → scale up and retry step
                if cfg.auto_scale_on_truncation and _is_truncated(stop_reason, tool_call_blocks):
                    consecutive_truncations += 1
                    if consecutive_truncations > max_truncation_retries:
                        cost = cost_tracker.cost if cost_tracker else 0.0
                        return _finalize_run(
                            status="error",
                            text=all_text or "",
                            final_text="",
                            messages=messages,
                            steps=step + 1,
                            total_input_tokens=total_input_tokens,
                            total_output_tokens=total_output_tokens,
                            tool_calls_count=tool_calls_count,
                            error=f"Repeated truncation ({consecutive_truncations}x) — model output exceeds capacity",
                            trace=trace,
                            trace_total_cost_usd=cost or 0.0,
                        )
                    self._fire("on_truncation")
                    # Also fire the unified stream-retry hook so UIs that
                    # already handle transient-error duplication can use the
                    # same visual treatment for truncation-recovery replays.
                    self._fire("on_stream_retry", "truncation", step_text or "")
                    current_max_tokens = min(
                        current_max_tokens * cfg.token_scale_factor,
                        cfg.max_tokens_cap,
                    )
                    # Remove the partial step text
                    all_text = all_text[: -len(step_text)] if step_text else all_text
                    continue

                # Successful step — reset truncation counter
                consecutive_truncations = 0

                # Append assistant message (allow caller to transform)
                if cfg.truncate_large_inputs:
                    content = final_message.get("content", [])
                    final_message = dict(final_message)
                    final_message["content"] = _truncate_tool_call_blocks(content, step_batch)

                cb_am = self._callbacks.on_assistant_message
                if cb_am is not None:
                    transformed = cb_am(final_message)
                    if transformed is not None:
                        final_message = transformed
                messages.append(final_message)

                # Fire on_tool_batch for multi-tool batches
                if len(tool_call_blocks) > 1:
                    self._fire("on_tool_batch", len(tool_call_blocks))

                # Fire on_tool_start for all tool calls BEFORE execution
                for block in tool_call_blocks:
                    self._fire("on_tool_start", block.name, block.input)

                # Execute tools
                # ── Cooperative cancellation check (before tools) ────
                if self._is_cancelled():
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
                        self._fire("on_tool_end", block.name, "Cancelled", True)
                    break

                tools_t0 = time.perf_counter()
                tool_results_tuples = _execute_tools(
                    tool_call_blocks,
                    step_batch,
                    cfg.concurrent_tools,
                    cfg.max_tool_workers,
                    cfg.tool_timeout,
                    executor=self._executor,
                    partial_side_effects=self._partial_side_effects,
                )
                tools_dur_ms = (time.perf_counter() - tools_t0) * 1000

                for block, result in tool_results_tuples:
                    tool_calls_count += 1
                    content_str = _truncate_result(result.content, cfg.max_tool_result_chars, result.is_error)
                    self._fire("on_tool_end", block.name, content_str, result.is_error)

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
                    batch_key = _StuckLoopDetector._make_key(
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
                    self._fire("on_warning", stuck_feedback)
                    cost = cost_tracker.cost if cost_tracker else 0.0
                    if trace:
                        trace.add_span(step=step, event_type="dedup_break", success=False, error=stuck_feedback)
                    return _finalize_run(
                        status="dedup_break",
                        text=all_text or "",
                        final_text=final_text,
                        messages=messages,
                        steps=step + 1,
                        total_input_tokens=total_input_tokens,
                        total_output_tokens=total_output_tokens,
                        tool_calls_count=tool_calls_count,
                        error=stuck_feedback,
                        trace=trace,
                        trace_total_cost_usd=cost or 0.0,
                    )

                # ── Cooperative cancellation check (after tools) ─────
                if self._is_cancelled():
                    break

                # ── Mid-loop reflection / critic step ─────────────────
                # After meaningful work boundaries (every-N cadence or
                # sub-agent return), append a short progress-check user
                # message so the next LLM call re-grounds on the task.
                # Avoids drift on long-horizon rollouts.
                if cfg.reflect_every_n_tool_calls > 0 or cfg.reflect_after_subagent:
                    batch_names = [b.name for b, _ in tool_results_tuples]
                    inject, reason = reflection.should_reflect(
                        reflect_every_n=cfg.reflect_every_n_tool_calls,
                        reflect_after_subagent=cfg.reflect_after_subagent,
                        tool_calls_total=tool_calls_count,
                        tool_calls_at_last_reflection=last_reflection_count,
                        batch_tool_names=batch_names,
                        messages=messages,
                    )
                    if inject:
                        prompt = reflection.build_reflection_prompt(
                            reason, tool_calls_count,
                        )
                        messages.append({"role": "user", "content": prompt})
                        last_reflection_count = tool_calls_count
                        self._fire("on_reflection", reason)
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
                    self._write_checkpoint(
                        step=step,
                        messages=messages,
                        total_input_tokens=total_input_tokens,
                        total_output_tokens=total_output_tokens,
                        tool_calls_count=tool_calls_count,
                        status="in_progress",
                    )

            # ── Cooperative cancellation — early exit ────────────────
            if self._is_cancelled():
                return _finalize_run(
                    status="interrupted",
                    text=all_text or "",
                    final_text=final_text,
                    messages=messages,
                    steps=step + 1,
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=total_output_tokens,
                    tool_calls_count=tool_calls_count,
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
                    fallback_context = dict(context)
                    fallback_system = context["system_prompt"] + "\n\n[SYSTEM: You have reached the maximum number of tool-use steps. Please provide your best answer now WITHOUT using any tools.]"
                    fallback_context["system_prompt"] = fallback_system

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

                    # Remove tools from fallback context to ensure no tool use
                    if "tools" in fallback_context:
                        del fallback_context["tools"]

                    if cfg.streaming:
                        fallback_text, _, _, fallback_message = self._call_streaming(fallback_context, fallback_opts)
                    else:
                        fallback_text, _, _, fallback_message = self._call_complete(fallback_context, fallback_opts)

                    # Accumulate usage
                    usage = fallback_message.get("usage", {})
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)
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
                        if cost_tracker.has_unpriced_usage and not unpriced_warned:
                            unpriced_warned = True
                            self._fire(
                                "on_warning",
                                "cost_tracker: at least one call lacked pricing data; "
                                "budget enforcement uses the priced subtotal only",
                            )

                    # Apply truncation if enabled
                    if cfg.truncate_large_inputs:
                        content = fallback_message.get("content", [])
                        fallback_message = dict(fallback_message)
                        fallback_message["content"] = _truncate_tool_call_blocks(content, step_batch)

                    # Append fallback response
                    messages.append(fallback_message)

                    # Return completed since we got a final answer.
                    # Note: previously this path skipped trace.finish() — the
                    # max_steps trace block below was unreachable on success.
                    cost = cost_tracker.cost if cost_tracker else 0.0
                    return _finalize_run(
                        status="completed",
                        text=fallback_text or all_text,
                        final_text=fallback_text,
                        messages=messages,
                        steps=cfg.max_steps,
                        total_input_tokens=total_input_tokens,
                        total_output_tokens=total_output_tokens,
                        tool_calls_count=tool_calls_count,
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
                text=all_text or "",
                final_text=final_text,
                messages=messages,
                steps=cfg.max_steps,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                tool_calls_count=tool_calls_count,
                trace=trace,
                trace_total_cost_usd=cost or 0.0,
            )

        except KeyboardInterrupt:
            return _finalize_run(
                status="interrupted",
                text=all_text + "\n\n[Interrupted by user]" if all_text else "[Interrupted by user]",
                final_text="",
                messages=messages,
                steps=step + 1,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                tool_calls_count=tool_calls_count,
                trace=trace,
            )
        except Exception as e:
            return _finalize_run(
                status="error",
                text=all_text or "",
                final_text="",
                messages=messages,
                steps=step + 1,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                tool_calls_count=tool_calls_count,
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
