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
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    # Normalized message TypedDict — used only as a type annotation on
    # ``AgentLoop.run`` and ``_run_impl``.  Kept under TYPE_CHECKING to
    # avoid a runtime import cycle (llm.types -> runtime.loop.llm_types).
    from ...llm.types import Message

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
from .callbacks import LoopCallbacks  # canonical public surface — imported via engine
from .config import LoopConfig, LoopResult  # canonical public surface — imported via engine
from ._thread_helpers import LoopThreadHelper  # cancel/_fire helpers


_logger = logging.getLogger(__name__)


# ─── Core types ──────────────────────────────────────────────────────────────
# ``ToolCallRequest`` lives in ``runtime/loop/llm_types.py`` (sibling to
# ``LLMOptions`` and ``ModelSpec``); engine imports it here only because
# its own method signatures reference it.  Callers must import from
# ``llm_types`` directly — the legacy back-compat re-export was retired
# (commit migrating tests off ``engine.ToolCallRequest``).

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
# tool-execution helpers live in the leaf modules under
# ``runtime/loop/``.
# Engine imports only what its own body uses.  Other call sites import
# directly from the leaf module (``tool_executor``, ``cost``, ``stuck_loop``,
# ``finalize``, ``compaction``).  Re-exports from this module are no longer
# offered for back-compat — the per-step body in
# ``runtime/loop/step.py::run_step`` calls ``tool_executor.execute_tools``
# directly, so any patching/monkeypatching for tests must target the leaf
# module, not ``engine``.
from .tool_pool import get_tool_dispatch_executor  # noqa: E402

# Helpers consumed by ``AgentLoop._call_llm_with_retry`` (a thin shim
# over ``_retry_llm_call``) and the max_steps fallback path.
#
# ``is_transient_error`` is re-exported here primarily for back-compat
# with tests that monkeypatch ``engine.is_transient_error`` to inject
# transient failures.  After the retry-loop consolidation (2026-05) the
# only functional caller is ``_retry_llm_call`` inside ``llm_runner``,
# so patches against ``engine.is_transient_error`` are now effectively
# no-ops — but the import is kept so existing tests keep importing
# cleanly.  New tests should patch ``llm_runner.is_transient_error``.
from .llm_runner import (  # noqa: E402
    is_transient_error,
    build_runtime_options,
    _retry_llm_call,
)

# Terminal-path helper used by ``AgentLoop.run()``.  ``step.py`` imports
# ``finalize_run`` directly from ``.finalize`` — both call sites use the same
# public name, no alias layer.
from .finalize import finalize_run  # noqa: E402

# Compaction helper used inside ``AgentLoop`` to trim oversized tool blocks.
from .compaction import truncate_tool_call_blocks  # noqa: E402


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
        #
        # We deliberately do NOT cache the executor on ``self``.  When a
        # later AgentLoop with a larger ``max_tool_workers`` triggers
        # ``get_tool_dispatch_executor`` to grow the shared pool, the
        # growth path shuts the old pool down (``shutdown(wait=False)``)
        # and replaces it.  A cached reference would then point at a
        # dead pool and the next ``executor.submit`` would raise
        # ``RuntimeError: cannot schedule new futures after shutdown``.
        # Resolving the executor lazily via the ``_executor`` property
        # below (which always returns the *current* shared pool) makes
        # the growth path safe for older still-running AgentLoop
        # instances.  The fast path inside
        # ``get_tool_dispatch_executor`` is a single attribute check, so
        # the per-step cost is negligible.
        # Warm the pool now so the first tool dispatch doesn't pay
        # creation latency, but discard the return value — readers go
        # through the property.
        get_tool_dispatch_executor(config.max_tool_workers)
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

    @property
    def _executor(self):
        """Live reference to the shared tool-dispatch executor.

        Resolved on every access (not cached on ``self``) so a parallel
        AgentLoop that grew the shared pool — which shuts the old pool
        down — cannot leave THIS loop holding a dead reference.  The
        fast path inside ``get_tool_dispatch_executor`` is a single
        attribute check, so per-dispatch cost is negligible.

        Test escape hatch: assigning ``loop._executor = X`` records X
        as an override (via the setter below) and the property returns
        it on subsequent reads.  Production code never assigns
        ``_executor`` — it didn't before this property landed either —
        so this only matters for tests that build loops with ``__new__``
        and want a hand-picked executor.
        """
        override = self.__dict__.get("_executor_override")
        if override is not None:
            return override
        return get_tool_dispatch_executor(self._config.max_tool_workers)

    @_executor.setter
    def _executor(self, value) -> None:
        """Test-only override — see the property docstring."""
        self.__dict__["_executor_override"] = value

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
        total_cache_creation_tokens: int = 0,
        total_cache_read_tokens: int = 0,
        api_calls: int = 0,
        error: str | None = None,
    ) -> None:
        """Persist a LoopCheckpoint if checkpointing is enabled.

        ``step`` may be an int (regular step boundary) or ``"final"``
        (terminal exit).  Errors are logged via ``on_warning`` — never
        propagated, checkpointing must never break a run.

        ``total_cache_creation_tokens`` / ``total_cache_read_tokens`` /
        ``api_calls`` were added 2026-05 to match the cadence call site
        in ``runtime/loop/step.py::_maybe_checkpoint``, which had been
        passing them since the cache-token plumbing landed.  They are
        keyword-only with default 0 so every existing engine call site
        keeps working unchanged; missing values just record zero in the
        checkpoint.  Codex review caught the latent ``TypeError`` that
        any user enabling ``checkpoint_every_n_steps`` would have hit.
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
                total_cache_creation_tokens=total_cache_creation_tokens,
                total_cache_read_tokens=total_cache_read_tokens,
                api_calls=api_calls,
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
        messages: "list[Message]",
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
            # finalize_run() call site.  Copy defensively so a caller that
            # retains the returned list can't mutate the AgentLoop's internal
            # state on the next run.
            result.partial_side_effects = list(self._partial_side_effects)
            if self._config.checkpoint_dir:
                # Terminal ("final") checkpoint — includes status + error.
                # Forward cache-token / api-call counters so ``final.json``
                # matches ``LoopResult``; the periodic checkpoint already
                # passes them (see ``step_bookkeeping._maybe_checkpoint``)
                # and omitting them here silently zeroed those fields in
                # the persisted record despite the in-memory result
                # having the correct values.
                self._write_checkpoint(
                    step="final",
                    messages=result.messages,
                    total_input_tokens=result.total_input_tokens,
                    total_output_tokens=result.total_output_tokens,
                    tool_calls_count=result.tool_calls_count,
                    total_cache_creation_tokens=result.total_cache_creation_tokens,
                    total_cache_read_tokens=result.total_cache_read_tokens,
                    api_calls=result.api_calls,
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
        messages: "list[Message]",
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
        # threaded into 5 ``finalize_run`` calls and ``cost_tracker`` into
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
                return finalize_run(
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
            # Defense-in-depth: the canonical finalize_run() path always
            # strips dangling [VERIFICATION] (idempotently), so we no longer
            # need a guarded pre-strip here.  The boundary guard at the
            # gate (step + 1 < cfg.max_steps) should already prevent the
            # leak, but finalize_run cleans up unconditionally as belt-
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
                #
                # Order-of-operations contract: every call that can raise
                # (``_call_streaming`` / ``_call_complete`` /
                # ``truncate_tool_call_blocks``) runs BEFORE any state
                # mutation.  All ``state.*``, ``cost_tracker.record``, and
                # ``messages.append`` writes are deferred to the commit
                # block at the bottom, so a fallback failure cannot leave
                # poisoned counters on the max_steps result that the
                # ``except Exception`` handler then silently reports.
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

                    # ── Phase 1: can-fail work (LLM call + truncation) ──
                    # Anything that raises here unwinds cleanly because no
                    # state mutation has happened yet.
                    if cfg.streaming:
                        fallback_text, _, _, fallback_message = self._call_streaming(fallback_context, fallback_opts)
                    else:
                        fallback_text, _, _, fallback_message = self._call_complete(fallback_context, fallback_opts)

                    # Apply truncation if enabled — uses the last step_batch
                    # built by run_step (threaded via state.last_step_batch).
                    # Done before commit so a malformed fallback message
                    # cannot land in the transcript with poisoned counters.
                    if cfg.truncate_large_inputs:
                        content = fallback_message.get("content", [])
                        fallback_message = dict(fallback_message)
                        fallback_message["content"] = truncate_tool_call_blocks(content, state.last_step_batch)

                    # ── Phase 2: commit (no can-fail calls below this line) ──
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
                    return finalize_run(
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
                except Exception as fb_exc:
                    # Surface the failure on the warning channel so it's
                    # visible to outer layers — the previous bare ``pass``
                    # silently turned a fallback crash into a plain
                    # max_steps result with no breadcrumb.  No state
                    # rollback is needed: the order-of-operations contract
                    # above guarantees that any pre-commit failure left
                    # ``state.*`` / ``cost_tracker`` / ``messages``
                    # untouched.
                    self._fire(
                        "on_warning",
                        f"max_steps fallback failed; reporting max_steps "
                        f"without final answer: "
                        f"{type(fb_exc).__name__}: {fb_exc}",
                    )

            # ── max_steps exit ─────────────────────────────────────────
            cost = cost_tracker.cost if cost_tracker else 0.0
            return finalize_run(
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
            return finalize_run(
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
            return finalize_run(
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

        Thin shim over the shared ``_retry_llm_call`` helper.  We keep
        the shim on ``AgentLoop`` (rather than routing straight to
        ``LLMRunner.call_with_retry``) so the retry loop dispatches
        through ``self._call_streaming`` / ``self._call_complete`` —
        several tests and internal diagnostics override those methods
        on a subclass or monkeypatch them on the instance to inject
        transient failures, and that contract is preserved.

        Returns ``(step_text, tool_call_blocks, stop_reason, final_message)``.
        """
        return _retry_llm_call(
            config=self._config,
            context=context,
            options=options,
            call_streaming=self._call_streaming,
            call_complete=self._call_complete,
            fire=self._fire,
            is_cancelled=self._is_cancelled,
            cancellable_sleep=self._cancellable_sleep,
            # Resolve via this module's globals so tests that patch
            # ``engine.is_transient_error`` see their override take
            # effect — the name binding is looked up at call time.
            is_transient=is_transient_error,
        )

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
