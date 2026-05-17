# Sub-Agent Tool — Spawn focused child agents for parallel/specialized subtasks.
#
# The parent (lead) agent decomposes a complex task into sub-problems,
# dispatches sub-agents for each, and synthesizes results.
#
# Sub-agents run silently (no terminal streaming), have their own context
# window and message history, and return only their final answer to the parent.

import os
import json
import sys
import time
import collections
import traceback
import threading
import contextvars
import concurrent.futures
from dataclasses import dataclass

from ..config import (
    DEFAULT_MAX_STEPS, STREAM_TIMEOUT,
    get_active_model_spec, get_reasoning_config_for_provider,
)
from ..runtime.loop.engine import AgentLoop, LoopConfig, LoopCallbacks
from ..runtime.tools.registry import get_registry
from ..llm import LLMOptions, LLMOwner
from ..runtime.tools.result import ToolResult
from ..runtime.stats import get_stats
from ..ui.subagent_status import (
    _subagent_tracker,
    COLOR_DIM, COLOR_RESET, COLOR_GREEN, COLOR_RED,
)



# ─── Defaults ────────────────────────────────────────────────────────────────

# Sub-agent step budget mirrors the main agent (``DEFAULT_MAX_STEPS`` resolves
# from the ``AGENT_MAX_STEPS`` env var, default 100). Sub-agents handle focused
# subtasks, so using the same budget avoids spurious ``max_steps`` exits when
# the main agent delegates non-trivial work.
_DEFAULT_MAX_STEPS = DEFAULT_MAX_STEPS
_DEFAULT_MAX_TOKENS_PER_RESPONSE = 8192
_SUBAGENT_STATUS_COMPLETED = "completed"
_SUBAGENT_STATUS_MAX_STEPS = "max_steps"
_SUBAGENT_STATUS_API_ERROR = "api_error"

# Hard cap on action="wait" blocking time for check_agent. Mirrors
# _BG_WAIT_MAX_SECONDS in tools/core.py for check_background.
_BG_AGENT_WAIT_MAX_SECONDS = 300

# Track nesting to prevent runaway recursion.
# Uses contextvars.ContextVar + explicit copy_context().run() because sub-agents
# run in ThreadPoolExecutor worker threads.  ThreadPoolExecutor does NOT auto-
# propagate ContextVars; we snapshot the context after incrementing depth and
# pass ctx.run(fn, ...) to executor.submit() so the worker inherits it.
_nesting_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_nesting_depth", default=0,
)
_MAX_NESTING = 2  # sub-agent can spawn sub-sub-agent, but no deeper


# ─── Parent → child cancel mirror ─────────────────────────────────────────────


def _install_cancel_mirror(
    parent: "threading.Event | None",
    child: threading.Event,
) -> "callable[[], None]":
    """Mirror ``parent.set()`` onto ``child.set()`` until the returned
    ``stop()`` callable fires.

    Returns a no-op when ``parent`` is None (no parent cancel event
    plumbed from the runtime).  When parent is provided, spawns a
    daemon watcher thread that wakes either when the parent fires or
    every 500ms to re-check whether the caller has called ``stop()``.

    Why polling instead of ``parent.wait(timeout=None)``?  A pure-wait
    watcher leaks a thread per dispatch — once spawned it sits blocked
    forever on a parent event that may never fire, and the thread can
    only die at process exit.  The poll-and-stop pattern lets the
    caller release the watcher when the sub-agent's future returns.
    """
    if parent is None:
        return lambda: None
    stop = threading.Event()

    def _watch() -> None:
        while not stop.is_set():
            # Returns True if parent fired, False on timeout.
            if parent.wait(timeout=0.5):
                child.set()
                return

    threading.Thread(
        target=_watch,
        name="jyagent-subagent-cancel-mirror",
        daemon=True,
    ).start()
    return stop.set


# ─── Runtime owner access ─────────────────────────────────────────────────────

_runtime_owner = None


def set_runtime_owner(runtime_owner):
    """Called during agent startup to share the active runtime owner."""
    global _runtime_owner
    _runtime_owner = runtime_owner


def _get_runtime_owner():
    """Get the shared runtime owner, creating a default one if needed."""
    global _runtime_owner
    if _runtime_owner is None:
        _runtime_owner = LLMOwner(get_active_model_spec())
    return _runtime_owner



# ─── Sub-agent system prompt ─────────────────────────────────────────────────

_SUBAGENT_SYSTEM_PROMPT = """You are a focused sub-agent. You have been dispatched by a lead agent to complete a specific task.

Rules:
1. Complete the task described in the user message. Be thorough but efficient.
2. Use tools as needed — you have the same capabilities as the lead agent.
3. When done, provide a clear, structured answer. Include key findings, data, and citations.
4. Do NOT ask clarifying questions — work with what you have.
5. Do NOT use dispatch_agent to spawn further sub-agents unless absolutely necessary.
6. Stay focused on your assigned task. Do not go off on tangents.
7. If you cannot complete the task, explain what you tried and why it failed."""

# ─── Helpers ─────────────────────────────────────────────────────────────────

# ─── Sub-agent memory injection (scoped) ────────────────────────────────────
#
# Before 2026-05-01 this was a blanket 4KB dump of MEMORY.md into every
# sub-agent system prompt.  Codex's context-management review flagged the
# default as over-broad: the parent's durable user/project memory leaked
# into every isolated task, many of which had no need for it.  The fix is
# a two-mode scope gate controlled by the ``memory_mode`` schema param:
#
#   * "none"    — strict isolation (default).  No MEMORY.md, no topics,
#                 no journal.  The sub-agent sees only ``task`` + optional
#                 ``context``.  Maximally self-contained.
#
#   * "matched" — retrieve only the top-K BM25 matches from topics + recent
#                 journal against the task text.  The sub-agent sees
#                 relevant durable knowledge without the full index.
#                 Recommended when the caller wants some knowledge transfer
#                 but cares about token cost / relevance.
#
# This file does NOT call ``build_memory_context`` from ``memory.context``
# (which always injects MEMORY.md) — that helper is deliberately scoped
# to the main loop's system prompt.


_SUBAGENT_MEMORY_MODES = ("none", "matched")
_SUBAGENT_MEMORY_DEFAULT = "none"
_SUBAGENT_MATCHED_TOP_K = 5
_SUBAGENT_MATCHED_MAX_CHARS = 2000


def _get_memory_context(query: str = "", mode: str = _SUBAGENT_MEMORY_DEFAULT) -> str:
    """Build the memory block injected into the sub-agent system prompt.

    ``mode`` controls scope — see the module-level design note above.
    Returns an empty string for mode="none" (default), for unknown modes,
    for empty query in "matched" mode, or when the underlying source is
    empty / unreadable.  Never raises — memory is best-effort.
    """
    if mode not in _SUBAGENT_MEMORY_MODES:
        return ""

    if mode == "none":
        return ""

    # mode == "matched"
    if not query:
        return ""
    try:
        from ..memory.search import search_memory, render_hits
    except Exception:
        return ""
    try:
        hits = search_memory(query, top_k=_SUBAGENT_MATCHED_TOP_K)
    except Exception:
        return ""
    if not hits:
        return ""
    rendered = render_hits(hits, max_body_chars=400)
    if len(rendered) > _SUBAGENT_MATCHED_MAX_CHARS:
        rendered = rendered[:_SUBAGENT_MATCHED_MAX_CHARS] + "\n[... truncated ...]"
    return f"\n\n## Relevant Memory (BM25-matched to task)\n{rendered}"


def _extract_text_blocks(message):
    """Extract concatenated text from a normalized assistant message."""
    text_parts = []
    for block in message.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(block.get("text", ""))
    return "\n".join(text_parts)


def _make_subagent_outcome(
    status, content, steps, input_tokens, output_tokens, tool_calls,
    error=None,
    *,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    api_calls: int = 0,
):
    """Build a structured terminal result for the wrapper.

    ``cache_creation_tokens`` / ``cache_read_tokens`` / ``api_calls`` are
    plumbed through from ``LoopResult`` so parent stats can record the
    sub-agent's real cache activity and LLM call count.  Default to 0
    for callers that don't yet propagate them (e.g. external tests);
    ``stats.add_subagent_usage`` falls back to a +1 floor on the
    api_calls path so the dispatch still shows up in session counts.
    """
    outcome = {
        "status": status,
        "content": content,
        "steps": steps,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tool_calls": tool_calls,
        "cache_creation_tokens": cache_creation_tokens,
        "cache_read_tokens": cache_read_tokens,
        "api_calls": api_calls,
    }
    if error:
        outcome["error"] = error
    return outcome


def _format_subagent_failure(message, partial_output="", final_answer=""):
    """Format a predictable error body while preserving useful output."""
    parts = [message]
    final_answer = (final_answer or "").strip()
    partial_output = (partial_output or "").strip()

    if final_answer:
        parts.extend(["", "Best-effort final answer:", final_answer])

    if partial_output and partial_output != final_answer:
        parts.extend(["", "Partial output:", partial_output])

    return "\n".join(parts)


def _best_effort_final_answer(runtime_owner, messages, model_spec):
    """Ask the model for one last no-tools answer after max-step exhaustion."""
    _FINAL_SUFFIX = (
        "\n\n[SYSTEM: You have reached the maximum number of tool-use steps. "
        "Provide your best answer now WITHOUT using any tools.]"
    )
    response = runtime_owner.complete(
        {
            "system_prompt": _SUBAGENT_SYSTEM_PROMPT + _FINAL_SUFFIX,
            "messages": messages,
        },
        options=LLMOptions(
            max_output_tokens=_DEFAULT_MAX_TOKENS_PER_RESPONSE,
            timeout=STREAM_TIMEOUT,
            reasoning=get_reasoning_config_for_provider(
                model_spec.provider,
                max_output_tokens=_DEFAULT_MAX_TOKENS_PER_RESPONSE,
                model=model_spec.model,
            ),
            metadata={
                "component": "subagent",
                "mode": "fallback_complete",
                "fallback": True,
            },
        ),
        model_spec=model_spec,
    )
    usage = response.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return _extract_text_blocks(response), input_tokens, output_tokens


def _run_subagent(task, context, model_spec, max_steps, tool_schemas, tool_functions,
                  agent_id=None, custom_system_prompt=None, cancel_event=None,
                  progress_ids=None, memory_mode=_SUBAGENT_MEMORY_DEFAULT):
    """Run a sub-agent's tool loop to completion via AgentLoop engine.

    Delegates the entire step loop, tool execution, retry, context compaction,
    and truncation recovery to the shared engine.  Runs silently (no callbacks).

    ``memory_mode``: see ``_get_memory_context``.  Default is "none" for
    strict isolation — callers that want parent memory visible must opt in.
    """
    runtime_owner = _get_runtime_owner()

    # Build system prompt with optional scoped memory context.  The task
    # is the BM25 query when mode="matched"; for "none" it is ignored.
    system_prompt = custom_system_prompt or _SUBAGENT_SYSTEM_PROMPT
    system_prompt += _get_memory_context(query=task, mode=memory_mode)

    # Build initial messages
    user_content = task
    if context:
        user_content = f"Context:\n{context}\n\nTask:\n{task}"
    messages = [{"role": "user", "content": user_content}]

    # Pre-filtered tool source (closure over the whitelist-filtered lists)
    def tool_source():
        return tool_schemas, tool_functions

    # Configure the engine — conservative settings for sub-agents
    config = LoopConfig(
        max_steps=max_steps,
        initial_max_tokens=16_384,
        auto_scale_on_truncation=True,
        concurrent_tools=True,
        max_tool_workers=2,
        compact_messages=True,
        retry_attempts=10,
        streaming=False,
    )

    # Step progress callback → update the global tracker + background registry.
    # Uses a mutable container shared with the caller so bg_id can be updated
    # after handoff/dispatch without the worker needing to know about ID changes.
    _progress_ids = progress_ids if progress_ids is not None else {"spinner_id": agent_id, "bg_id": None}

    def _on_step_progress(step: int, max_s: int) -> None:
        sid = _progress_ids.get("spinner_id")
        if sid is not None:
            _subagent_tracker.update_progress(sid, step, max_s)
        bid = _progress_ids.get("bg_id")
        if bid is not None:
            _bg_registry.update_progress(bid, step, max_s)

    callbacks = LoopCallbacks(on_step_progress=_on_step_progress)

    # TODO: parent run-id propagation for
    # cross-process checkpoint correlation is not yet implemented.  Sub-
    # agents get a fresh run id from ``new_run_id()`` (assigned inside
    # ``RunState.prepare_for_run``) when checkpointing is enabled at
    # the sub-agent level, with no link back to the parent run id that
    # spawned them.  Future work: thread an explicit ``parent_run_id``
    # kwarg through ``_run_subagent`` and call ``loop.set_run_id(...)``
    # before ``loop.run(...)`` so checkpoint files form a navigable tree.
    loop = AgentLoop(
        runtime_owner=runtime_owner,
        config=config,
        callbacks=callbacks,
        tool_source=tool_source,
        model_spec=model_spec,
        cancel_event=cancel_event,
    )
    result = loop.run(system_prompt, messages)

    # ── Convert LoopResult → outcome dict ────────────────────────────────

    if result.status == "completed":
        content = result.text or "[Sub-agent completed but produced no text output]"
        return _make_subagent_outcome(
            _SUBAGENT_STATUS_COMPLETED,
            content,
            result.steps,
            result.total_input_tokens,
            result.total_output_tokens,
            result.tool_calls_count,
            cache_creation_tokens=result.total_cache_creation_tokens,
            cache_read_tokens=result.total_cache_read_tokens,
            api_calls=result.api_calls,
        )

    if result.status == "max_steps":
        final_answer = ""
        extra_in = extra_out = 0
        try:
            final_answer, extra_in, extra_out = _best_effort_final_answer(
                runtime_owner, messages, model_spec,
            )
        except Exception:
            pass

        content = _format_subagent_failure(
            f"Error: Sub-agent reached max_steps ({max_steps}).",
            partial_output=result.text,
            final_answer=final_answer,
        )
        return _make_subagent_outcome(
            _SUBAGENT_STATUS_MAX_STEPS,
            content,
            max_steps,
            result.total_input_tokens + extra_in,
            result.total_output_tokens + extra_out,
            result.tool_calls_count,
            cache_creation_tokens=result.total_cache_creation_tokens,
            cache_read_tokens=result.total_cache_read_tokens,
            api_calls=result.api_calls,
            error=f"max_steps:{max_steps}",
        )

    if result.status == "error":
        content = _format_subagent_failure(
            f"Error: Sub-agent API failure: {result.error}",
            partial_output=result.text,
        )
        return _make_subagent_outcome(
            _SUBAGENT_STATUS_API_ERROR,
            content,
            result.steps,
            result.total_input_tokens,
            result.total_output_tokens,
            result.tool_calls_count,
            cache_creation_tokens=result.total_cache_creation_tokens,
            cache_read_tokens=result.total_cache_read_tokens,
            api_calls=result.api_calls,
            error=result.error,
        )

    # interrupted or unknown status
    content = _format_subagent_failure(
        "Error: Sub-agent was interrupted.",
        partial_output=result.text,
    )
    return _make_subagent_outcome(
        _SUBAGENT_STATUS_API_ERROR,
        content,
        result.steps,
        result.total_input_tokens,
        result.total_output_tokens,
        result.tool_calls_count,
        cache_creation_tokens=result.total_cache_creation_tokens,
        cache_read_tokens=result.total_cache_read_tokens,
        api_calls=result.api_calls,
        error=result.status,
    )



# ─── Background Agent Registry ──────────────────────────────────────────────

# Lazy pool initialisation, mirroring
# ``tool_executor.get_tool_dispatch_executor``.  Eager creation here used
# to spin up a 5-worker daemon thread pool and register an ``atexit``
# shutdown hook on every ``import jyagent.tools.subagent``, even for
# callers that never dispatch a sub-agent (CLI subcommands, unit tests
# that only touch other tools, etc.).  The double-checked-locking pattern
# keeps the fast path lock-free after the first dispatch.

_bg_executor: concurrent.futures.ThreadPoolExecutor | None = None
_bg_executor_lock = threading.Lock()


def _get_bg_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the background sub-agent thread pool, creating it on first use."""
    global _bg_executor
    if _bg_executor is not None:
        return _bg_executor
    with _bg_executor_lock:
        if _bg_executor is not None:
            return _bg_executor
        _bg_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=5, thread_name_prefix="bg-subagent",
        )
        import atexit as _subagent_atexit
        _subagent_atexit.register(_bg_executor.shutdown, wait=False)
        return _bg_executor


# ─── Subagent outcome persistence ────────────────────────────────────────────
#
# 2026-05 (G2 + G9 fix): completed subagent outcomes are now persisted to disk
# at ``data/sessions/subagents/<pid>-<agent_id>.json`` so that:
#
#   1. ``check_agent(agent_id)`` is **idempotent** — the agent record is NOT
#      removed after the first successful read. Multiple reads return the
#      same outcome.
#   2. If ``check_agent(action='wait')`` is called from a tool wrapper that
#      times out client-side (e.g. the 120s outer cap), the agent's outcome
#      is preserved on disk and can be recovered on a follow-up call,
#      instead of being silently discarded.
#   3. Reasonably bounded RAM: the in-memory registry caps the number of
#      completed agents kept hot (newer ones evict older ones); evicted
#      outcomes live on disk and are loaded back on miss.
#
# Disk filename uses ``<pid>-<agent_id>`` so concurrent or successive jyagent
# processes don't collide on the per-process monotonic ``_next_id``.


_MAX_COMPLETED_IN_MEMORY = 100


def _subagent_persist_dir() -> str:
    """Return (and lazily create) the on-disk subagent outcome directory."""
    from ..config import SESSIONS_DIR
    path = os.path.join(SESSIONS_DIR, "subagents")
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    return path


def _subagent_outcome_path(agent_id: int) -> str:
    """File path for a given agent_id, namespaced by current process PID."""
    return os.path.join(
        _subagent_persist_dir(), f"{os.getpid()}-{agent_id}.json",
    )


def _persist_subagent_outcome(agent: "_BackgroundAgent") -> None:
    """Write a completed agent's outcome + metadata to disk (best-effort).

    Safe to call multiple times — overwrites atomically. Failures are
    swallowed: persistence is a recovery aid, not a correctness requirement.
    """
    if agent.outcome is None:
        return
    record = {
        "agent_id": agent.agent_id,
        "task": agent.task,
        "model": agent.model,
        "started_at": agent.started_at,
        "done_at": agent.done_at,
        "steps": agent.current_step,
        "max_steps": agent.current_max_steps,
        "outcome": agent.outcome,
    }
    path = _subagent_outcome_path(agent.agent_id)
    try:
        # Atomic write: tmp file + rename
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass


def _load_subagent_outcome_from_disk(agent_id: int) -> dict | None:
    """Best-effort load of a previously persisted outcome record for agent_id.

    Returns the full record dict (with ``outcome`` key) or None on miss /
    error. Only looks up the **current process's** record — across-process
    collisions are avoided by the PID-prefixed filename.
    """
    path = _subagent_outcome_path(agent_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@dataclass
class _BackgroundAgent:
    agent_id: int
    task: str                           # task preview (first 80 chars)
    future: concurrent.futures.Future
    cancel_event: threading.Event
    started_at: float
    model: str = ""
    current_step: int = 0
    current_max_steps: int = _DEFAULT_MAX_STEPS
    outcome: dict | None = None         # filled when future completes
    stats_recorded: bool = False
    done_at: float | None = None        # wall-clock time outcome was captured
    persisted: bool = False              # set once written to disk


class _BackgroundAgentRegistry:
    """Thread-safe registry for background sub-agent jobs.

    Lifecycle (post-2026-05 G2+G9 fix):

      register() → agent enters ``_agents`` as running.
      Future completes → ``mark_done()`` records outcome + persists to disk.
      ``get(id)`` → returns the agent record, alive OR completed, until
                    the agent is evicted by the ``_MAX_COMPLETED_IN_MEMORY``
                    cap. Reads are idempotent.
      Disk fallback: if ``get(id)`` misses (evicted or process restart of
                     the same id range), the caller may consult
                     ``_load_subagent_outcome_from_disk(id)`` directly to
                     reconstruct the answer. See ``check_agent``.
      ``cancel_all()`` → wipes RAM + on-disk outcomes (used by test fixture
                         and shutdown).
    """

    _MAX_CONCURRENT = 5

    def __init__(self):
        self._lock = threading.Lock()
        self._agents: dict[int, _BackgroundAgent] = {}
        # Insertion-ordered completed-agent IDs for FIFO eviction.
        self._completed_order: collections.deque[int] = collections.deque()
        self._next_id = 0

    def register(self, task, future, cancel_event, max_steps, model, started_at=None) -> int:
        """Register a background agent.  Returns agent_id.

        Also attaches a done callback to the future for automatic stats
        recording when the agent completes (even if nobody polls for it).
        """
        with self._lock:
            agent_id = self._next_id
            self._next_id += 1
            agent = _BackgroundAgent(
                agent_id=agent_id,
                task=task[:80] if len(task) > 80 else task,
                future=future,
                cancel_event=cancel_event,
                started_at=started_at or time.time(),
                model=model,
                current_max_steps=max_steps,
            )
            self._agents[agent_id] = agent

        # Auto-record stats when future completes (fire-and-forget)
        def _on_done(fut):
            try:
                _record_bg_stats(agent)
            except Exception:
                pass
        future.add_done_callback(_on_done)

        return agent_id

    def get(self, agent_id: int) -> _BackgroundAgent | None:
        with self._lock:
            return self._agents.get(agent_id)

    def remove(self, agent_id: int) -> None:
        """Hard-remove an agent from in-memory registry.

        Note: as of the 2026-05 G2+G9 fix, normal ``check_agent`` reads no
        longer call this — completed agents stay resident until evicted by
        the FIFO cap. ``remove`` survives for ``cancel_all`` and explicit
        kill paths only.
        """
        with self._lock:
            self._agents.pop(agent_id, None)
            try:
                self._completed_order.remove(agent_id)
            except ValueError:
                pass

    def mark_done(self, agent_id: int, outcome: dict) -> None:
        """Record a completed agent's outcome, persist to disk, evict if needed.

        Idempotent — calling twice is a no-op. The outcome is captured on
        the agent record, the agent is appended to the FIFO completion
        queue, and the record is written to disk. When more than
        ``_MAX_COMPLETED_IN_MEMORY`` completed agents are resident, the
        oldest is dropped from RAM (its on-disk record remains, so it can
        still be loaded by ``_load_subagent_outcome_from_disk``).
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                return
            if agent.outcome is None:
                agent.outcome = outcome
            if agent.done_at is None:
                agent.done_at = time.time()
            if agent_id not in self._completed_order:
                self._completed_order.append(agent_id)
        # Persist outside the lock — disk I/O shouldn't block other ops.
        if not agent.persisted:
            _persist_subagent_outcome(agent)
            agent.persisted = True
        # FIFO eviction of oldest completed agents past the cap.
        self._evict_excess_completed()

    def _evict_excess_completed(self) -> None:
        """Drop oldest completed agents from RAM once over the in-memory cap."""
        with self._lock:
            while len(self._completed_order) > _MAX_COMPLETED_IN_MEMORY:
                oldest = self._completed_order.popleft()
                self._agents.pop(oldest, None)

    def list_active(self) -> list[dict]:
        """Return summary of all active background agents."""
        now = time.time()
        with self._lock:
            result = []
            for a in self._agents.values():
                done = a.future.done()
                result.append({
                    "agent_id": a.agent_id,
                    "task": a.task,
                    "status": "done" if done else "running",
                    "elapsed_seconds": round(now - a.started_at, 1),
                    "step": a.current_step,
                    "max_steps": a.current_max_steps,
                })
            return result

    def update_progress(self, agent_id: int, step: int, max_steps: int = 0):
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is not None:
                agent.current_step = step
                if max_steps:
                    agent.current_max_steps = max_steps

    def cancel_all(self):
        """Cancel all background agents and clear persistence (for test
        cleanup and shutdown).

        Wipes both RAM and on-disk outcome records for the current process,
        so test fixtures that call ``cancel_all()`` between tests don't
        leak stale outcomes into later tests that happen to reuse the same
        ``agent_id`` (the per-process ``_next_id`` is not reset).
        """
        with self._lock:
            agents = list(self._agents.values())
        for a in agents:
            a.cancel_event.set()
            try:
                a.future.result(timeout=10)
            except Exception:
                pass
        with self._lock:
            self._agents.clear()
            self._completed_order.clear()
        # Best-effort wipe of this process's on-disk records.
        try:
            persist_dir = _subagent_persist_dir()
            prefix = f"{os.getpid()}-"
            for fname in os.listdir(persist_dir):
                if fname.startswith(prefix) and (
                    fname.endswith(".json") or fname.endswith(".json.tmp")
                ):
                    try:
                        os.remove(os.path.join(persist_dir, fname))
                    except Exception:
                        pass
        except Exception:
            pass


# Module-level singleton
_bg_registry = _BackgroundAgentRegistry()


def _record_bg_stats(agent: _BackgroundAgent) -> None:
    """Record token usage for a background agent in parent stats (best-effort).

    Thread-safe: uses the stats_recorded flag as a one-shot guard.
    """
    if agent.stats_recorded:
        return
    agent.stats_recorded = True
    try:
        outcome = agent.outcome
        if outcome is None and agent.future.done():
            try:
                outcome = agent.future.result(timeout=0)
                agent.outcome = outcome
            except Exception:
                return
        if outcome is None:
            return
        parent_stats = get_stats()
        elapsed = time.time() - agent.started_at
        parent_stats.record_subagent_usage(
            outcome.get("input_tokens", 0),
            outcome.get("output_tokens", 0),
            "",  # provider — not tracked in bg agent currently
            agent.model,
            task_preview=agent.task,
            elapsed=elapsed,
            status=outcome.get("status", "unknown"),
            steps=outcome.get("steps", 0),
            tool_calls=outcome.get("tool_calls", 0),
            cache_creation_tokens=outcome.get("cache_creation_tokens", 0),
            cache_read_tokens=outcome.get("cache_read_tokens", 0),
            api_calls=outcome.get("api_calls", 0),
        )
    except Exception:
        pass  # stats recording is best-effort



def _outcome_from_disk_record(record: dict) -> ToolResult:
    """Reconstruct a ``check_agent`` ToolResult from a persisted disk record.

    Used when an agent_id has been evicted from the in-memory registry
    (or referenced by a follow-up call after a prior ``check_agent``).
    """
    outcome = record.get("outcome") or {}
    status = outcome.get("status", "completed")
    answer = outcome.get("content", "")
    return ToolResult(answer, is_error=(status != _SUBAGENT_STATUS_COMPLETED))


def check_agent(
    agent_id: int = -1,
    action: str = "status",
    wait_timeout_seconds: int = 60,
) -> ToolResult:
    """Check on or manage background sub-agents.

    Reads are idempotent — calling ``check_agent`` on a completed agent
    multiple times returns the same outcome until it is evicted from
    the in-memory registry by the FIFO cap (after which it is still
    recoverable from disk, see ``_load_subagent_outcome_from_disk``).

    On a registry miss, falls back to disk: an agent whose outcome
    survived a prior client-side wait-timeout can still be retrieved by
    its id.
    """
    if action == "list":
        return ToolResult(json.dumps(_bg_registry.list_active()))

    if agent_id < 0:
        return ToolResult(
            "Error: agent_id is required for status/kill/wait actions.",
            is_error=True,
        )

    agent = _bg_registry.get(agent_id)
    if agent is None:
        # Disk fallback — the agent may have been evicted from RAM or its
        # outcome may have been persisted by a prior, lost wait-timeout.
        record = _load_subagent_outcome_from_disk(agent_id)
        if record is not None:
            return _outcome_from_disk_record(record)
        return ToolResult(
            f"Error: No background agent with id={agent_id}. "
            "Use check_agent(action='list') to see active agents.",
            is_error=True,
        )

    if action == "wait":
        # Block up to the cap; no-op if already done. After the wait, fall
        # through to the same "status" handling below so the response shape
        # is identical regardless of whether the agent finished during the
        # wait or is still running.
        if not agent.future.done():
            wait_s = max(
                1, min(int(wait_timeout_seconds), _BG_AGENT_WAIT_MAX_SECONDS)
            )
            try:
                agent.future.result(timeout=wait_s)
            except concurrent.futures.TimeoutError:
                pass  # still running — caller can re-wait or check status
            except Exception:
                # Real failure — let the status branch surface it via the
                # standard error path below.
                pass
        # Fall through to status handling.

    elif action == "kill":
        agent.cancel_event.set()  # cooperative cancel
        try:
            agent.future.result(timeout=10)  # wait for clean shutdown
        except concurrent.futures.TimeoutError:
            # Agent is still running — report honest status
            return ToolResult(json.dumps({
                "status": "cancelling",
                "agent_id": agent_id,
                "message": (
                    f"Cancel signal sent but agent is still running after 10s. "
                    f"It will stop at the next loop boundary. "
                    f"Use check_agent(agent_id={agent_id}) to poll."
                ),
            }))
        except Exception:
            pass
        _record_bg_stats(agent)
        _bg_registry.remove(agent_id)
        return ToolResult(f"Agent {agent_id} cancelled.")

    # action == "status" (or "wait" falling through after blocking)
    if agent.future.done():
        try:
            outcome = agent.future.result(timeout=0)
        except Exception as e:
            # Real exception from the worker — keep the record so a follow-up
            # call returns the same error rather than "not found".
            err_outcome = {"status": "exception", "content": f"Background agent failed: {e}"}
            if agent.outcome is None:
                agent.outcome = err_outcome
            _bg_registry.mark_done(agent_id, agent.outcome)
            return ToolResult(
                f"Error: Background agent failed: {e}", is_error=True,
            )

        # Capture outcome on the agent record, record stats, persist + cap.
        # All three operations are idempotent — safe to call on every
        # check_agent invocation after completion.
        if agent.outcome is None:
            agent.outcome = outcome
        _record_bg_stats(agent)
        _bg_registry.mark_done(agent_id, outcome)

        elapsed = (agent.done_at or time.time()) - agent.started_at
        status = outcome.get("status", "completed")
        answer = outcome.get("content", "")
        steps = outcome.get("steps", 0)
        in_tok = outcome.get("input_tokens", 0)
        out_tok = outcome.get("output_tokens", 0)
        tool_calls = outcome.get("tool_calls", 0)

        # Terminal summary — print only ONCE per agent (first observation).
        # ``summary_printed`` is an attribute we set lazily so the dataclass
        # default doesn't need a migration.
        if not getattr(agent, "summary_printed", False):
            status_icon = "✓" if status == _SUBAGENT_STATUS_COMPLETED else "✗"
            status_color = COLOR_GREEN if status == _SUBAGENT_STATUS_COMPLETED else COLOR_RED
            sys.stdout.write(
                f"{status_color}  {status_icon} 🤖 Background agent done{COLOR_RESET}"
                f"{COLOR_DIM} ({status}, {elapsed:.1f}s, {steps} steps, {tool_calls} tool calls, "
                f"{in_tok}+{out_tok} tokens){COLOR_RESET}\n"
            )
            sys.stdout.flush()
            agent.summary_printed = True

        return ToolResult(answer, is_error=(status != _SUBAGENT_STATUS_COMPLETED))
    else:
        # Still running
        elapsed = time.time() - agent.started_at
        return ToolResult(json.dumps({
            "status": "running",
            "agent_id": agent_id,
            "elapsed_seconds": round(elapsed, 1),
            "step": agent.current_step,
            "max_steps": agent.current_max_steps,
            "task": agent.task,
        }))


# ─── Main tool function ──────────────────────────────────────────────────────

_BG_GRACE_PERIOD = 30  # seconds to wait before backgrounding
_FG_DEFAULT_TIMEOUT = 600   # default foreground timeout
_BG_DEFAULT_TIMEOUT = 1800  # default background timeout


def dispatch_agent(
    task: str,
    context: str = "",
    tool_whitelist: list = None,
    background: bool = False,
    timeout: int = 0,
    memory_mode: str = _SUBAGENT_MEMORY_DEFAULT,
    _cancel_event: "threading.Event | None" = None,
) -> ToolResult:
    """Spawn a sub-agent to handle a focused subtask.

    ``memory_mode``: scope of parent memory exposed to the sub-agent.
    Defaults to "none" (strict isolation); see ``_get_memory_context``.
    Invalid values silently degrade to "none" — we never fail a sub-agent
    dispatch on a bad memory knob.

    ``_cancel_event`` (runtime-injected, optional): when the parent
    agent's cancel event fires, the sub-agent's own cancel event is
    mirrored from it via a small daemon watcher thread, so a Ctrl-C in
    the parent reaches the foreground sub-agent's loop on the next
    cancel-aware checkpoint without waiting for ``effective_timeout``.
    Background sub-agents are unaffected (they survive their parent by
    design — that's what background means).
    """
    if memory_mode not in _SUBAGENT_MEMORY_MODES:
        memory_mode = _SUBAGENT_MEMORY_DEFAULT
    max_steps = _DEFAULT_MAX_STEPS
    custom_system_prompt = None

    # Guard: nesting depth
    depth = _nesting_depth.get()
    if depth >= _MAX_NESTING:
        return ToolResult(
            f"Error: Maximum sub-agent nesting depth ({_MAX_NESTING}) reached. "
            f"Cannot spawn deeper sub-agents.",
            is_error=True,
        )

    # Resolve effective timeout
    if timeout > 0:
        effective_timeout = max(60, min(3600, timeout))
    elif background:
        effective_timeout = _BG_DEFAULT_TIMEOUT
    else:
        effective_timeout = _FG_DEFAULT_TIMEOUT

    runtime_owner = _get_runtime_owner()
    model_spec = runtime_owner.model_spec

    # Build tool schemas & functions for the sub-agent.
    # freeze() gives a batch-atomic deep-copied snapshot where schemas
    # can't be mutated post-freeze.
    registry = get_registry()
    _batch = registry.freeze()
    all_schemas = list(_batch.schemas)
    all_functions = dict(_batch.functions)

    if tool_whitelist:
        # Filter to only whitelisted tools (always include dispatch_agent if not excluded)
        whitelist_set = set(tool_whitelist)
        tool_schemas = [s for s in all_schemas if s["name"] in whitelist_set]
        tool_functions = {k: v for k, v in all_functions.items() if k in whitelist_set}
    else:
        tool_schemas = all_schemas
        tool_functions = all_functions

    # Remove dispatch_agent from sub-agent tools if at max nesting - 1
    if depth >= _MAX_NESTING - 1:
        tool_schemas = [s for s in tool_schemas if s["name"] != "dispatch_agent"]
        tool_functions = {k: v for k, v in tool_functions.items() if k != "dispatch_agent"}

    task_preview = task[:80] if len(task) > 80 else task
    cancel_event = threading.Event()
    # If the runtime injected a parent cancel event (Ctrl-C / programmatic
    # cancel from the outer agent loop), mirror it onto our local
    # ``cancel_event`` so the sub-agent's loop observes parent cancellation
    # within ~500ms.  ``_stop_cancel_mirror()`` is called in the finally
    # blocks below to retire the watcher thread when the sub-agent
    # completes naturally — without it, every dispatch_agent invocation
    # would leak a daemon watcher until process exit.
    _stop_cancel_mirror = _install_cancel_mirror(_cancel_event, cancel_event)

    # Increment nesting depth, then snapshot the context so the worker
    # thread inherits the updated value.  ThreadPoolExecutor does NOT
    # auto-propagate ContextVars; we must use copy_context().run().
    _depth_token = _nesting_depth.set(depth + 1)
    ctx = contextvars.copy_context()

    if background:
        # ── Background path ─────────────────────────────────────────────
        # Register spinner for the grace period
        spinner_id = _subagent_tracker.add(task, max_steps)
        t0 = time.time()

        # Shared mutable dict so the worker's progress callback can target
        # the correct bg_id once we register it after the grace period.
        progress_ids = {"spinner_id": spinner_id, "bg_id": None}

        future = None
        try:
            future = _get_bg_executor().submit(
                ctx.run, _run_subagent, task, context, model_spec,
                max_steps, tool_schemas, tool_functions,
                spinner_id, custom_system_prompt, cancel_event,
                progress_ids, memory_mode,
            )

            # Grace period: wait up to 30s for fast finish
            try:
                outcome = future.result(timeout=_BG_GRACE_PERIOD)
            except concurrent.futures.TimeoutError:
                # Still running — register in background registry, remove spinner
                _subagent_tracker.remove(spinner_id)
                bg_id = _bg_registry.register(
                    task_preview, future, cancel_event, max_steps,
                    model_spec.model, started_at=t0,
                )
                # Update the shared progress_ids so the running worker
                # reports progress to the correct bg_id going forward.
                progress_ids["bg_id"] = bg_id
                progress_ids["spinner_id"] = None  # stop updating dead spinner
                return ToolResult(json.dumps({
                    "status": "dispatched",
                    "agent_id": bg_id,
                    "message": f"Background agent running. Call check_agent(agent_id={bg_id}) to get results.",
                    "task": task_preview,
                }))
        except KeyboardInterrupt:
            cancel_event.set()
            _subagent_tracker.remove(spinner_id)
            raise
        except Exception as e:
            _subagent_tracker.remove(spinner_id)
            error_text = str(e)
            error_detail = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            return ToolResult(
                f"Error: Sub-agent failed: {error_text}\n{error_detail}",
                is_error=True,
            )
        finally:
            _nesting_depth.reset(_depth_token)
            _subagent_tracker.remove(spinner_id)
            # Retire the parent→child cancel mirror UNLESS the future is
            # still running (handoff-to-bg case): in that case the bg
            # registry takes ownership and we want the mirror to keep
            # propagating parent cancel into the long-running sub-agent.
            # The mirror is a daemon thread so leaking it on handoff is
            # bounded by process lifetime.
            if future is None or future.done():
                _stop_cancel_mirror()

        # Fast finish — return inline (same as sync path)
        elapsed = time.time() - t0
        return _finalize_outcome(outcome, elapsed, model_spec, task_preview)

    else:
        # ── Foreground (sync) path ──────────────────────────────────────
        # Register with global tracker (spinner)
        agent_id = _subagent_tracker.add(task, max_steps)
        t0 = time.time()
        _depth_reset = False  # guard against double-reset of ContextVar token

        # Shared mutable dict for progress routing after potential handoff.
        progress_ids = {"spinner_id": agent_id, "bg_id": None}

        future = None
        try:
            # Submit to the shared executor (NOT a per-call `with ThreadPoolExecutor`
            # which blocks on __exit__ with shutdown(wait=True), making timeout
            # handoff impossible).
            future = _get_bg_executor().submit(
                ctx.run, _run_subagent, task, context, model_spec,
                max_steps, tool_schemas, tool_functions,
                agent_id, custom_system_prompt, cancel_event,
                progress_ids, memory_mode,
            )
            try:
                outcome = future.result(timeout=effective_timeout)
            except concurrent.futures.TimeoutError:
                # Soft handoff: register as background agent instead of erroring.
                # Because we use the shared _bg_executor (no `with` block),
                # this return is non-blocking — the worker keeps running.
                _subagent_tracker.remove(agent_id)
                bg_id = _bg_registry.register(
                    task_preview, future, cancel_event, max_steps,
                    model_spec.model, started_at=t0,
                )
                # Update progress routing
                progress_ids["bg_id"] = bg_id
                progress_ids["spinner_id"] = None
                _nesting_depth.reset(_depth_token)
                _depth_reset = True
                return ToolResult(json.dumps({
                    "status": "timeout_handoff",
                    "agent_id": bg_id,
                    "message": (
                        f"Agent exceeded {effective_timeout}s. Handed off to background. "
                        f"Call check_agent(agent_id={bg_id}) to get results."
                    ),
                }))
        except KeyboardInterrupt:
            cancel_event.set()
            _subagent_tracker.remove(agent_id)
            raise
        except Exception as e:
            _subagent_tracker.remove(agent_id)
            error_text = str(e)
            error_detail = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            return ToolResult(
                f"Error: Sub-agent failed: {error_text}\n{error_detail}",
                is_error=True,
            )
        finally:
            if not _depth_reset:
                _nesting_depth.reset(_depth_token)
            _subagent_tracker.remove(agent_id)
            # Retire the parent→child cancel mirror UNLESS the future is
            # still running (timeout-handoff case): in that case the bg
            # registry takes ownership and we want the mirror to keep
            # propagating parent cancel into the long-running sub-agent.
            if future is None or future.done():
                _stop_cancel_mirror()

        elapsed = time.time() - t0
        return _finalize_outcome(outcome, elapsed, model_spec, task_preview)


def _format_subagent_envelope(
    *,
    status: str,
    answer: str,
    elapsed: float,
    steps: int,
    tool_calls: int,
    input_tokens: int,
    output_tokens: int,
    error: str = "",
) -> str:
    """Wrap a sub-agent's free-form answer in a structured Markdown envelope.

    The envelope is stable and machine-parseable-ish — the parent LLM sees
    status and cost metadata upfront, which substantially improves its
    ability to reason about sub-agent outputs without ambiguity (this was
    a flagged P1 from the 2026-04-18 joint Codex + Claude Code review).

    Headings are plain Markdown so the parent model parses them trivially.
    """
    lines = [
        "## Sub-agent Result",
        f"**Status:** {status}",
        (
            f"**Stats:** {steps} step(s) · {tool_calls} tool call(s) · "
            f"{input_tokens}+{output_tokens} tokens · {elapsed:.1f}s"
        ),
    ]
    if error:
        lines.append(f"**Error:** {error}")
    body = (answer or "").rstrip()
    if body:
        lines.extend(["", "### Response", body])
    else:
        lines.extend(["", "### Response", "_(sub-agent produced no text output)_"])
    return "\n".join(lines)


def _finalize_outcome(outcome, elapsed, model_spec, task_preview):
    """Print terminal summary, record stats, return ToolResult for a completed sub-agent.

    Output format: structured Markdown envelope (see ``_format_subagent_envelope``)
    so the parent model sees status + cost + response as distinct sections,
    rather than having to infer them from free-form text.
    """
    status = outcome.get("status", _SUBAGENT_STATUS_COMPLETED)
    answer = outcome.get("content", "")
    steps = outcome.get("steps", 0)
    in_tok = outcome.get("input_tokens", 0)
    out_tok = outcome.get("output_tokens", 0)
    tool_calls = outcome.get("tool_calls", 0)
    err = outcome.get("error", "")

    status_icon = "✓" if status == _SUBAGENT_STATUS_COMPLETED else "✗"
    status_color = COLOR_GREEN if status == _SUBAGENT_STATUS_COMPLETED else COLOR_RED
    sys.stdout.write(
        f"{status_color}  {status_icon} 🤖 Sub-agent done{COLOR_RESET}"
        f"{COLOR_DIM} ({status}, {elapsed:.1f}s, {steps} steps, {tool_calls} tool calls, "
        f"{in_tok}+{out_tok} tokens){COLOR_RESET}\n"
    )
    sys.stdout.flush()

    # Record token usage in parent stats
    try:
        parent_stats = get_stats()
        parent_stats.record_subagent_usage(
            in_tok, out_tok, model_spec.provider, model_spec.model,
            task_preview=task_preview,
            elapsed=elapsed,
            status=status,
            steps=steps,
            tool_calls=tool_calls,
            cache_creation_tokens=outcome.get("cache_creation_tokens", 0),
            cache_read_tokens=outcome.get("cache_read_tokens", 0),
            api_calls=outcome.get("api_calls", 0),
        )
    except Exception:
        pass  # stats recording is best-effort

    envelope = _format_subagent_envelope(
        status=status,
        answer=answer,
        elapsed=elapsed,
        steps=steps,
        tool_calls=tool_calls,
        input_tokens=in_tok,
        output_tokens=out_tok,
        error=err,
    )
    return ToolResult(envelope, is_error=(status != _SUBAGENT_STATUS_COMPLETED))
