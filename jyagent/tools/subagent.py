# Sub-Agent Tool — Spawn focused child agents for parallel/specialized subtasks.
#
# The parent (lead) agent decomposes a complex task into sub-problems,
# dispatches sub-agents for each, and synthesizes results.
#
# Sub-agents run silently (no terminal streaming), have their own context
# window and message history, and return only their final answer to the parent.

import json
import sys
import time
import traceback
import threading
import contextvars
import concurrent.futures
from dataclasses import dataclass, field

try:
    from ..config import (
        STREAM_TIMEOUT, get_active_model_spec, get_reasoning_config_for_provider,
    )
    from ..loop_engine import AgentLoop, LoopConfig, LoopCallbacks
    from ..registry import get_registry
    from ..runtime import RuntimeOptions, RuntimeOwner
    from ..toolresult import ToolResult
    from ..session_stats import get_stats
except ImportError:
    from jyagent.config import (
        STREAM_TIMEOUT, get_active_model_spec, get_reasoning_config_for_provider,
    )
    from jyagent.loop_engine import AgentLoop, LoopConfig, LoopCallbacks
    from jyagent.registry import get_registry
    from jyagent.runtime import RuntimeOptions, RuntimeOwner
    from jyagent.toolresult import ToolResult
    from jyagent.session_stats import get_stats


# ─── Defaults ────────────────────────────────────────────────────────────────

_DEFAULT_MAX_STEPS = 30
_DEFAULT_MAX_TOKENS_PER_RESPONSE = 8192
_SUBAGENT_STATUS_COMPLETED = "completed"
_SUBAGENT_STATUS_MAX_STEPS = "max_steps"
_SUBAGENT_STATUS_API_ERROR = "api_error"

# Track nesting to prevent runaway recursion.
# Uses contextvars.ContextVar + explicit copy_context().run() because sub-agents
# run in ThreadPoolExecutor worker threads.  ThreadPoolExecutor does NOT auto-
# propagate ContextVars; we snapshot the context after incrementing depth and
# pass ctx.run(fn, ...) to executor.submit() so the worker inherits it.
_nesting_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_nesting_depth", default=0,
)
_MAX_NESTING = 2  # sub-agent can spawn sub-sub-agent, but no deeper


# ─── Runtime owner access ─────────────────────────────────────────────────────

_runtime_owner = None
_client = None


def set_runtime_owner(runtime_owner):
    """Called during agent startup to share the active runtime owner."""
    global _runtime_owner, _client
    _runtime_owner = runtime_owner
    _client = None


def set_client(runtime_owner):
    """Backward-compatible alias for older call sites/tests."""
    global _runtime_owner, _client
    if isinstance(runtime_owner, RuntimeOwner):
        set_runtime_owner(runtime_owner)
        return
    _runtime_owner = None
    _client = runtime_owner


def _get_client():
    """Backward-compatible helper for tests that monkeypatch the raw client."""
    return _client


class _LegacyClientRuntimeOwner:
    """Compatibility shim for tests that still provide a fake Anthropic client.

    DEPRECATED: Tests should use RuntimeOwner with mock adapters instead
    of injecting raw Anthropic SDK clients. This shim exists only to keep
    existing tests working during the migration.
    """

    def __init__(self, client):
        self._client = client
        self.model_spec = get_active_model_spec()

    def complete(self, context, options=None, model_spec=None):
        # Lazy imports: only needed when running through the legacy shim
        from ..runtime.reasoning import build_anthropic_request_reasoning
        from ..runtime.providers._anthropic_helpers import (
            assistant_from_response, convert_messages,
        )

        model_spec = model_spec or self.model_spec
        max_tokens = _DEFAULT_MAX_TOKENS_PER_RESPONSE
        timeout = STREAM_TIMEOUT
        if options is not None:
            max_tokens = options.max_output_tokens or max_tokens
            timeout = options.timeout or timeout

        kwargs = {
            "model": model_spec.model,
            "max_tokens": max_tokens,
            "system": context.get("system_prompt", ""),
            "messages": convert_messages(model_spec, context.get("messages", [])),
        }
        if options is not None and options.reasoning is not None:
            thinking, output_config = build_anthropic_request_reasoning(options.reasoning, model=model_spec.model)
            if thinking is not None:
                kwargs["thinking"] = thinking
            if output_config is not None:
                kwargs["output_config"] = output_config
        if context.get("tools"):
            kwargs["tools"] = context["tools"]
        stream_fn = getattr(self._client.messages, "stream", None)
        if callable(stream_fn):
            with stream_fn(**kwargs, timeout=timeout) as stream:
                for _event in stream:
                    pass
                response = stream.get_final_message()
        else:
            response = self._client.messages.create(**kwargs, timeout=timeout)
        return assistant_from_response(model_spec, response)


def _get_runtime_owner():
    """Get the shared runtime owner, creating a default one if needed."""
    global _runtime_owner
    client = _get_client()
    if client is not None:
        return _LegacyClientRuntimeOwner(client)
    if _runtime_owner is None:
        _runtime_owner = RuntimeOwner(get_active_model_spec())
    return _runtime_owner


# ─── Schema ───────────────────────────────────────────────────────────────────

TOOL_SCHEMA = {
    "name": "dispatch_agent",
    "description": (
        "Spawn a focused sub-agent to handle a specific subtask. The sub-agent gets its own "
        "context window, runs silently, and returns its final answer. Use this for: "
        "(1) parallel research — dispatch multiple sub-agents for different search queries, "
        "(2) specialized tasks — give a sub-agent a focused job like 'analyze this file', "
        "(3) context isolation — prevent a large subtask from polluting your main context. "
        "The sub-agent has access to the same tools as you (or a subset via tool_whitelist). "
        "Keep task descriptions specific and self-contained — the sub-agent has NO access to "
        "your conversation history. "
        "Set background=true for tasks that may take over 2 minutes; you'll get an agent_id "
        "and can poll with check_agent()."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "Clear, specific task description. Must be self-contained — include all "
                    "context the sub-agent needs. Example: 'Search the web for Python 3.14 "
                    "breaking changes and summarize the top 5 issues with links.'"
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Optional additional context or data for the sub-agent. "
                    "Use this to pass relevant information from your conversation."
                ),
            },
            "tool_whitelist": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of tool names the sub-agent can use. "
                    "Empty = all tools available. Example: ['web_fetch', 'read_file']"
                ),
            },
            "background": {
                "type": "boolean",
                "description": (
                    "If true, run in background and return immediately with agent_id. "
                    "Poll with check_agent(). Use for tasks likely to exceed 2 minutes "
                    "(research, multi-step analysis)."
                ),
            },
            "timeout": {
                "type": "integer",
                "minimum": 60,
                "maximum": 1800,
                "description": (
                    "Max runtime in seconds (default: 300 foreground, 900 background). "
                    "Clamp: 60-1800."
                ),
            },
        },
        "required": ["task"],
    },
}


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

def _get_memory_context() -> str:
    """Load the first 4KB of MEMORY.md and return it as a formatted context block.

    Returns an empty string if the file doesn't exist or is empty.
    """
    try:
        from ..config import MEMORY_MD_FILE
    except ImportError:
        from jyagent.config import MEMORY_MD_FILE

    import os
    if not os.path.isfile(MEMORY_MD_FILE):
        return ""
    try:
        with open(MEMORY_MD_FILE, "r", encoding="utf-8") as f:
            content = f.read(4096)
        content = content.strip()
        if not content:
            return ""
        return f"\n\n## Project Memory\n{content}"
    except Exception:
        return ""


def _extract_text_blocks(message):
    """Extract concatenated text from a normalized assistant message."""
    text_parts = []
    for block in message.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(block.get("text", ""))
    return "\n".join(text_parts)


def _make_subagent_outcome(status, content, steps, input_tokens, output_tokens, tool_calls, error=None):
    """Build a structured terminal result for the wrapper."""
    outcome = {
        "status": status,
        "content": content,
        "steps": steps,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tool_calls": tool_calls,
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
        options=RuntimeOptions(
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
                  progress_ids=None):
    """Run a sub-agent's tool loop to completion via AgentLoop engine.

    Delegates the entire step loop, tool execution, retry, context compaction,
    and truncation recovery to the shared engine.  Runs silently (no callbacks).
    """
    runtime_owner = _get_runtime_owner()

    # Build system prompt with optional memory context
    system_prompt = custom_system_prompt or _SUBAGENT_SYSTEM_PROMPT
    system_prompt += _get_memory_context()

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
        retry_attempts=2,
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
        error=result.status,
    )


# ─── Terminal status display ─────────────────────────────────────────────────

COLOR_DIM = "\033[2m"
COLOR_RESET = "\033[0m"
COLOR_CYAN = "\033[1;36m"
COLOR_GREEN = "\033[0;32m"
COLOR_RED = "\033[1;31m"
COLOR_YELLOW = "\033[1;33m"


class _SubagentTracker:
    """Global consolidated spinner for all active sub-agents.

    Thread-safe.  Automatically starts/stops the animation thread when
    the first/last sub-agent registers/deregisters.

    Single subagent:  "🤖 Sub-agent: task preview (45s)"
    Multiple:         "🤖 3 sub-agents running (45s)"
    With steps:       "🤖 Sub-agent: task preview (3/30 steps, 45s)"
    """

    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self):
        self._lock = threading.Lock()
        self._agents: dict[int, dict] = {}  # id -> {task, t0, step, max_steps}
        self._next_id = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add(self, task: str, max_steps: int = 0) -> int:
        """Register a new sub-agent.  Returns a unique agent id."""
        preview = task[:60] + "..." if len(task) > 60 else task
        with self._lock:
            agent_id = self._next_id
            self._next_id += 1
            self._agents[agent_id] = {
                "task": preview,
                "t0": time.time(),
                "step": 0,
                "max_steps": max_steps,
            }
            if len(self._agents) == 1:
                self._start_animation()
        return agent_id

    def remove(self, agent_id: int) -> None:
        """Deregister a completed sub-agent."""
        with self._lock:
            self._agents.pop(agent_id, None)
            if not self._agents:
                self._stop_animation()

    def update_progress(self, agent_id: int, step: int, max_steps: int = 0) -> None:
        """Update the step progress for an active sub-agent."""
        with self._lock:
            info = self._agents.get(agent_id)
            if info is not None:
                info["step"] = step
                if max_steps:
                    info["max_steps"] = max_steps

    # ── Animation lifecycle ──────────────────────────────────────────────

    def _start_animation(self):
        """Start the animation thread.  Caller must hold self._lock."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def _stop_animation(self):
        """Stop the animation thread.  Caller must hold self._lock."""
        self._stop.set()
        t = self._thread
        self._thread = None
        if t is not None:
            # Release the lock during join to avoid deadlock
            self._lock.release()
            try:
                t.join(timeout=1)
            finally:
                self._lock.acquire()

    def _animate(self):
        idx = 0
        while not self._stop.is_set():
            with self._lock:
                agents = dict(self._agents)

            if not agents:
                break

            frame = self._FRAMES[idx % len(self._FRAMES)]
            now = time.time()
            count = len(agents)

            if count == 1:
                info = next(iter(agents.values()))
                elapsed = now - info["t0"]
                step = info["step"]
                max_steps = info["max_steps"]
                if step and max_steps:
                    time_info = f"({step}/{max_steps} steps, {elapsed:.0f}s)"
                elif step:
                    time_info = f"(step {step}, {elapsed:.0f}s)"
                else:
                    time_info = f"({elapsed:.0f}s)"
                line = f"\r{COLOR_DIM}  {frame} 🤖 Sub-agent: {info['task']} {time_info}{COLOR_RESET}"
            else:
                # Use the oldest t0 for elapsed
                oldest_t0 = min(a["t0"] for a in agents.values())
                elapsed = now - oldest_t0
                line = f"\r{COLOR_DIM}  {frame} 🤖 {count} sub-agents running ({elapsed:.0f}s){COLOR_RESET}"

            sys.stdout.write(line)
            sys.stdout.flush()
            idx += 1
            self._stop.wait(0.1)

        # Clear the spinner line
        sys.stdout.write("\r" + " " * 120 + "\r")
        sys.stdout.flush()


# Global singleton
_subagent_tracker = _SubagentTracker()


# ─── Background Agent Registry ──────────────────────────────────────────────

_bg_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=5, thread_name_prefix="bg-subagent",
)

import atexit as _subagent_atexit
_subagent_atexit.register(_bg_executor.shutdown, wait=False)


@dataclass
class _BackgroundAgent:
    agent_id: int
    task: str                           # task preview (first 80 chars)
    future: concurrent.futures.Future
    cancel_event: threading.Event
    started_at: float
    model: str = ""
    current_step: int = 0
    current_max_steps: int = 30
    outcome: dict | None = None         # filled when future completes
    stats_recorded: bool = False


class _BackgroundAgentRegistry:
    """Thread-safe registry for background sub-agent jobs."""

    _MAX_CONCURRENT = 5

    def __init__(self):
        self._lock = threading.Lock()
        self._agents: dict[int, _BackgroundAgent] = {}
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
        with self._lock:
            self._agents.pop(agent_id, None)

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
        """Cancel all background agents (for cleanup on exit)."""
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
        )
    except Exception:
        pass  # stats recording is best-effort


# ─── check_agent tool ───────────────────────────────────────────────────────

CHECK_AGENT_SCHEMA = {
    "name": "check_agent",
    "description": (
        "Check status of a background sub-agent or manage background agents. "
        "Use after dispatch_agent(background=True) to poll for results. "
        "action='status' returns progress/result. action='kill' cancels the agent. "
        "action='list' shows all active background agents."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "integer",
                "description": "ID of the background agent (from dispatch_agent response). Required for status/kill.",
            },
            "action": {
                "type": "string",
                "enum": ["status", "kill", "list"],
                "description": "Action: 'status' (default) check progress, 'kill' cancel agent, 'list' show all active.",
            },
        },
        "required": [],
    },
}


def check_agent(agent_id: int = -1, action: str = "status") -> ToolResult:
    """Check on or manage background sub-agents."""
    if action == "list":
        return ToolResult(json.dumps(_bg_registry.list_active()))

    if agent_id < 0:
        return ToolResult(
            "Error: agent_id is required for status/kill actions.",
            is_error=True,
        )

    agent = _bg_registry.get(agent_id)
    if agent is None:
        return ToolResult(
            f"Error: No background agent with id={agent_id}. "
            "Use check_agent(action='list') to see active agents.",
            is_error=True,
        )

    if action == "kill":
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

    # action == "status"
    if agent.future.done():
        try:
            outcome = agent.future.result(timeout=0)
        except Exception as e:
            _bg_registry.remove(agent_id)
            return ToolResult(
                f"Error: Background agent failed: {e}", is_error=True,
            )

        # Record stats, print terminal summary, return result
        agent.outcome = outcome
        _record_bg_stats(agent)
        _bg_registry.remove(agent_id)

        elapsed = time.time() - agent.started_at
        status = outcome.get("status", "completed")
        answer = outcome.get("content", "")
        steps = outcome.get("steps", 0)
        in_tok = outcome.get("input_tokens", 0)
        out_tok = outcome.get("output_tokens", 0)
        tool_calls = outcome.get("tool_calls", 0)

        # Terminal summary
        status_icon = "✓" if status == _SUBAGENT_STATUS_COMPLETED else "✗"
        status_color = COLOR_GREEN if status == _SUBAGENT_STATUS_COMPLETED else COLOR_RED
        sys.stdout.write(
            f"{status_color}  {status_icon} 🤖 Background agent done{COLOR_RESET}"
            f"{COLOR_DIM} ({status}, {elapsed:.1f}s, {steps} steps, {tool_calls} tool calls, "
            f"{in_tok}+{out_tok} tokens){COLOR_RESET}\n"
        )
        sys.stdout.flush()

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
_FG_DEFAULT_TIMEOUT = 300  # default foreground timeout
_BG_DEFAULT_TIMEOUT = 900  # default background timeout


def dispatch_agent(
    task: str,
    context: str = "",
    tool_whitelist: list = None,
    background: bool = False,
    timeout: int = 0,
) -> ToolResult:
    """Spawn a sub-agent to handle a focused subtask."""
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
        effective_timeout = max(60, min(1800, timeout))
    elif background:
        effective_timeout = _BG_DEFAULT_TIMEOUT
    else:
        effective_timeout = _FG_DEFAULT_TIMEOUT

    runtime_owner = _get_runtime_owner()
    model_spec = runtime_owner.model_spec

    # Build tool schemas & functions for the sub-agent
    registry = get_registry()
    _, all_schemas, all_functions = registry.snapshot()

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

        try:
            future = _bg_executor.submit(
                ctx.run, _run_subagent, task, context, model_spec,
                max_steps, tool_schemas, tool_functions,
                spinner_id, custom_system_prompt, cancel_event,
                progress_ids,
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

        try:
            # Submit to the shared executor (NOT a per-call `with ThreadPoolExecutor`
            # which blocks on __exit__ with shutdown(wait=True), making timeout
            # handoff impossible).
            future = _bg_executor.submit(
                ctx.run, _run_subagent, task, context, model_spec,
                max_steps, tool_schemas, tool_functions,
                agent_id, custom_system_prompt, cancel_event,
                progress_ids,
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
    rather than having to infer them from free-form text.  Set
    ``JY_SUBAGENT_FLAT_RESULT=1`` in the environment to opt out and get
    the legacy raw-answer string (mainly for backwards compatibility).
    """
    import os as _os

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
        )
    except Exception:
        pass  # stats recording is best-effort

    # Structured envelope is the new default; `JY_SUBAGENT_FLAT_RESULT=1`
    # opts out for callers that rely on the legacy raw-answer string.
    if _os.environ.get("JY_SUBAGENT_FLAT_RESULT", "").lower() in ("1", "true", "yes"):
        return ToolResult(answer, is_error=(status != _SUBAGENT_STATUS_COMPLETED))

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
