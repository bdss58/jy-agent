# Sub-Agent Tool — Spawn focused child agents for parallel/specialized subtasks.
#
# The parent (lead) agent decomposes a complex task into sub-problems,
# dispatches sub-agents for each, and synthesizes results.
#
# Sub-agents run silently (no terminal streaming), have their own context
# window and message history, and return only their final answer to the parent.

import sys
import time
import traceback
import threading
import concurrent.futures

try:
    from ..config import (
        MAX_TOOL_RESULT_CHARS, STREAM_TIMEOUT, get_active_model_spec, get_reasoning_config_for_provider,
        get_subagent_model_spec,
    )
    from ..registry import get_registry
    from ..runtime.reasoning import build_anthropic_request_reasoning
    from ..runtime.providers._anthropic_helpers import assistant_from_response, convert_messages
    from ..runtime import RuntimeOptions, RuntimeOwner
    from ..toolresult import ToolResult
    from ..validation import validate_tool_input
    from ..session_stats import get_stats
except ImportError:
    from jyagent.config import (
        MAX_TOOL_RESULT_CHARS, STREAM_TIMEOUT, get_active_model_spec, get_reasoning_config_for_provider,
        get_subagent_model_spec,
    )
    from jyagent.registry import get_registry
    from jyagent.runtime.reasoning import build_anthropic_request_reasoning
    from jyagent.runtime.providers._anthropic_helpers import assistant_from_response, convert_messages
    from jyagent.runtime import RuntimeOptions, RuntimeOwner
    from jyagent.toolresult import ToolResult
    from jyagent.validation import validate_tool_input
    from jyagent.session_stats import get_stats


# ─── Model tiers ──────────────────────────────────────────────────────────────

_DEFAULT_MAX_STEPS = 30
_DEFAULT_MAX_TOKENS_PER_RESPONSE = 8192
_SUBAGENT_TIMEOUT = 300  # 5 min wall-clock per sub-agent
_SUBAGENT_STATUS_COMPLETED = "completed"
_SUBAGENT_STATUS_MAX_STEPS = "max_steps"
_SUBAGENT_STATUS_API_ERROR = "api_error"

# Track nesting to prevent runaway recursion
_nesting_depth = threading.local()
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
    """Compatibility shim for tests that still provide a fake Anthropic client."""

    def __init__(self, client):
        self._client = client
        self.model_spec = get_active_model_spec()

    def complete(self, context, options=None, model_spec=None):
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
            "messages": self._convert_messages(context.get("messages", [])),
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
        return self._normalize_response(model_spec, response)

    def _convert_messages(self, messages):
        return convert_messages(self.model_spec, messages)

    def _normalize_response(self, model_spec, response):
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
        "your conversation history."
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
            "model": {
                "type": "string",
                "enum": ["fast", "default", "strong"],
                "description": (
                    "Model tier. 'fast' = Haiku (cheap, simple tasks), "
                    "'default' = same model as parent (balanced), "
                    "'strong' = best available (complex reasoning). Default: 'default'."
                ),
            },
            "max_steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "description": "Max tool-use steps (default 30). Lower = cheaper/faster.",
            },
            "tool_whitelist": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of tool names the sub-agent can use. "
                    "Empty = all tools available. Example: ['web_fetch', 'read_file']"
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

_SUBAGENT_FINAL_NO_TOOLS_SUFFIX = (
    "\n\n[SYSTEM: You have reached the maximum number of tool-use steps. "
    "Provide your best answer now WITHOUT using any tools.]"
)


# ─── Silent tool loop (the sub-agent engine) ─────────────────────────────────

def _execute_tool_silent(tool_name, tool_input, tool_functions):
    """Execute a single tool call. Same as planner._execute_tool but standalone."""
    fn = tool_functions.get(tool_name)
    if fn is None:
        return ToolResult(
            f"Error: Unknown tool '{tool_name}'. Available: {sorted(tool_functions.keys())[:20]}",
            is_error=True,
        )

    tool_schema = get_registry().get_schema(tool_name)
    validation_error = validate_tool_input(tool_name, tool_input, fn, tool_schema)
    if validation_error:
        return ToolResult(validation_error, is_error=True)

    try:
        if tool_input is None:
            tool_input = {}
        raw = fn(**tool_input)
        if isinstance(raw, ToolResult):
            return raw
        return ToolResult(str(raw))
    except Exception as e:
        return ToolResult(f"Error calling tool {tool_name}: {e}", is_error=True)


def _truncate(text, max_chars=MAX_TOOL_RESULT_CHARS):
    """Truncate text for context management."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.85)
    tail = int(max_chars * 0.10)
    return (
        text[:head]
        + f"\n\n[... truncated {len(text) - head - tail} chars ...]\n\n"
        + text[-tail:]
    )


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
    response = runtime_owner.complete(
        {
            "system_prompt": _SUBAGENT_SYSTEM_PROMPT + _SUBAGENT_FINAL_NO_TOOLS_SUFFIX,
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


def _run_subagent(task, context, model_spec, max_steps, tool_schemas, tool_functions):
    """Run a sub-agent's tool loop to completion.

    This is the core engine for silent sub-agent execution.
    Provider transport may still stream under the hood.
    Runs entirely in the calling thread (no terminal output).
    """
    runtime_owner = _get_runtime_owner()

    # Build initial messages
    user_content = task
    if context:
        user_content = f"Context:\n{context}\n\nTask:\n{task}"

    messages = [{"role": "user", "content": user_content}]

    all_text = ""
    total_input_tokens = 0
    total_output_tokens = 0
    tool_calls_count = 0

    for step in range(max_steps):
        runtime_context = {
            "system_prompt": _SUBAGENT_SYSTEM_PROMPT,
            "messages": messages,
        }
        if tool_schemas:
            runtime_context["tools"] = tool_schemas

        try:
            response = runtime_owner.complete(
                runtime_context,
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
                        "mode": "loop_complete",
                        "step": step + 1,
                    },
                ),
                model_spec=model_spec,
            )
        except Exception as e:
            step_num = step + 1
            error_text = str(e)
            content = _format_subagent_failure(
                f"Error: Sub-agent API failure at step {step_num}: {error_text}",
                partial_output=all_text,
            )
            return _make_subagent_outcome(
                _SUBAGENT_STATUS_API_ERROR,
                content,
                step_num,
                total_input_tokens,
                total_output_tokens,
                tool_calls_count,
                error=error_text,
            )

        # Track tokens
        usage = response.get("usage", {})
        total_input_tokens += usage.get("input_tokens", 0)
        total_output_tokens += usage.get("output_tokens", 0)

        # Extract text and tool_call blocks
        text_parts = []
        tool_use_blocks = []
        for block in response.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_call":
                tool_use_blocks.append(block)

        step_text = "\n".join(text_parts)
        all_text += step_text

        # No tool calls → sub-agent is done
        if not tool_use_blocks:
            if not all_text:
                all_text = "[Sub-agent completed but produced no text output]"
            return _make_subagent_outcome(
                _SUBAGENT_STATUS_COMPLETED,
                all_text,
                step + 1,
                total_input_tokens,
                total_output_tokens,
                tool_calls_count,
            )

        # Execute tool calls and build results
        messages.append(response)

        tool_results = []
        for block in tool_use_blocks:
            tool_calls_count += 1
            result = _execute_tool_silent(block["name"], block.get("arguments", {}), tool_functions)
            content_str = _truncate(result.content)
            tool_results.append({
                "role": "tool_result",
                "tool_call_id": block["id"],
                "tool_name": block["name"],
                "content": content_str,
                "is_error": result.is_error,
            })

        messages.extend(tool_results)

    final_answer = ""
    try:
        final_answer, extra_in, extra_out = _best_effort_final_answer(runtime_owner, messages, model_spec)
        total_input_tokens += extra_in
        total_output_tokens += extra_out
    except Exception:
        pass

    content = _format_subagent_failure(
        f"Error: Sub-agent reached max_steps ({max_steps}).",
        partial_output=all_text,
        final_answer=final_answer,
    )
    return _make_subagent_outcome(
        _SUBAGENT_STATUS_MAX_STEPS,
        content,
        max_steps,
        total_input_tokens,
        total_output_tokens,
        tool_calls_count,
        error=f"max_steps:{max_steps}",
    )


# ─── Terminal status display ─────────────────────────────────────────────────

COLOR_DIM = "\033[2m"
COLOR_RESET = "\033[0m"
COLOR_CYAN = "\033[1;36m"
COLOR_GREEN = "\033[0;32m"
COLOR_RED = "\033[1;31m"
COLOR_YELLOW = "\033[1;33m"

_spinner_lock = threading.Lock()


class _SubagentSpinner:
    """Animated spinner shown while a sub-agent is running."""
    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, task_preview):
        self._preview = task_preview[:60] + "..." if len(task_preview) > 60 else task_preview
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)

    def _animate(self):
        idx = 0
        t0 = time.time()
        while not self._stop.is_set():
            elapsed = time.time() - t0
            frame = self._FRAMES[idx % len(self._FRAMES)]
            with _spinner_lock:
                sys.stdout.write(
                    f"\r{COLOR_DIM}  {frame} 🤖 Sub-agent: {self._preview} ({elapsed:.0f}s){COLOR_RESET}"
                )
                sys.stdout.flush()
            idx += 1
            self._stop.wait(0.1)
        # Clear the spinner line
        with _spinner_lock:
            sys.stdout.write("\r" + " " * 100 + "\r")
            sys.stdout.flush()


# ─── Main tool function ──────────────────────────────────────────────────────

def dispatch_agent(
    task: str,
    context: str = "",
    model: str = "default",
    max_steps: int = _DEFAULT_MAX_STEPS,
    tool_whitelist: list = None,
) -> ToolResult:
    """Spawn a sub-agent to handle a focused subtask."""
    # Guard: nesting depth
    depth = getattr(_nesting_depth, "value", 0)
    if depth >= _MAX_NESTING:
        return ToolResult(
            f"Error: Maximum sub-agent nesting depth ({_MAX_NESTING}) reached. "
            f"Cannot spawn deeper sub-agents.",
            is_error=True,
        )

    # Resolve model
    runtime_owner = _get_runtime_owner()
    model_spec = get_subagent_model_spec(model, runtime_owner.model_spec)

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

    # Show spinner
    spinner = _SubagentSpinner(task)
    spinner.start()
    t0 = time.time()

    try:
        # Increment nesting depth for this thread
        _nesting_depth.value = depth + 1

        # Run with wall-clock timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _run_subagent, task, context, model_spec,
                max_steps, tool_schemas, tool_functions,
            )
            try:
                outcome = future.result(timeout=_SUBAGENT_TIMEOUT)
            except concurrent.futures.TimeoutError:
                future.cancel()
                return ToolResult(
                    f"Error: Sub-agent timed out after {_SUBAGENT_TIMEOUT}s. "
                    f"Task may be too complex — try breaking it down further.",
                    is_error=True,
                )
    except KeyboardInterrupt:
        spinner.stop()
        raise
    except Exception as e:
        spinner.stop()
        error_text = str(e)
        error_detail = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return ToolResult(
            f"Error: Sub-agent failed: {error_text}\n{error_detail}",
            is_error=True,
        )
    finally:
        _nesting_depth.value = depth  # restore
        spinner.stop()

    elapsed = time.time() - t0

    # Print summary to terminal
    status = outcome.get("status", _SUBAGENT_STATUS_COMPLETED)
    answer = outcome.get("content", "")
    steps = outcome.get("steps", 0)
    in_tok = outcome.get("input_tokens", 0)
    out_tok = outcome.get("output_tokens", 0)
    tool_calls = outcome.get("tool_calls", 0)

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
        parent_stats.record_subagent_usage(in_tok, out_tok, model_spec.provider, model_spec.model)
    except Exception:
        pass  # stats recording is best-effort

    return ToolResult(answer, is_error=(status != _SUBAGENT_STATUS_COMPLETED))
