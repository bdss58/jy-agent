# Sub-Agent Tool — Spawn focused child agents for parallel/specialized subtasks.
#
# The parent (lead) agent decomposes a complex task into sub-problems,
# dispatches sub-agents for each, and synthesizes results.
#
# Sub-agents run silently (no streaming to terminal), have their own context
# window and message history, and return only their final answer to the parent.

import os
import sys
import time
import traceback
import threading
import concurrent.futures

try:
    from ..config import (
        MAX_TOOL_RESULT_CHARS, STREAM_TIMEOUT, SUBAGENT_MODEL_TIERS,
    )
    from ..registry import get_registry
    from ..toolresult import ToolResult
    from ..validation import validate_tool_input
    from ..session_stats import get_stats
except ImportError:
    from jyagent.config import (
        MAX_TOOL_RESULT_CHARS, STREAM_TIMEOUT, SUBAGENT_MODEL_TIERS,
    )
    from jyagent.registry import get_registry
    from jyagent.toolresult import ToolResult
    from jyagent.validation import validate_tool_input
    from jyagent.session_stats import get_stats


# ─── Model tiers ──────────────────────────────────────────────────────────────

_DEFAULT_MAX_STEPS = 30
_DEFAULT_MAX_TOKENS_PER_RESPONSE = 8192
_SUBAGENT_TIMEOUT = 300  # 5 min wall-clock per sub-agent

# Track nesting to prevent runaway recursion
_nesting_depth = threading.local()
_MAX_NESTING = 2  # sub-agent can spawn sub-sub-agent, but no deeper


# ─── Client access ────────────────────────────────────────────────────────────

_client = None


def set_client(client):
    """Called once during agent startup to share the Anthropic client."""
    global _client
    _client = client


def _get_client():
    """Get the shared Anthropic client, creating if needed."""
    global _client
    if _client is not None:
        return _client
    # Fallback: create a new client (should not happen in normal flow)
    import httpx
    import anthropic
    kwargs = {}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if base_url:
        kwargs["base_url"] = base_url
    if auth_token:
        kwargs["api_key"] = auth_token
    kwargs["http_client"] = httpx.Client(verify=False)
    _client = anthropic.Anthropic(**kwargs)
    return _client


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


# ─── Non-streaming tool loop (the sub-agent engine) ──────────────────────────

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


def _serialize_blocks(content_blocks):
    """Serialize SDK content blocks to plain dicts."""
    out = []
    for block in content_blocks:
        if hasattr(block, "model_dump"):
            d = block.model_dump(exclude_none=True)
        elif hasattr(block, "dict"):
            d = block.dict(exclude_none=True)
        else:
            out.append(block)
            continue

        if d.get("type") == "tool_use":
            out.append({
                "type": "tool_use",
                "id": d["id"],
                "name": d["name"],
                "input": d.get("input", {}),
            })
        elif d.get("type") == "text":
            out.append({"type": "text", "text": d.get("text", "")})
        else:
            out.append(d)
    return out


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


def _run_subagent(task, context, model_name, max_steps, tool_schemas, tool_functions):
    """Run a sub-agent's tool loop to completion. Returns (answer_text, stats_dict).

    This is the core engine — a non-streaming version of plan_next_action.
    Runs entirely in the calling thread (no terminal output).
    """
    client = _get_client()

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
        # API call (non-streaming for simplicity and thread-safety)
        api_kwargs = dict(
            model=model_name,
            max_tokens=_DEFAULT_MAX_TOKENS_PER_RESPONSE,
            system=_SUBAGENT_SYSTEM_PROMPT,
            messages=messages,
        )
        if tool_schemas:
            api_kwargs["tools"] = tool_schemas

        try:
            response = client.messages.create(**api_kwargs, timeout=STREAM_TIMEOUT)
        except Exception as e:
            return f"[Sub-agent API error at step {step}: {e}]", {
                "steps": step,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "tool_calls": tool_calls_count,
                "error": str(e),
            }

        # Track tokens
        if hasattr(response, "usage"):
            total_input_tokens += getattr(response.usage, "input_tokens", 0)
            total_output_tokens += getattr(response.usage, "output_tokens", 0)

        # Extract text and tool_use blocks
        text_parts = []
        tool_use_blocks = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_use_blocks.append(block)

        step_text = "\n".join(text_parts)
        all_text += step_text

        # No tool calls → sub-agent is done
        if not tool_use_blocks:
            break

        # Execute tool calls and build results
        serialized = _serialize_blocks(response.content)
        messages.append({"role": "assistant", "content": serialized})

        tool_results = []
        for block in tool_use_blocks:
            tool_calls_count += 1
            result = _execute_tool_silent(block.name, block.input, tool_functions)
            content_str = _truncate(result.content)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content_str,
                "is_error": result.is_error,
            })

        messages.append({"role": "user", "content": tool_results})

    stats = {
        "steps": step + 1 if 'step' in dir() else 0,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "tool_calls": tool_calls_count,
    }

    if not all_text:
        all_text = "[Sub-agent completed but produced no text output]"

    return all_text, stats


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
    model_name = SUBAGENT_MODEL_TIERS.get(model, SUBAGENT_MODEL_TIERS["default"])

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
                _run_subagent, task, context, model_name,
                max_steps, tool_schemas, tool_functions,
            )
            try:
                answer, stats = future.result(timeout=_SUBAGENT_TIMEOUT)
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
        return ToolResult(
            f"Error: Sub-agent failed: {e}\n{traceback.format_exc()}",
            is_error=True,
        )
    finally:
        _nesting_depth.value = depth  # restore
        spinner.stop()

    elapsed = time.time() - t0

    # Print summary to terminal
    steps = stats.get("steps", 0)
    in_tok = stats.get("input_tokens", 0)
    out_tok = stats.get("output_tokens", 0)
    tool_calls = stats.get("tool_calls", 0)
    error = stats.get("error")

    status_icon = "✓" if not error else "✗"
    status_color = COLOR_GREEN if not error else COLOR_RED
    sys.stdout.write(
        f"{status_color}  {status_icon} 🤖 Sub-agent done{COLOR_RESET}"
        f"{COLOR_DIM} ({elapsed:.1f}s, {steps} steps, {tool_calls} tool calls, "
        f"{in_tok}+{out_tok} tokens){COLOR_RESET}\n"
    )
    sys.stdout.flush()

    # Record token usage in parent stats
    try:
        parent_stats = get_stats()
        parent_stats.record_subagent_usage(in_tok, out_tok, model_name)
    except Exception:
        pass  # stats recording is best-effort

    return ToolResult(answer)
