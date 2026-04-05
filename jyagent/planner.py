# Planner — Streaming tool-use loop with structured results, concurrent execution, unified timeouts.
# Honesty rules are in the main SYSTEM_PROMPT (agent.py), not duplicated here.
import json
import sys
import time
import atexit
import logging
import threading
import traceback
import concurrent.futures
from dataclasses import dataclass
from .observability import format_traceback, log_event, scrub_string
from .registry import get_registry
from .toolresult import ToolResult
from .validation import validate_tool_input
from .session_stats import get_stats
from .runtime import RuntimeOwner, RuntimeOptions
from .config import (
    DEFAULT_MAX_TOKENS, MAX_TOKENS_CAP, DEFAULT_MAX_STEPS,
    MAX_TOOL_RESULT_CHARS, MAX_TOOL_USE_INPUT_CHARS,
    MAX_WORKING_TOKENS, DEFAULT_TOOL_TIMEOUT, STREAM_TIMEOUT,
    COMPACT_TOOL_RESULT_CHARS, get_reasoning_config_for_provider,
)

# Shared executor for tool timeouts.  max_workers=4 so a hung tool doesn't
# block subsequent calls.  NOTE: future.cancel() can't stop a *running* thread,
# so 4 hung tools would exhaust this pool.  atexit ensures clean shutdown.
_tool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
atexit.register(_tool_executor.shutdown, wait=False)

TOOL_COLOR = "\033[0;33m"  # yellow for tool info
COLOR_RESET = "\033[0m"
COLOR_DIM = "\033[2m"
COLOR_YELLOW = "\033[1;33m"
COLOR_CYAN = "\033[1;36m"
COLOR_RED = "\033[1;31m"
COLOR_GREEN = "\033[0;32m"
COLOR_MAGENTA = "\033[0;35m"
COLOR_DIM_YELLOW = "\033[2;33m"

logger = logging.getLogger(__name__)


@dataclass
class _ToolCallRequest:
    id: str
    name: str
    input: dict


def _is_error_result(result) -> bool:
    """Detect if a tool result indicates an error."""
    if isinstance(result, ToolResult):
        return result.is_error
    # Fallback for any non-ToolResult values
    s = str(result)
    return s.startswith("Error:") or s.startswith("Error calling tool")


def _result_content(result) -> str:
    """Extract string content from a tool result."""
    if isinstance(result, ToolResult):
        return result.content
    return str(result)


# ─── Output helpers ───────────────────────────────────────────────────────────

def _stream_write(text: str):
    """Write streamed text to stdout."""
    sys.stdout.write(text)
    sys.stdout.flush()


# ─── Thinking spinner ────────────────────────────────────────────────────────

class _ThinkingSpinner:
    """Animated spinner shown while waiting for the first token.

    Uses a background thread to animate; call stop() when the first
    text/tool_use delta arrives.  Thread-safe.
    """
    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str = "Thinking"):
        self._label = label
        self._stop_event = threading.Event()
        self._thread = None
        self._started = False

    def start(self):
        if self._started:
            return
        self._started = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._started:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1)
        self._started = False
        # Clear the spinner line
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _animate(self):
        idx = 0
        t0 = time.time()
        while not self._stop_event.is_set():
            elapsed = time.time() - t0
            frame = self._FRAMES[idx % len(self._FRAMES)]
            sys.stdout.write(f"\r{COLOR_DIM}  {frame} {self._label}... ({elapsed:.1f}s){COLOR_RESET}")
            sys.stdout.flush()
            idx += 1
            self._stop_event.wait(0.08)


# ─── Tool output formatting ──────────────────────────────────────────────────

_TOOL_ICONS = {
    "run_shell": "⚡", "read_file": "📄", "write_file": "📝",
    "edit_file": "✏️", "list_directory": "📁", "glob_files": "🔍",
    "grep_files": "🔎", "web_fetch": "🌐", "manage_memory": "🧠",
    "manage_skills": "📦", "mcp": "🔌",
}


def _tool_info(msg: str):
    """Print a visible tool/status message."""
    sys.stdout.write(f"\n{TOOL_COLOR}  🔧 {msg}{COLOR_RESET}\n")
    sys.stdout.flush()


def _tool_call_header(tool_name: str, tool_input: dict):
    """Print a compact, visually distinct tool call header."""
    icon = _TOOL_ICONS.get(tool_name, "🔧")
    # Build compact arg summary
    args_preview = _format_tool_args(tool_name, tool_input)
    sys.stdout.write(f"\n{TOOL_COLOR}  {icon} {tool_name}{COLOR_RESET}")
    if args_preview:
        sys.stdout.write(f"{COLOR_DIM} {args_preview}{COLOR_RESET}")
    sys.stdout.write("\n")
    sys.stdout.flush()


def _format_tool_args(tool_name: str, tool_input: dict) -> str:
    """Format tool arguments for display — show key info, hide verbosity."""
    if not tool_input:
        return ""
    # Special formatting per tool
    if tool_name == "run_shell":
        cmd = tool_input.get("command", "")
        if len(cmd) > 120:
            cmd = cmd[:117] + "..."
        return f"$ {cmd}"
    if tool_name in ("read_file", "write_file", "edit_file"):
        path = tool_input.get("path", "")
        extras = []
        if tool_input.get("operation"):
            extras.append(tool_input["operation"])
        if tool_input.get("insert_at_line"):
            extras.append(f"L{tool_input['insert_at_line']}")
        if tool_input.get("dry_run"):
            extras.append("dry-run")
        suffix = f" ({', '.join(extras)})" if extras else ""
        return f"{path}{suffix}"
    if tool_name == "list_directory":
        return tool_input.get("path", ".") or "."
    if tool_name in ("glob_files", "grep_files"):
        pattern = tool_input.get("pattern", "")
        path = tool_input.get("path", "")
        return f"'{pattern}'" + (f" in {path}" if path else "")
    if tool_name == "web_fetch":
        url = tool_input.get("url", "")
        if len(url) > 100:
            url = url[:97] + "..."
        return url
    if tool_name in ("manage_memory", "manage_skills"):
        action = tool_input.get("action", "")
        name = tool_input.get("name", "")
        return f"{action}" + (f" {name}" if name else "")
    if tool_name == "mcp":
        return tool_input.get("action", "") + (" " + tool_input.get("server", "") if tool_input.get("server") else "")
    # Generic: show first 120 chars of stringified input
    s = str(tool_input)
    return s[:120] + "..." if len(s) > 120 else s


def _tool_result_preview(result_str: str, tool_name: str = "", is_error: bool = False):
    """Print a compact tool result summary with smart formatting."""
    lines = result_str.split('\n')
    n_lines = len(lines)
    n_chars = len(result_str)

    if is_error:
        # Errors: show full (they're usually short)
        preview = result_str[:300].replace('\n', ' ↵ ')
        if n_chars > 300:
            preview += "..."
        sys.stdout.write(f"{COLOR_RED}  ✗ {preview}{COLOR_RESET}\n")
        sys.stdout.flush()
        return

    # Detect edit_file diffs and show them nicely
    if tool_name == "edit_file" and any(ln.strip().startswith(">") for ln in lines[:20]):
        _render_edit_diff(result_str)
        return

    # Compact display: first line as summary + dims
    first_line = lines[0].strip() if lines else ""
    if len(first_line) > 150:
        first_line = first_line[:147] + "..."

    size_info = f"{n_chars} chars" if n_lines <= 1 else f"{n_lines} lines, {n_chars} chars"
    sys.stdout.write(f"{COLOR_GREEN}  ✓{COLOR_RESET} {first_line}")
    sys.stdout.write(f" {COLOR_DIM}({size_info}){COLOR_RESET}\n")
    sys.stdout.flush()


def _render_edit_diff(result_str: str):
    """Render edit_file output with color-coded diff lines."""
    lines = result_str.split('\n')
    # First line is the summary (e.g. "Edited foo.py: replaced 3 lines...")
    summary = lines[0] if lines else ""
    sys.stdout.write(f"{COLOR_GREEN}  ✓ {summary}{COLOR_RESET}\n")

    # Render context lines with diff coloring
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith(">"):
            # Changed line — highlight in green
            sys.stdout.write(f"\033[32m    {line}{COLOR_RESET}\n")
        elif stripped.startswith("L") or stripped.startswith(" "):
            # Context line — dim
            sys.stdout.write(f"{COLOR_DIM}    {line}{COLOR_RESET}\n")
        elif line.strip():
            sys.stdout.write(f"    {line}\n")
    sys.stdout.flush()


def _interrupted_msg():
    """Print interruption message."""
    sys.stdout.write(f"\n{COLOR_YELLOW}⚠ Interrupted by Ctrl-C{COLOR_RESET}\n")
    sys.stdout.flush()


def _runtime_warning(msg: str):
    """Print a short runtime recovery warning without aborting the turn."""
    sys.stdout.write(f"\n{COLOR_DIM_YELLOW}  ⚠ {msg}{COLOR_RESET}\n")
    sys.stdout.flush()


# ─── Working-message size management ────────────────────────────────────────

def _truncate_tool_result(result: str, is_error: bool = False, max_chars: int = None) -> str:
    """Truncate a tool result for working_messages if it exceeds max_chars.

    Error messages are never truncated (they're usually short and critical).
    """
    if max_chars is None:
        max_chars = MAX_TOOL_RESULT_CHARS
    if len(result) <= max_chars or is_error:
        return result

    head = int(max_chars * 0.85)
    tail = int(max_chars * 0.10)
    return (
        result[:head]
        + f"\n\n[... truncated {len(result) - head - tail} chars "
        + f"(total: {len(result)} chars) ...]\n\n"
        + result[-tail:]
    )


def _truncate_tool_call_blocks(blocks: list) -> list:
    """Truncate large tool_call argument fields in normalized assistant content."""
    registry = get_registry()
    out = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "tool_call":
            large_keys = registry.get_large_input_keys(block.get("name", ""))
            if large_keys:
                inp = block.get("arguments", {})
                truncated_inp = {}
                did_truncate = False
                for k, v in inp.items():
                    if k in large_keys and isinstance(v, str) and len(v) > MAX_TOOL_USE_INPUT_CHARS:
                        truncated_inp[k] = (
                            v[:MAX_TOOL_USE_INPUT_CHARS]
                            + f"\n[... truncated, {len(v)} chars total ...]"
                        )
                        did_truncate = True
                    else:
                        truncated_inp[k] = v
                if did_truncate:
                    block = dict(block)
                    block["arguments"] = truncated_inp
        out.append(block)
    return out


def _compact_working_messages(messages: list, max_tokens: int = None) -> list:
    """Aggressively truncate older tool results when working_messages is too large.

    Keeps the last 2 messages (most recent tool call + result) intact so Claude
    can reason about the latest tool output.
    """
    if max_tokens is None:
        max_tokens = MAX_WORKING_TOKENS

    from .memory import estimate_conversation_tokens
    estimated = estimate_conversation_tokens(messages)
    if estimated <= max_tokens:
        return messages

    AGGRESSIVE_LIMIT = COMPACT_TOOL_RESULT_CHARS
    compacted = [dict(m) for m in messages]  # shallow copy each message

    # Work from oldest, skip the last 2 messages
    for i in range(len(compacted) - 2):
        msg = compacted[i]
        if msg.get("role") == "tool_result":
            result_text = str(msg.get("content", ""))
            if len(result_text) > AGGRESSIVE_LIMIT:
                compacted[i]["content"] = (
                    result_text[:AGGRESSIVE_LIMIT]
                    + f"\n[... compacted from {len(result_text)} chars ...]"
                )
            continue

        content = msg.get("content", "")
        if isinstance(content, list):
            new_blocks = []
            for block in _truncate_tool_call_blocks(content):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    result_text = str(block.get("content", ""))
                    if len(result_text) > AGGRESSIVE_LIMIT:
                        block = dict(block)
                        block["content"] = (
                            result_text[:AGGRESSIVE_LIMIT]
                            + f"\n[... compacted from {len(result_text)} chars ...]"
                        )
                new_blocks.append(block)
            compacted[i]["content"] = new_blocks

    return compacted


# ─── Tool execution ─────────────────────────────────────────────────────────

def _execute_tool(tool_name: str, tool_input: dict, tool_functions: dict) -> ToolResult:
    """Execute a tool call and return a ToolResult.
    
    Returns ToolResult with is_error=True for:
    - Unknown tool names
    - Missing required parameters
    - Exceptions during execution
    - Timeout exceeded
    """
    fn = tool_functions.get(tool_name)
    if fn is None:
        return ToolResult(
            f"Error: Unknown tool '{tool_name}'. Available: {sorted(tool_functions.keys())[:20]}",
            is_error=True
        )

    tool_schema = get_registry().get_schema(tool_name)
    validation_error = validate_tool_input(tool_name, tool_input, fn, tool_schema)
    if validation_error:
        return ToolResult(validation_error, is_error=True)

    try:
        if tool_input is None:
            tool_input = {}
        raw_result = fn(**tool_input)
        # All tools should return ToolResult; wrap legacy string returns for safety
        if isinstance(raw_result, ToolResult):
            return raw_result
        return ToolResult(str(raw_result))
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return ToolResult(
            f"Error calling tool {tool_name}: {e}\n{traceback.format_exc()}",
            is_error=True
        )


def _execute_tool_with_timeout(tool_name: str, tool_input: dict, tool_functions: dict,
                                timeout: int = None) -> ToolResult:
    """Execute a tool with a unified timeout.

    Timeout is determined from registry metadata (timeout_hint).
    Tools without a hint use DEFAULT_TOOL_TIMEOUT.
    run_shell is special: its timeout comes from its own input parameter.
    """
    if timeout is None:
        timeout = DEFAULT_TOOL_TIMEOUT

    registry = get_registry()
    hint = registry.get_timeout_hint(tool_name)
    if hint is not None:
        timeout = max(timeout, hint)

    # run_shell manages its own timeout — give it extra slack
    if tool_name == "run_shell":
        user_timeout = tool_input.get("timeout", 60)
        timeout = max(timeout, int(user_timeout) + 10)

    future = _tool_executor.submit(_execute_tool, tool_name, tool_input, tool_functions)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        future.cancel()
        return ToolResult(
            f"Error: Tool '{tool_name}' timed out after {timeout}s. "
            f"Consider breaking the operation into smaller steps.",
            is_error=True
        )
    except KeyboardInterrupt:
        future.cancel()
        raise


# ─── Multi-tool dispatch ─────────────────────────────────────────────────────

def _execute_tools(tool_blocks: list, tool_functions: dict) -> list:
    """Execute tool calls with selective parallelization.

    Parallel-safe tools (read_file, glob_files, etc.) run concurrently.
    State-mutating tools run sequentially in original order.
    Sequential tools act as barriers between parallel batches.
    Results are always returned in original order.
    """
    registry = get_registry()

    # Fast path: single tool or no parallel-safe tools → sequential
    if len(tool_blocks) <= 1 or not any(
        registry.is_parallel_safe(b.name) for b in tool_blocks
    ):
        results = []
        for block in tool_blocks:
            result = _execute_tool_with_timeout(block.name, block.input, tool_functions)
            results.append((block, result))
        return results

    # Partition into contiguous groups and execute
    results = [None] * len(tool_blocks)
    i = 0
    while i < len(tool_blocks):
        if registry.is_parallel_safe(tool_blocks[i].name):
            # Collect contiguous parallel-safe tools
            parallel_batch = []
            while i < len(tool_blocks) and registry.is_parallel_safe(tool_blocks[i].name):
                parallel_batch.append((i, tool_blocks[i]))
                i += 1

            # Execute batch concurrently via the shared executor
            # (_execute_tool_with_timeout already submits to _tool_executor for
            # timeout enforcement, so we call _execute_tool directly here to
            # avoid a double thread-pool layer.)
            futures = {
                _tool_executor.submit(
                    _execute_tool, block.name, block.input, tool_functions
                ): (idx, block)
                for idx, block in parallel_batch
            }
            for future in concurrent.futures.as_completed(futures):
                idx, block = futures[future]
                try:
                    results[idx] = (block, future.result())
                except Exception as exc:
                    results[idx] = (block, ToolResult(
                        f"Error calling tool {block.name}: {exc}\n{traceback.format_exc()}",
                        is_error=True
                    ))
        else:
            # Sequential tool — execute immediately as a barrier
            block = tool_blocks[i]
            result = _execute_tool_with_timeout(block.name, block.input, tool_functions)
            results[i] = (block, result)
            i += 1

    return results


def _assistant_tool_calls(message: dict) -> list[_ToolCallRequest]:
    return [
        _ToolCallRequest(
            id=block["id"],
            name=block["name"],
            input=block.get("arguments", {}),
        )
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_call"
    ]


def _assistant_text(message: dict) -> str:
    return "".join(
        block.get("text", "")
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _is_truncated_tool_call(stop_reason: str, tool_use_blocks: list[_ToolCallRequest]) -> bool:
    """Detect if a response likely stopped while emitting tool calls."""
    return stop_reason == "length" and bool(tool_use_blocks)


# ─── Streaming helper ────────────────────────────────────────────────────────

def _stream_response(
    runtime_owner: RuntimeOwner,
    context: dict,
    max_output_tokens: int,
    metadata: dict | None = None,
) -> tuple:
    """
    Stream a single runtime response with thinking spinner and token tracking.
    Returns (step_text, tool_use_blocks, stop_reason, final_message).
    """
    text_parts = []
    stop_reason = "stop"
    final_message = None
    first_token = True
    spinner = _ThinkingSpinner()
    spinner.start()
    stream = None

    try:
        stream = runtime_owner.stream(
            context,
            options=RuntimeOptions(
                max_output_tokens=max_output_tokens,
                timeout=STREAM_TIMEOUT,
                reasoning=get_reasoning_config_for_provider(
                    runtime_owner.model_spec.provider,
                    max_output_tokens=max_output_tokens,
                ),
                metadata=metadata,
            ),
        )
        for event in stream:
            event_type = event.get("type")
            if event_type == "text_delta":
                if first_token:
                    spinner.stop()
                    first_token = False
                _stream_write(event.get("text", ""))
                text_parts.append(event.get("text", ""))
            elif event_type in {"tool_call_delta", "thinking_delta"}:
                if first_token:
                    spinner.stop()
                    first_token = False

        final_message = stream.get_final_message()
        for warning in final_message.get("runtime_warnings", []):
            _runtime_warning(warning)
        stop_reason = final_message.get("stop_reason", "stop")
        tool_use_blocks = _assistant_tool_calls(final_message)

        stats = get_stats()
        stats.record_usage(
            final_message.get("usage", {}),
            provider=final_message.get("provider", ""),
            model=final_message.get("model", ""),
        )
    finally:
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
        spinner.stop()  # Ensure spinner is always cleaned up

    return "".join(text_parts), tool_use_blocks, stop_reason, final_message


# ─── Main planner loop ───────────────────────────────────────────────────────

_STREAM_MAX_RETRIES = 2


def _stream_with_retry(
    runtime_owner: RuntimeOwner,
    context: dict,
    max_output_tokens: int,
    step: int,
    all_text: str,
    working_messages: list,
):
    """Stream a response with auto-retry on transient failures.

    Returns (step_text, tool_use_blocks, stop_reason, final_message) on success.
    Returns None and a tuple (all_text, final_text, working_messages) on fatal error
    via a special sentinel: raises _StreamAbort with the return tuple.
    """
    for attempt in range(_STREAM_MAX_RETRIES + 1):
        try:
            return _stream_response(
                runtime_owner,
                context,
                max_output_tokens,
                metadata={
                    "component": "planner",
                    "mode": "stream",
                    "step": step + 1,
                    "attempt": attempt + 1,
                },
            )
        except KeyboardInterrupt:
            _interrupted_msg()
            msg = all_text + "\n\n[Response interrupted by user]" if all_text else "[Response interrupted by user]"
            raise _StreamAbort(msg, "", working_messages)
        except Exception as stream_err:
            is_transient = isinstance(stream_err, json.JSONDecodeError) or any(kw in str(stream_err).lower() for kw in [
                "incomplete", "peer closed", "connection reset", "timeout",
                "eof", "broken pipe", "overloaded", "529", "server_error",
                "response.completed",  # OpenAI stream missing terminal event
            ])
            if is_transient and attempt < _STREAM_MAX_RETRIES:
                retry_delay = 2 ** (attempt + 1)  # 2s, 4s
                log_event(
                    logger,
                    logging.WARNING,
                    "planner.stream.retry",
                    component="planner",
                    step=step + 1,
                    attempt=attempt + 1,
                    retry_in_seconds=retry_delay,
                    transient=True,
                    error_type=type(stream_err).__name__,
                    error_message=scrub_string(str(stream_err), max_text_chars=500),
                )
                retry_msg = f"\n⚡ Stream interrupted ({stream_err}), retrying in {retry_delay}s... (attempt {attempt + 2}/{_STREAM_MAX_RETRIES + 1})\n"
                sys.stdout.write(f"{COLOR_YELLOW}{retry_msg}{COLOR_RESET}")
                sys.stdout.flush()
                time.sleep(retry_delay)  # responds to KeyboardInterrupt
                continue
            # Non-transient error or retries exhausted
            log_event(
                logger,
                logging.ERROR,
                "planner.stream.failed",
                component="planner",
                step=step + 1,
                attempt=attempt + 1,
                transient=is_transient,
                error_type=type(stream_err).__name__,
                error_message=scrub_string(str(stream_err), max_text_chars=500),
                traceback=scrub_string(format_traceback(stream_err), max_text_chars=4000),
            )
            error_msg = f"\n[Stream error at step {step+1}: {stream_err}]\n"
            sys.stdout.write(f"{COLOR_RED}{error_msg}{COLOR_RESET}")
            sys.stdout.flush()
            if all_text:
                raise _StreamAbort(
                    all_text + f"\n\n[Error: streaming interrupted at step {step+1}: {stream_err}]",
                    "", working_messages
                )
            raise _StreamAbort(f"Error during streaming: {stream_err}", "", working_messages)


class _StreamAbort(Exception):
    """Raised by _stream_with_retry to signal an early return from plan_next_action."""
    def __init__(self, all_text, final_text, working_messages):
        self.result = (all_text, final_text, working_messages)


def _handle_tool_calls(tool_use_blocks, tool_functions, working_messages, all_text):
    """Execute tool calls and append results to working_messages.

    Returns the list of normalized tool_result messages.
    Raises KeyboardInterrupt (with synthetic results already appended) if interrupted.
    """
    try:
        stats = get_stats()
        n_tools = len(tool_use_blocks)
        if n_tools > 1:
            _tool_info(f"Executing {n_tools} tool calls...")

        # Print headers for all tool calls
        for block in tool_use_blocks:
            _tool_call_header(block.name, block.input if hasattr(block, 'input') else {})

        executed = _execute_tools(tool_use_blocks, tool_functions)

        tool_results = []
        for block, result in executed:
            stats.record_tool_call()
            content_str = _truncate_tool_result(result.content, is_error=result.is_error)
            _tool_result_preview(content_str, tool_name=block.name, is_error=result.is_error)

            tool_result_block = {
                "role": "tool_result",
                "tool_call_id": block.id,
                "tool_name": block.name,
                "content": content_str,
                "is_error": result.is_error,
            }

            tool_results.append(tool_result_block)

        return tool_results

    except KeyboardInterrupt:
        _interrupted_msg()
        synthetic_results = [
            {
                "role": "tool_result",
                "tool_call_id": b.id,
                "tool_name": b.name,
                "content": "[Tool execution interrupted by user]",
                "is_error": True,
            }
            for b in tool_use_blocks
        ]
        working_messages.extend(synthetic_results)
        msg = all_text + "\n\n[Tool execution interrupted by user]" if all_text else "[Tool execution interrupted by user]"
        raise _StreamAbort(msg, "", working_messages)

def plan_next_action(runtime_owner: RuntimeOwner, messages: list, system_prompt: str, max_steps: int = None) -> tuple:
    """
    Streams a provider-neutral runtime response token-by-token to stdout.
    Handles tool use in a loop: when the model calls tools, execute them and
    continue streaming the next response.
    Returns (full_text, final_step_text, working_messages).
    """
    if max_steps is None:
        max_steps = DEFAULT_MAX_STEPS

    # Defaults for the outer except handler (overwritten inside try)
    working_messages = []
    all_text = ""

    try:
        stats = get_stats()
        stats.new_turn()
        working_messages = messages.copy()
        all_text = ""
        final_text = ""  # Only the last step's text (for markdown rendering)
        current_max_tokens = DEFAULT_MAX_TOKENS

        registry = get_registry()
        reg_version, tool_schemas, tool_functions = registry.snapshot()

        for step in range(max_steps):
            # Guard: compact working_messages if token count is too high
            working_messages = _compact_working_messages(working_messages)

            context = {
                "system_prompt": system_prompt,
                "messages": working_messages,
            }
            if tool_schemas:
                context["tools"] = tool_schemas

            # Stream with auto-retry on transient failures
            step_text, tool_use_blocks, stop_reason, final_message = _stream_with_retry(
                runtime_owner, context, current_max_tokens, step, all_text, working_messages
            )

            all_text += step_text
            final_text = step_text  # Always track latest step's text

            # If there are no tool calls, we're done
            if not tool_use_blocks:
                if not step_text:
                    final_text = _assistant_text(final_message)
                    all_text = final_text or all_text
                if step_text:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                result = all_text if all_text else "I processed your request but had no text response to return."
                working_messages.append(final_message)
                return (result, final_text, working_messages)

            # --- Handle tool calls ---
            if step_text:
                sys.stdout.write("\n")
                sys.stdout.flush()

            # Detect truncated responses
            if _is_truncated_tool_call(stop_reason, tool_use_blocks):
                _tool_info("⚠️ Response truncated (max_tokens hit). Retrying with higher token limit...")
                current_max_tokens = min(current_max_tokens * 2, MAX_TOKENS_CAP)
                all_text = all_text[:-len(step_text)] if step_text else all_text
                continue

            # Append assistant's full normalized response (with large write inputs truncated)
            final_message = dict(final_message)
            final_message["content"] = _truncate_tool_call_blocks(final_message.get("content", []))
            working_messages.append(final_message)

            # Execute tool calls and build results
            tool_results = _handle_tool_calls(tool_use_blocks, tool_functions, working_messages, all_text)

            working_messages.extend(tool_results)

            # Refresh tool snapshots if registry changed (e.g. MCP connect mid-turn)
            if registry.version != reg_version:
                reg_version, tool_schemas, tool_functions = registry.snapshot()

            # Show step progress for long operations
            if step >= 5:
                sys.stdout.write(f"{COLOR_DIM}  [Step {step+1}/{max_steps}]{COLOR_RESET}\n")
                sys.stdout.flush()

        # Max steps reached
        max_step_msg = f"\n\n⚠️ Reached maximum reasoning steps ({max_steps}). My response may be incomplete."
        sys.stdout.write(f"{COLOR_YELLOW}{max_step_msg}{COLOR_RESET}\n")
        sys.stdout.flush()

        if all_text:
            return (all_text + max_step_msg, final_text, working_messages)

        # Try a final non-tool response
        fallback_text = ""
        try:
            final_message = runtime_owner.complete(
                {
                    "system_prompt": system_prompt + "\n\n[SYSTEM: You have reached the maximum number of tool-use steps. Please provide your best answer now WITHOUT using any tools.]",
                    "messages": working_messages,
                },
                options=RuntimeOptions(
                    max_output_tokens=DEFAULT_MAX_TOKENS,
                    timeout=STREAM_TIMEOUT,
                    reasoning=get_reasoning_config_for_provider(
                        runtime_owner.model_spec.provider,
                        max_output_tokens=DEFAULT_MAX_TOKENS,
                    ),
                    metadata={
                        "component": "planner",
                        "mode": "fallback_complete",
                        "step": max_steps,
                        "fallback": True,
                    },
                ),
            )
            stats.record_usage(
                final_message.get("usage", {}),
                provider=final_message.get("provider", ""),
                model=final_message.get("model", ""),
            )
            fallback_text = _assistant_text(final_message)
            if fallback_text:
                _stream_write(fallback_text)
            sys.stdout.write("\n")
            sys.stdout.flush()
            working_messages.append(final_message)
            return (fallback_text, fallback_text, working_messages)
        except KeyboardInterrupt:
            _interrupted_msg()
            msg = fallback_text + "\n\n[Response interrupted by user]" if fallback_text else "[Response interrupted by user]"
            return (msg, "", working_messages)
        except Exception as fallback_err:
            log_event(
                logger,
                logging.ERROR,
                "planner.fallback.failed",
                component="planner",
                step=max_steps,
                fallback=True,
                error_type=type(fallback_err).__name__,
                error_message=scrub_string(str(fallback_err), max_text_chars=500),
                traceback=scrub_string(format_traceback(fallback_err), max_text_chars=4000),
            )
            sys.stdout.write(f"{COLOR_RED}\n  Fallback response failed: {fallback_err}{COLOR_RESET}\n")
            sys.stdout.flush()

        return ("I've reached my maximum reasoning steps. Please try rephrasing your request.", "", working_messages)

    except _StreamAbort as abort:
        return abort.result
    except KeyboardInterrupt:
        _interrupted_msg()
        msg = all_text + "\n\n[Interrupted by user]" if all_text else "[Interrupted by user]"
        return (msg, "", working_messages)
    except Exception as e:
        error_detail = traceback.format_exc()
        log_event(
            logger,
            logging.ERROR,
            "planner.unhandled_exception",
            component="planner",
            error_type=type(e).__name__,
            error_message=scrub_string(str(e), max_text_chars=500),
            traceback=scrub_string(format_traceback(e), max_text_chars=4000),
        )
        sys.stdout.write(f"{COLOR_RED}\nFatal error in planner: {e}\n{error_detail}{COLOR_RESET}\n")
        sys.stdout.flush()
        return (f"Error during planning: {e}", "", working_messages)
