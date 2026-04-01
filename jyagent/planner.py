# Planner — Streaming tool-use loop with structured results, concurrent execution, unified timeouts.
# Honesty rules are in the main SYSTEM_PROMPT (agent.py), not duplicated here.
import sys
import time
import atexit
import traceback
import concurrent.futures
from .registry import get_registry
from .toolresult import ToolResult
from .validation import validate_tool_input
from .config import (
    DEFAULT_MAX_TOKENS, MAX_TOKENS_CAP, DEFAULT_MAX_STEPS,
    MAX_TOOL_RESULT_CHARS, MAX_TOOL_USE_INPUT_CHARS,
    MAX_WORKING_TOKENS, DEFAULT_TOOL_TIMEOUT, STREAM_TIMEOUT,
    AGENT_MODEL, COMPACT_TOOL_RESULT_CHARS,
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


def _tool_info(msg: str):
    """Print a visible tool/status message."""
    sys.stdout.write(f"\n{TOOL_COLOR}  🔧 {msg}{COLOR_RESET}\n")
    sys.stdout.flush()


def _tool_result_preview(result_str: str, is_error: bool = False, max_len: int = 200):
    """Print a short preview of the tool result."""
    preview = result_str[:max_len].replace('\n', ' ')
    if len(result_str) > max_len:
        preview += "..."
    color = COLOR_RED if is_error else COLOR_DIM
    prefix = "✗" if is_error else "↳"
    sys.stdout.write(f"{color}  {prefix} ({len(result_str)} chars) {preview}{COLOR_RESET}\n")
    sys.stdout.flush()


def _interrupted_msg():
    """Print interruption message."""
    sys.stdout.write(f"\n{COLOR_YELLOW}⚠ Interrupted by Ctrl-C{COLOR_RESET}\n")
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


def _truncate_tool_use_inputs(blocks: list) -> list:
    """Truncate large tool_use inputs in serialized blocks.

    Uses registry metadata (large_input_keys) to identify which tools/keys to truncate.
    Claude already generated these — it doesn't need them echoed back in full.
    """
    registry = get_registry()
    out = []
    for block in blocks:
        if (isinstance(block, dict)
                and block.get("type") == "tool_use"):
            large_keys = registry.get_large_input_keys(block.get("name", ""))
            if large_keys:
                inp = block.get("input", {})
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
                    block["input"] = truncated_inp
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
        content = msg.get("content", "")
        if isinstance(content, list):
            new_blocks = []
            for block in content:
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


def _serialize_content_blocks(content_blocks) -> list:
    """Serialize SDK content block objects to plain dicts for Bedrock compatibility."""
    serialized = []
    for block in content_blocks:
        if hasattr(block, 'model_dump'):
            d = block.model_dump(exclude_none=True)
        elif hasattr(block, 'dict'):
            d = block.dict(exclude_none=True)
        else:
            serialized.append(block)
            continue

        if d.get("type") == "tool_use":
            serialized.append({
                "type": "tool_use",
                "id": d["id"],
                "name": d["name"],
                "input": d.get("input", {}),
            })
        elif d.get("type") == "text":
            serialized.append({
                "type": "text",
                "text": d.get("text", ""),
            })
        else:
            serialized.append(d)
    return serialized


def _is_truncated_tool_call(stop_reason: str, tool_use_blocks: list) -> bool:
    """Detect if tool_use blocks were truncated due to max_tokens."""
    if stop_reason != "max_tokens" or not tool_use_blocks:
        return False
    return any(block.input is None for block in tool_use_blocks)


# ─── Streaming helper ────────────────────────────────────────────────────────

def _stream_response(client, api_kwargs: dict) -> tuple:
    """
    Stream a single API response.
    Returns (step_text, tool_use_blocks, stop_reason, final_message).
    """
    text_parts = []
    tool_use_blocks = []
    stop_reason = None
    final_message = None

    with client.messages.stream(**api_kwargs, timeout=STREAM_TIMEOUT) as stream:
        for event in stream:
            if hasattr(event, 'type') and event.type == 'content_block_delta':
                delta = event.delta
                if hasattr(delta, 'text'):
                    _stream_write(delta.text)
                    text_parts.append(delta.text)

        final_message = stream.get_final_message()
        stop_reason = final_message.stop_reason

        for block in final_message.content:
            if block.type == "tool_use":
                tool_use_blocks.append(block)

    return "".join(text_parts), tool_use_blocks, stop_reason, final_message


# ─── Main planner loop ───────────────────────────────────────────────────────

_STREAM_MAX_RETRIES = 2


def _stream_with_retry(client, api_kwargs: dict, step: int, all_text: str, working_messages: list):
    """Stream a response with auto-retry on transient failures.

    Returns (step_text, tool_use_blocks, stop_reason, final_message) on success.
    Returns None and a tuple (all_text, final_text, working_messages) on fatal error
    via a special sentinel: raises _StreamAbort with the return tuple.
    """
    for attempt in range(_STREAM_MAX_RETRIES + 1):
        try:
            return _stream_response(client, api_kwargs)
        except KeyboardInterrupt:
            _interrupted_msg()
            msg = all_text + "\n\n[Response interrupted by user]" if all_text else "[Response interrupted by user]"
            raise _StreamAbort(msg, "", working_messages)
        except Exception as stream_err:
            is_transient = any(kw in str(stream_err).lower() for kw in [
                "incomplete", "peer closed", "connection reset", "timeout",
                "eof", "broken pipe", "overloaded", "529", "server_error",
            ])
            if is_transient and attempt < _STREAM_MAX_RETRIES:
                retry_delay = 2 ** (attempt + 1)  # 2s, 4s
                retry_msg = f"\n⚡ Stream interrupted ({stream_err}), retrying in {retry_delay}s... (attempt {attempt + 2}/{_STREAM_MAX_RETRIES + 1})\n"
                sys.stdout.write(f"{COLOR_YELLOW}{retry_msg}{COLOR_RESET}")
                sys.stdout.flush()
                time.sleep(retry_delay)  # responds to KeyboardInterrupt
                continue
            # Non-transient error or retries exhausted
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

    Returns the list of tool_result dicts.
    Raises KeyboardInterrupt (with synthetic results already appended) if interrupted.
    """
    try:
        n_tools = len(tool_use_blocks)
        if n_tools > 1:
            _tool_info(f"Executing {n_tools} tool calls...")

        for block in tool_use_blocks:
            input_preview = str(block.input)
            if len(input_preview) > 150:
                input_preview = input_preview[:150] + "..."
            _tool_info(f"Calling: {block.name}({input_preview})")

        executed = _execute_tools(tool_use_blocks, tool_functions)

        tool_results = []
        for block, result in executed:
            content_str = _truncate_tool_result(result.content, is_error=result.is_error)
            _tool_result_preview(content_str, is_error=result.is_error)

            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content_str,
            }
            if result.is_error:
                tool_result_block["is_error"] = True

            tool_results.append(tool_result_block)

        return tool_results

    except KeyboardInterrupt:
        _interrupted_msg()
        synthetic_results = [
            {"type": "tool_result", "tool_use_id": b.id,
             "content": "[Tool execution interrupted by user]", "is_error": True}
            for b in tool_use_blocks
        ]
        working_messages.append({"role": "user", "content": synthetic_results})
        msg = all_text + "\n\n[Tool execution interrupted by user]" if all_text else "[Tool execution interrupted by user]"
        raise _StreamAbort(msg, "", working_messages)

def plan_next_action(client, messages: list, system_prompt: str, max_steps: int = None) -> tuple:
    """
    Streams Claude's response token-by-token to stdout.
    Handles tool use in a loop: when Claude calls tools, we execute them
    and continue streaming the next response.
    Returns (full_text, final_step_text, working_messages).
    """
    if max_steps is None:
        max_steps = DEFAULT_MAX_STEPS

    # Defaults for the outer except handler (overwritten inside try)
    working_messages = []
    all_text = ""

    try:
        working_messages = messages.copy()
        all_text = ""
        final_text = ""  # Only the last step's text (for markdown rendering)
        current_max_tokens = DEFAULT_MAX_TOKENS

        registry = get_registry()
        reg_version, tool_schemas, tool_functions = registry.snapshot()

        for step in range(max_steps):
            # Guard: compact working_messages if token count is too high
            working_messages = _compact_working_messages(working_messages)

            api_kwargs = dict(
                model=AGENT_MODEL,
                max_tokens=current_max_tokens,
                system=system_prompt,
                messages=working_messages,
            )

            if tool_schemas:
                api_kwargs["tools"] = tool_schemas

            # Stream with auto-retry on transient failures
            step_text, tool_use_blocks, stop_reason, final_message = _stream_with_retry(
                client, api_kwargs, step, all_text, working_messages
            )

            all_text += step_text
            final_text = step_text  # Always track latest step's text

            # If there are no tool calls, we're done
            if not tool_use_blocks:
                if step_text:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                result = all_text if all_text else "I processed your request but had no text response to return."
                # Append the final assistant message so history doesn't end
                # on a user-role tool_result from the previous step.
                serialized = _serialize_content_blocks(final_message.content)
                working_messages.append({"role": "assistant", "content": serialized})
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

            # Append assistant's full response (with large write inputs truncated)
            serialized = _serialize_content_blocks(final_message.content)
            serialized = _truncate_tool_use_inputs(serialized)
            working_messages.append({"role": "assistant", "content": serialized})

            # Execute tool calls and build results
            tool_results = _handle_tool_calls(tool_use_blocks, tool_functions, working_messages, all_text)

            working_messages.append({"role": "user", "content": tool_results})

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
            api_kwargs_final = dict(
                model=AGENT_MODEL,
                max_tokens=DEFAULT_MAX_TOKENS,
                system=system_prompt + "\n\n[SYSTEM: You have reached the maximum number of tool-use steps. Please provide your best answer now WITHOUT using any tools.]",
                messages=working_messages,
            )
            with client.messages.stream(**api_kwargs_final, timeout=STREAM_TIMEOUT) as stream:
                for event in stream:
                    if hasattr(event, 'type') and event.type == 'content_block_delta':
                        delta = event.delta
                        if hasattr(delta, 'text'):
                            _stream_write(delta.text)
                            fallback_text += delta.text
                sys.stdout.write("\n")
                sys.stdout.flush()
                # Append fallback assistant message to maintain valid history
                working_messages.append({"role": "assistant", "content": [{"type": "text", "text": fallback_text}]})
                return (fallback_text, fallback_text, working_messages)
        except KeyboardInterrupt:
            _interrupted_msg()
            msg = fallback_text + "\n\n[Response interrupted by user]" if fallback_text else "[Response interrupted by user]"
            return (msg, "", working_messages)
        except Exception as fallback_err:
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
        sys.stdout.write(f"{COLOR_RED}\nFatal error in planner: {e}\n{error_detail}{COLOR_RESET}\n")
        sys.stdout.flush()
        return (f"Error during planning: {e}", "", working_messages)
