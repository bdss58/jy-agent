# Planner — Streaming tool-use loop with structured results, concurrent execution, unified timeouts.
# Honesty enforced via system prompt injection (HONESTY_SYSTEM_ADDENDUM).
import os
import sys
import time
import inspect
import traceback
import concurrent.futures
from .registry import get_registry

TOOL_COLOR = "\033[0;33m"  # yellow for tool info
COLOR_RESET = "\033[0m"
COLOR_DIM = "\033[2m"
COLOR_YELLOW = "\033[1;33m"
COLOR_CYAN = "\033[1;36m"
COLOR_RED = "\033[1;31m"

# Configurable max_tokens via environment variable
DEFAULT_MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "16384"))
MAX_TOKENS_CAP = int(os.environ.get("ANTHROPIC_MAX_TOKENS_CAP", "128000"))

# Practical step limit
DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "100"))

# Tool result truncation — prevents working_messages from growing unboundedly
MAX_TOOL_RESULT_CHARS = int(os.environ.get("AGENT_MAX_TOOL_RESULT_CHARS", "8000"))
MAX_TOOL_USE_INPUT_CHARS = int(os.environ.get("AGENT_MAX_TOOL_USE_INPUT_CHARS", "4000"))
MAX_WORKING_TOKENS = int(os.environ.get("AGENT_MAX_WORKING_TOKENS", "100000"))

# Unified tool timeout (P2) — applies to all tool calls, not just run_shell
DEFAULT_TOOL_TIMEOUT = int(os.environ.get("AGENT_TOOL_TIMEOUT", "120"))

HONESTY_SYSTEM_ADDENDUM = """

CRITICAL HONESTY RULES — You MUST follow these at all times:
1. NEVER present information as if you fetched, retrieved, or verified it from external sources unless you actually used a tool (run_shell, read_file, etc.) to obtain that information in this conversation.
2. If asked about recent events, changelogs, version numbers, release dates, or other rapidly-changing factual information, you MUST use tools to look it up. Do NOT fabricate or hallucinate this information.
3. If you cannot verify information via tools, explicitly say so: "I don't have verified information about this. Let me try to look it up." Then use a tool.
4. NEVER claim information was "sourced from", "fetched from", "pulled from", or "verified against" any external source unless you actually performed a tool call to access that source.
5. If tools are unavailable or fail, clearly state that you were unable to verify the information rather than presenting unverified claims as facts.
6. It is always better to say "I'm not sure" or "I'd need to look that up" than to fabricate detailed information.
7. When presenting information from your training data, clearly distinguish it as such: "Based on my training data (which may be outdated)..." rather than implying it's freshly sourced.
8. IMPORTANT: Using a tool to read one file does NOT verify claims about other files, URLs, APIs, or topics you did not explicitly check with tools. Each claim of verification must correspond to an actual tool call for that specific resource.
9. When reporting findings from code review or file analysis, only make claims about content you actually read via tools. Do not extrapolate or invent issues in files/sections you did not examine.
"""


# ─── Structured Tool Result ──────────────────────────────────────────────────

class ToolResult:
    """Structured tool result with explicit error flag.
    
    Benefits over raw strings:
    - Anthropic API supports `is_error: true` in tool_result blocks, which helps
      Claude reason better about failures vs. successful results containing "Error"
    - Error results are never truncated
    - Clear programmatic distinction between success and failure
    """
    __slots__ = ('content', 'is_error')
    
    def __init__(self, content: str, is_error: bool = False):
        self.content = content
        self.is_error = is_error
    
    def __str__(self):
        return self.content
    
    def __repr__(self):
        return f"ToolResult(is_error={self.is_error}, content={self.content[:80]!r}...)"


def _is_error_result(result) -> bool:
    """Detect if a tool result indicates an error.
    
    Supports both:
    - ToolResult objects (structured, preferred)
    - Legacy string returns starting with "Error:" (backward compatible)
    """
    if isinstance(result, ToolResult):
        return result.is_error
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
    """Truncate large tool_use inputs (write_file, evolve_self, edit_file) in serialized blocks.

    Claude already generated these — it doesn't need them echoed back in full.
    """
    LARGE_INPUT_TOOLS = {"write_file", "evolve_self", "edit_file"}
    LARGE_INPUT_KEYS = {"content", "new_text", "old_text"}
    out = []
    for block in blocks:
        if (isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("name") in LARGE_INPUT_TOOLS):
            inp = block.get("input", {})
            truncated_inp = {}
            did_truncate = False
            for k, v in inp.items():
                if k in LARGE_INPUT_KEYS and isinstance(v, str) and len(v) > MAX_TOOL_USE_INPUT_CHARS:
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

    from .self_memory import estimate_conversation_tokens
    estimated = estimate_conversation_tokens(messages)
    if estimated <= max_tokens:
        return messages

    AGGRESSIVE_LIMIT = 2000
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


def _inject_honesty_rules(system_prompt: str) -> str:
    """Append honesty rules to the system prompt."""
    return system_prompt + HONESTY_SYSTEM_ADDENDUM


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

    if tool_input is None:
        tool_input = {}

    # Validate required parameters
    try:
        sig = inspect.signature(fn)
        missing = [
            pname for pname, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty
            and param.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
            and pname not in tool_input
        ]
        if missing:
            return ToolResult(
                f"Error: Tool {tool_name} called with missing required parameters: {missing}. "
                f"Received: {list(tool_input.keys())}. "
                f"Try breaking the operation into smaller steps.",
                is_error=True
            )
    except (ValueError, TypeError):
        pass

    try:
        raw_result = fn(**tool_input)
        # Support tools that already return ToolResult
        if isinstance(raw_result, ToolResult):
            return raw_result
        # Legacy string results — detect errors by convention
        result_str = str(raw_result)
        is_error = result_str.startswith("Error:") or result_str.startswith("Error calling tool")
        return ToolResult(result_str, is_error=is_error)
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
    
    Tools that manage their own timeout (run_shell) are given extra slack.
    MCP tools and other external calls benefit most from this safety net.
    """
    if timeout is None:
        timeout = DEFAULT_TOOL_TIMEOUT
    
    # run_shell has its own timeout parameter — give it extra slack
    if tool_name == "run_shell":
        user_timeout = tool_input.get("timeout", 60)
        timeout = max(timeout, int(user_timeout) + 10)
    
    # web_fetch can be slow — give it more time
    if tool_name == "web_fetch":
        timeout = max(timeout, 180)
    
    # MCP tools may involve browser automation — generous timeout
    if tool_name.startswith("mcp__"):
        timeout = max(timeout, 180)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_execute_tool, tool_name, tool_input, tool_functions)
            return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return ToolResult(
            f"Error: Tool '{tool_name}' timed out after {timeout}s. "
            f"Consider breaking the operation into smaller steps.",
            is_error=True
        )
    except KeyboardInterrupt:
        raise


# ─── Concurrent tool execution (P1) ─────────────────────────────────────────

def _execute_tools_concurrently(tool_blocks: list, tool_functions: dict) -> list:
    """Execute multiple tool calls concurrently when they are independent.
    
    Claude may return multiple tool_use blocks in a single response.
    These are independent by definition (Claude chose to emit them together),
    so we can safely execute them in parallel.
    
    Returns list of (block, ToolResult) tuples in original order.
    """
    if len(tool_blocks) <= 1:
        # Single tool call — no need for thread pool overhead
        block = tool_blocks[0]
        result = _execute_tool_with_timeout(block.name, block.input, tool_functions)
        return [(block, result)]
    
    results = [None] * len(tool_blocks)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tool_blocks), 4)) as executor:
        future_to_idx = {}
        for i, block in enumerate(tool_blocks):
            future = executor.submit(
                _execute_tool_with_timeout, block.name, block.input, tool_functions
            )
            future_to_idx[future] = i
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = (tool_blocks[idx], future.result())
            except KeyboardInterrupt:
                raise
            except Exception as e:
                results[idx] = (tool_blocks[idx], ToolResult(
                    f"Error: Unexpected failure executing {tool_blocks[idx].name}: {e}",
                    is_error=True
                ))
    
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
    step_text = ""
    tool_use_blocks = []
    stop_reason = None
    final_message = None

    with client.messages.stream(**api_kwargs, timeout=300) as stream:
        for event in stream:
            if hasattr(event, 'type') and event.type == 'content_block_delta':
                delta = event.delta
                if hasattr(delta, 'text'):
                    _stream_write(delta.text)
                    step_text += delta.text

        final_message = stream.get_final_message()
        stop_reason = final_message.stop_reason

        for block in final_message.content:
            if block.type == "tool_use":
                tool_use_blocks.append(block)

    return step_text, tool_use_blocks, stop_reason, final_message


# ─── Main planner loop ───────────────────────────────────────────────────────

def plan_next_action(client, messages: list, system_prompt: str, max_steps: int = None) -> str:
    """
    Streams Claude's response token-by-token to stdout.
    Handles tool use in a loop: when Claude calls tools, we execute them
    and continue streaming the next response.
    Returns the full concatenated text response.
    """
    if max_steps is None:
        max_steps = DEFAULT_MAX_STEPS

    try:
        working_messages = messages.copy()
        all_text = ""
        current_max_tokens = DEFAULT_MAX_TOKENS

        augmented_system_prompt = _inject_honesty_rules(system_prompt)

        for step in range(max_steps):
            registry = get_registry()
            tool_schemas = registry.get_schemas()
            tool_functions = registry.get_functions()

            api_kwargs = dict(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
                max_tokens=current_max_tokens,
                system=augmented_system_prompt,
                messages=working_messages,
            )

            if tool_schemas:
                api_kwargs["tools"] = tool_schemas

            # Guard: compact working_messages if token count is too high
            working_messages = _compact_working_messages(working_messages)

            # Stream with auto-retry on transient failures (e.g. API connection drops)
            _STREAM_MAX_RETRIES = 2
            for _stream_attempt in range(_STREAM_MAX_RETRIES + 1):
                try:
                    step_text, tool_use_blocks, stop_reason, final_message = _stream_response(client, api_kwargs)
                    break  # Success
                except KeyboardInterrupt:
                    _interrupted_msg()
                    return all_text + "\n\n[Response interrupted by user]" if all_text else "[Response interrupted by user]"
                except Exception as stream_err:
                    is_transient = any(kw in str(stream_err).lower() for kw in [
                        "incomplete", "peer closed", "connection reset", "timeout",
                        "eof", "broken pipe", "overloaded", "529", "server_error",
                    ])
                    if is_transient and _stream_attempt < _STREAM_MAX_RETRIES:
                        _retry_delay = 2 ** (_stream_attempt + 1)  # 2s, 4s
                        retry_msg = f"\n⚡ Stream interrupted ({stream_err}), retrying in {_retry_delay}s... (attempt {_stream_attempt + 2}/{_STREAM_MAX_RETRIES + 1})\n"
                        sys.stdout.write(f"{COLOR_YELLOW}{retry_msg}{COLOR_RESET}")
                        sys.stdout.flush()
                        time.sleep(_retry_delay)
                        continue
                    # Non-transient error or retries exhausted
                    error_msg = f"\n[Stream error at step {step+1}: {stream_err}]\n"
                    sys.stdout.write(f"{COLOR_RED}{error_msg}{COLOR_RESET}")
                    sys.stdout.flush()
                    if all_text:
                        return all_text + f"\n\n[Error: streaming interrupted at step {step+1}: {stream_err}]"
                    return f"Error during streaming: {stream_err}"

            all_text += step_text

            # If there are no tool calls, we're done
            if not tool_use_blocks:
                if step_text:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                return all_text if all_text else "I processed your request but had no text response to return."

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

            # Execute tool calls — concurrently if multiple (P1)
            try:
                n_tools = len(tool_use_blocks)
                if n_tools > 1:
                    _tool_info(f"Executing {n_tools} tool calls concurrently...")

                for block in tool_use_blocks:
                    input_preview = str(block.input)
                    if len(input_preview) > 150:
                        input_preview = input_preview[:150] + "..."
                    _tool_info(f"Calling: {block.name}({input_preview})")

                # Execute (concurrent if multiple, sequential if single)
                executed = _execute_tools_concurrently(tool_use_blocks, tool_functions)

                # Build tool_results in original order
                tool_results = []
                for block, result in executed:
                    content_str = _truncate_tool_result(result.content, is_error=result.is_error)
                    _tool_result_preview(content_str, is_error=result.is_error)

                    # Build the tool_result block — Anthropic API supports is_error
                    tool_result_block = {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content_str,
                    }
                    if result.is_error:
                        tool_result_block["is_error"] = True

                    tool_results.append(tool_result_block)

            except KeyboardInterrupt:
                _interrupted_msg()
                return all_text + "\n\n[Tool execution interrupted by user]" if all_text else "[Tool execution interrupted by user]"

            working_messages.append({"role": "user", "content": tool_results})

            # Show step progress for long operations
            if step >= 5:
                sys.stdout.write(f"{COLOR_DIM}  [Step {step+1}/{max_steps}]{COLOR_RESET}\n")
                sys.stdout.flush()

        # Max steps reached
        max_step_msg = f"\n\n⚠️ Reached maximum reasoning steps ({max_steps}). My response may be incomplete."
        sys.stdout.write(f"{COLOR_YELLOW}{max_step_msg}{COLOR_RESET}\n")
        sys.stdout.flush()

        if all_text:
            return all_text + max_step_msg

        # Try a final non-tool response
        fallback_text = ""
        try:
            api_kwargs_final = dict(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
                max_tokens=DEFAULT_MAX_TOKENS,
                system=augmented_system_prompt + "\n\n[SYSTEM: You have reached the maximum number of tool-use steps. Please provide your best answer now WITHOUT using any tools.]",
                messages=working_messages,
            )
            with client.messages.stream(**api_kwargs_final, timeout=300) as stream:
                for event in stream:
                    if hasattr(event, 'type') and event.type == 'content_block_delta':
                        delta = event.delta
                        if hasattr(delta, 'text'):
                            _stream_write(delta.text)
                            fallback_text += delta.text
                sys.stdout.write("\n")
                sys.stdout.flush()
                return fallback_text
        except KeyboardInterrupt:
            _interrupted_msg()
            return fallback_text + "\n\n[Response interrupted by user]" if fallback_text else "[Response interrupted by user]"
        except Exception:
            pass

        return "I've reached my maximum reasoning steps. Please try rephrasing your request."

    except KeyboardInterrupt:
        _interrupted_msg()
        return all_text + "\n\n[Interrupted by user]" if all_text else "[Interrupted by user]"
    except Exception as e:
        error_detail = traceback.format_exc()
        sys.stdout.write(f"{COLOR_RED}\nFatal error in planner: {e}\n{error_detail}{COLOR_RESET}\n")
        sys.stdout.flush()
        return f"Error during planning: {e}"
