# Planner — Thin wrapper around AgentLoop engine.
# Wires terminal UX (spinner, headers, previews) to the loop via callbacks.
import sys
import time
import threading
from .registry import get_registry
from .session_stats import get_stats
from .runtime import RuntimeOwner, RuntimeOptions
from .loop_engine import (
    AgentLoop,
    LoopCallbacks,
    LoopConfig,
    LoopResult,
)
from .config import (
    DEFAULT_MAX_TOKENS, MAX_TOKENS_CAP, DEFAULT_MAX_STEPS,
    MAX_TOOL_RESULT_CHARS, MAX_TOOL_USE_INPUT_CHARS,
    MAX_WORKING_TOKENS, DEFAULT_TOOL_TIMEOUT, STREAM_TIMEOUT,
    COMPACT_TOOL_RESULT_CHARS, get_reasoning_config_for_provider,
)

# ─── Colors ──────────────────────────────────────────────────────────────────

TOOL_COLOR = "\033[0;33m"  # yellow for tool info
COLOR_RESET = "\033[0m"
COLOR_DIM = "\033[2m"
COLOR_YELLOW = "\033[1;33m"
COLOR_CYAN = "\033[1;36m"
COLOR_RED = "\033[1;31m"
COLOR_GREEN = "\033[0;32m"
COLOR_MAGENTA = "\033[0;35m"
COLOR_DIM_YELLOW = "\033[2;33m"


# ─── Output helpers ──────────────────────────────────────────────────────────

def _stream_write(text: str):
    """Write streamed text to stdout."""
    sys.stdout.write(text)
    sys.stdout.flush()


# ─── Thinking spinner ───────────────────────────────────────────────────────

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


# ─── Tool output formatting ─────────────────────────────────────────────────

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


# ─── Main planner entry point ───────────────────────────────────────────────

def plan_next_action(runtime_owner: RuntimeOwner, messages: list, system_prompt: str, max_steps: int = None) -> tuple:
    """
    Streams a provider-neutral runtime response token-by-token to stdout.
    Handles tool use in a loop: when the model calls tools, execute them and
    continue streaming the next response.
    Returns (full_text, final_step_text, working_messages).
    """
    if max_steps is None:
        max_steps = DEFAULT_MAX_STEPS

    stats = get_stats()
    stats.new_turn()

    working_messages = messages.copy()

    # ── Mutable state shared by callbacks ──
    spinner = _ThinkingSpinner()
    needs_newline = False  # True when text was streamed and tools/completion follow

    def _on_text_delta(text: str):
        nonlocal needs_newline
        spinner.stop()
        _stream_write(text)
        needs_newline = True

    def _on_thinking_start():
        spinner.start()

    def _on_thinking_stop():
        spinner.stop()

    def _on_tool_start(name: str, tool_input: dict):
        nonlocal needs_newline
        spinner.stop()
        if needs_newline:
            sys.stdout.write("\n")
            sys.stdout.flush()
            needs_newline = False
        _tool_call_header(name, tool_input)

    def _on_tool_end(name: str, content: str, is_error: bool):
        stats.record_tool_call()
        _tool_result_preview(content, tool_name=name, is_error=is_error)

    def _on_retry(attempt: int, error: Exception):
        retry_msg = (
            f"\n⚡ Stream interrupted ({error}), "
            f"retrying... "
            f"(attempt {attempt + 1})\n"
        )
        sys.stdout.write(f"{COLOR_YELLOW}{retry_msg}{COLOR_RESET}")
        sys.stdout.flush()

    def _on_usage(usage: dict):
        stats.record_usage(
            usage,
            provider=runtime_owner.model_spec.provider,
            model=runtime_owner.model_spec.model,
        )

    def _on_step_progress(step: int, max_steps_val: int):
        nonlocal needs_newline
        needs_newline = False
        if step >= 5:
            sys.stdout.write(f"{COLOR_DIM}  [Step {step + 1}/{max_steps_val}]{COLOR_RESET}\n")
            sys.stdout.flush()

    def _on_compaction(before_len: int, after_len: int):
        pass  # silent — could add diagnostic output here

    def _on_assistant_message(msg: dict) -> dict:
        """Truncate large tool_call inputs before the engine appends to messages."""
        msg = dict(msg)
        msg["content"] = _truncate_tool_call_blocks(msg.get("content", []))
        return msg

    def _on_warning(warning: str):
        _runtime_warning(warning)

    def _on_truncation():
        _tool_info("⚠️ Response truncated (max_tokens hit). Retrying with higher token limit...")

    def _on_tool_batch(n_tools: int):
        _tool_info(f"Executing {n_tools} tool calls...")

    callbacks = LoopCallbacks(
        on_text_delta=_on_text_delta,
        on_thinking_start=_on_thinking_start,
        on_thinking_stop=_on_thinking_stop,
        on_tool_start=_on_tool_start,
        on_tool_end=_on_tool_end,
        on_retry=_on_retry,
        on_usage=_on_usage,
        on_step_progress=_on_step_progress,
        on_compaction=_on_compaction,
        on_assistant_message=_on_assistant_message,
        on_warning=_on_warning,
        on_truncation=_on_truncation,
        on_tool_batch=_on_tool_batch,
    )

    config = LoopConfig(
        max_steps=max_steps,
        initial_max_tokens=DEFAULT_MAX_TOKENS,
        max_tokens_cap=MAX_TOKENS_CAP,
        auto_scale_on_truncation=True,
        token_scale_factor=2,
        concurrent_tools=True,
        max_tool_workers=4,
        tool_timeout=DEFAULT_TOOL_TIMEOUT,
        retry_attempts=2,
        retry_base_delay=2.0,
        compact_messages=True,
        max_working_tokens=MAX_WORKING_TOKENS,
        compact_tool_result_chars=COMPACT_TOOL_RESULT_CHARS,
        max_tool_result_chars=MAX_TOOL_RESULT_CHARS,
        streaming=True,
    )

    registry = get_registry()
    tool_source = lambda: registry.snapshot()[1:]  # (schemas, functions) — skip version

    loop = AgentLoop(runtime_owner, config, callbacks=callbacks, tool_source=tool_source)

    try:
        result: LoopResult = loop.run(system_prompt, working_messages)
    except KeyboardInterrupt:
        spinner.stop()
        _interrupted_msg()
        msg = "\n\n[Interrupted by user]"
        return (msg, "", working_messages)

    spinner.stop()  # ensure spinner is always cleaned up

    # ── Translate LoopResult to the legacy return tuple ──
    if result.status == "completed":
        if needs_newline:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return (result.text, result.final_text, result.messages)

    if result.status == "max_steps":
        max_step_msg = f"\n\n⚠️ Reached maximum reasoning steps ({max_steps}). My response may be incomplete."
        sys.stdout.write(f"{COLOR_YELLOW}{max_step_msg}{COLOR_RESET}\n")
        sys.stdout.flush()

        if result.text:
            return (result.text + max_step_msg, result.final_text, result.messages)

        # Try a final non-tool response
        return _fallback_completion(runtime_owner, system_prompt, result.messages, max_steps)

    if result.status == "interrupted":
        _interrupted_msg()
        return (result.text, "", result.messages)

    if result.status == "error":
        error_msg = f"\n[Error: {result.error}]\n"
        sys.stdout.write(f"{COLOR_RED}{error_msg}{COLOR_RESET}")
        sys.stdout.flush()
        if result.text:
            return (result.text + f"\n\n[Error: {result.error}]", "", result.messages)
        return (f"Error during planning: {result.error}", "", result.messages)

    # Fallback — should not happen
    return (result.text or "Unknown error", "", result.messages)


def _fallback_completion(
    runtime_owner: RuntimeOwner,
    system_prompt: str,
    working_messages: list,
    max_steps: int,
) -> tuple:
    """Try a final non-tool response when max_steps is reached."""
    stats = get_stats()
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
                    model=runtime_owner.model_spec.model,
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
        # Extract text from fallback response
        fallback_text = "".join(
            block.get("text", "")
            for block in final_message.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
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
        sys.stdout.write(
            f"{COLOR_RED}\n  Fallback response failed: "
            f"{fallback_err}"
            f"{COLOR_RESET}\n"
        )
        sys.stdout.flush()

    return ("I've reached my maximum reasoning steps. Please try rephrasing your request.", "", working_messages)
