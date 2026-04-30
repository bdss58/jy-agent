# Terminal UX — Formatting, colors, spinner, and streaming callbacks for AgentLoop.

import sys
import time
import threading
from dataclasses import dataclass
from ..runtime.tools.registry import get_registry
from ..runtime.loop.engine import LoopCallbacks
from .cli import console


# ─── Stream state ────────────────────────────────────────────────────────────

@dataclass
class StreamState:
    """Typed state shared between streaming callbacks."""
    needs_newline: bool = False


# ─── Output helpers ──────────────────────────────────────────────────────────

def _stream_write(text: str):
    """Write streamed text to stdout (char-level streaming — Rich can't do this)."""
    sys.stdout.write(text)
    sys.stdout.flush()


# ─── Thinking spinner ───────────────────────────────────────────────────────

class ThinkingSpinner:
    """Animated spinner shown while waiting for the first token.

    Uses a background thread to animate; call stop() when the first
    text/tool_use delta arrives.  Thread-safe and idempotent.
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
            sys.stdout.write(f"\r\033[2m  {frame} {self._label}... ({elapsed:.1f}s)\033[0m")
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
    # Flush raw stdout first to maintain ordering with Rich output
    sys.stdout.flush()
    console.print(f"\n  🔧 {msg}", style="yellow")


def _tool_call_header(tool_name: str, tool_input: dict):
    """Print a compact, visually distinct tool call header."""
    icon = _TOOL_ICONS.get(tool_name, "🔧")
    # Build compact arg summary
    args_preview = _format_tool_args(tool_name, tool_input)
    # Flush raw stdout first to maintain ordering with Rich output
    sys.stdout.flush()
    from rich.text import Text
    text = Text()
    text.append(f"\n  {icon} {tool_name}", style="yellow")
    if args_preview:
        text.append(f" {args_preview}", style="dim")
    console.print(text)


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
    from rich.text import Text
    # Flush raw stdout to maintain ordering with Rich output
    sys.stdout.flush()

    lines = result_str.split('\n')
    n_lines = len(lines)
    n_chars = len(result_str)

    if is_error:
        # Errors: show full (they're usually short)
        preview = result_str[:300].replace('\n', ' ↵ ')
        if n_chars > 300:
            preview += "..."
        text = Text(f"  ✗ {preview}", style="bold red")
        console.print(text)
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
    text = Text()
    text.append("  ✓", style="green")
    text.append(f" {first_line}")
    text.append(f" ({size_info})", style="dim")
    console.print(text)


def _render_edit_diff(result_str: str):
    """Render edit_file output with color-coded diff lines using Rich Text."""
    from rich.text import Text
    # Flush raw stdout to maintain ordering with Rich output
    sys.stdout.flush()

    lines = result_str.split('\n')
    # First line is the summary (e.g. "Edited foo.py: replaced 3 lines...")
    summary = lines[0] if lines else ""
    console.print(Text(f"  ✓ {summary}", style="green"))

    # Render context lines with diff coloring
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith(">"):
            # Changed line — highlight in green
            console.print(Text(f"    {line}", style="green"))
        elif stripped.startswith("L") or stripped.startswith(" "):
            # Context line — dim
            console.print(Text(f"    {line}", style="dim"))
        elif line.strip():
            console.print(f"    {line}")


def _interrupted_msg():
    """Print interruption message."""
    sys.stdout.flush()
    console.print("\n[bold yellow]⚠ Interrupted by Ctrl-C[/bold yellow]")


def _runtime_warning(msg: str):
    """Print a short runtime recovery warning without aborting the turn."""
    sys.stdout.flush()
    from rich.text import Text
    text = Text(f"\n  ⚠ {msg}", style="dim yellow")
    console.print(text)


# ─── Streaming callbacks factory ────────────────────────────────────────────

def build_streaming_callbacks(stats, runtime_owner) -> tuple[LoopCallbacks, ThinkingSpinner]:
    """
    Build LoopCallbacks for streaming AgentLoop with terminal UX.
    Returns (callbacks, spinner) tuple.
    """
    # Mutable state shared by callbacks
    spinner = ThinkingSpinner()
    stream_state = StreamState()

    def _on_text_delta(text: str):
        spinner.stop()
        _stream_write(text)
        stream_state.needs_newline = True

    def _on_thinking_start():
        spinner.start()

    def _on_thinking_stop():
        spinner.stop()

    def _on_tool_start(name: str, tool_input: dict):
        spinner.stop()
        if stream_state.needs_newline:
            sys.stdout.write("\n")
            sys.stdout.flush()
            stream_state.needs_newline = False
        _tool_call_header(name, tool_input)

    def _on_tool_end(name: str, content: str, is_error: bool):
        stats.record_tool_call()
        _tool_result_preview(content, tool_name=name, is_error=is_error)

    def _on_retry(attempt: int, error: Exception):
        sys.stdout.flush()
        console.print(
            f"\n⚡ Stream interrupted ({error}), retrying... (attempt {attempt + 1})",
            style="bold yellow",
        )

    def _on_usage(usage: dict):
        stats.record_usage(
            usage,
            provider=runtime_owner.model_spec.provider,
            model=runtime_owner.model_spec.model,
        )

    def _on_step_progress(step: int, max_steps_val: int):
        stream_state.needs_newline = False
        if step >= 5:
            sys.stdout.flush()
            console.print(f"  [Step {step + 1}/{max_steps_val}]", style="dim")

    def _on_compaction(before_len: int, after_len: int):
        pass  # silent — could add diagnostic output here

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
        on_warning=_on_warning,
        on_truncation=_on_truncation,
        on_tool_batch=_on_tool_batch,
    )

    # Attach the stream state to the callbacks object for access by caller
    callbacks._stream_state = stream_state

    return callbacks, spinner