# Terminal UX — Formatting, colors, spinner, and streaming callbacks for AgentLoop.

import sys
import time
import threading
from dataclasses import dataclass, field

from rich.text import Text

from ..runtime.loop.engine import LoopCallbacks
from .output import _STDOUT_LOCK, console


# ─── Stream state ────────────────────────────────────────────────────────────

@dataclass
class StreamState:
    """Typed state shared between streaming callbacks."""
    needs_newline: bool = False


# ─── Reasoning (thinking) preview state ─────────────────────────────────────
#
# Tier-A reasoning UX: stream the first N lines of a thinking block in dim
# italic, then suppress further deltas and print a fold marker when the block
# ends.  Each completed block is recorded so the ``/think`` slash command can
# re-render the most recent turn's reasoning expanded.
#
# State machine (per turn — a fresh StreamingUI is built per turn in agent.py):
#
#   on_thinking_delta(chunk) → _ReasoningStreamer.feed(chunk)
#       streams up to ``preview_lines`` newlines worth of text, then flips
#       ``in_fold`` and stops writing (still buffers for the record).
#
#   on_thinking_block_end(text, reason) → _ReasoningStreamer.finalize(...)
#       newline-terminates the preview if mid-line, prints the fold marker
#       footer if the block was truncated, and appends a ReasoningBlock to
#       ``StreamingUI.reasoning_blocks``.
#
#   on_stream_retry(reason, partial) → _ReasoningStreamer.discard_last()
#       The runner already flushed a block with reason="error" on its way
#       out; we drop it from the record so the upcoming replay doesn't
#       produce a duplicate entry.  We do NOT erase the on-screen output
#       (the user has already seen it); the retry marker that the existing
#       on_stream_retry handler prints is the user's cue.

@dataclass
class ReasoningBlock:
    """One completed reasoning/thinking block within a turn."""
    text: str
    reason: str  # "end" | "tool_interrupt" | "error"


@dataclass
class _ReasoningStreamer:
    """Mutable per-turn state for the reasoning preview renderer."""
    preview_lines: int = 5
    # Lines newline-terminated so far in the current block.
    lines_emitted: int = 0
    # True once we've hit ``preview_lines`` and stopped writing chunks.
    in_fold: bool = False
    # True if the current preview line has unterminated chars (needs a
    # trailing newline before we print the fold marker / next block).
    needs_newline: bool = False
    # Completed blocks for the current turn, in order.
    blocks: list[ReasoningBlock] = field(default_factory=list)

    def _reset_block(self) -> None:
        self.lines_emitted = 0
        self.in_fold = False
        self.needs_newline = False

    def feed(self, chunk: str) -> None:
        """Stream a thinking_delta chunk to stdout, capped at preview_lines."""
        if not chunk or self.in_fold:
            return
        # Write up to (preview_lines - lines_emitted) newlines worth of
        # content; once we'd cross the cap, write only the portion up to
        # the final allowed newline and stop.
        remaining = self.preview_lines - self.lines_emitted
        if remaining <= 0:
            self.in_fold = True
            return

        # Find newline positions in this chunk.
        nls = [i for i, c in enumerate(chunk) if c == "\n"]
        if len(nls) < remaining:
            # Whole chunk fits within budget.
            self._write_dim(chunk)
            self.lines_emitted += len(nls)
            # If the chunk doesn't end with \n, we have an open line.
            self.needs_newline = not chunk.endswith("\n")
        else:
            # Cap at the ``remaining``-th newline (inclusive).
            cut = nls[remaining - 1] + 1
            self._write_dim(chunk[:cut])
            self.lines_emitted += remaining
            self.needs_newline = False
            self.in_fold = True

    @staticmethod
    def _write_dim(text: str) -> None:
        """Write text in dim italic style via raw ANSI (Rich can't stream chars)."""
        # ESC[2;3m = dim + italic; ESC[0m = reset.
        # We wrap each chunk independently so a forgotten reset can never
        # bleed into subsequent normal text.  Take the shared stdout lock
        # so this cannot interleave mid-line with the spinner / sub-agent
        # status / streaming text writers — see ``output._STDOUT_LOCK``.
        with _STDOUT_LOCK:
            sys.stdout.write(f"\033[2;3m{text}\033[0m")
            sys.stdout.flush()

    def finalize(self, full_text: str, reason: str) -> None:
        """End-of-block: print fold marker if needed, record the block."""
        # Terminate any open preview line so the marker/next output starts
        # on its own row.
        if self.needs_newline:
            _locked_newline()
            self.needs_newline = False

        total_lines = full_text.count("\n") + (0 if full_text.endswith("\n") else 1) if full_text else 0
        # If we never entered fold mode and the whole thing fit, no marker
        # is needed — the preview WAS the full content.
        if self.in_fold and total_lines > self.lines_emitted:
            hidden = total_lines - self.lines_emitted
            reason_tag = "" if reason == "end" else f" · {reason}"
            marker = Text()
            marker.append(f"  ▸ {hidden} more line{'s' if hidden != 1 else ''} folded", style="dim italic")
            marker.append(f" · /think to expand{reason_tag}", style="dim")
            console.print(marker)

        self.blocks.append(ReasoningBlock(text=full_text, reason=reason))
        self._reset_block()

    def discard_last(self) -> None:
        """Drop the most recently recorded block (used on stream-retry).

        Called from on_stream_retry: the runner just fired
        on_thinking_block_end(reason="error") on its way out of a failed
        stream attempt; the retry will replay the reasoning, so we don't
        want a duplicate entry in ``blocks``.  The text already written to
        stdout stays — the user has seen it, and the on_stream_retry banner
        printed by the existing handler is the signal that what follows is
        a re-emission.
        """
        if self.blocks and self.blocks[-1].reason == "error":
            self.blocks.pop()
        # Also reset the in-flight preview counters: the replay starts a
        # fresh block, so the preview budget should reset.
        self._reset_block()


# ─── Output helpers ──────────────────────────────────────────────────────────

# Raw-stdout writes serialize on the package-shared lock exported from
# :mod:`jyagent.ui.output`.  The ``ThinkingSpinner`` runs a background
# thread that writes spinner frames, while the main thread streams model
# text via ``_stream_write`` and clears the spinner line in
# ``ThinkingSpinner.stop()``.  Without serialization, a spinner frame
# written between the main thread's clear and the first streamed
# character leaves stale glyphs on the line.  Every raw-stdout writer in
# this module (spinner, ``_stream_write``, ``_ReasoningStreamer._write_dim``)
# must take this lock; Rich's ``console.print`` has its own internal lock
# and is excluded.  The same lock is acquired by the sub-agent status
# spinner in :mod:`jyagent.ui.subagent_status` so threaded ANSI from two
# UI modules cannot interleave mid-line.


def _stream_write(text: str):
    """Write streamed text to stdout (char-level streaming — Rich can't do this)."""
    with _STDOUT_LOCK:
        sys.stdout.write(text)
        sys.stdout.flush()


def _locked_newline() -> None:
    """Write a single newline under the shared stdout lock.

    Used by callers that need to terminate an in-flight raw-stdout line
    (streaming text, reasoning preview) before printing the next thing.
    Equivalent to ``_stream_write("\\n")`` but named for the common case.
    """
    with _STDOUT_LOCK:
        sys.stdout.write("\n")
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
            # Join WITHOUT a timeout: the daemon thread's wait cycle is
            # 80 ms, so it returns promptly once the event is set.  A
            # ``join(timeout=1)`` followed by an unconditional clear was
            # racy — if the daemon happened to be mid-write when the join
            # gave up, we'd clear the line and the daemon would then write
            # one final spinner frame on top of the now-streaming text.
            # Daemon=True still protects against pathological hangs at
            # interpreter shutdown.
            self._thread.join()
        self._started = False
        # Clear the spinner line.  Take the shared stdout lock so this can
        # never interleave with a (vestigial) spinner write.
        with _STDOUT_LOCK:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    def _animate(self):
        idx = 0
        t0 = time.time()
        while not self._stop_event.is_set():
            elapsed = time.time() - t0
            frame = self._FRAMES[idx % len(self._FRAMES)]
            with _STDOUT_LOCK:
                # Re-check the stop flag inside the lock: stop() may have
                # fired between the while-check and acquiring the lock; in
                # that case skip the write so we don't paint a frame after
                # the line was cleared.
                if self._stop_event.is_set():
                    break
                sys.stdout.write(f"\r\033[2m  {frame} {self._label}... ({elapsed:.1f}s)\033[0m")
                sys.stdout.flush()
            idx += 1
            self._stop_event.wait(0.08)


# ─── Tool output formatting ─────────────────────────────────────────────────
#
# Per-tool icon + argument formatter live in ``jyagent.tools.display`` —
# this module asks the tools package what to show, never embeds a per-tool
# switch table here.  Adding a new tool with custom display only touches
# that one module.
from ..tools.display import get_icon, format_tool_args


def _tool_info(msg: str):
    """Print a visible tool/status message."""
    # Flush raw stdout first to maintain ordering with Rich output
    sys.stdout.flush()
    console.print(f"\n  🔧 {msg}", style="yellow")


def _tool_call_header(tool_name: str, tool_input: dict):
    """Print a compact, visually distinct tool call header."""
    icon = get_icon(tool_name)
    args_preview = format_tool_args(tool_name, tool_input)
    # Flush raw stdout first to maintain ordering with Rich output
    sys.stdout.flush()
    text = Text()
    text.append(f"\n  {icon} {tool_name}", style="yellow")
    if args_preview:
        text.append(f" {args_preview}", style="dim")
    console.print(text)


def _format_duration(duration_ms: float | None) -> str:
    """Format a ms duration as a compact, human-friendly string.

    <1 ms  → "<1ms"       (don't bother)
    <1 s   → "123ms"      (integer ms)
    <10 s  → "1.23s"      (2 decimals)
    <60 s  → "12.3s"      (1 decimal)
    ≥60 s  → "1m23s"      (mm:ss rollup)
    None   → ""           (silent — keeps cancel-shim and legacy callers clean)
    """
    if duration_ms is None:
        return ""
    if duration_ms < 1:
        return "<1ms"
    if duration_ms < 1000:
        return f"{duration_ms:.0f}ms"
    s = duration_ms / 1000.0
    if s < 10:
        return f"{s:.2f}s"
    if s < 60:
        return f"{s:.1f}s"
    m, rem = divmod(int(s), 60)
    return f"{m}m{rem:02d}s"


def _tool_result_preview(result_str: str, tool_name: str = "", is_error: bool = False, duration_ms: float | None = None):
    """Print a compact tool result summary with smart formatting."""
    # Flush raw stdout to maintain ordering with Rich output
    sys.stdout.flush()

    lines = result_str.split('\n')
    n_lines = len(lines)
    n_chars = len(result_str)

    dur_str = _format_duration(duration_ms)

    if is_error:
        # Errors: show full (they're usually short)
        preview = result_str[:300].replace('\n', ' ↵ ')
        if n_chars > 300:
            preview += "..."
        text = Text(f"  ✗ {preview}", style="bold red")
        if dur_str:
            text.append(f" ({dur_str})", style="dim red")
        console.print(text)
        return

    # Detect edit_file diffs and show them nicely
    if tool_name == "edit_file" and any(ln.strip().startswith(">") for ln in lines[:20]):
        _render_edit_diff(result_str, duration_ms=duration_ms)
        return

    # Compact display: first line as summary + dims
    first_line = lines[0].strip() if lines else ""
    if len(first_line) > 150:
        first_line = first_line[:147] + "..."

    size_info = f"{n_chars} chars" if n_lines <= 1 else f"{n_lines} lines, {n_chars} chars"
    if dur_str:
        size_info = f"{size_info} · {dur_str}"
    text = Text()
    text.append("  ✓", style="green")
    text.append(f" {first_line}")
    text.append(f" ({size_info})", style="dim")
    console.print(text)


def _render_edit_diff(result_str: str, duration_ms: float | None = None):
    """Render edit_file output with color-coded diff lines using Rich Text."""
    # Flush raw stdout to maintain ordering with Rich output
    sys.stdout.flush()

    lines = result_str.split('\n')
    # First line is the summary (e.g. "Edited foo.py: replaced 3 lines...")
    summary = lines[0] if lines else ""
    header = Text()
    header.append(f"  ✓ {summary}", style="green")
    dur_str = _format_duration(duration_ms)
    if dur_str:
        header.append(f" ({dur_str})", style="dim")
    console.print(header)

    # Render context lines with diff coloring.
    # NOTE: check ``stripped`` for content-bearing prefixes (">", "L<n>:"),
    # but check the ORIGINAL ``line`` for the leading-space marker — after
    # ``strip()`` the leading whitespace is gone, so a `stripped.startswith(" ")`
    # branch is provably unreachable (was a real bug, caught in code review).
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith(">"):
            # Changed line — highlight in green
            console.print(Text(f"    {line}", style="green"))
        elif stripped.startswith("L") or line.startswith(" "):
            # Context line (numbered "L42:" form, or simply indented) — dim
            console.print(Text(f"    {line}", style="dim"))
        elif stripped:
            console.print(f"    {line}")


def interrupted_msg():
    """Print interruption message."""
    sys.stdout.flush()
    console.print("\n[bold yellow]⚠ Interrupted by Ctrl-C[/bold yellow]")


def render_final_text(text: str, *, markdown: bool = True) -> None:
    """Render the assistant's final text answer as a Markdown panel.

    **Dual output is intentional**, not a bug.  By the time we get here the
    streaming callbacks have already written the raw assistant text to
    stdout char-by-char (that's the only way to get a true streaming feel
    — Rich can't stream Markdown mid-render).  After the stream completes,
    we *re-render* the same text through Rich's Markdown renderer so the
    user sees both:

      1. the raw stream (preserved for select-and-copy, scrollback, and
         any consumer piping our stdout), and
      2. a polished Markdown panel below it (headings, lists, code
         highlighting via ``monokai``).

    Skipped when:

      * ``markdown=False`` (user toggled it off via ``/markdown``), or
      * the text is empty / a status marker like
        ``[Response interrupted by user]`` — re-rendering those adds no value.

    Centralised here so the UI package owns all rendering decisions; callers
    only need to pass the final text and a toggle.
    """
    from rich.markdown import Markdown

    if not markdown:
        return
    stripped = text.strip() if text else ""
    if not stripped or stripped.startswith("["):
        return

    # No Panel/border around the Markdown body: borders get included when
    # users select-and-copy the rendered output.  A thin header + trailing
    # hint gives the visual cue without polluting copy-paste.
    console.print()
    console.print("[bold green]📝 Rendered[/bold green] [dim](/markdown to toggle)[/dim]")
    try:
        console.print(Markdown(text, code_theme="monokai"))
    except Exception as e:
        # Markdown rendering is best-effort — never crash the turn over
        # it.  But also don't leave the user staring at a header with no
        # body: surface a short notice and re-print the raw text plainly
        # so the dual-output contract still holds.
        console.print(f"[dim](markdown render failed: {type(e).__name__}; showing raw text)[/dim]")
        console.print(text)


def _runtime_warning(msg: str):
    """Print a short runtime recovery warning without aborting the turn."""
    sys.stdout.flush()
    text = Text(f"\n  ⚠ {msg}", style="dim yellow")
    console.print(text)


# ─── Streaming callbacks factory ────────────────────────────────────────────

@dataclass
class StreamingUI:
    """Typed bundle returned by ``build_streaming_callbacks``.

    Replaces the old ``(callbacks, spinner)`` tuple plus the
    ``callbacks._stream_state`` side-channel with a small, named API:

      ui.callbacks  — wired LoopCallbacks for the AgentLoop
      ui.spinner    — ThinkingSpinner; caller must stop() in finally
      ui.flush_trailing_newline()  — write a stdout newline iff the last
                                     thing emitted was a partial text line
                                     (i.e. no terminator written yet).

    Keeping ``flush_trailing_newline`` as a method (not a flag) hides the
    state from callers and lets us change the heuristic later (e.g. switch
    to an ``on_text_done`` engine event) without touching ``agent.py``.
    """
    callbacks: LoopCallbacks
    spinner: "ThinkingSpinner"
    _stream_state: StreamState
    # Per-turn reasoning preview state.  Exposed so the CLI's ``/think``
    # slash command can re-render this turn's reasoning expanded.
    _reasoning: "_ReasoningStreamer" = field(default_factory=_ReasoningStreamer)

    @property
    def reasoning_blocks(self) -> list[ReasoningBlock]:
        """Completed reasoning blocks from this turn (in order)."""
        return list(self._reasoning.blocks)

    def flush_trailing_newline(self) -> None:
        if self._stream_state.needs_newline:
            _locked_newline()
            self._stream_state.needs_newline = False


def build_streaming_callbacks(
    stats,
    runtime_owner,
    *,
    reasoning_show: bool = True,
    reasoning_preview_lines: int = 5,
) -> StreamingUI:
    """
    Build LoopCallbacks for streaming AgentLoop with terminal UX.

    Returns a :class:`StreamingUI` bundle.  Use ``ui.callbacks`` to pass to
    ``AgentLoop``, ``ui.spinner`` to ensure stop() in a ``finally``, and
    ``ui.flush_trailing_newline()`` after the loop finishes to terminate
    the streamed text line cleanly.

    ``reasoning_show=True`` (default) streams the first
    ``reasoning_preview_lines`` lines of each thinking block in dim italic
    and prints a fold marker after long blocks; the full text is recorded
    on the returned ``StreamingUI`` so a ``/think`` slash command can
    re-render it.  Set ``reasoning_show=False`` to fall back to the
    spinner-only behaviour (reasoning text discarded for display, full
    text still preserved into the assistant message by the engine).
    """
    # Mutable state shared by callbacks
    spinner = ThinkingSpinner()
    stream_state = StreamState()
    reasoning_streamer = _ReasoningStreamer(preview_lines=reasoning_preview_lines)

    def _on_text_delta(text: str):
        spinner.stop()
        # If a thinking-preview line was open (no trailing newline), close
        # it before answer text begins.  This keeps the reasoning preview
        # and the answer on separate visual rows.
        if reasoning_show and reasoning_streamer.needs_newline:
            _locked_newline()
            reasoning_streamer.needs_newline = False
        _stream_write(text)
        stream_state.needs_newline = True

    def _on_thinking_start():
        spinner.start()

    def _on_thinking_stop():
        spinner.stop()

    def _on_thinking_delta(chunk: str):
        if not reasoning_show:
            return
        # Stop the spinner the moment real reasoning text starts streaming;
        # otherwise the spinner line fights with our preview output.
        spinner.stop()
        reasoning_streamer.feed(chunk)

    def _on_thinking_block_end(full_text: str, reason: str):
        if not reasoning_show:
            return
        reasoning_streamer.finalize(full_text, reason)

    def _on_stream_retry(reason: str, partial_text: str):
        # Existing UX banner first (mirrors _on_retry for transient errors,
        # but on_stream_retry fires for every retry — both transient and
        # truncation — with partial-text context).
        sys.stdout.flush()
        # Drop the partial reasoning block the runner just flushed with
        # reason="error" so the replay doesn't double-record it.  The text
        # already on screen stays — the user has seen it, and the banner
        # below is the signal that what follows is a re-emission.
        if reasoning_show:
            reasoning_streamer.discard_last()
        console.print(
            f"\n⚡ Stream retry ({reason}); replaying...",
            style="bold yellow",
        )

    def _on_tool_start(name: str, tool_input: dict):
        spinner.stop()
        if stream_state.needs_newline:
            _locked_newline()
            stream_state.needs_newline = False
        # Also terminate any open thinking-preview line so the tool header
        # doesn't appear glued to the tail of a reasoning chunk.
        if reasoning_show and reasoning_streamer.needs_newline:
            _locked_newline()
            reasoning_streamer.needs_newline = False
        _tool_call_header(name, tool_input)

    def _on_tool_end(name: str, content: str, is_error: bool, duration_ms: float | None = None):
        stats.record_tool_call()
        _tool_result_preview(content, tool_name=name, is_error=is_error, duration_ms=duration_ms)

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
        on_thinking_delta=_on_thinking_delta,
        on_thinking_block_end=_on_thinking_block_end,
        on_tool_start=_on_tool_start,
        on_tool_end=_on_tool_end,
        on_retry=_on_retry,
        on_stream_retry=_on_stream_retry,
        on_usage=_on_usage,
        on_step_progress=_on_step_progress,
        on_compaction=_on_compaction,
        on_warning=_on_warning,
        on_truncation=_on_truncation,
        on_tool_batch=_on_tool_batch,
    )

    return StreamingUI(
        callbacks=callbacks,
        spinner=spinner,
        _stream_state=stream_state,
        _reasoning=reasoning_streamer,
    )
