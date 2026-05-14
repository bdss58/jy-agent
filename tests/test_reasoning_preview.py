"""Tests for the tier-A reasoning preview + /think slash command.

Covers:
  - _ReasoningStreamer state machine (preview cap, line counting, fold marker
    suppression when content fits, mid-line newline injection, discard_last
    on retry).
  - End-to-end wiring through build_streaming_callbacks: feeding
    on_thinking_delta and on_thinking_block_end via the LoopCallbacks and
    asserting StreamingUI.reasoning_blocks reflects the result.
  - _cmd_think behaviour: no-blocks message, single-block, indexed block,
    out-of-range error, non-integer arg error.

These tests use captured stdout / a fake CLI; they never hit a real LLM.
"""
from __future__ import annotations

import io
import sys

import pytest


# ─── Streamer unit tests ─────────────────────────────────────────────────────


def _capture_stdout(fn, *args, **kw):
    buf = io.StringIO()
    real = sys.stdout
    sys.stdout = buf
    try:
        result = fn(*args, **kw)
    finally:
        sys.stdout = real
    return result, buf.getvalue()


def test_streamer_passes_through_when_under_cap():
    from jyagent.ui.terminal import _ReasoningStreamer

    s = _ReasoningStreamer(preview_lines=10)
    _, out = _capture_stdout(s.feed, "alpha\nbeta\n")
    assert s.lines_emitted == 2
    assert s.in_fold is False
    assert s.needs_newline is False
    # Output should contain the text (wrapped in ANSI; just check substring).
    assert "alpha" in out and "beta" in out


def test_streamer_caps_at_preview_lines_and_stops_writing():
    from jyagent.ui.terminal import _ReasoningStreamer

    s = _ReasoningStreamer(preview_lines=3)
    _, out = _capture_stdout(
        s.feed, "l1\nl2\nl3\nl4\nl5\n"
    )
    assert s.lines_emitted == 3
    assert s.in_fold is True
    # Only l1..l3 should be in the captured output.
    assert "l1" in out and "l2" in out and "l3" in out
    assert "l4" not in out and "l5" not in out


def test_streamer_subsequent_feed_after_fold_is_silent():
    from jyagent.ui.terminal import _ReasoningStreamer

    s = _ReasoningStreamer(preview_lines=2)
    _capture_stdout(s.feed, "l1\nl2\nl3\n")
    _, out = _capture_stdout(s.feed, "more text that should be silent")
    assert out == ""
    assert s.in_fold is True


def test_streamer_tracks_open_line_when_no_trailing_newline():
    from jyagent.ui.terminal import _ReasoningStreamer

    s = _ReasoningStreamer(preview_lines=5)
    _capture_stdout(s.feed, "partial line without newline")
    assert s.lines_emitted == 0
    assert s.needs_newline is True
    assert s.in_fold is False


def test_finalize_no_marker_when_content_fits():
    from jyagent.ui.terminal import _ReasoningStreamer

    s = _ReasoningStreamer(preview_lines=10)
    _capture_stdout(s.feed, "a\nb\n")
    _, marker = _capture_stdout(s.finalize, "a\nb\n", "end")
    # No fold marker because in_fold was never set.
    assert "more line" not in marker
    assert len(s.blocks) == 1
    assert s.blocks[0].text == "a\nb\n"
    assert s.blocks[0].reason == "end"


def test_finalize_prints_fold_marker_when_truncated():
    from jyagent.ui.terminal import _ReasoningStreamer

    s = _ReasoningStreamer(preview_lines=2)
    _capture_stdout(s.feed, "l1\nl2\nl3\nl4\nl5\n")
    _, marker = _capture_stdout(s.finalize, "l1\nl2\nl3\nl4\nl5\n", "end")
    assert "3 more lines folded" in marker
    assert "/think to expand" in marker


def test_finalize_marker_shows_reason_when_not_clean_end():
    from jyagent.ui.terminal import _ReasoningStreamer

    s = _ReasoningStreamer(preview_lines=1)
    _capture_stdout(s.feed, "l1\nl2\nl3\n")
    _, marker = _capture_stdout(s.finalize, "l1\nl2\nl3\n", "tool_interrupt")
    assert "tool_interrupt" in marker


def test_finalize_injects_newline_when_preview_ended_mid_line():
    from jyagent.ui.terminal import _ReasoningStreamer

    s = _ReasoningStreamer(preview_lines=5)
    _capture_stdout(s.feed, "no newline here")
    _, marker = _capture_stdout(s.finalize, "no newline here", "end")
    # The finalize should have emitted a "\n" before the (absent) marker
    # so subsequent output starts on its own line.  Since content fit, no
    # marker is printed, but a single "\n" should still appear because
    # needs_newline was True.
    assert marker.startswith("\n")


def test_discard_last_drops_error_block_only():
    from jyagent.ui.terminal import _ReasoningStreamer, ReasoningBlock

    s = _ReasoningStreamer(preview_lines=5)
    s.blocks.append(ReasoningBlock(text="clean", reason="end"))
    s.blocks.append(ReasoningBlock(text="failed", reason="error"))
    s.discard_last()
    assert len(s.blocks) == 1
    assert s.blocks[0].reason == "end"

    # Calling discard_last when the last block is NOT an error must be a no-op
    # (we don't want a stray on_stream_retry to delete clean history).
    s.discard_last()
    assert len(s.blocks) == 1


def test_discard_last_resets_in_flight_preview_state():
    from jyagent.ui.terminal import _ReasoningStreamer

    s = _ReasoningStreamer(preview_lines=2)
    _capture_stdout(s.feed, "l1\nl2\nl3\n")
    assert s.in_fold is True
    s.discard_last()
    # Replay starts fresh: preview budget reset.
    assert s.in_fold is False
    assert s.lines_emitted == 0


# ─── End-to-end through build_streaming_callbacks ────────────────────────────


class _Stub:
    """Minimal stats / runtime_owner stub."""

    def __init__(self):
        self.tool_calls = 0
        self.usages: list[dict] = []
        # runtime_owner.model_spec.provider / .model are read by on_usage.
        class _Spec:
            provider = "stub"
            model = "stub-model"
        self.model_spec = _Spec()

    def record_tool_call(self):
        self.tool_calls += 1

    def record_usage(self, usage, provider=None, model=None):
        self.usages.append(usage)


def test_build_streaming_callbacks_records_reasoning_blocks():
    from jyagent.ui.terminal import build_streaming_callbacks

    stub = _Stub()
    ui = build_streaming_callbacks(stub, stub, reasoning_preview_lines=2)
    cb = ui.callbacks

    # Simulate engine fires.
    _capture_stdout(cb.on_thinking_start)
    _capture_stdout(cb.on_thinking_delta, "l1\nl2\nl3\nl4\n")
    _capture_stdout(cb.on_thinking_block_end, "l1\nl2\nl3\nl4\n", "end")

    blocks = ui.reasoning_blocks
    assert len(blocks) == 1
    assert blocks[0].text == "l1\nl2\nl3\nl4\n"
    assert blocks[0].reason == "end"


def test_build_streaming_callbacks_respects_reasoning_show_false():
    from jyagent.ui.terminal import build_streaming_callbacks

    stub = _Stub()
    ui = build_streaming_callbacks(stub, stub, reasoning_show=False)
    cb = ui.callbacks

    # With reasoning_show=False, deltas + block_end must be no-ops for
    # recording purposes.
    _capture_stdout(cb.on_thinking_delta, "hidden\n")
    _capture_stdout(cb.on_thinking_block_end, "hidden\n", "end")
    assert ui.reasoning_blocks == []


def test_build_streaming_callbacks_stream_retry_drops_error_block():
    from jyagent.ui.terminal import build_streaming_callbacks

    stub = _Stub()
    ui = build_streaming_callbacks(stub, stub, reasoning_preview_lines=2)
    cb = ui.callbacks

    # Attempt 1: fails mid-stream → runner fires block_end with reason="error".
    _capture_stdout(cb.on_thinking_delta, "partial\nattempt\n")
    _capture_stdout(cb.on_thinking_block_end, "partial\nattempt\n", "error")
    # Then on_stream_retry fires.
    _capture_stdout(cb.on_stream_retry, "transient_error", "partial\nattempt\n")

    # Attempt 2: succeeds.
    _capture_stdout(cb.on_thinking_delta, "final\nresult\n")
    _capture_stdout(cb.on_thinking_block_end, "final\nresult\n", "end")

    blocks = ui.reasoning_blocks
    # Error block should have been dropped; only the clean replay remains.
    assert len(blocks) == 1
    assert blocks[0].reason == "end"
    assert blocks[0].text == "final\nresult\n"


# ─── /think slash command ────────────────────────────────────────────────────


class _FakeCLI:
    def __init__(self):
        self.systems: list[str] = []
        self.errors: list[str] = []

    def print_system(self, msg):
        self.systems.append(str(msg))

    def print_error(self, msg):
        self.errors.append(str(msg))


def test_cmd_think_empty_state_prints_notice():
    from jyagent.agent import AgentRunState
    from jyagent.agent_commands import _cmd_think

    cli = _FakeCLI()
    state = AgentRunState()
    _capture_stdout(_cmd_think, cli=cli, state=state, user_input="/think")
    assert any("No reasoning" in m for m in cli.systems)


def test_cmd_think_renders_all_blocks_when_no_arg():
    from jyagent.agent import AgentRunState
    from jyagent.agent_commands import _cmd_think
    from jyagent.ui.terminal import ReasoningBlock

    cli = _FakeCLI()
    state = AgentRunState()
    state.last_reasoning_blocks = [
        ReasoningBlock(text="first\nblock", reason="end"),
        ReasoningBlock(text="second\nblock", reason="tool_interrupt"),
    ]
    _, out = _capture_stdout(_cmd_think, cli=cli, state=state, user_input="/think")
    assert "block 1/2" in out
    assert "block 2/2" in out
    assert "tool_interrupt" in out


def test_cmd_think_indexed_block():
    from jyagent.agent import AgentRunState
    from jyagent.agent_commands import _cmd_think
    from jyagent.ui.terminal import ReasoningBlock

    cli = _FakeCLI()
    state = AgentRunState()
    state.last_reasoning_blocks = [
        ReasoningBlock(text="A", reason="end"),
        ReasoningBlock(text="B", reason="end"),
        ReasoningBlock(text="C", reason="end"),
    ]
    _, out = _capture_stdout(_cmd_think, cli=cli, state=state, user_input="/think 2")
    assert "block 2/3" in out
    # The other blocks' bodies should NOT be present.
    assert "block 1/3" not in out and "block 3/3" not in out


def test_cmd_think_non_integer_arg_errors():
    from jyagent.agent import AgentRunState
    from jyagent.agent_commands import _cmd_think
    from jyagent.ui.terminal import ReasoningBlock

    cli = _FakeCLI()
    state = AgentRunState()
    state.last_reasoning_blocks = [ReasoningBlock(text="A", reason="end")]
    _cmd_think(cli=cli, state=state, user_input="/think foo")
    assert any("expected a block number" in e for e in cli.errors)


def test_cmd_think_out_of_range_arg_errors():
    from jyagent.agent import AgentRunState
    from jyagent.agent_commands import _cmd_think
    from jyagent.ui.terminal import ReasoningBlock

    cli = _FakeCLI()
    state = AgentRunState()
    state.last_reasoning_blocks = [ReasoningBlock(text="A", reason="end")]
    _cmd_think(cli=cli, state=state, user_input="/think 5")
    assert any("out of range" in e for e in cli.errors)
