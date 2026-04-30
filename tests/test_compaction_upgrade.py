# tests/test_compaction_upgrade.py — Tests for the context compaction improvements.
#
# Covers:
#   Observation masking, thinking pruning, and cache-friendly signatures.
#   File reinjection, compaction priority, and the 9-section prompt.

import os
import tempfile
from jyagent.runtime.tools.registry import get_registry
import pytest

# ─── Observation masking + thinking pruning ─────────────────────────────────

def test_thinking_blocks_pruned_from_old_messages():
    """Tier 0: thinking blocks are stripped from old messages, EXCEPT when
    they sit alongside a tool-invocation block in the same assistant message
    (Anthropic extended-thinking signatures are bound to the adjacent
    tool_use/tool_call and must not be separated — stripping them would
    invalidate the cryptographic signature and the provider would reject
    the next turn).
    """
    from jyagent.runtime.loop.engine import _compact_messages
    import jyagent.tools  # noqa: triggers registration

    messages = []
    for i in range(6):
        # Alternate: even-indexed assistants have a tool_call + thinking
        # (must be preserved together); odd-indexed have thinking but no
        # tool_call (free to strip).
        if i % 2 == 0:
            assistant_content = [
                {"type": "thinking", "thinking": f"Thinking #{i} " + "x" * 200},
                {"type": "text", "text": f"Response #{i}"},
                {"type": "tool_call", "id": f"tc_{i}", "name": "run_shell",
                 "arguments": {"command": "ls"}},
            ]
        else:
            assistant_content = [
                {"type": "thinking", "thinking": f"Thinking #{i} " + "x" * 200},
                {"type": "text", "text": f"Response #{i}"},
            ]
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({
            "role": "tool_result", "tool_call_id": f"tc_{i}",
            "tool_name": "run_shell", "content": "output " * 500,
        })

    result = _compact_messages(messages, max_tokens=1000, compact_chars=2000, batch=get_registry().freeze())
    assert result is not messages

    # Walk old messages (all but last 2) and check thinking-block state.
    for i, msg in enumerate(result[:-2]):
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        has_tool_block = any(
            isinstance(b, dict) and b.get("type") in ("tool_call", "tool_use")
            for b in content
        )
        thinking_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        if has_tool_block:
            # Must keep the thinking block — signature integrity.
            assert len(thinking_blocks) > 0, (
                f"Message {i} has a tool_call but its thinking block was "
                f"stripped — Anthropic signature would break"
            )
        else:
            # Safe to strip.
            assert len(thinking_blocks) == 0, (
                f"Message {i} still has {len(thinking_blocks)} thinking "
                f"blocks and has no tool_call to protect them"
            )


def test_observation_masking_clears_far_tool_results():
    """Tier 1: tool results beyond OBSERVATION_MASK_DISTANCE are fully cleared."""
    from jyagent.runtime.loop.engine import _compact_messages
    import jyagent.tools  # noqa

    messages = []
    # 10 pairs = 20 messages. With mask_distance=5, messages 0-14 are "far"
    for i in range(10):
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": f"Step {i}"}],
        })
        messages.append({
            "role": "tool_result", "tool_call_id": f"tc_{i}",
            "tool_name": "run_shell", "content": "long output " * 500,
        })

    result = _compact_messages(messages, max_tokens=500, compact_chars=2000, batch=get_registry().freeze())

    # Far-away ephemeral results should be fully cleared
    for i in [1, 3, 5, 7, 9, 11, 13]:  # tool_results at odd indices
        distance = len(result) - 1 - i
        content = result[i].get("content", "")
        if distance > 5:
            assert content == "[Tool result cleared]", \
                f"Message {i} (distance={distance}) not cleared: {content[:60]}"


def test_ephemeral_tools_cleared_even_when_close():
    """Tier 2: ephemeral tools are cleared regardless of distance."""
    from jyagent.runtime.loop.engine import _compact_messages
    import jyagent.tools  # noqa

    # 6 messages: assistant+tool_result x 3. Message indices 0-5.
    # Last 2 kept intact (4,5). Messages 0-3 compacted.
    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": "listing"}]},
        {"role": "tool_result", "tool_call_id": "tc_0",
         "tool_name": "list_directory", "content": "lots of dirs " * 500},
        {"role": "assistant", "content": [{"type": "text", "text": "searching"}]},
        {"role": "tool_result", "tool_call_id": "tc_1",
         "tool_name": "grep_files", "content": "match results " * 500},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]

    result = _compact_messages(messages, max_tokens=100, compact_chars=2000, batch=get_registry().freeze())

    # Both list_directory and grep_files are ephemeral → cleared
    assert result[1]["content"] == "[Tool result cleared]"
    assert result[3]["content"] == "[Tool result cleared]"


def test_persistent_tool_results_retained():
    """Persistent tool results (read_file, web_fetch) are NOT fully cleared."""
    from jyagent.runtime.loop.engine import _compact_messages
    import jyagent.tools  # noqa

    content_4k = "line of code\n" * 300  # ~3900 chars

    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": "reading"}]},
        {"role": "tool_result", "tool_call_id": "tc_0",
         "tool_name": "read_file", "content": content_4k},
        {"role": "assistant", "content": [{"type": "text", "text": "got it"}]},
        {"role": "tool_result", "tool_call_id": "tc_1",
         "tool_name": "run_shell", "content": content_4k},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
    ]

    result = _compact_messages(messages, max_tokens=100, compact_chars=2000, batch=get_registry().freeze())

    # read_file (persistent): should NOT be "[Tool result cleared]" but may be truncated
    rf_content = result[1]["content"]
    assert rf_content != "[Tool result cleared]", "Persistent read_file was fully cleared"
    assert "line of code" in rf_content, "Persistent content should have some data"

    # run_shell (ephemeral): should be fully cleared
    rs_content = result[3]["content"]
    assert rs_content == "[Tool result cleared]", f"Ephemeral run_shell not cleared: {rs_content[:60]}"


def test_last_two_messages_always_intact():
    """The last 2 messages are never modified regardless of content."""
    from jyagent.runtime.loop.engine import _compact_messages
    import jyagent.tools  # noqa

    big_thinking = {"type": "thinking", "thinking": "x" * 10000}
    messages = [
        {"role": "user", "content": "start"},
        {"role": "tool_result", "tool_call_id": "tc_0",
         "tool_name": "run_shell", "content": "y" * 10000},
        {"role": "user", "content": "end question"},
        {"role": "assistant", "content": [big_thinking, {"type": "text", "text": "answer"}]},
    ]

    result = _compact_messages(messages, max_tokens=100, compact_chars=500, batch=get_registry().freeze())

    # Last 2 messages unchanged
    assert result[-1]["content"][0]["thinking"] == "x" * 10000
    assert result[-2]["content"] == "end question"


# ─── Cache-friendly compaction signature ────────────────────────────────────

def test_compact_conversation_accepts_system_prompt():
    """compact_conversation accepts system_prompt parameter for cache reuse."""
    import inspect
    from jyagent.memory.compaction import compact_conversation
    sig = inspect.signature(compact_conversation)
    assert "system_prompt" in sig.parameters


def test_summarize_if_needed_accepts_system_prompt():
    """summarize_if_needed passes system_prompt through."""
    import inspect
    from jyagent.memory.compaction import summarize_if_needed
    sig = inspect.signature(summarize_if_needed)
    assert "system_prompt" in sig.parameters


def test_compaction_system_prompt_includes_base_and_memory(monkeypatch):
    """Auto-compaction should reuse the full base+memory prompt prefix."""
    import shutil
    import jyagent.config as config
    import jyagent.agent as agent
    from jyagent.memory.operations import ensure_dirs, write_memory_md

    tmpdir = tempfile.mkdtemp(prefix="jy_compact_prompt_test_")
    monkeypatch.setattr(config, "MEMORY_DIR", os.path.join(tmpdir, "memory"))
    monkeypatch.setattr(config, "TOPICS_DIR", os.path.join(tmpdir, "memory", "topics"))
    monkeypatch.setattr(config, "JOURNAL_DIR", os.path.join(tmpdir, "memory", "journal"))
    monkeypatch.setattr(config, "MEMORY_MD_FILE", os.path.join(tmpdir, "memory", "MEMORY.md"))

    try:
        ensure_dirs()
        write_memory_md("# Agent Memory\n\n[tip] compaction-memory-marker\n")
        agent._cached_memory_context = None

        prompt = agent._build_compaction_system_prompt("trigger compaction")

        assert "TOOL-FIRST PRINCIPLE" in prompt
        assert "SELF-USE MEMORY" in prompt
        assert "compaction-memory-marker" in prompt
    finally:
        agent._cached_memory_context = None
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─── File re-injection ──────────────────────────────────────────────────────

def test_file_access_tracker_basic():
    """FileAccessTracker records and returns files in order."""
    from jyagent.memory.compaction import FileAccessTracker

    tracker = FileAccessTracker()
    tracker.record("/a.py")
    tracker.record("/b.py")
    tracker.record("/c.py")

    assert tracker.recent(2) == ["/c.py", "/b.py"]
    assert tracker.recent(5) == ["/c.py", "/b.py", "/a.py"]
    assert len(tracker) == 3


def test_file_access_tracker_dedup_and_reorder():
    """Re-accessing a file moves it to the most recent position."""
    from jyagent.memory.compaction import FileAccessTracker

    tracker = FileAccessTracker()
    tracker.record("/a.py")
    tracker.record("/b.py")
    tracker.record("/a.py")  # re-access

    assert tracker.recent(5) == ["/a.py", "/b.py"]
    assert len(tracker) == 2


def test_file_reinjection_reads_real_files():
    """_build_file_reinjection_content reads actual file contents."""
    from jyagent.memory.compaction import (
        _build_file_reinjection_content, get_file_tracker,
    )

    tracker = get_file_tracker()
    tracker.clear()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def hello():\n    print('world')\n")
        tmp_path = f.name

    try:
        tracker.record(tmp_path)
        content = _build_file_reinjection_content()
        assert "def hello()" in content
        assert "Post-compaction context" in content
    finally:
        os.unlink(tmp_path)
        tracker.clear()


def test_file_reinjection_skips_missing_files():
    """Non-existent files are silently skipped during re-injection."""
    from jyagent.memory.compaction import (
        _build_file_reinjection_content, get_file_tracker,
    )

    tracker = get_file_tracker()
    tracker.clear()

    tracker.record("/nonexistent/file.py")
    content = _build_file_reinjection_content()
    assert content == ""  # no files readable → empty
    tracker.clear()


# ─── Compaction priority ────────────────────────────────────────────────────

def test_compaction_priority_in_registry():
    """Registry correctly stores and retrieves compaction_priority."""
    from jyagent.runtime.tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register("test_tool", lambda: None, {"name": "test_tool"},
                 compaction_priority="ephemeral")
    # Use ``freeze()`` rather than the deprecated live-read accessor.  The
    # per-step ToolBatch is the canonical metadata API; the registry-level
    # method is a footgun.
    batch = reg.freeze()
    assert batch.get_compaction_priority("test_tool") == "ephemeral"
    assert batch.get_compaction_priority("unknown_tool") == "standard"  # default


def test_builtin_tools_have_correct_priorities():
    """Built-in tools registered with expected compaction priorities."""
    import jyagent.tools  # noqa
    from jyagent.runtime.tools.registry import get_registry
    # Use freeze() rather than the deprecated registry-level live-read
    # accessor.
    batch = get_registry().freeze()

    assert batch.get_compaction_priority("run_shell") == "ephemeral"
    assert batch.get_compaction_priority("list_directory") == "ephemeral"
    assert batch.get_compaction_priority("read_file") == "persistent"
    assert batch.get_compaction_priority("web_fetch") == "persistent"
    assert batch.get_compaction_priority("write_file") == "standard"


# ─── Enhanced 9-section summary prompt ──────────────────────────────────────

def test_compact_prompt_has_nine_sections():
    """The compact prompt includes all 9 required sections."""
    from jyagent.memory.compaction import COMPACT_PROMPT

    sections = [l.strip() for l in COMPACT_PROMPT.splitlines() if l.strip().startswith("## ")]
    expected = [
        "## Task Context",
        "## Files Modified",
        "## Key Decisions",
        "## Errors & Failures",
        "## Technical Details",
        "## Environment State",
        "## Current State",
        "## Working Hypotheses",
        "## Pending Tasks",
    ]
    assert sections == expected, f"Sections mismatch:\n  got:      {sections}\n  expected: {expected}"


def test_format_messages_excludes_thinking():
    """_format_messages_for_compact strips thinking blocks."""
    from jyagent.memory.compaction import _format_messages_for_compact

    messages = [
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "SECRET THINKING"},
            {"type": "text", "text": "Visible response"},
        ]},
    ]
    formatted = _format_messages_for_compact(messages)
    assert "SECRET THINKING" not in formatted
    assert "Visible response" in formatted


# ─── Integration: file tracking from tools ────────────────────────────────────

def test_track_file_from_tool_functions():
    """_track_file in core.py records to the global tracker."""
    from jyagent.tools.core import _track_file
    from jyagent.memory.compaction import get_file_tracker

    tracker = get_file_tracker()
    tracker.clear()

    _track_file("/test/integration.py")
    assert "/test/integration.py" in tracker.recent(5)
    tracker.clear()
