# Tests for memory.session.checkpoint_session — the per-turn durability hook.
#
# Lock-in invariants (post-2026-05 simplification):
#   1. checkpoint_session writes ONLY the per-session event log + a tiny
#      latest.txt pointer — NO snapshots, NO timestamped archives.
#   2. checkpoint_session is idempotent: many calls in a row don't churn
#      the session-listing space.
#   3. load_session roundtrips a checkpointed session (session_id + messages).
#   4. The agent helper _checkpoint_turn is a no-op on empty/None input.
#   5. end_session clears the latest pointer but leaves the log resumable
#      by id.

import os
import tempfile

import pytest

from jyagent import config
from jyagent.memory import (
    ConversationMemory,
    checkpoint_session,
    load_session,
    end_session,
    has_saved_session,
    list_sessions,
)
from jyagent.memory.event_log import event_log_path


@pytest.fixture
def tmp_sessions_dir(monkeypatch):
    td = tempfile.mkdtemp(prefix="jyagent_session_test_")
    monkeypatch.setattr(config, "SESSIONS_DIR", td)
    # Bypass the once-per-process migration latch so each test starts clean.
    import jyagent.memory.session as session_mod
    monkeypatch.setattr(session_mod, "_MIGRATION_DONE", False)
    yield td


def _top_level_files(sessions_dir: str) -> list[str]:
    """Files at the top level of SESSIONS_DIR (excluding the events/ subdir)."""
    return sorted(
        f for f in os.listdir(sessions_dir)
        if f != "events"
    )


def _event_files(sessions_dir: str) -> list[str]:
    events_dir = os.path.join(sessions_dir, "events")
    if not os.path.isdir(events_dir):
        return []
    return sorted(os.listdir(events_dir))


def test_checkpoint_writes_only_log_and_pointer(tmp_sessions_dir):
    c = ConversationMemory()
    c.add_message("user", "hello")
    sid = checkpoint_session(c, metadata={"provider": "x", "model": "y"})
    assert sid == c.session_id

    # Top level: only latest.txt — no latest.json, no timestamped archives.
    assert _top_level_files(tmp_sessions_dir) == ["latest.txt"]
    # The event log file exists under events/.
    assert _event_files(tmp_sessions_dir) == [f"{c.session_id}.jsonl"]
    # Pointer points at this session.
    with open(os.path.join(tmp_sessions_dir, "latest.txt")) as f:
        assert f.read().strip() == c.session_id


def test_checkpoint_is_idempotent_no_churn(tmp_sessions_dir):
    c = ConversationMemory()
    c.add_message("user", "a")
    for _ in range(10):
        c.add_message("assistant", "b")
        checkpoint_session(c)
    # Still exactly one event-log file — no archive creep.
    assert _event_files(tmp_sessions_dir) == [f"{c.session_id}.jsonl"]
    assert _top_level_files(tmp_sessions_dir) == ["latest.txt"]


def test_checkpoint_empty_conversation_is_noop(tmp_sessions_dir):
    c = ConversationMemory()
    assert checkpoint_session(c) == ""
    # No log, no pointer.
    assert _top_level_files(tmp_sessions_dir) == []
    assert _event_files(tmp_sessions_dir) == []


def test_checkpoint_roundtrip(tmp_sessions_dir):
    c = ConversationMemory()
    c.add_message("user", "q")
    c.add_message("assistant", "a")
    checkpoint_session(c)

    c2 = ConversationMemory()
    assert c2.session_id != c.session_id  # fresh
    result = load_session(c2)
    assert result["loaded"] is True
    assert result["message_count"] == 2
    assert c2.session_id == c.session_id
    assert c2.messages == c.messages


def test_has_saved_session_reflects_pointer(tmp_sessions_dir):
    assert has_saved_session() is False
    c = ConversationMemory()
    c.add_message("user", "hi")
    checkpoint_session(c)
    assert has_saved_session() is True


def test_end_session_clears_pointer_but_keeps_log(tmp_sessions_dir):
    c = ConversationMemory()
    c.add_message("user", "q")
    checkpoint_session(c)
    sid = c.session_id

    end_session(c, reason="new")

    # Pointer is gone — bare /continue should NOT auto-resume.
    assert has_saved_session() is False
    # But the log is preserved — discoverable / resumable by id.
    assert os.path.isfile(event_log_path(sid, tmp_sessions_dir))
    entries = list_sessions()
    assert any(e["session_id"] == sid and e["ended"] for e in entries)


def test_agent_helper_is_safe_on_empty_or_none(tmp_sessions_dir):
    from jyagent.agent import _checkpoint_turn

    # Should not raise on None.
    _checkpoint_turn(None)

    class Fake:
        messages = []

    _checkpoint_turn(Fake())  # empty messages — no-op
