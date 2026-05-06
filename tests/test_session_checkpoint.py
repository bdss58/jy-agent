# Tests for memory.session.checkpoint_session — the per-turn durability hook.
#
# Lock-in invariants:
#   1. checkpoint_session writes latest.json only — NO timestamped archive.
#   2. checkpoint_session is safe to call many times in a row without churn.
#   3. load_session roundtrips a checkpointed session (session_id + messages).
#   4. The agent helper _checkpoint_turn is a no-op on empty/None input.

import os
import tempfile

import pytest

from jyagent import config
from jyagent.memory import (
    ConversationMemory,
    checkpoint_session,
    load_session,
    save_session,
)


@pytest.fixture
def tmp_sessions_dir(monkeypatch):
    td = tempfile.mkdtemp(prefix="jyagent_session_test_")
    monkeypatch.setattr(config, "SESSIONS_DIR", td)
    monkeypatch.setattr(config, "LATEST_SESSION_FILE", os.path.join(td, "latest.json"))
    yield td


def _snapshot_files(sessions_dir: str) -> list[str]:
    """Return snapshot-layer filenames (excluding the events/ subdir)."""
    return sorted(
        f for f in os.listdir(sessions_dir)
        if f != "events"
    )


def test_checkpoint_writes_only_latest(tmp_sessions_dir):
    c = ConversationMemory()
    c.add_message("user", "hello")
    path = checkpoint_session(c, metadata={"provider": "x", "model": "y"})
    assert path == config.LATEST_SESSION_FILE
    # Snapshot layer: only latest.json, no timestamped archive.
    assert _snapshot_files(tmp_sessions_dir) == ["latest.json"]


def test_checkpoint_is_idempotent_no_churn(tmp_sessions_dir):
    c = ConversationMemory()
    c.add_message("user", "a")
    for _ in range(10):
        c.add_message("assistant", "b")
        checkpoint_session(c)
    # Still exactly one snapshot file — no archive creep from per-turn saves.
    assert _snapshot_files(tmp_sessions_dir) == ["latest.json"]


def test_checkpoint_empty_conversation_is_noop(tmp_sessions_dir):
    c = ConversationMemory()
    assert checkpoint_session(c) == ""
    assert os.listdir(tmp_sessions_dir) == []


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


def test_save_session_still_archives(tmp_sessions_dir):
    # Regression: checkpoint_session must NOT have broken save_session's archive.
    c = ConversationMemory()
    c.add_message("user", "q")
    save_session(c)
    files = _snapshot_files(tmp_sessions_dir)
    assert "latest.json" in files
    # Exactly one timestamped archive alongside latest.
    ts_archives = [f for f in files if f != "latest.json"]
    assert len(ts_archives) == 1


def test_agent_helper_is_safe_on_empty_or_none():
    from jyagent.agent import _checkpoint_turn

    # Should not raise on None.
    _checkpoint_turn(None)

    class Fake:
        messages = []

    _checkpoint_turn(Fake())  # empty messages — no-op
