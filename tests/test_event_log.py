# Tests for the append-only event log.
#
# The critical invariants under test:
#   - Pre-compaction messages survive on disk even after the live view
#     (ConversationMemory.messages) has been compacted.
#   - Replaying the log alone reconstructs the live view exactly.
#   - With the snapshot layer removed (2026-05), the log + latest.txt
#     pointer are the entire on-disk state.

import json
import os
import tempfile

import pytest

from jyagent import config
from jyagent.memory import (
    ConversationMemory,
    EventLog,
    checkpoint_session,
    event_log_path,
    load_session,
    replay_from_events,
)


@pytest.fixture
def tmp_sessions_dir(monkeypatch):
    td = tempfile.mkdtemp(prefix="jyagent_evlog_test_")
    monkeypatch.setattr(config, "SESSIONS_DIR", td)
    yield td


# ─── EventLog unit tests ──────────────────────────────────────────────────────

def test_event_log_emits_sequentially(tmp_sessions_dir):
    log = EventLog("sid-1", event_log_path("sid-1", tmp_sessions_dir))
    assert len(log) == 0
    seq0 = log.emit({"kind": "message", "message": {"role": "user", "content": "a"}})
    seq1 = log.emit({"kind": "message", "message": {"role": "assistant", "content": "b"}})
    assert seq0 == 0 and seq1 == 1
    assert len(log) == 2

    events = log.get_events()
    assert len(events) == 2
    assert events[0]["seq"] == 0 and events[0]["kind"] == "message"
    assert events[1]["seq"] == 1
    assert "ts" in events[0]


def test_event_log_emit_many_is_atomic_batch(tmp_sessions_dir):
    log = EventLog("sid-batch", event_log_path("sid-batch", tmp_sessions_dir))
    last = log.emit_many([
        {"kind": "message", "message": {"role": "user", "content": "1"}},
        {"kind": "message", "message": {"role": "assistant", "content": "2"}},
        {"kind": "message", "message": {"role": "user", "content": "3"}},
    ])
    assert last == 2
    events = log.get_events()
    assert [e["seq"] for e in events] == [0, 1, 2]


def test_event_log_resumes_seq_on_reopen(tmp_sessions_dir):
    path = event_log_path("sid-rs", tmp_sessions_dir)
    log1 = EventLog("sid-rs", path)
    log1.emit({"kind": "message", "message": {"role": "user", "content": "x"}})
    log1.emit({"kind": "message", "message": {"role": "assistant", "content": "y"}})
    log1.close()

    log2 = EventLog("sid-rs", path)
    assert len(log2) == 2
    seq = log2.emit({"kind": "message", "message": {"role": "user", "content": "z"}})
    assert seq == 2


def test_event_log_get_events_range(tmp_sessions_dir):
    log = EventLog("sid-r", event_log_path("sid-r", tmp_sessions_dir))
    for i in range(5):
        log.emit({"kind": "message", "message": {"role": "user", "content": str(i)}})
    sliced = log.get_events(start=1, end=4)
    assert [e["seq"] for e in sliced] == [1, 2, 3]
    assert log.get_events(start=3) == [e for e in log.get_events() if e["seq"] >= 3]


# ─── Integration: ConversationMemory + checkpoint ─────────────────────────────

def test_checkpoint_emits_session_start_then_messages(tmp_sessions_dir):
    """A fresh checkpoint creates the log with seq 0 = session_start, then messages."""
    c = ConversationMemory()
    c.add_message("user", "hi")
    checkpoint_session(c, metadata={"provider": "anthropic", "model": "x"})

    log_file = event_log_path(c.session_id, tmp_sessions_dir)
    with open(log_file) as f:
        events = [json.loads(l) for l in f]
    # Seq 0: session_start with metadata
    assert events[0]["seq"] == 0
    assert events[0]["kind"] == "session_start"
    assert events[0]["metadata"]["provider"] == "anthropic"
    # Seq 1: the user message
    assert events[1]["seq"] == 1
    assert events[1]["kind"] == "message"
    assert events[1]["message"] == {"role": "user", "content": "hi"}


def test_checkpoint_is_incremental(tmp_sessions_dir):
    """Each checkpoint flushes only the new tail, never re-emits prior messages."""
    c = ConversationMemory()
    c.add_message("user", "q1")
    checkpoint_session(c)
    c.add_message("assistant", "a1")
    checkpoint_session(c)
    c.add_message("user", "q2")
    checkpoint_session(c)

    log_file = event_log_path(c.session_id, tmp_sessions_dir)
    with open(log_file) as f:
        events = [json.loads(l) for l in f]
    # session_start + 3 messages, all sequential.
    assert [e["seq"] for e in events] == [0, 1, 2, 3]
    kinds = [e["kind"] for e in events]
    assert kinds == ["session_start", "message", "message", "message"]
    msgs = [e["message"]["content"] for e in events if e["kind"] == "message"]
    assert msgs == ["q1", "a1", "q2"]


def test_session_meta_emitted_on_metadata_change(tmp_sessions_dir):
    """Changing metadata mid-session emits a session_meta event (for /list)."""
    c = ConversationMemory()
    c.add_message("user", "hi")
    checkpoint_session(c, metadata={"provider": "anthropic", "model": "a"})
    c.add_message("assistant", "ok")
    checkpoint_session(c, metadata={"provider": "anthropic", "model": "a"})  # no change
    c.add_message("user", "switch")
    checkpoint_session(c, metadata={"provider": "openai", "model": "b"})  # change!

    log_file = event_log_path(c.session_id, tmp_sessions_dir)
    with open(log_file) as f:
        events = [json.loads(l) for l in f]
    kinds = [e["kind"] for e in events]
    # session_start + msg + msg + session_meta + msg
    assert kinds.count("session_start") == 1
    assert kinds.count("session_meta") == 1
    meta_evt = next(e for e in events if e["kind"] == "session_meta")
    assert meta_evt["metadata"]["provider"] == "openai"


def test_load_session_attaches_existing_log(tmp_sessions_dir):
    c = ConversationMemory()
    c.add_message("user", "hi")
    c.add_message("assistant", "hello")
    checkpoint_session(c)
    sid = c.session_id

    c2 = ConversationMemory()
    result = load_session(c2)
    assert result["loaded"]
    assert result["session_id"] == sid
    assert c2.session_id == sid
    assert c2.messages == [{"role": "user", "content": "hi"},
                           {"role": "assistant", "content": "hello"}]
    assert c2._event_log is not None
    # Cursor at end — future appends are incremental.
    assert c2._recorded_seq == len(c2.messages)

    # Append → goes to the SAME log file, no re-emission.
    before_len = len(c2._event_log)
    c2.add_message("user", "follow-up")
    checkpoint_session(c2)
    assert len(c2._event_log) == before_len + 1
    events = c2._event_log.get_events()
    assert events[-1]["message"]["content"] == "follow-up"


# ─── The core invariant: compaction preserves history in log ─────────────────

def test_compaction_event_preserves_predrop_messages_in_log(tmp_sessions_dir):
    c = ConversationMemory()
    sid = c.session_id
    for i in range(3):
        c.add_message("user", f"q{i}")
        checkpoint_session(c)
        c.add_message("assistant", f"a{i}")
        checkpoint_session(c)

    assert len(c.messages) == 6
    # Log: 1 session_start + 6 messages = 7 events.
    assert len(c._event_log) == 7

    # Simulate compaction: drop first 4, replace with synthetic summary.
    split = 4
    new_messages = [
        {"role": "user", "content": "[Compacted summary of 4 msgs]"},
        {"role": "assistant", "content": "Understood, continuing."},
    ]
    recent = c.messages[split:]
    c._event_log.emit({
        "kind": "compaction",
        "drop_count": split,
        "replacement_messages": list(new_messages),
        "summary": "summary text",
        "before_tokens": 100,
        "after_tokens": 30,
    })
    c.messages = new_messages + recent
    c.mark_recorded()

    # Post-compaction turn
    c.add_message("user", "after compact")
    checkpoint_session(c)
    c.add_message("assistant", "ok")
    checkpoint_session(c)

    # Live view: 2 synth + 2 kept + 2 new = 6.
    assert len(c.messages) == 6
    assert c.messages[0]["content"].startswith("[Compacted")

    # Log retains EVERY pre-compaction message.
    events = c._event_log.get_events()
    kinds = [e["kind"] for e in events]
    assert kinds == ["session_start"] + ["message"] * 6 + ["compaction"] + ["message"] * 2

    msg_contents = [e["message"]["content"] for e in events if e["kind"] == "message"]
    assert msg_contents[:6] == ["q0", "a0", "q1", "a1", "q2", "a2"]


def test_replay_from_events_reconstructs_compacted_view(tmp_sessions_dir):
    """End-to-end: compaction via event emit, then replay rebuilds the same view."""
    c = ConversationMemory()
    sid = c.session_id
    for i in range(3):
        c.add_message("user", f"q{i}")
        checkpoint_session(c)
        c.add_message("assistant", f"a{i}")
        checkpoint_session(c)

    split = 4
    new_messages = [
        {"role": "user", "content": "[SUMMARY]"},
        {"role": "assistant", "content": "ok"},
    ]
    c._event_log.emit({
        "kind": "compaction",
        "drop_count": split,
        "replacement_messages": list(new_messages),
        "summary": "s",
        "before_tokens": 1, "after_tokens": 1,
    })
    c.messages = new_messages + c.messages[split:]
    c.mark_recorded()

    c.add_message("user", "post")
    checkpoint_session(c)

    live_view = list(c.messages)

    # Replay from events alone.
    c.detach_event_log()
    replayed = replay_from_events(sid)
    assert replayed.messages == live_view
    assert replayed.session_id == sid

    before_len = len(replayed._event_log)
    replayed.add_message("assistant", "pong")
    checkpoint_session(replayed)
    after_len = len(replayed._event_log)
    assert after_len == before_len + 1  # only the new message appended


def test_clear_detaches_event_log(tmp_sessions_dir):
    c = ConversationMemory()
    c.add_message("user", "hi")
    checkpoint_session(c)
    assert c._event_log is not None

    c.clear()
    assert c._event_log is None
    assert c._recorded_seq == 0
    new_sid = c.session_id
    c.add_message("user", "fresh")
    checkpoint_session(c)
    assert os.path.isfile(event_log_path(new_sid, tmp_sessions_dir))


# ─── Metadata dedup ──────────────────────────────────────────────────────────

def test_metadata_dedup_is_O1_per_checkpoint(tmp_sessions_dir):
    """Repeated checkpoints with unchanged metadata must NOT re-emit
    session_meta events, and the dedup must not require re-reading the log
    on every call (perf regression guard).
    """
    c = ConversationMemory()
    md = {"provider": "anthropic", "model": "claude-x"}
    c.add_message("user", "hi")
    checkpoint_session(c, metadata=md)

    log = c._event_log
    base_len = len(log)

    # 50 more checkpoints with the same metadata — none should emit session_meta.
    for i in range(50):
        c.add_message("assistant" if i % 2 == 0 else "user", f"m{i}")
        checkpoint_session(c, metadata=md)

    # Final log: base events + 50 message events. Zero new session_meta events.
    assert len(log) == base_len + 50
    kinds = [e["kind"] for e in log.get_events()]
    assert kinds.count("session_meta") == 0
    assert kinds.count("session_start") == 1


def test_metadata_change_emits_session_meta(tmp_sessions_dir):
    """When metadata actually changes (e.g. /model switch), emit session_meta."""
    c = ConversationMemory()
    c.add_message("user", "hi")
    checkpoint_session(c, metadata={"provider": "a", "model": "x"})
    c.add_message("user", "again")
    checkpoint_session(c, metadata={"provider": "a", "model": "y"})  # model changed

    events = c._event_log.get_events()
    kinds = [e["kind"] for e in events]
    assert kinds.count("session_meta") == 1
    meta_evt = next(e for e in events if e["kind"] == "session_meta")
    assert meta_evt["metadata"]["model"] == "y"

