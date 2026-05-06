# Tests for the append-only event log (bug #2 fix).
#
# The critical invariant under test: pre-compaction messages survive on disk
# even after the live view (ConversationMemory.messages) has been compacted.
# Replaying the log alone must reconstruct the live view exactly.

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
    monkeypatch.setattr(config, "LATEST_SESSION_FILE", os.path.join(td, "latest.json"))
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

    # Re-open — next seq must continue at 2
    log2 = EventLog("sid-rs", path)
    assert len(log2) == 2
    seq = log2.emit({"kind": "message", "message": {"role": "user", "content": "z"}})
    assert seq == 2


def test_event_log_get_events_range(tmp_sessions_dir):
    log = EventLog("sid-r", event_log_path("sid-r", tmp_sessions_dir))
    for i in range(5):
        log.emit({"kind": "message", "message": {"role": "user", "content": str(i)}})
    # Half-open [start, end)
    sliced = log.get_events(start=1, end=4)
    assert [e["seq"] for e in sliced] == [1, 2, 3]
    assert log.get_events(start=3) == [e for e in log.get_events() if e["seq"] >= 3]


# ─── Integration: ConversationMemory + checkpoint ─────────────────────────────

def test_checkpoint_flushes_messages_to_log(tmp_sessions_dir):
    c = ConversationMemory()
    c.add_message("user", "hi")
    checkpoint_session(c)

    log_file = event_log_path(c.session_id, tmp_sessions_dir)
    assert os.path.isfile(log_file)
    with open(log_file) as f:
        lines = f.readlines()
    assert len(lines) == 1
    evt = json.loads(lines[0])
    assert evt["kind"] == "message"
    assert evt["message"] == {"role": "user", "content": "hi"}
    assert evt["seq"] == 0


def test_checkpoint_is_incremental(tmp_sessions_dir):
    # Each checkpoint should only flush the new tail, not re-emit prior messages.
    c = ConversationMemory()
    c.add_message("user", "q1")
    checkpoint_session(c)
    c.add_message("assistant", "a1")
    checkpoint_session(c)
    c.add_message("user", "q2")
    checkpoint_session(c)

    log_file = event_log_path(c.session_id, tmp_sessions_dir)
    with open(log_file) as f:
        lines = [json.loads(l) for l in f]
    assert [e["seq"] for e in lines] == [0, 1, 2]
    assert [e["message"]["content"] for e in lines] == ["q1", "a1", "q2"]


def test_snapshot_embeds_last_event_seq(tmp_sessions_dir):
    c = ConversationMemory()
    c.add_message("user", "hello")
    c.add_message("assistant", "world")
    checkpoint_session(c)

    with open(config.LATEST_SESSION_FILE) as f:
        snap = json.load(f)
    assert snap["last_event_seq"] == 1
    assert snap["version"] == 1


def test_load_session_attaches_existing_log(tmp_sessions_dir):
    # First session: write some messages and checkpoint.
    c = ConversationMemory()
    c.add_message("user", "hi")
    c.add_message("assistant", "hello")
    checkpoint_session(c)
    sid = c.session_id

    # Fresh ConversationMemory loads the snapshot and should auto-attach the log.
    c2 = ConversationMemory()
    result = load_session(c2)
    assert result["loaded"]
    assert result["event_log_status"] == "synced"
    assert result["event_log_seq"] == 1
    assert c2._event_log is not None
    # Cursor is at end — future appends will be incremental.
    assert c2._recorded_seq == len(c2.messages)

    # Append & re-checkpoint → goes to the SAME log file.
    c2.add_message("user", "follow-up")
    checkpoint_session(c2)
    events = c2._event_log.get_events()
    assert len(events) == 3
    assert events[2]["message"]["content"] == "follow-up"


def test_load_session_without_log_is_clean(tmp_sessions_dir):
    # Simulate a legacy snapshot with no event log on disk.
    import json as _json
    legacy_sid = "legacy-session-xyz"
    payload = {
        "version": 1,
        "session_id": legacy_sid,
        "saved_at": "2026-01-01T00:00:00+08:00",
        "message_count": 1,
        "metadata": {},
        "messages": [{"role": "user", "content": "legacy"}],
        # No last_event_seq — older snapshot.
    }
    with open(config.LATEST_SESSION_FILE, "w") as f:
        _json.dump(payload, f)

    c = ConversationMemory()
    result = load_session(c)
    assert result["loaded"]
    assert result["event_log_status"] == "absent"
    assert c._event_log is None

    # First checkpoint after load should create a fresh log.
    c.add_message("assistant", "resumed")
    checkpoint_session(c)
    assert c._event_log is not None
    assert os.path.isfile(event_log_path(legacy_sid, tmp_sessions_dir))


# ─── The core bug-#2 invariant: compaction preserves history in log ──────────

def test_compaction_event_preserves_predrop_messages_in_log(tmp_sessions_dir):
    # Build up a 6-message conversation, checkpointing each turn.
    c = ConversationMemory()
    sid = c.session_id
    for i in range(3):
        c.add_message("user", f"q{i}")
        checkpoint_session(c)
        c.add_message("assistant", f"a{i}")
        checkpoint_session(c)

    assert len(c.messages) == 6
    assert len(c._event_log) == 6

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

    # Live view is now compacted (6 msgs: 2 synth + 2 kept + 2 new)
    assert len(c.messages) == 6
    assert c.messages[0]["content"].startswith("[Compacted")

    # But the log retains EVERY pre-compaction message (q0-q1, a0-a1).
    # Log events: 6 messages + 1 compaction + 2 post-compaction = 9
    events = c._event_log.get_events()
    kinds = [e["kind"] for e in events]
    assert kinds == ["message"] * 6 + ["compaction"] + ["message"] * 2
    # Verify the dropped messages are still in the log:
    dropped_contents = [e["message"]["content"] for e in events[:4]]
    assert dropped_contents == ["q0", "a0", "q1", "a1"]


def test_replay_from_events_reconstructs_compacted_view(tmp_sessions_dir):
    # End-to-end: compaction via manual event emit, then replay reconstructs
    # the same view the live session has.
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

    # Now replay from events alone.
    c.detach_event_log()
    replayed = replay_from_events(sid)
    assert replayed.messages == live_view
    assert replayed.session_id == sid
    # Replayed conv attaches the log with cursor at end — no re-emission on
    # next checkpoint.
    before_len = len(replayed._event_log)
    replayed.add_message("assistant", "pong")
    checkpoint_session(replayed)
    after_len = len(replayed._event_log)
    assert after_len == before_len + 1  # only the new message got appended


def test_clear_detaches_event_log(tmp_sessions_dir):
    c = ConversationMemory()
    c.add_message("user", "hi")
    checkpoint_session(c)
    assert c._event_log is not None

    c.clear()
    assert c._event_log is None
    assert c._recorded_seq == 0
    # Session ID changed
    # Next checkpoint on new session creates a fresh log file.
    new_sid = c.session_id
    c.add_message("user", "fresh")
    checkpoint_session(c)
    assert os.path.isfile(event_log_path(new_sid, tmp_sessions_dir))


# ─── Log-ahead-of-snapshot recovery (P2-2 from Codex review) ─────────────────

def test_load_session_recovers_when_log_is_ahead_of_snapshot(tmp_sessions_dir):
    """Crash between log flush and snapshot write must be recoverable.

    Scenario: turn 2's events made it to the JSONL log, but the snapshot
    write crashed before persisting. On /continue, the snapshot has stale
    `last_event_seq` and message_count. We must replay from the log and
    NOT silently keep the stale snapshot view (which would orphan the
    ahead events forever).
    """
    # Turn 1: clean checkpoint (log + snapshot in sync at seq 0,1).
    c = ConversationMemory()
    sid = c.session_id
    c.add_message("user", "q1")
    checkpoint_session(c)
    c.add_message("assistant", "a1")
    checkpoint_session(c)

    # Read the synced snapshot (we'll restore it after simulating a crash).
    with open(config.LATEST_SESSION_FILE) as f:
        synced_snapshot = f.read()

    # Turn 2: append messages to the LOG only, simulating a crash before
    # the snapshot write completed.  We do this by manually emitting and
    # NOT calling checkpoint_session.
    c._event_log.emit_many([
        {"kind": "message", "message": {"role": "user", "content": "q2-orphan"}},
        {"kind": "message", "message": {"role": "assistant", "content": "a2-orphan"}},
    ])

    # Snapshot on disk is still the turn-1 snapshot (last_event_seq=1).
    # Simulate the process dying here — the log has 4 events, snapshot has 2.

    # Now /continue from a fresh process.
    c2 = ConversationMemory()
    result = load_session(c2)

    assert result["loaded"] is True
    assert result["event_log_status"] == "log_ahead_recovered"
    assert result["log_ahead_recovered"] is True
    # Recovered view has all 4 messages from the log, not the 2 from the snapshot.
    assert len(c2.messages) == 4
    assert c2.messages[2]["content"] == "q2-orphan"
    assert c2.messages[3]["content"] == "a2-orphan"
    # Cursor must be at len(view) so the next checkpoint doesn't re-emit.
    assert c2._recorded_seq == 4

    # Next checkpoint must NOT lose the recovered events.
    c2.add_message("user", "q3")
    checkpoint_session(c2)
    events = c2._event_log.get_events()
    assert len(events) == 5
    assert events[4]["message"]["content"] == "q3"


def test_load_session_synced_when_snapshot_matches_log(tmp_sessions_dir):
    # Sanity: when snapshot and log are in sync, status is "synced", not
    # "log_ahead_recovered".
    c = ConversationMemory()
    c.add_message("user", "hi")
    checkpoint_session(c)
    c2 = ConversationMemory()
    result = load_session(c2)
    assert result["event_log_status"] == "synced"
    assert result["log_ahead_recovered"] is False
