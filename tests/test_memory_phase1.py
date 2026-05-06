# Tests for Phase 1 memory enhancements.
#
# Test targets:
#   1. Topic frontmatter (timestamps)
#   2. Session persistence (save/load)
#   3. MAX_MEMORY_PROMPT_CHARS raised
#   4. Proactive extraction (should_extract logic)

import json
import os
import sys
import tempfile
import shutil
from uuid import UUID

# Ensure we import from the worktree, not the main checkout
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Point config at temp directories before importing modules
_tmpdir = tempfile.mkdtemp(prefix="jy_memory_test_")
os.environ["AGENT_PROVIDER"] = "anthropic"

# Need to patch config before imports
import jyagent.config as config
config.MEMORY_DIR = os.path.join(_tmpdir, "memory")
config.TOPICS_DIR = os.path.join(_tmpdir, "memory", "topics")
config.MEMORY_MD_FILE = os.path.join(_tmpdir, "memory", "MEMORY.md")
config.SESSIONS_DIR = os.path.join(_tmpdir, "sessions")

from jyagent.memory.operations import (
    write_topic, read_topic, read_topic_body, read_topic_meta,
    list_topics, delete_topic, remember, show_memory,
    read_memory_md, write_memory_md,
    _parse_frontmatter, _build_frontmatter,
    _extract_topic_description, _add_topic_index_entry, _remove_topic_index_entry,
)
from jyagent.memory.session import (
    checkpoint_session, load_session, has_saved_session, delete_session,
    end_session, list_sessions,
)
from jyagent.memory.event_log import event_log_path
from jyagent.memory.conversation import ConversationMemory
from jyagent.memory.extraction import should_extract, _extract_text


def setup():
    """Clean temp dirs before each test group."""
    for d in [config.MEMORY_DIR, config.TOPICS_DIR, config.SESSIONS_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)


def teardown():
    """Remove temp dirs."""
    shutil.rmtree(_tmpdir, ignore_errors=True)


# ─── Test 1: Topic Frontmatter ───────────────────────────────────────────────

def test_write_topic_adds_frontmatter():
    setup()
    write_topic("test-topic", "# Hello\nSome content here.")
    raw = read_topic("test-topic")
    assert raw.startswith("---\n"), f"Expected frontmatter, got: {raw[:100]}"
    meta = read_topic_meta("test-topic")
    assert "created" in meta, f"Missing 'created' in meta: {meta}"
    assert "updated" in meta, f"Missing 'updated' in meta: {meta}"
    body = read_topic_body("test-topic")
    assert "# Hello" in body, f"Body missing content: {body[:100]}"
    assert "---" not in body, f"Body should not contain frontmatter: {body[:100]}"
    print("  ✅ write_topic adds frontmatter with created/updated")


def test_write_topic_preserves_created():
    setup()
    write_topic("test-preserve", "First version")
    meta1 = read_topic_meta("test-preserve")
    created1 = meta1["created"]

    # Write again (update)
    write_topic("test-preserve", "Second version")
    meta2 = read_topic_meta("test-preserve")
    assert meta2["created"] == created1, f"created changed: {created1} -> {meta2['created']}"
    body = read_topic_body("test-preserve")
    assert "Second version" in body, f"Body not updated: {body}"
    print("  ✅ write_topic preserves original created timestamp")


def test_write_topic_with_caller_frontmatter():
    setup()
    content = "---\nauthor: jianyong\n---\n# My Topic\nContent"
    write_topic("caller-fm", content)
    meta = read_topic_meta("caller-fm")
    assert meta.get("author") == "jianyong", f"Caller meta lost: {meta}"
    assert "created" in meta, f"Missing timestamps: {meta}"
    body = read_topic_body("caller-fm")
    assert "# My Topic" in body
    print("  ✅ write_topic merges caller frontmatter with timestamps")


def test_parse_frontmatter_no_fm():
    meta, body = _parse_frontmatter("# Hello\nWorld")
    assert meta == {}
    assert body == "# Hello\nWorld"
    print("  ✅ _parse_frontmatter handles no-frontmatter case")


def test_parse_frontmatter_with_fm():
    raw = "---\ncreated: 2025-01-01\nupdated: 2025-06-01\n---\n# Content\nHere"
    meta, body = _parse_frontmatter(raw)
    assert meta["created"] == "2025-01-01"
    assert meta["updated"] == "2025-06-01"
    assert body.startswith("# Content")
    print("  ✅ _parse_frontmatter parses YAML frontmatter correctly")


def test_show_memory_includes_timestamps():
    setup()
    write_topic("ts-test", "Content for timestamp test")
    result = show_memory()
    assert "updated" in result, f"show_memory missing timestamp info: {result[:200]}"
    print("  ✅ show_memory displays topic timestamps")


# ─── Test 2: Session Persistence ─────────────────────────────────────────────

def _reset_migration_latch():
    """Each test resets SESSIONS_DIR; also reset the once-per-process migration flag."""
    import jyagent.memory.session as session_mod
    session_mod._MIGRATION_DONE = False


def test_checkpoint_and_load_session():
    setup()
    _reset_migration_latch()
    conv = ConversationMemory()
    conv.add_message("user", "Hello")
    conv.add_message("assistant", "Hi there!")
    session_id = conv.session_id

    sid = checkpoint_session(conv)
    assert sid == session_id, "checkpoint_session returned wrong id"
    # Event log file exists.
    assert os.path.isfile(event_log_path(session_id, config.SESSIONS_DIR))
    # Pointer exists.
    assert os.path.isfile(os.path.join(config.SESSIONS_DIR, "latest.txt"))

    conv2 = ConversationMemory()
    result = load_session(conv2)
    assert result["loaded"] is True, f"Load failed: {result}"
    assert result["message_count"] == 2
    assert result["session_id"] == session_id
    assert conv2.session_id == session_id
    assert len(conv2.messages) == 2
    assert conv2.messages[0]["content"] == "Hello"
    assert conv2.messages[1]["content"] == "Hi there!"
    print("  ✅ checkpoint_session + load_session roundtrip works")


def test_has_saved_session():
    setup()
    _reset_migration_latch()
    assert not has_saved_session(), "Should not find session before checkpoint"
    conv = ConversationMemory()
    conv.add_message("user", "test")
    checkpoint_session(conv)
    assert has_saved_session(), "Should find session after checkpoint"
    print("  ✅ has_saved_session works correctly")


def test_delete_session_clears_pointer():
    setup()
    _reset_migration_latch()
    conv = ConversationMemory()
    conv.add_message("user", "test")
    checkpoint_session(conv)
    assert has_saved_session()
    # delete_session() with no args clears only the pointer (log survives).
    deleted = delete_session()
    assert deleted
    assert not has_saved_session()
    # The log is still on disk (for manual recovery via /continue <id>).
    assert os.path.isfile(event_log_path(conv.session_id, config.SESSIONS_DIR))
    print("  ✅ delete_session() clears latest.txt but preserves the log")


def test_checkpoint_empty_conversation():
    setup()
    _reset_migration_latch()
    conv = ConversationMemory()
    result = checkpoint_session(conv)
    assert result == "", "Should return empty for empty conversation"
    print("  ✅ checkpoint_session skips empty conversation")


def test_load_nonexistent_session():
    setup()
    _reset_migration_latch()
    conv = ConversationMemory()
    result = load_session(conv, query="does-not-exist-9999")
    assert result["loaded"] is False
    assert "error" in result
    print("  ✅ load_session handles missing session gracefully")


def test_end_session_marks_ended_but_keeps_log():
    setup()
    _reset_migration_latch()
    conv = ConversationMemory()
    conv.add_message("user", "Hello")
    checkpoint_session(conv)
    sid = conv.session_id

    end_session(conv, reason="new")

    # Pointer cleared → bare /continue should not pick this session.
    assert not has_saved_session()
    # But log file remains, discoverable + resumable by id.
    assert os.path.isfile(event_log_path(sid, config.SESSIONS_DIR))
    entries = list_sessions()
    assert any(e["session_id"] == sid and e["ended"] for e in entries)
    print("  ✅ end_session clears pointer but preserves log")


def test_event_log_structure():
    setup()
    _reset_migration_latch()
    conv = ConversationMemory()
    conv.add_message("user", "Hello")
    conv.add_message("assistant", "World")
    checkpoint_session(conv, metadata={"provider": "anthropic", "model": "claude-sonnet-4-6"})

    # Log: session_start + 2 messages.
    log_file = event_log_path(conv.session_id, config.SESSIONS_DIR)
    with open(log_file) as f:
        events = [json.loads(l) for l in f]
    assert len(events) == 3
    assert events[0]["kind"] == "session_start"
    assert events[0]["metadata"]["provider"] == "anthropic"
    assert events[0]["metadata"]["model"] == "claude-sonnet-4-6"
    assert events[1]["kind"] == "message"
    assert events[2]["kind"] == "message"
    # Session id is a valid UUID.
    UUID(conv.session_id)
    print("  ✅ event log structure: session_start + messages")


def test_legacy_snapshot_migrates_on_first_call():
    setup()
    _reset_migration_latch()
    # Drop a legacy latest.json + a timestamped archive on disk.
    legacy_sid_a = "legacy-aaaa-bbbb-cccc-dddd-eeeeffff0011"
    legacy_sid_b = "legacy-aaaa-bbbb-cccc-dddd-eeeeffff0022"
    os.makedirs(config.SESSIONS_DIR, exist_ok=True)
    with open(os.path.join(config.SESSIONS_DIR, "latest.json"), "w") as f:
        json.dump({
            "version": 1, "session_id": legacy_sid_a,
            "saved_at": "2026-04-29T12:00:00+08:00", "message_count": 1,
            "metadata": {"provider": "anthropic", "model": "legacy-model"},
            "messages": [{"role": "user", "content": "legacy-a"}],
        }, f)
    with open(os.path.join(config.SESSIONS_DIR, "20260428_120000.json"), "w") as f:
        json.dump({
            "version": 1, "session_id": legacy_sid_b,
            "saved_at": "2026-04-28T12:00:00+08:00", "message_count": 1,
            "metadata": {},
            "messages": [{"role": "user", "content": "legacy-b"}],
        }, f)

    # Any API call triggers migration.
    entries = list_sessions()
    sids = {e["session_id"] for e in entries}
    assert legacy_sid_a in sids
    assert legacy_sid_b in sids
    # Legacy files were renamed.
    assert not os.path.isfile(os.path.join(config.SESSIONS_DIR, "latest.json"))
    assert os.path.isfile(os.path.join(config.SESSIONS_DIR, "latest.json.legacy"))
    # Pointer was seeded from latest.json → points at legacy_sid_a.
    with open(os.path.join(config.SESSIONS_DIR, "latest.txt")) as f:
        assert f.read().strip() == legacy_sid_a
    # Replay works on the migrated log.
    conv = ConversationMemory()
    result = load_session(conv, query=legacy_sid_b)
    assert result["loaded"]
    assert conv.messages[0]["content"] == "legacy-b"
    print("  ✅ legacy snapshots migrate to event log + pointer on first call")


def test_conversation_clear_rotates_session_id():
    conv = ConversationMemory()
    original_session_id = conv.session_id
    conv.add_message("user", "Hello")

    conv.clear()

    assert conv.session_id != original_session_id
    UUID(conv.session_id)
    assert conv.messages == []
    print("  ✅ conversation clear rotates session_id")


# ─── Test 3: Config Change ──────────────────────────────────────────────────

def test_max_memory_prompt_chars():
    assert config.MAX_MEMORY_PROMPT_CHARS == 10000, f"Expected 10000, got {config.MAX_MEMORY_PROMPT_CHARS}"
    print("  ✅ MAX_MEMORY_PROMPT_CHARS raised to 10000")


# ─── Test 4: Proactive Extraction Logic ──────────────────────────────────────

def test_extract_text_string():
    assert _extract_text("hello") == "hello"
    print("  ✅ _extract_text handles plain string")


def test_extract_text_blocks():
    content = [
        {"type": "text", "text": "Part 1"},
        {"type": "tool_use", "name": "read_file"},
        {"type": "text", "text": "Part 2"},
    ]
    result = _extract_text(content)
    assert "Part 1" in result
    assert "Part 2" in result
    print("  ✅ _extract_text handles block content")


def test_should_extract_interval():
    import jyagent.memory.extraction as ext
    ext._messages_since_extraction = 0

    # First 3 calls should return False (interval = 4)
    for i in range(3):
        result = should_extract("x" * 50)
        assert result is False, f"should_extract returned True at message {i+1}"

    # 4th call should return True
    result = should_extract("x" * 50)
    assert result is True, "should_extract should return True at interval"
    print("  ✅ should_extract respects interval")


def test_should_extract_short_message():
    import jyagent.memory.extraction as ext
    ext._messages_since_extraction = 3  # at interval boundary

    result = should_extract("hi")  # too short
    assert result is False, "should_extract should skip short messages"
    print("  ✅ should_extract skips short messages")


# ─── Test 5: Topic Index Auto-Update ─────────────────────────────────────────

def test_extract_topic_description_heading():
    desc = _extract_topic_description("# My Cool Topic\nSome body text")
    assert desc == "My Cool Topic", f"Expected heading text, got: {desc}"
    print("  ✅ _extract_topic_description extracts heading")


def test_extract_topic_description_no_heading():
    desc = _extract_topic_description("Just some plain text here that is the body.")
    assert desc == "Just some plain text here that is the body.", f"Got: {desc}"
    print("  ✅ _extract_topic_description falls back to first line")


def test_extract_topic_description_long_line():
    # Cap was raised from 80 → 120 chars on 2026-05-05 to match the
    # _MAX_TOPIC_DESC_CHARS constant after sanitisation hardening.
    long_line = "A" * 200
    desc = _extract_topic_description(long_line)
    assert len(desc) <= 120 + 1  # 120 chars + "…"
    assert desc.endswith("…"), f"Should truncate with ellipsis: {desc}"
    print("  ✅ _extract_topic_description truncates long lines")


def test_extract_topic_description_empty():
    desc = _extract_topic_description("")
    assert desc == "(no description)"
    print("  ✅ _extract_topic_description handles empty body")


def test_write_topic_auto_indexes_new_topic():
    setup()
    # Seed MEMORY.md with basic content
    write_memory_md("# Agent Memory\n\n## User Profile\n- Name: Test\n")
    write_topic("test-auto-index", "# My Test Topic\nSome content.")
    content = read_memory_md()
    assert "## Topic Files Index" in content, f"Index section missing:\n{content}"
    assert "**test-auto-index.md**" in content, f"Entry missing:\n{content}"
    assert "My Test Topic" in content, f"Description missing:\n{content}"
    print("  ✅ write_topic auto-indexes new topic in MEMORY.md")


def test_write_topic_no_duplicate_index():
    setup()
    write_memory_md("# Agent Memory\n\n## User Profile\n- Name: Test\n")
    write_topic("no-dup", "# First Version\nBody")
    # Overwrite same topic
    write_topic("no-dup", "# Second Version\nNew body")
    content = read_memory_md()
    count = content.count("**no-dup.md**")
    assert count == 1, f"Expected 1 index entry, found {count}:\n{content}"
    print("  ✅ write_topic doesn't duplicate index on overwrite")


def test_delete_topic_removes_index_entry():
    setup()
    write_memory_md("# Agent Memory\n\n## User Profile\n- Name: Test\n")
    write_topic("del-me", "# Delete Test\nContent")
    content = read_memory_md()
    assert "**del-me.md**" in content, "Precondition: entry should exist"
    delete_topic("del-me")
    content = read_memory_md()
    assert "**del-me.md**" not in content, f"Entry not removed:\n{content}"
    print("  ✅ delete_topic removes index entry from MEMORY.md")


def test_delete_topic_removes_empty_section():
    setup()
    write_memory_md("# Agent Memory\n\n## User Profile\n- Name: Test\n")
    write_topic("only-one", "# Only Topic\nContent")
    delete_topic("only-one")
    content = read_memory_md()
    assert "## Topic Files Index" not in content, \
        f"Empty section should be removed:\n{content}"
    print("  ✅ delete_topic removes empty Topic Files Index section")


def test_add_index_creates_section_before_later_sections():
    setup()
    write_memory_md(
        "# Agent Memory\n\n"
        "## User Profile\n- Name: Test\n\n"
        "## Repo Snapshot\n- Some repo info\n"
    )
    write_topic("placement-test", "# Placement Test\nBody")
    content = read_memory_md()
    idx_pos = content.index("## Topic Files Index")
    repo_pos = content.index("## Repo Snapshot")
    assert idx_pos < repo_pos, \
        f"Topic index ({idx_pos}) should come before Repo Snapshot ({repo_pos})"
    print("  ✅ Topic Files Index section inserted before Repo Snapshot")


# ─── Run all tests ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # 1. Frontmatter
        test_write_topic_adds_frontmatter,
        test_write_topic_preserves_created,
        test_write_topic_with_caller_frontmatter,
        test_parse_frontmatter_no_fm,
        test_parse_frontmatter_with_fm,
        test_show_memory_includes_timestamps,
        # 2. Session persistence
        test_checkpoint_and_load_session,
        test_has_saved_session,
        test_delete_session_clears_pointer,
        test_checkpoint_empty_conversation,
        test_load_nonexistent_session,
        test_end_session_marks_ended_but_keeps_log,
        test_event_log_structure,
        test_legacy_snapshot_migrates_on_first_call,
        test_conversation_clear_rotates_session_id,
        # 3. Config
        test_max_memory_prompt_chars,
        # 4. Extraction
        test_extract_text_string,
        test_extract_text_blocks,
        test_should_extract_interval,
        test_should_extract_short_message,
        # 5. Topic index auto-update
        test_extract_topic_description_heading,
        test_extract_topic_description_no_heading,
        test_extract_topic_description_long_line,
        test_extract_topic_description_empty,
        test_write_topic_auto_indexes_new_topic,
        test_write_topic_no_duplicate_index,
        test_delete_topic_removes_index_entry,
        test_delete_topic_removes_empty_section,
        test_add_index_creates_section_before_later_sections,
    ]

    print(f"\n🧪 Running {len(tests)} Phase 1 memory tests...\n")
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__}: {e}")
            failed += 1

    teardown()
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed:
        sys.exit(1)
    print("🎉 All tests passed!")
