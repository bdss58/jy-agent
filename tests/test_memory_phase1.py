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
config.LATEST_SESSION_FILE = os.path.join(_tmpdir, "sessions", "latest.json")

from jyagent.memory.operations import (
    write_topic, read_topic, read_topic_body, read_topic_meta,
    list_topics, delete_topic, remember, show_memory,
    read_memory_md, write_memory_md,
    _parse_frontmatter, _build_frontmatter,
    _extract_topic_description, _add_topic_index_entry, _remove_topic_index_entry,
)
from jyagent.memory.session import (
    save_session, load_session, has_saved_session, delete_session,
    _prune_archives,
)
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

def test_save_and_load_session():
    setup()
    conv = ConversationMemory()
    conv.add_message("user", "Hello")
    conv.add_message("assistant", "Hi there!")
    session_id = conv.session_id

    path = save_session(conv)
    assert path, "save_session returned empty path"
    assert os.path.isfile(config.LATEST_SESSION_FILE), "latest.json not created"

    # Load into fresh conversation
    conv2 = ConversationMemory()
    result = load_session(conv2)
    assert result["loaded"] is True, f"Load failed: {result}"
    assert result["message_count"] == 2
    assert result["session_id"] == session_id
    assert conv2.session_id == session_id
    assert len(conv2.messages) == 2
    assert conv2.messages[0]["content"] == "Hello"
    assert conv2.messages[1]["content"] == "Hi there!"
    print("  ✅ save_session + load_session roundtrip works")


def test_has_saved_session():
    setup()
    assert not has_saved_session(), "Should not find session before save"
    conv = ConversationMemory()
    conv.add_message("user", "test")
    save_session(conv)
    assert has_saved_session(), "Should find session after save"
    print("  ✅ has_saved_session works correctly")


def test_delete_session():
    setup()
    conv = ConversationMemory()
    conv.add_message("user", "test")
    save_session(conv)
    assert has_saved_session()
    deleted = delete_session()
    assert deleted
    assert not has_saved_session()
    print("  ✅ delete_session removes latest.json")


def test_load_empty_conversation():
    setup()
    conv = ConversationMemory()
    result = save_session(conv)
    assert result == "", "Should return empty for empty conversation"
    print("  ✅ save_session skips empty conversation")


def test_load_nonexistent_session():
    setup()
    conv = ConversationMemory()
    result = load_session(conv, path="/nonexistent/path.json")
    assert result["loaded"] is False
    assert "error" in result
    print("  ✅ load_session handles missing file gracefully")


def test_session_archive_created():
    setup()
    conv = ConversationMemory()
    conv.add_message("user", "Hello")
    save_session(conv)

    archive_files = [f for f in os.listdir(config.SESSIONS_DIR) if f != "latest.json"]
    assert len(archive_files) >= 1, f"No archive created. Files: {os.listdir(config.SESSIONS_DIR)}"
    print("  ✅ save_session creates timestamped archive")


def test_session_prune():
    setup()
    # Create many archives
    for i in range(25):
        conv = ConversationMemory()
        conv.add_message("user", f"msg {i}")
        save_session(conv)

    all_files = [f for f in os.listdir(config.SESSIONS_DIR) if f.endswith('.json') and f != "latest.json"]
    assert len(all_files) <= 20, f"Prune failed, {len(all_files)} archives remain"
    print("  ✅ session archive pruning works (keeps <= 20)")


def test_session_json_structure():
    setup()
    conv = ConversationMemory()
    conv.add_message("user", "Hello")
    conv.add_message("assistant", "World")
    save_session(conv, metadata={"provider": "anthropic", "model": "claude-sonnet-4-6"})

    with open(config.LATEST_SESSION_FILE) as f:
        data = json.load(f)
    assert data["version"] == 1
    assert data["session_id"] == conv.session_id
    UUID(data["session_id"])
    assert data["message_count"] == 2
    assert data["metadata"]["provider"] == "anthropic"
    assert len(data["messages"]) == 2
    print("  ✅ session JSON structure is correct")


def test_session_archive_preserves_session_id():
    setup()
    conv = ConversationMemory()
    conv.add_message("user", "Hello")
    save_session(conv)

    with open(config.LATEST_SESSION_FILE) as f:
        latest = json.load(f)

    archive_files = [
        f for f in os.listdir(config.SESSIONS_DIR)
        if f.endswith(".json") and f != "latest.json"
    ]
    assert len(archive_files) == 1, f"Expected one archive, got: {archive_files}"
    with open(os.path.join(config.SESSIONS_DIR, archive_files[0])) as f:
        archived = json.load(f)

    assert latest["session_id"] == conv.session_id
    assert archived["session_id"] == conv.session_id
    print("  ✅ session archives preserve session_id")


def test_load_legacy_session_without_session_id():
    setup()
    legacy_path = os.path.join(config.SESSIONS_DIR, "legacy.json")
    with open(legacy_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": 1,
            "saved_at": "2026-04-29T12:00:00+08:00",
            "message_count": 1,
            "metadata": {},
            "messages": [{"role": "user", "content": "legacy"}],
        }, f)

    conv = ConversationMemory()
    result = load_session(conv, path=legacy_path)

    assert result["loaded"] is True, f"Load failed: {result}"
    assert result["session_id"] == conv.session_id
    UUID(result["session_id"])
    assert conv.messages[0]["content"] == "legacy"
    print("  ✅ legacy sessions without session_id still load")


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
        test_save_and_load_session,
        test_has_saved_session,
        test_delete_session,
        test_load_empty_conversation,
        test_load_nonexistent_session,
        test_session_archive_created,
        test_session_prune,
        test_session_json_structure,
        test_session_archive_preserves_session_id,
        test_load_legacy_session_without_session_id,
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
