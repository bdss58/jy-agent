# Tests for the four 2026-04-25 memory upgrades:
#   1. BM25 search across topics + journal (jyagent.memory.search)
#   2. Reconciliation in extraction (ADD / UPDATE / NOOP directives)
#   3. supersede() — non-destructive update with strikethrough
#   4. Reflection pass at compaction (writes [reflection] candidates to journal)
#   5. Section-level topic reads (read_topic_section, list_topic_sections)
#
# Each test patches config to point at a tmpdir before importing memory
# modules — same isolation pattern as test_memory_tiers.py.

import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_tmpdir = tempfile.mkdtemp(prefix="jy_memory_upgrades_test_")
os.environ["AGENT_PROVIDER"] = "anthropic"

import jyagent.config as config
config.MEMORY_DIR = os.path.join(_tmpdir, "memory")
config.TOPICS_DIR = os.path.join(_tmpdir, "memory", "topics")
config.JOURNAL_DIR = os.path.join(_tmpdir, "memory", "journal")
config.MEMORY_MD_FILE = os.path.join(_tmpdir, "memory", "MEMORY.md")

from jyagent.memory.operations import (
    write_memory_md, read_memory_md,
    write_topic, read_topic,
    read_topic_section, list_topic_sections,
    supersede, remember,
    append_journal, list_journals,
    ensure_dirs,
)
from jyagent.memory.search import (
    search_memory, render_hits, _tokenize, _split_sections, SearchHit,
)
from jyagent.memory import extraction
from jyagent.memory import compaction
from jyagent.tools.facades import manage_memory


def setup():
    """Wipe and recreate tmp memory dirs."""
    if os.path.exists(_tmpdir):
        shutil.rmtree(_tmpdir, ignore_errors=True)
    ensure_dirs()


# ─── 1. SEARCH ────────────────────────────────────────────────────────────────

def test_tokenize_ascii_dotted_paths():
    toks = _tokenize("Run jyagent.tools.facades on Python 3.14")
    assert "jyagent.tools.facades" in toks
    assert "python" in toks
    assert "3.14" in toks
    # stop words dropped
    assert "on" not in toks


def test_tokenize_cjk_bigrams():
    toks = _tokenize("用户偏好")
    assert "用户" in toks
    assert "户偏" in toks
    assert "偏好" in toks


def test_split_sections_basic():
    body = "intro line\n\n## A\nfirst\n\n## B\nsecond\n\n### B.1\nnested"
    chunks = _split_sections(body)
    headers = [h for h, _ in chunks]
    assert "" in headers          # preamble
    assert "A" in headers
    assert "B" in headers
    assert "B.1" in headers
    # section A's body must NOT include section B
    a_body = next(b for h, b in chunks if h == "A")
    assert "first" in a_body
    assert "second" not in a_body


def test_search_returns_relevant_topic_hits():
    setup()
    write_topic("kafka_notes", "## Producers\nuse acks=all\n\n## Consumers\nrebalance protocol")
    write_topic("redis_notes", "## Persistence\naof vs rdb tradeoffs")

    hits = search_memory("kafka producer acks", top_k=3)
    assert hits, "expected at least one hit"
    assert hits[0].chunk.source == "topics/kafka_notes.md"
    assert "Producers" in hits[0].chunk.section


def test_search_includes_journal_when_no_topic_matches():
    setup()
    write_topic("misc", "## Random\nunrelated content")
    append_journal("Investigated the gnarly TLS handshake bug on wan2", "debug")

    hits = search_memory("TLS handshake wan2", top_k=3)
    assert hits, "expected journal hit"
    assert any(h.chunk.source.startswith("journal/") for h in hits)


def test_search_empty_query_returns_no_hits():
    setup()
    write_topic("foo", "## Bar\nbaz")
    assert search_memory("") == []
    assert search_memory("   ") == []


def test_render_hits_handles_empty():
    assert "No matching" in render_hits([])


# ─── 2. RECONCILIATION IN EXTRACTION ─────────────────────────────────────────

class _StubOwner:
    """Minimal stand-in for runtime_owner.complete_text — returns a canned reply."""
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[str] = []

    def complete_text(self, prompt: str, max_output_tokens: int = 0) -> str:
        self.calls.append(prompt)
        return self.reply


def _wait_for_extraction():
    """Extraction runs in a daemon thread; join all named threads."""
    import threading
    for t in threading.enumerate():
        if t.name == "memory-extraction" and t.is_alive():
            t.join(timeout=5)


def test_extract_directive_add_appends_new_line():
    setup()
    write_memory_md("# Agent Memory\n\n## Rules\n- existing line\n")
    owner = _StubOwner("ADD::[tip] Use uv sync from worktree before testing")

    # Force extraction by bypassing the cooldown
    extraction._messages_since_extraction = 999
    extraction.extract_and_remember(
        owner,
        "user msg long enough to pass the 30-char gate xx",
        "assistant reply also long enough to pass",
    )
    _wait_for_extraction()

    content = read_memory_md()
    assert "Use uv sync from worktree" in content
    assert "[tip]" in content


def test_extract_directive_update_supersedes_old_line():
    setup()
    write_memory_md(
        "# Agent Memory\n\n"
        "[user_stated] K8s test host: wan2.think-force.com port 31555\n"
    )
    owner = _StubOwner(
        "UPDATE::wan2.think-force.com::[user_stated] K8s test host moved to wan3.think-force.com port 31555"
    )

    extraction._messages_since_extraction = 999
    extraction.extract_and_remember(
        owner,
        "we moved the test host to wan3 today, please remember",
        "noted, updating memory",
    )
    _wait_for_extraction()

    content = read_memory_md()
    # Old line is struck-through, not deleted
    assert "~~" in content
    assert "wan2.think-force.com" in content
    # New line present
    assert "wan3.think-force.com" in content


def test_extract_directive_noop_writes_nothing():
    setup()
    write_memory_md("# Agent Memory\n\n[tip] existing\n")
    before = read_memory_md()
    owner = _StubOwner("NOOP::already covered by existing line")

    extraction._messages_since_extraction = 999
    extraction.extract_and_remember(
        owner,
        "tell me about the existing tip in memory please",
        "see existing line",
    )
    _wait_for_extraction()

    assert read_memory_md() == before


def test_extract_handles_NONE_response():
    setup()
    write_memory_md("# Agent Memory\n\n[tip] existing\n")
    before = read_memory_md()
    owner = _StubOwner("NONE")

    extraction._messages_since_extraction = 999
    extraction.extract_and_remember(
        owner,
        "this is a sufficiently long user message to pass the gate",
        "assistant reply",
    )
    _wait_for_extraction()

    assert read_memory_md() == before


# ─── 3. SUPERSEDE ────────────────────────────────────────────────────────────

def test_supersede_marks_old_appends_new():
    setup()
    write_memory_md(
        "# Agent Memory\n\n"
        "[user_stated] K8s test host: wan2.think-force.com\n"
        "[tip] unrelated line\n"
    )
    msg = supersede("wan2.think-force.com", "K8s test host: wan3.think-force.com", "user_stated")

    content = read_memory_md()
    assert "Superseded 1" in msg
    # Old line marked, not removed
    assert "~~[user_stated] K8s test host: wan2.think-force.com~~" in content
    assert "(superseded " in content
    # New line appended
    assert "wan3.think-force.com" in content
    # Unrelated line untouched
    assert "[tip] unrelated line" in content
    assert "~~[tip] unrelated line~~" not in content


def test_supersede_no_match_returns_message_no_write():
    setup()
    write_memory_md("# Agent Memory\n\n[tip] foo\n")
    before = read_memory_md()
    msg = supersede("nonexistent", "this should not land", "tip")
    assert "No entries matched" in msg
    assert read_memory_md() == before


def test_supersede_idempotent_on_already_struck():
    setup()
    write_memory_md(
        "# Agent Memory\n\n"
        "[user_stated] foo bar baz\n"
    )
    supersede("foo bar", "foo bar quux", "user_stated")
    after_first = read_memory_md()

    # Second call with same keyword should NOT double-wrap the existing
    # ~~struck~~ line.
    supersede("foo bar", "foo bar zzz", "user_stated")
    after_second = read_memory_md()

    # `~~~~` would indicate double-wrap
    assert "~~~~" not in after_second
    # But the new entry from the 2nd call IS appended (it doesn't start with ~~)
    assert "foo bar zzz" in after_second


def test_supersede_via_facade():
    setup()
    write_memory_md("# Agent Memory\n\n[tip] old style approach\n")
    res = manage_memory("supersede", text="old style approach|new shiny approach", category="tip")
    assert res.is_error is False
    assert "Superseded" in res.content
    content = read_memory_md()
    assert "~~[tip] old style approach~~" in content
    assert "new shiny approach" in content


# ─── 4. REFLECTION PASS AT COMPACTION ────────────────────────────────────────

def test_reflection_writes_journal_candidates():
    setup()
    summary = (
        "## Task Context\nDebugging cert chain on wan2.\n\n"
        "## Errors & Failures\nhttpx default verify=True fails on broken CA bundle.\n"
    )
    owner = _StubOwner(
        "[gotcha] httpx defaults to verify=True; fall back to verify=False on broken CA bundles\n"
        "[tip] always test cert chain with curl -v before suspecting code\n"
    )

    n = compaction._run_reflection_pass(owner, summary)
    assert n == 2
    # Find the journal entry
    months = list_journals()
    assert months, "journal month should exist after reflection"
    journal_text = ""
    for m in months:
        from jyagent.memory.operations import read_journal
        journal_text += read_journal(m)
    assert "[reflection]" in journal_text
    assert "httpx defaults to verify=True" in journal_text
    assert "do NOT promote blindly" in journal_text  # safety preamble


def test_reflection_NONE_writes_nothing():
    setup()
    owner = _StubOwner("NONE")
    n = compaction._run_reflection_pass(owner, "## Task\nTrivial Q&A nothing learned")
    assert n == 0
    assert list_journals() == []


def test_reflection_swallows_owner_exceptions():
    setup()
    class _Boom:
        def complete_text(self, *a, **kw):
            raise RuntimeError("LLM down")
    n = compaction._run_reflection_pass(_Boom(), "summary text")
    assert n == 0


def test_reflection_caps_at_three_candidates():
    setup()
    owner = _StubOwner(
        "[tip] one\n[tip] two\n[tip] three\n[tip] four\n[tip] five\n"
    )
    # All five lines have body length < 10 — they will be filtered. Use longer.
    owner.reply = "\n".join(f"[tip] candidate number {i} with sufficient length" for i in range(5))
    n = compaction._run_reflection_pass(owner, "summary")
    assert n == 3


# ─── 5. SECTION-LEVEL TOPIC READS ────────────────────────────────────────────

def test_list_topic_sections():
    setup()
    write_topic("guide", "preamble\n\n## Setup\nfoo\n\n## Usage\nbar\n\n### Edge cases\nbaz")
    sections = list_topic_sections("guide")
    assert sections == ["Setup", "Usage", "Edge cases"]


def test_read_topic_section_returns_one_section():
    setup()
    write_topic("guide", "## Setup\nfoo line\n\n## Usage\nbar line")
    s = read_topic_section("guide", "Usage")
    assert "## Usage" in s
    assert "bar line" in s
    assert "foo line" not in s


def test_read_topic_section_includes_nested_subsections():
    setup()
    write_topic("guide", "## Setup\nfoo\n\n### Detail\nnested\n\n## Other\nbar")
    s = read_topic_section("guide", "Setup")
    assert "### Detail" in s
    assert "nested" in s
    assert "## Other" not in s


def test_read_topic_section_case_insensitive_and_strips_hashes():
    setup()
    write_topic("guide", "## Setup Notes\nfoo")
    assert "foo" in read_topic_section("guide", "setup notes")
    assert "foo" in read_topic_section("guide", "## Setup Notes")
    assert "foo" in read_topic_section("guide", "SETUP NOTES  ")


def test_read_topic_section_returns_empty_for_missing():
    setup()
    write_topic("guide", "## A\nfoo")
    assert read_topic_section("guide", "Nonexistent") == ""
    assert read_topic_section("does_not_exist", "A") == ""


def test_facade_topic_read_with_section():
    setup()
    write_topic("guide", "## Setup\nfoo\n\n## Usage\nbar")
    res = manage_memory("topic", text="read:guide#Usage")
    assert res.is_error is False
    assert "## Usage" in res.content
    assert "bar" in res.content
    assert "foo" not in res.content


def test_facade_topic_sections_command():
    setup()
    write_topic("guide", "## A\n1\n\n## B\n2")
    res = manage_memory("topic", text="sections:guide")
    assert res.is_error is False
    assert "A" in res.content
    assert "B" in res.content


def test_facade_search_action():
    setup()
    write_topic("kafka_notes", "## Producers\nuse acks=all\n")
    res = manage_memory("search", text="kafka producer acks")
    assert res.is_error is False
    assert "kafka_notes" in res.content


def test_search_with_plural_query_matches_singular_body():
    """Stemming: 'producers' should retrieve a body that says 'producer'."""
    setup()
    write_topic("kafka", "## Notes\na producer pushes to a partition")
    hits = search_memory("producers partition", top_k=3)
    assert hits, "stemming should let 'producers' match 'producer'"
    assert hits[0].chunk.source == "topics/kafka.md"


# ─── 6. CRITICAL / HIGH FIX REGRESSIONS (post-review hardening) ──────────────

def test_append_memory_md_heals_missing_trailing_newline():
    """C1: a hand-edited MEMORY.md without a trailing newline must not glue
    two unrelated entries onto one line on the next append."""
    setup()
    from jyagent.memory.operations import append_memory_md
    # Write WITHOUT trailing newline directly, simulating a hand edit.
    with open(config.MEMORY_MD_FILE, "w") as f:
        f.write("# Agent Memory\n\nLAST LINE NO NEWLINE")
    append_memory_md("[tip] new entry")
    content = read_memory_md()
    # Two distinct lines, not glued
    assert "LAST LINE NO NEWLINE\n[tip] new entry" in content
    assert "LAST LINE NO NEWLINE[tip]" not in content


def test_supersede_rejects_short_keyword():
    """H2: keywords shorter than the minimum can hit dozens of unrelated
    lines. Refuse before any RMW."""
    setup()
    write_memory_md("# Agent Memory\n\n[tip] aaa\n[tip] bbb\n")
    res = supersede("aa", "this should not land", "tip")
    assert "Error" in res
    # Original content unchanged
    assert read_memory_md() == "# Agent Memory\n\n[tip] aaa\n[tip] bbb\n"


def test_supersede_skips_protected_headers_and_rules():
    """H2: the LLM must not be able to strike-through a markdown heading or
    a Behavioral Rules entry via UPDATE. (`Critical` matches the heading
    text 'Behavioral Rules (CRITICAL)'.)"""
    setup()
    write_memory_md(
        "# Agent Memory\n\n"
        "## Behavioral Rules (CRITICAL)\n"
        "- Never fabricate command results\n"
        "\n"
        "## Random Section\n"
        "- ordinary line about CRITICAL bugs\n"
    )
    # 'CRITICAL' appears in the heading, in the Behavioral Rules child line,
    # and in the ordinary line. Only the ordinary line should be marked.
    res = supersede("CRITICAL bugs", "ordinary line about important bugs", "tip")
    content = read_memory_md()
    # Heading not touched
    assert "## Behavioral Rules (CRITICAL)" in content
    assert "~~## Behavioral Rules" not in content
    # Behavioral rule child line not touched
    assert "- Never fabricate command results" in content
    assert "~~- Never fabricate" not in content
    # Ordinary line WAS struck
    assert "~~- ordinary line about CRITICAL bugs~~" in content
    assert "Superseded 1" in res


def test_supersede_only_protected_matches_returns_no_op():
    """H2: if the *only* matches were protected, return a no-op message and
    do NOT append the new line."""
    setup()
    write_memory_md(
        "# Agent Memory\n\n"
        "## Behavioral Rules (CRITICAL)\n"
        "- protected unique phrase\n"
    )
    before = read_memory_md()
    res = supersede("protected unique phrase", "replacement text here", "tip")
    assert "No entries matched" in res
    assert "protected" in res
    # Memory unchanged
    assert read_memory_md() == before


def test_apply_directive_rejects_no_match_update_as_skip():
    """C3: an UPDATE directive whose keyword matches nothing must NOT count
    as a successful update — otherwise hallucinated UPDATEs crowd out real
    ADDs against the per-turn cap."""
    from jyagent.memory.extraction import _apply_directive
    setup()
    write_memory_md("# Agent Memory\n\n[tip] only entry\n")
    out = _apply_directive(
        "UPDATE::keyword_that_does_not_exist::[tip] this should not count"
    )
    assert out is not None
    kind, msg = out
    assert kind == "skip"
    assert "No entries matched" in msg


def test_apply_directive_rejects_overlong_body():
    """H3: enforce the prompt's '<120 chars, one line' rule in the parser."""
    from jyagent.memory.extraction import _apply_directive
    setup()
    write_memory_md("# Agent Memory\n\n[tip] existing\n")
    too_long = "x" * 200
    out = _apply_directive(f"ADD::[tip] {too_long}")
    assert out is None
    # Header-shaped body rejected (LLM must not be able to inject a heading)
    out = _apply_directive("ADD::[tip] # malicious heading injected")
    assert out is None
    # Strikethrough-shaped body rejected
    out = _apply_directive("ADD::[tip] ~~struck content~~ inject")
    assert out is None
    # Body shorter than 10 chars rejected
    out = _apply_directive("ADD::[tip] short")
    assert out is None
    # Valid body still accepted
    out = _apply_directive("ADD::[tip] this is a valid durable rule")
    assert out is not None and out[0] == "add"


def test_topic_path_rejects_traversal():
    """H1: ../escape, absolute paths, and special chars must be refused."""
    from jyagent.memory.operations import _topic_path, write_topic, delete_topic
    assert _topic_path("../escape") is None
    assert _topic_path("/abs/path") is None
    assert _topic_path("foo/bar") is None
    assert _topic_path("..") is None
    assert _topic_path(".hidden") is None
    assert _topic_path("") is None
    # Valid ones pass
    assert _topic_path("kafka") is not None
    assert _topic_path("kafka_notes_v2") is not None
    assert _topic_path("agent-loop-changelog") is not None


def test_write_topic_refuses_traversal():
    """Bad names raise ValueError; good names work."""
    setup()
    import pytest
    with pytest.raises(ValueError):
        write_topic("../escape", "malicious content")
    with pytest.raises(ValueError):
        write_topic("subdir/file", "x")
    # Good name: works (read_topic returns frontmatter + body, look for body)
    write_topic("legit", "## Hi\nbody")
    assert "## Hi" in read_topic("legit")
    assert "body" in read_topic("legit")


def test_delete_topic_refuses_traversal():
    """Bad names return False without removing anything."""
    from jyagent.memory.operations import delete_topic
    setup()
    write_topic("real", "body")
    # Create a sentinel sibling: write_topic auto-creates MEMORY.md (one
    # level up from TOPICS_DIR). Capture its content and assert that
    # `delete_topic('../MEMORY')` is refused without touching it.
    sibling_dir = os.path.dirname(config.TOPICS_DIR)
    sentinel_path = os.path.join(sibling_dir, "MEMORY.md")
    with open(sentinel_path, "r") as f:
        before = f.read()
    assert delete_topic("../MEMORY") is False
    with open(sentinel_path, "r") as f:
        assert f.read() == before
    # Legit delete still works
    assert delete_topic("real") is True


def test_facade_topic_write_returns_error_for_bad_name():
    setup()
    res = manage_memory("topic", text="write:../escape|content")
    assert res.is_error is True
    assert "invalid topic name" in res.content.lower() or "error" in res.content.lower()


def test_supersede_concurrent_writes_dont_lose_data():
    """C2: concurrent supersede + remember on MEMORY.md must not lose either
    write. We can't reliably reproduce a race in a unit test, but we can at
    least verify the lock is acquired (reentrant), so nested supersede→remember
    doesn't deadlock and both writes land."""
    setup()
    write_memory_md("# Agent Memory\n\n[user_stated] before-supersede\n")
    res = supersede("before-supersede", "after-supersede text", "user_stated")
    assert "Superseded 1" in res
    content = read_memory_md()
    # Both halves present
    assert "~~[user_stated] before-supersede~~" in content
    assert "after-supersede text" in content


# ─── runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("tokenize_ascii_dotted_paths", test_tokenize_ascii_dotted_paths),
        ("tokenize_cjk_bigrams", test_tokenize_cjk_bigrams),
        ("split_sections_basic", test_split_sections_basic),
        ("search_returns_relevant_topic_hits", test_search_returns_relevant_topic_hits),
        ("search_includes_journal_when_no_topic_matches", test_search_includes_journal_when_no_topic_matches),
        ("search_empty_query_returns_no_hits", test_search_empty_query_returns_no_hits),
        ("render_hits_handles_empty", test_render_hits_handles_empty),
        ("extract_directive_add_appends_new_line", test_extract_directive_add_appends_new_line),
        ("extract_directive_update_supersedes_old_line", test_extract_directive_update_supersedes_old_line),
        ("extract_directive_noop_writes_nothing", test_extract_directive_noop_writes_nothing),
        ("extract_handles_NONE_response", test_extract_handles_NONE_response),
        ("supersede_marks_old_appends_new", test_supersede_marks_old_appends_new),
        ("supersede_no_match_returns_message_no_write", test_supersede_no_match_returns_message_no_write),
        ("supersede_idempotent_on_already_struck", test_supersede_idempotent_on_already_struck),
        ("supersede_via_facade", test_supersede_via_facade),
        ("reflection_writes_journal_candidates", test_reflection_writes_journal_candidates),
        ("reflection_NONE_writes_nothing", test_reflection_NONE_writes_nothing),
        ("reflection_swallows_owner_exceptions", test_reflection_swallows_owner_exceptions),
        ("reflection_caps_at_three_candidates", test_reflection_caps_at_three_candidates),
        ("list_topic_sections", test_list_topic_sections),
        ("read_topic_section_returns_one_section", test_read_topic_section_returns_one_section),
        ("read_topic_section_includes_nested_subsections", test_read_topic_section_includes_nested_subsections),
        ("read_topic_section_case_insensitive_and_strips_hashes", test_read_topic_section_case_insensitive_and_strips_hashes),
        ("read_topic_section_returns_empty_for_missing", test_read_topic_section_returns_empty_for_missing),
        ("facade_topic_read_with_section", test_facade_topic_read_with_section),
        ("facade_topic_sections_command", test_facade_topic_sections_command),
        ("facade_search_action", test_facade_search_action),
        ("search_with_plural_query_matches_singular_body", test_search_with_plural_query_matches_singular_body),
        ("append_memory_md_heals_missing_trailing_newline", test_append_memory_md_heals_missing_trailing_newline),
        ("supersede_rejects_short_keyword", test_supersede_rejects_short_keyword),
        ("supersede_skips_protected_headers_and_rules", test_supersede_skips_protected_headers_and_rules),
        ("supersede_only_protected_matches_returns_no_op", test_supersede_only_protected_matches_returns_no_op),
        ("apply_directive_rejects_no_match_update_as_skip", test_apply_directive_rejects_no_match_update_as_skip),
        ("apply_directive_rejects_overlong_body", test_apply_directive_rejects_overlong_body),
        ("topic_path_rejects_traversal", test_topic_path_rejects_traversal),
        ("write_topic_refuses_traversal", test_write_topic_refuses_traversal),
        ("delete_topic_refuses_traversal", test_delete_topic_refuses_traversal),
        ("facade_topic_write_returns_error_for_bad_name", test_facade_topic_write_returns_error_for_bad_name),
        ("supersede_concurrent_writes_dont_lose_data", test_supersede_concurrent_writes_dont_lose_data),
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    shutil.rmtree(_tmpdir, ignore_errors=True)
    sys.exit(0 if failed == 0 else 1)
