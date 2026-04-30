# Tests for the four 2026-04-25 memory upgrades:
#   1. BM25 search across topics + journal (jyagent.memory.search)
#   2. Reconciliation in extraction (ADD / UPDATE / NOOP directives)
#   3. UPDATE replacement (forget old + journal-archive + remember new) —
#      replaced the old supersede() Tier-1 strikethrough behavior
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
    remember,
    append_journal, list_journals, read_journal,
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


def test_extract_directive_update_replaces_old_line():
    """UPDATE directive: old MEMORY.md line is removed (Tier 1 stays lean),
    archived to current month's journal (Tier 3 audit trail), and the new
    line is appended via remember()."""
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
    # Tier 1: old line gone, new line present
    assert "wan2.think-force.com" not in content
    assert "wan3.think-force.com" in content
    # No strikethrough syntax should leak into MEMORY.md any more
    assert "~~" not in content

    # Tier 3: old line archived to journal
    months = list_journals()
    assert months, "journal month should exist after UPDATE replacement"
    journal_text = "".join(read_journal(m) for m in months)
    assert "[memory_revision]" in journal_text
    assert "wan2.think-force.com" in journal_text
    assert "Replaced via UPDATE directive" in journal_text


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


# ─── 3. UPDATE REPLACEMENT (forget + journal-archive + remember) ─────────────
# These tests target ``extraction._replace_line``, which is the implementation
# behind the LLM-driven UPDATE directive after we removed the public
# supersede() action. The behavioural surface mirrors the old supersede
# safety rails (keyword length, protected sections, no-match → skip) but
# tier placement changed: old lines move to journal instead of staying in
# MEMORY.md as ``~~strikethrough~~``.

def test_update_replaces_old_archives_to_journal():
    setup()
    write_memory_md(
        "# Agent Memory\n\n"
        "[user_stated] K8s test host: wan2.think-force.com\n"
        "[tip] unrelated line\n"
    )
    status, msg = extraction._replace_line(
        "wan2.think-force.com",
        "K8s test host: wan3.think-force.com",
        "user_stated",
    )
    assert status == "update"
    assert "Replaced 1" in msg

    content = read_memory_md()
    # Tier 1: old line gone, no strikethrough leakage, new line present
    assert "wan2.think-force.com" not in content
    assert "~~" not in content
    assert "wan3.think-force.com" in content
    # Unrelated line untouched
    assert "[tip] unrelated line" in content

    # Tier 3: old line archived under [memory_revision]
    months = list_journals()
    assert months
    journal_text = "".join(read_journal(m) for m in months)
    assert "[memory_revision]" in journal_text
    assert "wan2.think-force.com" in journal_text


def test_update_no_match_returns_skip_no_writes():
    setup()
    write_memory_md("# Agent Memory\n\n[tip] foo\n")
    before = read_memory_md()
    status, msg = extraction._replace_line(
        "nonexistent-keyword", "this should not land", "tip",
    )
    assert status == "skip"
    assert "No entries matched" in msg
    # MEMORY.md unchanged AND no journal write — without a match we have
    # nothing to archive.
    assert read_memory_md() == before
    assert list_journals() == []


def test_update_rejects_short_keyword():
    """H2-equivalent: keywords shorter than the minimum can hit dozens of
    unrelated lines. Refuse before any RMW."""
    setup()
    write_memory_md("# Agent Memory\n\n[tip] aaa\n[tip] bbb\n")
    before = read_memory_md()
    status, msg = extraction._replace_line("aa", "this should not land", "tip")
    assert status == "skip"
    assert "Error" in msg
    assert read_memory_md() == before
    assert list_journals() == []


def test_update_skips_protected_headers_and_rules():
    """H2-equivalent: the LLM must not be able to overwrite a markdown
    heading or a Behavioral Rules entry via UPDATE. (`Critical` matches
    the heading text 'Behavioral Rules (CRITICAL)'.)"""
    setup()
    write_memory_md(
        "# Agent Memory\n\n"
        "## Behavioral Rules (CRITICAL)\n"
        "- Never fabricate command results\n"
        "\n"
        "## Random Section\n"
        "- ordinary line about CRITICAL bugs\n"
    )
    status, msg = extraction._replace_line(
        "CRITICAL bugs", "ordinary line about important bugs", "tip",
    )
    content = read_memory_md()
    # Heading not touched
    assert "## Behavioral Rules (CRITICAL)" in content
    # Behavioral rule child line not touched
    assert "- Never fabricate command results" in content
    # Ordinary line WAS replaced (gone from MEMORY.md)
    assert "ordinary line about CRITICAL bugs" not in content
    # New line landed
    assert "important bugs" in content
    assert status == "update"
    assert "Replaced 1" in msg


def test_update_only_protected_matches_returns_skip():
    """If the only matches were protected, return skip and DO NOT write
    anything (neither MEMORY.md nor journal)."""
    setup()
    write_memory_md(
        "# Agent Memory\n\n"
        "## Behavioral Rules (CRITICAL)\n"
        "- protected unique phrase\n"
    )
    before = read_memory_md()
    status, msg = extraction._replace_line(
        "protected unique phrase", "replacement text here", "tip",
    )
    assert status == "skip"
    assert "No entries matched" in msg
    assert "protected" in msg
    # Memory unchanged, no journal entry
    assert read_memory_md() == before
    assert list_journals() == []


def test_supersede_action_no_longer_recognized():
    """Regression: the public 'supersede' action was removed. The facade
    must reject it cleanly so any old caller (or LLM emitting the old
    action name) gets an explicit error rather than a silent no-op."""
    setup()
    res = manage_memory("supersede", text="old|new")
    assert res.is_error is True
    assert "Unknown action" in res.content
    # The error lists valid actions; 'supersede' must NOT appear in that list
    valid_section = res.content.lower().split("valid:", 1)[-1]
    assert "supersede" not in valid_section


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
    """A hand-edited MEMORY.md without a trailing newline must not glue
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


def test_remember_rejects_prompt_shaping_or_oversized_entries():
    """Durable MEMORY.md entries are injected into the system prompt, so the
    public write path must reject markdown blocks and oversized lines."""
    setup()
    import pytest

    write_memory_md("# Agent Memory\n\n[tip] existing durable rule\n")
    before = read_memory_md()

    bad_inputs = [
        "first line\nsecond line",
        "## Injected heading",
        "~~struck-through injected rule~~",
        "x" * 401,
    ]
    for text in bad_inputs:
        with pytest.raises(ValueError):
            remember(text, "tip")

    with pytest.raises(ValueError):
        remember("valid text with a bad category", "not-a-category")

    assert read_memory_md() == before


def test_manage_memory_remember_and_goal_return_errors_for_invalid_entries():
    setup()
    before = read_memory_md()

    res = manage_memory("remember", text="## Injected heading", category="tip")
    assert res.is_error is True
    assert "Error" in res.content

    goal_res = manage_memory("goal", text="first line\nsecond line")
    assert goal_res.is_error is True
    assert "Error" in goal_res.content

    assert read_memory_md() == before



def test_apply_directive_rejects_no_match_update_as_skip():
    """An UPDATE directive whose keyword matches nothing must NOT count
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
    """Enforce the prompt's '<120 chars, one line' rule in the parser."""
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
    """../escape, absolute paths, and special chars must be refused."""
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


def test_topic_rewrite_updates_index_description():
    setup()
    write_memory_md("# Agent Memory\n\n## User Profile\n- Name: Test\n")

    write_topic("rewrite-index", "# Old Title\nbody")
    write_topic("rewrite-index", "# New Title\nbody")

    content = read_memory_md()
    assert content.count("**rewrite-index.md**") == 1
    assert "New Title" in content
    assert "Old Title" not in content


def test_concurrent_topic_index_and_remember_writes_do_not_lose_data():
    """Topic index upserts and remember() both mutate MEMORY.md; concurrent
    writers should not lose either side of the update."""
    import threading

    setup()
    write_memory_md("# Agent Memory\n\n## User Profile\n- Name: Test\n")
    barrier = threading.Barrier(20)
    errors: list[Exception] = []

    def writer(i: int) -> None:
        try:
            barrier.wait()
            if i % 2 == 0:
                write_topic(f"topic-{i}", f"# Topic {i}\nbody")
            else:
                remember(f"durable concurrent rule {i}", "tip")
        except Exception as e:  # pragma: no cover - surfaced by assert below
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent memory writes raised: {errors}"
    content = read_memory_md()
    for i in range(20):
        if i % 2 == 0:
            assert f"**topic-{i}.md**" in content
        else:
            assert f"durable concurrent rule {i}" in content


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
        ("extract_directive_update_replaces_old_line", test_extract_directive_update_replaces_old_line),
        ("extract_directive_noop_writes_nothing", test_extract_directive_noop_writes_nothing),
        ("extract_handles_NONE_response", test_extract_handles_NONE_response),
        ("update_replaces_old_archives_to_journal", test_update_replaces_old_archives_to_journal),
        ("update_no_match_returns_skip_no_writes", test_update_no_match_returns_skip_no_writes),
        ("update_rejects_short_keyword", test_update_rejects_short_keyword),
        ("update_skips_protected_headers_and_rules", test_update_skips_protected_headers_and_rules),
        ("update_only_protected_matches_returns_skip", test_update_only_protected_matches_returns_skip),
        ("supersede_action_no_longer_recognized", test_supersede_action_no_longer_recognized),
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
        ("apply_directive_rejects_no_match_update_as_skip", test_apply_directive_rejects_no_match_update_as_skip),
        ("apply_directive_rejects_overlong_body", test_apply_directive_rejects_overlong_body),
        ("topic_path_rejects_traversal", test_topic_path_rejects_traversal),
        ("write_topic_refuses_traversal", test_write_topic_refuses_traversal),
        ("delete_topic_refuses_traversal", test_delete_topic_refuses_traversal),
        ("facade_topic_write_returns_error_for_bad_name", test_facade_topic_write_returns_error_for_bad_name),
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
