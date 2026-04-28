# Tests for the three-tier memory redesign:
#   Tier 1 — MEMORY.md (always loaded, hard cap)
#   Tier 2 — topics/<name>.md (curated, on demand)  [pre-existing, covered elsewhere]
#   Tier 3 — journal/YYYY-MM.md (append-only, never auto-loaded)
#
# Plus: size-warning helper, consolidate analyzer, facade routing.
#
# Background: chronological "completed task" notes used to be appended to
# MEMORY.md as `[note] YYYY-MM-DD …` entries. That's an anti-pattern documented
# by Anthropic, Letta, Mem0, LangMem, Zep, A-MEM and the context-rot research
# (Chroma 2025, NoLiMa ICML 2025). These tests pin the new tier discipline.

import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Patch config BEFORE importing memory modules
_tmpdir = tempfile.mkdtemp(prefix="jy_memory_tiers_test_")
os.environ["AGENT_PROVIDER"] = "anthropic"

import jyagent.config as config
config.MEMORY_DIR = os.path.join(_tmpdir, "memory")
config.TOPICS_DIR = os.path.join(_tmpdir, "memory", "topics")
config.JOURNAL_DIR = os.path.join(_tmpdir, "memory", "journal")
config.MEMORY_MD_FILE = os.path.join(_tmpdir, "memory", "MEMORY.md")

from jyagent.memory.operations import (
    write_memory_md, read_memory_md,
    append_journal, list_journals, read_journal,
    memory_index_size_warning, consolidate_memory,
    remember, show_memory,
    ensure_dirs,
)
from jyagent.tools.facades import manage_memory


def setup():
    """Reset all memory state before each test group."""
    for d in (config.MEMORY_DIR, config.TOPICS_DIR, config.JOURNAL_DIR):
        if os.path.exists(d):
            shutil.rmtree(d)
    ensure_dirs()


def teardown():
    shutil.rmtree(_tmpdir, ignore_errors=True)


# ─── Tier 3: Journal ──────────────────────────────────────────────────────────

def test_append_journal_creates_monthly_file():
    setup()
    path = append_journal("worked on memory redesign", "session")
    assert path.startswith("data/memory/journal/")
    assert path.endswith(".md")
    months = list_journals()
    assert len(months) == 1
    body = read_journal(months[0])
    assert "worked on memory redesign" in body
    assert "[session]" in body
    assert body.startswith("# Journal — ")


def test_journal_appends_dont_overwrite():
    setup()
    append_journal("first entry", "note")
    append_journal("second entry", "debug")
    body = read_journal()
    assert "first entry" in body
    assert "second entry" in body
    assert "[note]" in body
    assert "[debug]" in body
    # Should not have duplicate journal headers
    assert body.count("# Journal — ") == 1


def test_journal_does_not_touch_memory_md():
    """Critical: writing to the journal MUST NOT bloat the always-loaded index."""
    setup()
    write_memory_md("# Agent Memory\n\n[gotcha] some durable rule\n")
    before = read_memory_md()
    for i in range(20):
        append_journal(f"task {i} done", "session")
    after = read_memory_md()
    assert before == after, "journal writes leaked into MEMORY.md (cache-invalidation bug)"


def test_list_journals_sorted_newest_first():
    setup()
    # Manually create files for distinct months
    ensure_dirs()
    for month in ("2026-01", "2026-04", "2025-12"):
        with open(os.path.join(config.JOURNAL_DIR, f"{month}.md"), "w") as f:
            f.write(f"# Journal — {month}\n")
    months = list_journals()
    assert months == ["2026-04", "2026-01", "2025-12"]


# ─── Size-warning helper ──────────────────────────────────────────────────────

def test_no_warning_for_small_memory():
    setup()
    write_memory_md("# Agent Memory\n\nshort and healthy\n")
    assert memory_index_size_warning() is None


def test_warning_for_too_many_lines():
    setup()
    big = "\n".join(f"[tip] line {i}" for i in range(config.MEMORY_INDEX_WARN_LINES + 5))
    write_memory_md(big)
    warn = memory_index_size_warning()
    assert warn is not None
    assert "lines" in warn
    assert "MEMORY.md" in warn
    assert "topics/" in warn  # actionable: tells caller where to move detail
    assert "journal/" in warn


def test_warning_for_too_many_bytes():
    setup()
    # ~20 KB on few lines → byte threshold trips, line threshold doesn't
    long_line = "[tip] " + ("x" * 1000)
    big = "\n".join(long_line for _ in range(20))
    write_memory_md(big)
    warn = memory_index_size_warning()
    assert warn is not None
    assert "bytes" in warn


def test_remember_returns_warning_when_oversized():
    """remember() should surface the warning so the caller actually sees it."""
    setup()
    write_memory_md("\n".join(f"[tip] line {i}" for i in range(config.MEMORY_INDEX_WARN_LINES)))
    msg = remember("one more durable rule", "tip")
    assert "Remembered:" in msg
    assert "approaching load cap" in msg


def test_remember_silent_when_healthy():
    setup()
    msg = remember("first rule", "tip")
    assert "Remembered:" in msg
    assert "approaching load cap" not in msg


# ─── Consolidate analyzer ─────────────────────────────────────────────────────

def test_consolidate_empty_memory():
    setup()
    out = consolidate_memory()
    assert "empty" in out.lower()


def test_consolidate_flags_oversized_lines():
    setup()
    fat = "[note] " + ("very long content " * 30)  # > 400 chars
    write_memory_md(f"# Agent Memory\n\n{fat}\n")
    out = consolidate_memory()
    assert "> 400 chars" in out


def test_consolidate_flags_dated_entries_for_journal():
    """The original sin: dated entries in the always-loaded index.
    Any category with a date (not only [note]) is flagged."""
    setup()
    write_memory_md(
        "# Agent Memory\n\n"
        "[note] 2026-04-18 — Agent-loop upgrade FULLY COMPLETE: 13 P0 bugs fixed\n"
        "[gotcha] 2025-11-21 — skill router silently broken\n"
        "[gotcha] some durable rule with no date\n"
    )
    out = consolidate_memory()
    assert "Dated entries" in out
    assert "journal/" in out
    # Both dated entries are flagged — including the [gotcha] one, not only [note]
    assert "[note]" in out
    assert "2026-04-18" in out
    assert "2025-11-21" in out


def test_consolidate_flags_large_categories():
    setup()
    lines = [f"[tip] tip number {i} about {topic}"
             for i, topic in enumerate(["a", "b", "c", "d", "e", "f", "g"])]
    write_memory_md("\n".join(lines))
    out = consolidate_memory()
    assert "consider consolidating" in out


def test_consolidate_no_false_positive_when_healthy():
    setup()
    write_memory_md(
        "# Agent Memory\n\n"
        "[gotcha] Python 3.14 venv has broken CA certs\n"
        "[preference] User prefers direct communication\n"
    )
    out = consolidate_memory()
    assert "No obvious consolidation candidates" in out


# ─── Facade routing ───────────────────────────────────────────────────────────

def test_manage_memory_journal_action():
    setup()
    res = manage_memory("journal", "did some research today", "session")
    assert not res.is_error
    assert "journal/" in res.content
    body = read_journal()
    assert "did some research today" in body


def test_manage_memory_consolidate_action():
    setup()
    write_memory_md("# Agent Memory\n\n[note] 2026-04-18 — some dated cruft\n")
    res = manage_memory("consolidate")
    assert not res.is_error
    assert "consolidation analysis" in res.content


def test_manage_memory_unknown_action_lists_new_actions():
    setup()
    res = manage_memory("totally_made_up_action")
    assert res.is_error
    # Should advertise the new actions in the error
    assert "journal" in res.content
    assert "consolidate" in res.content


def test_manage_memory_journal_requires_text():
    setup()
    res = manage_memory("journal", "")
    assert res.is_error


def test_show_memory_includes_journals():
    setup()
    append_journal("an entry", "note")
    out = show_memory()
    assert "JOURNAL" in out
    assert "2026" in out


def test_remember_is_still_for_durable_facts():
    """Sanity: remember() still appends to MEMORY.md (it's the durable tier)."""
    setup()
    msg = remember("a durable rule that prevents future mistakes", "gotcha")
    assert "Remembered:" in msg
    content = read_memory_md()
    assert "[gotcha] a durable rule" in content


# ─── Regression tests for Codex-review findings ──────────────────────────────

def test_append_journal_no_duplicate_header_under_concurrency():
    """Codex-review bug: `is_new = not exists(path); if is_new: write_header`
    is TOCTOU under parallel sub-agents. Fix uses O_CREAT|O_EXCL for the
    header install. Regression test: 16 threads race on a fresh month."""
    import threading
    setup()
    barrier = threading.Barrier(16)
    errors: list[Exception] = []

    def writer(i: int) -> None:
        try:
            barrier.wait()
            append_journal(f"parallel entry {i}", "concurrency")
        except Exception as e:  # pragma: no cover — surface for asserts
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(16)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert not errors, f"append_journal raised under concurrency: {errors}"
    body = read_journal()
    # The header and intro paragraph must appear exactly once despite 16
    # racing writers each thinking the file was new.
    assert body.count("# Journal — ") == 1
    assert body.count("Tier 3 / append-only") == 1
    # All 16 entries must be present (no writes lost).
    for i in range(16):
        assert f"parallel entry {i}" in body


def test_journal_path_rejects_traversal():
    """Codex-review bug: _journal_path didn't validate month → path traversal.
    A malicious or buggy caller passing '../../etc/passwd' should raise."""
    setup()
    import pytest
    with pytest.raises(ValueError):
        read_journal("../../etc/passwd")
    with pytest.raises(ValueError):
        read_journal("2026-04/../../etc")
    with pytest.raises(ValueError):
        read_journal("not-a-month")
    # Valid month still works.
    assert read_journal("2026-04") == ""


def test_size_warning_off_by_one_fixed():
    """Codex-review bug: `content.split('\\n')` over-counts lines by one for
    any newline-terminated file (which is all of them, since append_memory_md
    terminates lines). The fix switches to splitlines()."""
    setup()
    # Exactly 149 real lines, newline-terminated. Under the warn threshold
    # (default 150). With the buggy count this was 150 and tripped the warn.
    content = "\n".join(f"[tip] line {i}" for i in range(149)) + "\n"
    write_memory_md(content)
    assert len(content.splitlines()) == 149  # sanity
    # Bytes are small, so we're only testing the line threshold.
    assert memory_index_size_warning() is None, \
        "149 lines should NOT trip the 150-line warn threshold"

    # One more line → warn fires.
    content += "[tip] line 149\n"
    write_memory_md(content)
    assert memory_index_size_warning() is not None


def test_consolidate_cjk_dedup():
    """Codex-review bug: [A-Za-z] regex silently drops CJK content, so
    Chinese duplicates were never flagged. Fix extends token regex to CJK."""
    setup()
    write_memory_md(
        "# Agent Memory\n\n"
        "[tip] 用户偏好使用中文交流并且要求直接坦诚的沟通风格\n"
        "[tip] 用户偏好中文并要求直接坦诚的交流沟通风格习惯\n"
        "[tip] 完全无关的第三条提示关于网络代理配置的说明\n"
        "[tip] 另一条不相关的提示关于Docker镜像构建的流程\n"
    )
    out = consolidate_memory()
    # With the old regex these 4 lines produced empty word sets and no
    # overlap hints. With CJK support, the first two share several tokens.
    assert "Possible duplicate pairs" in out, \
        f"CJK dedup didn't flag overlapping Chinese entries; output was:\n{out}"


def test_consolidate_captures_version_numbers():
    """Codex-review note: the old regex dropped 'Python3.14' (the exact kind
    of high-signal token you want for dedup). Fix includes dotted identifiers."""
    setup()
    write_memory_md(
        "# Agent Memory\n\n"
        "[gotcha] Python3.14 venv has broken certificate authority bundle issue\n"
        "[gotcha] Python3.14 virtualenv certificate authority issue breaks pip\n"
        "[gotcha] Python3.14 venv cert authority configuration breaks verification\n"
        "[gotcha] Python3.14 env certificate issue with authority bundle verification\n"
    )
    out = consolidate_memory()
    # All four entries share "Python3.14" and several cert-related tokens
    assert "Possible duplicate pairs" in out, \
        f"version-number dedup didn't fire; output was:\n{out}"
    assert "python3.14" in out.lower()


def test_remember_suppress_warning_kwarg():
    """Codex-review improvement: programmatic callers need a way to get a
    clean return string without the embedded size warning."""
    setup()
    # Force oversized memory.
    big = "\n".join(f"[tip] line {i}" for i in range(config.MEMORY_INDEX_WARN_LINES + 5))
    write_memory_md(big)
    # Default path includes the warning.
    default = remember("a new durable rule", "gotcha")
    assert "MEMORY.md approaching" in default
    # suppress_warning strips it.
    quiet = remember("another durable rule", "gotcha", suppress_warning=True)
    assert "MEMORY.md approaching" not in quiet
    assert quiet.startswith("Remembered:")


def test_journal_read_creates_dir_on_first_call():
    """read_journal on a fresh install shouldn't crash because JOURNAL_DIR
    doesn't exist yet — it should return empty string."""
    setup()
    # Nuke journal dir to simulate a fresh install.
    if os.path.isdir(config.JOURNAL_DIR):
        shutil.rmtree(config.JOURNAL_DIR)
    # Should not raise, should return empty.
    assert read_journal() == ""
    assert read_journal("2025-01") == ""


def test_list_journals_ignores_non_month_filenames():
    """Defensive: list_journals should only return files matching YYYY-MM.md,
    not random junk someone drops in the journal dir."""
    setup()
    ensure_dirs()
    # Drop a non-month file alongside real journals.
    for fname in ("2026-04.md", "2026-03.md", "README.md", "notes.md", "backup.md.bak"):
        with open(os.path.join(config.JOURNAL_DIR, fname), "w") as f:
            f.write("irrelevant")
    months = list_journals()
    assert months == ["2026-04", "2026-03"]


def test_very_large_memory_md_doesnt_crash():
    """Defensive: a 200 KB MEMORY.md (way over cap) should still produce a
    warning and consolidate report without OOM or timeout."""
    setup()
    # 200 KB of entries — 10× the hard cap.
    giant = "\n".join(f"[tip] entry {i} with some content " * 5 for i in range(2000))
    write_memory_md(giant)
    warn = memory_index_size_warning()
    assert warn is not None
    assert "bytes" in warn
    out = consolidate_memory()
    assert "consider consolidating" in out  # [tip] has thousands of entries
    assert "approaching load cap" in out
