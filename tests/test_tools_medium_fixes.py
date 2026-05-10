"""Tests for medium-priority tool fixes (co-review with Codex, commit 2).

Covers:
  * read_file: offset/limit streams correct window
  * list_directory: bounds traversal by budget, flags truncation
  * grep_files files_only: short-circuits on first match
  * grep_files count: counts occurrences, not matching lines
  * web_fetch: errors on unknown strategy
"""
from __future__ import annotations
import os
import re
import tempfile
import pytest
from jyagent.tools.core import read_file, list_directory
from jyagent.tools.search import grep_files, _count_matches, _has_match
from jyagent.tools.web_fetch import web_fetch


# ─── read_file ───────────────────────────────────────────────────────────────

def test_read_file_offset_limit_window():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        f.write("".join(f"line{i}\n" for i in range(1, 101)))
        path = f.name
    try:
        result = read_file(path, offset=10, limit=3)
        assert not result.is_error
        assert "100 lines total" in result.content
        assert "L10-L12" in result.content
        assert "line10" in result.content
        assert "line13" not in result.content
        assert "line9" not in result.content
    finally:
        os.unlink(path)


def test_read_file_no_pagination_full():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        f.write("hello\nworld\n")
        path = f.name
    try:
        result = read_file(path)
        assert not result.is_error
        assert result.content == "hello\nworld\n"
    finally:
        os.unlink(path)


def test_read_file_line_numbers():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        f.write("a\nb\nc\n")
        path = f.name
    try:
        result = read_file(path, offset=2, limit=2, line_numbers=True)
        assert "L2: b" in result.content
        assert "L3: c" in result.content
        assert "L1:" not in result.content
    finally:
        os.unlink(path)


# ─── list_directory ──────────────────────────────────────────────────────────

def test_list_directory_bounds_traversal(tmp_path):
    # 50 dirs x 20 files = 1050 entries; limit=5 should stop early
    for i in range(50):
        d = tmp_path / f"dir{i:02d}"
        d.mkdir()
        for j in range(20):
            (d / f"file{j:02d}.txt").write_text("x")
    result = list_directory(str(tmp_path), depth=1, limit=5)
    assert not result.is_error
    assert "more" in result.content


def test_list_directory_exact_count_when_fits(tmp_path):
    for i in range(3):
        (tmp_path / f"f{i}.txt").write_text("x")
    result = list_directory(str(tmp_path), depth=1, limit=200)
    assert not result.is_error
    assert "3 entries" in result.content
    assert "truncated" not in result.content


# ─── grep_files helpers ──────────────────────────────────────────────────────

def test_has_match_short_circuits():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        f.write("no match here\nfoo bar\nno match\n" * 1000)
        path = f.name
    try:
        assert _has_match(path, re.compile("foo")) is True
        assert _has_match(path, re.compile("zzznomatch")) is False
    finally:
        os.unlink(path)


def test_count_matches_counts_occurrences_not_lines():
    # "foo foo" on one line = 2 occurrences, not 1
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        f.write("foo foo\nfoo\nbar\n")
        path = f.name
    try:
        assert _count_matches(path, re.compile("foo")) == 3
    finally:
        os.unlink(path)


def test_grep_files_only_mode(tmp_path):
    (tmp_path / "a.txt").write_text("hello world\n")
    (tmp_path / "b.txt").write_text("no match\n")
    (tmp_path / "c.txt").write_text("hello again\n")
    result = grep_files("hello", str(tmp_path), output_mode="files_only")
    assert not result.is_error
    assert "a.txt" in result.content
    assert "c.txt" in result.content
    assert "b.txt" not in result.content


def test_grep_files_count_mode_occurrences(tmp_path):
    (tmp_path / "f.txt").write_text("foo foo\nfoo\nbar\n")
    result = grep_files("foo", str(tmp_path), output_mode="count")
    assert not result.is_error
    # Should report 3 occurrences, not 2 matching lines
    assert "3" in result.content


# ─── web_fetch strategy validation (mocked — no network) ─────────────────────

def test_web_fetch_unknown_strategy_errors():
    # No network needed — validation fires before any fetcher is called.
    result = web_fetch("https://example.com", strategy="bogus")
    assert result.is_error
    assert "bogus" in result.content
    assert "unknown strategy" in result.content.lower()


def test_web_fetch_known_strategies_accepted(monkeypatch):
    # _STRATEGY_MAP holds direct function refs captured at import time, so we
    # patch the map itself rather than the module-level names.
    from jyagent.tools import web_fetch as wf_mod
    stub = lambda url: (503, "blocked")
    monkeypatch.setitem(wf_mod._STRATEGY_MAP, "auto",   [stub])
    monkeypatch.setitem(wf_mod._STRATEGY_MAP, "cffi",   [stub])
    monkeypatch.setitem(wf_mod._STRATEGY_MAP, "direct", [stub])
    monkeypatch.setitem(wf_mod._STRATEGY_MAP, "jina",   [stub])
    monkeypatch.setitem(wf_mod._STRATEGY_MAP, "chrome", [stub])
    monkeypatch.setitem(wf_mod._STRATEGY_MAP_JS_HEAVY, "auto", [stub])
    for strat in ("auto", "cffi", "direct", "jina", "chrome"):
        result = web_fetch("https://example.com", strategy=strat)
        if result.is_error:
            assert "unknown strategy" not in result.content.lower(), \
                f"strategy={strat!r} was incorrectly rejected as unknown"
