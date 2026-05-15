"""Tests for the G1-lite project-context block in jyagent.system_prompt.

When jy-agent is launched from inside a project repo that ships an
AGENTS.md or CLAUDE.md, that file should be auto-detected and surfaced as
a session-long block in the system prompt — capped, cache-stable, and
preferring AGENTS.md over CLAUDE.md.
"""
from __future__ import annotations

import pytest

from jyagent import system_prompt as sp


@pytest.fixture(autouse=True)
def _clean_project_context_cache():
    """Reset the module-level cache before AND after every test so cases
    don't leak state into each other."""
    sp.invalidate_project_context_cache()
    yield
    sp.invalidate_project_context_cache()


def _set_launch_dir(monkeypatch, path: str) -> None:
    """LAUNCH_DIR is module-level config; bind via the config module."""
    from jyagent import config as cfg
    monkeypatch.setattr(cfg, "LAUNCH_DIR", path)


# ─── Detection ───────────────────────────────────────────────────────────────


def test_no_project_file_returns_empty(tmp_path, monkeypatch):
    _set_launch_dir(monkeypatch, str(tmp_path))
    block = sp._build_project_context_block()
    assert block == ""
    assert sp.project_context_source() is None


def test_finds_agents_md_in_launch_dir(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("# Project rules\nUse pytest.", encoding="utf-8")
    _set_launch_dir(monkeypatch, str(tmp_path))

    block = sp._build_project_context_block()
    assert "<project_context>" in block
    assert "Use pytest." in block
    assert sp.project_context_source() == str(tmp_path / "AGENTS.md")


def test_finds_claude_md_when_no_agents_md(tmp_path, monkeypatch):
    (tmp_path / "CLAUDE.md").write_text("# Claude-specific\nTests live in tests/.", encoding="utf-8")
    _set_launch_dir(monkeypatch, str(tmp_path))

    block = sp._build_project_context_block()
    assert "Tests live in tests/." in block
    assert sp.project_context_source() == str(tmp_path / "CLAUDE.md")


def test_agents_md_preferred_over_claude_md_at_same_level(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("from AGENTS", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("from CLAUDE", encoding="utf-8")
    _set_launch_dir(monkeypatch, str(tmp_path))

    block = sp._build_project_context_block()
    assert "from AGENTS" in block
    assert "from CLAUDE" not in block


def test_walks_up_to_find_ancestor_file(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("repo-root rules", encoding="utf-8")
    sub = tmp_path / "src" / "deep"
    sub.mkdir(parents=True)
    _set_launch_dir(monkeypatch, str(sub))

    block = sp._build_project_context_block()
    assert "repo-root rules" in block
    assert sp.project_context_source() == str(tmp_path / "AGENTS.md")


def test_nearest_ancestor_wins(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("outer rules", encoding="utf-8")
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "AGENTS.md").write_text("inner rules", encoding="utf-8")
    _set_launch_dir(monkeypatch, str(inner))

    block = sp._build_project_context_block()
    assert "inner rules" in block
    assert "outer rules" not in block


def test_does_not_walk_past_home_dir(tmp_path, monkeypatch):
    """A file at $HOME or above should not be picked up — that's not a project."""
    fake_home = tmp_path
    (fake_home / "AGENTS.md").write_text("home-level (should be ignored)", encoding="utf-8")
    sub = fake_home / "project"
    sub.mkdir()
    _set_launch_dir(monkeypatch, str(sub))
    monkeypatch.setenv("HOME", str(fake_home))

    block = sp._build_project_context_block()
    # The walk reaches `sub` (no file), then `fake_home` (the home guard
    # stops the walk before looking inside it). So no file is found.
    assert block == ""


# ─── Truncation ──────────────────────────────────────────────────────────────


def test_oversized_file_is_truncated(tmp_path, monkeypatch):
    huge = "line\n" * 1000  # 5000 lines, well above the 300-line cap
    (tmp_path / "AGENTS.md").write_text(huge, encoding="utf-8")
    _set_launch_dir(monkeypatch, str(tmp_path))

    block = sp._build_project_context_block()
    assert "truncated by jy-agent" in block
    # Capped portion should be present but not all 1000 lines.
    assert block.count("line\n") < 1000


# ─── Caching ─────────────────────────────────────────────────────────────────


def test_cache_is_stable_across_calls(tmp_path, monkeypatch):
    f = tmp_path / "AGENTS.md"
    f.write_text("v1", encoding="utf-8")
    _set_launch_dir(monkeypatch, str(tmp_path))

    first = sp._build_project_context_block()
    # Modify the file on disk — the cache should NOT pick it up without
    # an explicit invalidate call (intentional design: project context
    # is per-session).
    f.write_text("v2 — changed after launch", encoding="utf-8")
    second = sp._build_project_context_block()
    assert first == second
    assert "v1" in second
    assert "v2" not in second

    # After explicit invalidation, the new content should be picked up.
    sp.invalidate_project_context_cache()
    third = sp._build_project_context_block()
    assert "v2" in third


# ─── Full system-prompt integration ──────────────────────────────────────────


def test_build_system_prompt_includes_project_context(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text(
        "# Conventions\nAlways run ruff before committing.", encoding="utf-8"
    )
    _set_launch_dir(monkeypatch, str(tmp_path))

    prompt = sp.build_system_prompt(user_input="hi", skill_mgr=None)
    assert "<project_context>" in prompt
    assert "Always run ruff before committing." in prompt


def test_build_system_prompt_no_project_context_when_absent(tmp_path, monkeypatch):
    _set_launch_dir(monkeypatch, str(tmp_path))
    prompt = sp.build_system_prompt(user_input="hi", skill_mgr=None)
    assert "<project_context>" not in prompt
