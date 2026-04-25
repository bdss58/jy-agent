"""Smoke tests for SkillManager default-path discovery.

Regression guard for the 2026-04-25 bug: after `git mv jyagent/skills.py
jyagent/runtime/skills.py`, the `os.path.dirname(...)` chain used to
resolve `DEFAULT_SKILLS_DIR` was off by one, silently pointing at
`<repo>/jyagent/skills` (which doesn't exist) instead of `<repo>/skills`.
Discovery returned `[]`, `manage_skills(action='list')` reported "No
skills found", but every test in the suite still passed because they
inject `skills_dir` explicitly via tmp_path fixtures.

These tests assert the real default path resolves to an existing
directory containing the in-repo skills. Any future "move module deeper
in package tree" refactor that breaks the depth count will fail here.
"""

from __future__ import annotations

import os

import pytest

from jyagent.runtime.skills import DEFAULT_SKILLS_DIR, SkillManager


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPECTED_SKILLS_DIR = os.path.join(REPO_ROOT, "skills")


def test_default_skills_dir_resolves_to_repo_skills():
    """DEFAULT_SKILLS_DIR must point at <repo_root>/skills, not some
    pseudo-path nested inside the package tree."""
    assert os.path.abspath(DEFAULT_SKILLS_DIR) == EXPECTED_SKILLS_DIR, (
        f"DEFAULT_SKILLS_DIR={DEFAULT_SKILLS_DIR!r} "
        f"expected={EXPECTED_SKILLS_DIR!r} — likely a __file__-depth bug "
        "after moving runtime/skills.py to a different package depth."
    )


def test_default_skills_dir_exists():
    """The resolved default path must actually exist on disk."""
    assert os.path.isdir(DEFAULT_SKILLS_DIR), (
        f"DEFAULT_SKILLS_DIR={DEFAULT_SKILLS_DIR!r} does not exist. "
        "Either the repo layout changed or __file__-depth math is wrong."
    )


def test_skill_manager_discovers_at_least_one_skill_from_default_path():
    """SkillManager() with no args must discover ≥1 skill on a real
    checkout. This is the user-facing symptom of the depth bug:
    `manage_skills(action='list')` returned empty because discovery
    silently scanned a non-existent directory."""
    mgr = SkillManager()
    discovered = mgr.discover()
    assert len(discovered) >= 1, (
        f"SkillManager() discovered 0 skills from default path "
        f"{mgr.skills_dir!r}. Expected ≥1 skill in <repo>/skills/. "
        "If the repo genuinely has no skills, delete this test."
    )
    # list_skills() should agree with discover()
    assert sorted(discovered) == mgr.list_skills()
