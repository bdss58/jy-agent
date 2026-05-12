"""Regression tests for ``SkillManager.create_skill``.

Closes the second pair of issues flagged by the 2026-05 codex review of
``jyagent/skills.py`` (the first pair — pin-path injection + silent
truncation — is covered by ``test_skill_cache_stability.TestPinnedBodySafety``):

  * MED — YAML-frontmatter escaping. ``description``/``metadata`` values
    were written into the file as f-string-interpolated text, so any
    embedded quote, newline, colon, or YAML-meaningful character
    corrupted the generated ``SKILL.md`` and caused the next
    ``discover()`` call to silently drop the skill.

  * MED — Symlink / path-traversal at the destination. The skill-name
    regex blocks ``..`` and ``/`` in the argument, but if an attacker
    pre-planted a symlink at ``skills/<name>`` pointing outside the
    skills tree, the old implementation would happily follow it and
    write through.

Both fixes preserve the public success contract (returns a
``"✅ Skill ..."`` string), so existing callers don't need to change.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from jyagent.skills import SkillManager, parse_skill_md


# ─── helpers ─────────────────────────────────────────────────────────────


def _new_mgr(tmp_path) -> tuple[SkillManager, Path]:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    mgr = SkillManager(str(skills_dir))
    mgr.discover()  # empty
    return mgr, skills_dir


# ─── YAML escaping (MED #4) ──────────────────────────────────────────────


class TestCreateSkillEscapesYAML:
    """The generated SKILL.md must be valid YAML frontmatter regardless
    of what characters appear in ``description`` / ``metadata`` values."""

    def test_description_with_double_quote_round_trips(self, tmp_path):
        mgr, skills_dir = _new_mgr(tmp_path)
        result = mgr.create_skill(
            "qtest",
            description='A skill that says "hello" — note the quotes.',
            instructions="Body.",
        )
        assert result.startswith("✅"), result
        parsed = parse_skill_md(str(skills_dir / "qtest" / "SKILL.md"))
        assert parsed is not None, "Generated SKILL.md failed to parse"
        assert parsed["description"] == 'A skill that says "hello" — note the quotes.'

    def test_description_with_newline_round_trips(self, tmp_path):
        mgr, skills_dir = _new_mgr(tmp_path)
        result = mgr.create_skill(
            "ntest",
            description="Line one.\nLine two.",
            instructions="Body.",
        )
        assert result.startswith("✅"), result
        parsed = parse_skill_md(str(skills_dir / "ntest" / "SKILL.md"))
        assert parsed is not None
        # The hand-rolled parser may collapse the literal '\n' from JSON
        # back to a space or keep it as ``\n`` — the important thing is
        # that the file parses at all (i.e. the YAML structure isn't
        # broken by the embedded newline).
        assert parsed["name"] == "ntest"
        assert "Line one" in parsed["description"]

    def test_description_with_colon_round_trips(self, tmp_path):
        """Bare colons are YAML-significant and would break a naive
        ``description: >- \\n  {value}`` block when the description
        contains its own ``key:`` pair on the first wrapped line."""
        mgr, skills_dir = _new_mgr(tmp_path)
        result = mgr.create_skill(
            "ctest",
            description="Trigger on: any task containing the word 'foo'.",
            instructions="Body.",
        )
        assert result.startswith("✅"), result
        parsed = parse_skill_md(str(skills_dir / "ctest" / "SKILL.md"))
        assert parsed is not None
        assert "Trigger on:" in parsed["description"]

    def test_description_starting_with_yaml_block_indicator(self, tmp_path):
        """A leading ``>``, ``|``, or ``-`` used to be parsed as a YAML
        scalar style indicator. JSON-quoting prevents that."""
        mgr, skills_dir = _new_mgr(tmp_path)
        result = mgr.create_skill(
            "btest",
            description="> Use this skill when ...",
            instructions="Body.",
        )
        assert result.startswith("✅"), result
        parsed = parse_skill_md(str(skills_dir / "btest" / "SKILL.md"))
        assert parsed is not None
        assert parsed["description"].startswith(">")

    def test_metadata_with_quotes_does_not_corrupt_frontmatter(self, tmp_path):
        """``metadata`` is callable only from internal Python code (not
        the tool surface), but it still needs to round-trip."""
        mgr, skills_dir = _new_mgr(tmp_path)
        result = mgr.create_skill(
            "mtest",
            description="x",
            instructions="Body.",
            metadata={"author": 'Someone "Quoted"', "version": "1.0"},
        )
        assert result.startswith("✅"), result
        parsed = parse_skill_md(str(skills_dir / "mtest" / "SKILL.md"))
        assert parsed is not None, (
            "metadata containing a quote corrupted the SKILL.md frontmatter"
        )
        assert parsed["metadata"] is not None
        assert parsed["metadata"]["author"] == 'Someone "Quoted"'


# ─── symlink / path-traversal (MED #2) ───────────────────────────────────


class TestCreateSkillRejectsSymlinkEscape:
    """If ``skills/<name>`` already exists as a symlink pointing outside
    the skills tree, ``create_skill`` must refuse rather than write
    through it.

    NOTE: ``os.symlink`` requires admin rights on Windows; these tests
    are skipped there.  Pytest doesn't pick up the platform skip
    automatically because we want explicit messaging."""

    @pytest.mark.skipif(
        os.name == "nt", reason="os.symlink requires admin on Windows"
    )
    def test_refuses_to_write_through_directory_symlink(self, tmp_path):
        # Pre-plant a symlink at skills/evil that points OUTSIDE the skills tree.
        outside = tmp_path / "outside"
        outside.mkdir()

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        evil_link = skills_dir / "evil"
        evil_link.symlink_to(outside, target_is_directory=True)

        mgr = SkillManager(str(skills_dir))
        mgr.discover()
        result = mgr.create_skill(
            "evil", description="x", instructions="Body.",
        )

        # The function must refuse — not write through the symlink.
        assert result.startswith("Error:"), result
        assert "outside the skills tree" in result or "symlink" in result.lower()
        # And the outside directory must be unchanged (no SKILL.md inside it).
        assert not (outside / "SKILL.md").exists(), (
            "create_skill wrote through a pre-planted symlink — the realpath "
            "check failed. This is the path-traversal regression."
        )

    @pytest.mark.skipif(
        os.name == "nt", reason="os.symlink requires admin on Windows"
    )
    def test_refuses_to_follow_symlink_at_skill_md_path(self, tmp_path):
        """Defense-in-depth: even if the directory check passes, the
        ``open(SKILL.md)`` call must refuse to follow a symlink at the
        SKILL.md path itself. (``O_NOFOLLOW`` on POSIX.)"""
        # Pre-create the legitimate skill dir, then plant a symlink AT
        # the SKILL.md path pointing somewhere outside.
        outside = tmp_path / "outside.txt"
        outside.write_text("original content")

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_dir = skills_dir / "trojan"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").symlink_to(outside)

        mgr = SkillManager(str(skills_dir))
        mgr.discover()
        result = mgr.create_skill(
            "trojan", description="x", instructions="Body.",
        )
        # Either the symlink-at-dir check catches it, OR the O_NOFOLLOW
        # open does — both are acceptable. The outside file must NOT be
        # overwritten.
        assert result.startswith("Error:"), result
        assert outside.read_text() == "original content", (
            "create_skill followed a symlink at the SKILL.md path and "
            "overwrote the target file. O_NOFOLLOW guard failed."
        )

    def test_normal_create_still_works(self, tmp_path):
        """Sanity: the security check doesn't break the happy path."""
        mgr, skills_dir = _new_mgr(tmp_path)
        result = mgr.create_skill(
            "happy", description="A normal skill.", instructions="Body.",
        )
        assert result.startswith("✅"), result
        assert (skills_dir / "happy" / "SKILL.md").exists()
        parsed = parse_skill_md(str(skills_dir / "happy" / "SKILL.md"))
        assert parsed is not None
        assert parsed["name"] == "happy"
        assert parsed["description"] == "A normal skill."
