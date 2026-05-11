"""P0 regressions for the Agent Skills layer.

These tests guard against three bugs caught in the 2026-05 codex review:

1. ``/skills`` and ``manage_skills(action="list")`` were reading
   ``entry["active"]`` from the catalog, but ``get_catalog()`` only ever
   emitted ``pinned``. Both call sites crashed with ``KeyError: 'active'``
   and there was no test covering either path.
2. ``discover()`` did not enforce the agentskills.io rule that a skill's
   frontmatter ``name`` must equal ``basename(skill_dir)``. A folder lying
   about its identity would silently load and break ``read_resource`` paths.
3. ``discover()`` was last-writer-wins on duplicate frontmatter ``name``,
   and the winner depended on unsorted ``glob.glob()`` order — a latent
   non-determinism bug across filesystems.
"""

from __future__ import annotations

import pytest

from jyagent.skills import SkillManager
from jyagent.tools.skills_tool import manage_skills as manage_skills_tool
from jyagent import skills as skills_mod


# ─── helpers ─────────────────────────────────────────────────────────────


def _make_skill(skills_dir, dir_name, *, name=None, description="x"):
    """Create one skills/<dir_name>/SKILL.md.

    By default frontmatter ``name`` matches ``dir_name`` (the spec-conformant
    case). Pass ``name=`` to deliberately create a mismatch.
    """
    if name is None:
        name = dir_name
    sdir = skills_dir / dir_name
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nBody for {name}.\n"
    )
    return sdir


# ─── P0-1: list paths must not KeyError on the catalog ───────────────────


class TestListCatalogKeyAccess:
    """``/skills`` and ``manage_skills(action="list")`` previously crashed
    with ``KeyError: 'active'`` because they read a key the catalog never
    emits. These tests pin the agreed contract: the catalog exposes
    ``name``, ``description``, ``pinned``."""

    def test_get_catalog_emits_only_documented_keys(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        _make_skill(skills_dir, "alpha")
        mgr = SkillManager(str(skills_dir))
        mgr.discover()

        catalog = mgr.get_catalog()
        assert catalog, "fixture should produce one catalog entry"
        assert set(catalog[0].keys()) == {"name", "description", "pinned"}, (
            "Catalog dict keys are part of the public contract — both "
            "the /skills CLI and manage_skills('list') tool read them. "
            "Adding keys is fine; removing or renaming requires updating "
            "every reader."
        )

    def test_manage_skills_list_does_not_error(
        self, tmp_path, monkeypatch
    ):
        """Direct repro of the codex-discovered crash:
        ``manage_skills('list')`` returned ``is_error=True`` with
        ``Error managing skills: 'active'``."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        _make_skill(skills_dir, "alpha", description="First skill")
        _make_skill(skills_dir, "beta", description="Second skill")

        # Point the singleton at our fixture.
        monkeypatch.setattr(skills_mod, "_manager", None)
        skills_mod.init_skills(str(skills_dir))

        result = manage_skills_tool("list")
        assert result.is_error is False, (
            f"manage_skills('list') errored: {result}"
        )
        text = str(result)
        assert "alpha" in text
        assert "beta" in text
        assert "2 skills" in text
        assert "0 pinned" in text

    def test_manage_skills_list_counts_pinned_correctly(
        self, tmp_path, monkeypatch
    ):
        """Once ``alpha`` is pinned, the totals line must say ``1 pinned``."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        _make_skill(skills_dir, "alpha")
        _make_skill(skills_dir, "beta")

        monkeypatch.setattr(skills_mod, "_manager", None)
        mgr = skills_mod.init_skills(str(skills_dir))
        mgr.pin("alpha")

        result = manage_skills_tool("list")
        assert result.is_error is False
        text = str(result)
        assert "2 skills" in text
        assert "1 pinned" in text
        # Pinned marker should appear next to alpha.
        assert "PINNED" in text

    def test_cmd_skills_handler_does_not_keyerror(
        self, tmp_path, monkeypatch
    ):
        """Same crash on the ``/skills`` CLI handler. We exercise the real
        handler with a stub CLI that just records the rendered string."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        _make_skill(skills_dir, "alpha")

        monkeypatch.setattr(skills_mod, "_manager", None)
        skills_mod.init_skills(str(skills_dir))

        from jyagent.agent import _cmd_skills

        captured = {}

        class _StubCLI:
            def print_system(self, msg):
                captured["system"] = msg

            def print_error(self, msg):  # pragma: no cover
                captured["error"] = msg

        # Must not raise. The bug used to raise KeyError('active') here.
        _cmd_skills(_StubCLI())
        assert "system" in captured, (
            "_cmd_skills did not produce any output — "
            "the KeyError('active') regression is back."
        )
        assert "alpha" in captured["system"]
        assert "0 pinned" in captured["system"]


# ─── P0-2: parent-dir / frontmatter-name match ───────────────────────────


class TestParentDirNameMustMatch:
    """agentskills.io rule: for ``skills/<dir>/SKILL.md`` the frontmatter
    ``name`` must equal ``basename(<dir>)``. Otherwise the resource path
    layer (which uses the on-disk dir) would diverge from what the model
    sees in the catalog (which uses the frontmatter name)."""

    def test_mismatched_name_is_rejected(self, tmp_path, capsys):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        # Folder is "alpha" but frontmatter declares name "imposter".
        _make_skill(skills_dir, "alpha", name="imposter")

        mgr = SkillManager(str(skills_dir))
        discovered = mgr.discover()

        assert discovered == [], (
            f"Mismatched skill should have been rejected, got {discovered}"
        )
        assert mgr.get_catalog() == []
        warn = capsys.readouterr().err
        assert "imposter" in warn and "alpha" in warn, (
            f"Rejection should warn on stderr with both names; got: {warn!r}"
        )

    def test_matched_name_is_accepted(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        _make_skill(skills_dir, "alpha")  # name == basename

        mgr = SkillManager(str(skills_dir))
        discovered = mgr.discover()
        assert discovered == ["alpha"]

    def test_one_mismatch_does_not_taint_a_valid_sibling(
        self, tmp_path, capsys
    ):
        """A bad skill must be skipped, not abort the whole discovery."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        _make_skill(skills_dir, "alpha")
        _make_skill(skills_dir, "bogus", name="not-bogus")

        mgr = SkillManager(str(skills_dir))
        discovered = mgr.discover()
        assert discovered == ["alpha"]
        warn = capsys.readouterr().err
        assert "not-bogus" in warn


# ─── P0-3: duplicate-name rejection is deterministic ─────────────────────


class TestDuplicateNameIsRejected:
    """Two on-disk skills declaring the same frontmatter ``name`` used to
    silently overwrite each other, with the winner depending on the
    unsorted ``glob.glob()`` order — non-deterministic across filesystems.
    The agreed behavior is now: first one wins (sorted glob), second is
    rejected with a stderr warning."""

    def test_duplicate_second_skill_is_rejected(self, tmp_path, capsys):
        """Sorted glob ensures the FIRST folder lexicographically wins; the
        second is rejected (either via parent-dir mismatch or, if its folder
        name happened to also match, via the duplicate check)."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        # First skill: folder name matches frontmatter — accepted.
        _make_skill(skills_dir, "dup-a", description="first")

        # Second folder: declares the same frontmatter name as the first
        # but lives in a different folder, so the parent-dir check rejects
        # it before the duplicate check fires. Either rejection path is
        # acceptable as long as exactly one 'dup-a' ends up loaded with
        # the FIRST description.
        sib = skills_dir / "dup-a-clone"
        sib.mkdir()
        (sib / "SKILL.md").write_text(
            "---\nname: dup-a\ndescription: second\n---\n\nBody.\n"
        )

        mgr = SkillManager(str(skills_dir))
        discovered = mgr.discover()

        assert discovered == ["dup-a"], (
            f"expected only the first 'dup-a' to load, got {discovered}"
        )
        loaded = mgr.get_skill("dup-a")
        assert loaded["description"] == "first"
        warn = capsys.readouterr().err
        assert warn, "rejected sibling should have produced a stderr warning"

    def test_true_duplicate_name_path(self, tmp_path, capsys, monkeypatch):
        """Direct exercise of the duplicate-name code path: monkeypatch
        glob so two distinct folders produce SKILL.md files whose folder
        names match their frontmatter names AND whose names collide.

        On a real filesystem this is impossible (two folders cannot share
        the same basename in the same parent), but the safety net should
        still catch it if a future refactor introduces nested layouts."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        # Two unrelated valid skills, each spec-conformant on its own.
        _make_skill(skills_dir, "twin", description="first")
        # Hand-craft a second valid SKILL.md somewhere else with the same
        # name and a folder basename that also matches.
        elsewhere = tmp_path / "other-root" / "twin"
        elsewhere.mkdir(parents=True)
        (elsewhere / "SKILL.md").write_text(
            "---\nname: twin\ndescription: second\n---\n\nBody.\n"
        )

        import glob as glob_mod
        real_glob = glob_mod.glob

        def fake_glob(pattern):
            # Return both SKILL.md paths, sorted, so the in-tmp_path one
            # wins and the elsewhere one trips the duplicate check.
            return sorted([
                str(skills_dir / "twin" / "SKILL.md"),
                str(elsewhere / "SKILL.md"),
            ])

        monkeypatch.setattr("jyagent.skills.glob.glob", fake_glob)

        mgr = SkillManager(str(skills_dir))
        discovered = mgr.discover()

        assert discovered == ["twin"]
        warn = capsys.readouterr().err
        assert "duplicate name" in warn, (
            f"expected duplicate-name warning, got: {warn!r}"
        )

    def test_discovery_order_is_deterministic(self, tmp_path):
        """The discovery list must be sorted, not glob-order-dependent."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        # Create in scrambled order; the result must still be sorted.
        for n in ["zeta", "alpha", "mu", "beta"]:
            _make_skill(skills_dir, n)

        mgr = SkillManager(str(skills_dir))
        discovered = mgr.discover()
        assert discovered == sorted(discovered), (
            f"discover() must return a deterministic, sorted list; got {discovered}"
        )
