"""Cache-stability invariants for the skills layer.

The MEMORY.md durable rule says: "Mutating Anthropic system_prompt breaks
prompt caching — inject dynamic context as a non-persisted tail message
block instead." Skills used to violate this — `build_prompt_context` was
called with a query every turn, ran the LLM router, then concatenated the
pinned-skill bodies into the system prompt. Any pin diff would mutate the
system prompt → invalidate the Anthropic prompt cache prefix → pay ~12×
cost on cache-heavy workloads.

These tests pin the post-refactor design (Design B, progressive disclosure):

  * Stage 1 catalog (build_catalog_block) goes into the system prompt and
    is byte-stable across pin changes — only the on-disk skills/ directory
    contents can change it.
  * Stage 2 pinned bodies (build_pinned_bodies_block) are emitted as a
    SEPARATE block that the agent attaches to the last user message, NOT
    to the system prompt.
  * There is NO per-turn automatic skill router — skills are loaded by
    the LLM (via `manage_skills(action='load')`) or pinned by the user
    (via `/skill`).  The previous opt-in `SKILL_PRE_ROUTER` flag was
    removed 2026-05.

Any future refactor that re-introduces pin-state into the catalog (or
pulls pinned bodies back into the system prompt) will fail here.
"""

from __future__ import annotations

import importlib

import pytest

from jyagent.skills import SkillManager


# ─── helpers ─────────────────────────────────────────────────────────────


def _make_skills_dir(tmp_path, names_and_descs):
    """Create a tmp skills dir with the given (name, description) pairs."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    for name, desc in names_and_descs:
        sdir = skills_dir / name
        sdir.mkdir()
        (sdir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\n---\n\nBody for {name}.\n"
        )
    return str(skills_dir)


# ─── catalog stability ───────────────────────────────────────────────────


class TestCatalogIsCacheStable:
    """build_catalog_block must NOT depend on the pinned set."""

    def test_catalog_unchanged_after_pin(self, tmp_path):
        skills_dir = _make_skills_dir(tmp_path, [
            ("alpha", "First skill"),
            ("beta", "Second skill"),
        ])
        mgr = SkillManager(skills_dir)
        mgr.discover()

        catalog_before = mgr.build_catalog_block()
        assert mgr.pin("alpha") is True
        catalog_after = mgr.build_catalog_block()

        assert catalog_before == catalog_after, (
            "build_catalog_block() changed after pin() — this would "
            "invalidate the Anthropic prompt-cache prefix on every pin "
            "diff. Pinned state must NOT leak into the catalog."
        )

    def test_catalog_unchanged_after_unpin(self, tmp_path):
        skills_dir = _make_skills_dir(tmp_path, [
            ("alpha", "First skill"),
            ("beta", "Second skill"),
        ])
        mgr = SkillManager(skills_dir)
        mgr.discover()
        mgr.pin("alpha")

        catalog_with_pinned = mgr.build_catalog_block()
        mgr.unpin("alpha")
        catalog_without_pinned = mgr.build_catalog_block()

        assert catalog_with_pinned == catalog_without_pinned

    def test_catalog_has_no_pinned_attribute(self, tmp_path):
        """`status="pinned"` in the XML would couple the catalog to pinned
        state and silently re-introduce the cache-invalidation bug."""
        skills_dir = _make_skills_dir(tmp_path, [("alpha", "x")])
        mgr = SkillManager(skills_dir)
        mgr.discover()
        mgr.pin("alpha")
        catalog = mgr.build_catalog_block()
        assert "status=" not in catalog, (
            "Catalog leaks pin state via a `status=` attribute. "
            "Pinned state belongs in build_pinned_bodies_block(), not here."
        )

    def test_catalog_changes_when_skills_dir_changes(self, tmp_path):
        """Sanity check: the catalog SHOULD change when the disk inventory
        changes — otherwise it'd be a useless constant."""
        skills_dir = _make_skills_dir(tmp_path, [("alpha", "x")])
        mgr = SkillManager(skills_dir)
        mgr.discover()
        catalog_one = mgr.build_catalog_block()

        # Add a second skill on disk and rediscover.
        sdir = tmp_path / "skills" / "beta"
        sdir.mkdir()
        (sdir / "SKILL.md").write_text(
            "---\nname: beta\ndescription: y\n---\n\nBody.\n"
        )
        mgr.discover()
        catalog_two = mgr.build_catalog_block()

        assert catalog_one != catalog_two
        assert "beta" in catalog_two


# ─── pinned bodies isolation ─────────────────────────────────────────────


class TestPinnedBodiesAreSeparate:
    """build_pinned_bodies_block must contain ONLY pinned-skill bodies, and
    must be empty when nothing is pinned."""

    def test_empty_when_no_skills_pinned(self, tmp_path):
        skills_dir = _make_skills_dir(tmp_path, [("alpha", "x")])
        mgr = SkillManager(skills_dir)
        mgr.discover()
        assert mgr.build_pinned_bodies_block() == ""

    def test_contains_only_pinned_skills(self, tmp_path):
        skills_dir = _make_skills_dir(tmp_path, [
            ("alpha", "first"),
            ("beta", "second"),
        ])
        mgr = SkillManager(skills_dir)
        mgr.discover()
        mgr.pin("alpha")
        bodies = mgr.build_pinned_bodies_block()

        assert 'name="alpha"' in bodies
        assert 'name="beta"' not in bodies

    def test_contains_no_catalog(self, tmp_path):
        """Bodies block must not duplicate the catalog — catalog goes in
        the system prompt, bodies ride with the user message."""
        skills_dir = _make_skills_dir(tmp_path, [("alpha", "x")])
        mgr = SkillManager(skills_dir)
        mgr.discover()
        mgr.pin("alpha")
        bodies = mgr.build_pinned_bodies_block()
        assert "<available_skills>" not in bodies


# ─── full agent system prompt invariant ──────────────────────────────────


class TestAgentSystemPromptIsStable:
    """The end-to-end invariant: build_system_prompt must produce
    byte-identical output before and after a skill is pinned."""

    def test_system_prompt_byte_identical_across_pin(
        self, tmp_path, monkeypatch
    ):
        skills_dir = _make_skills_dir(tmp_path, [
            ("alpha", "First skill"),
            ("beta", "Second skill"),
        ])
        mgr = SkillManager(skills_dir)
        mgr.discover()

        # Stub out memory loading so the test is independent of the user's
        # real MEMORY.md / topic files.
        from jyagent import system_prompt as sp
        monkeypatch.setattr(
            sp, "build_memory_context", lambda query=None: ""
        )

        prompt_before = sp.build_system_prompt(
            "any user input", mgr, force_rebuild=True,
        )
        mgr.pin("alpha")
        prompt_after = sp.build_system_prompt(
            "any user input", mgr, force_rebuild=True,
        )

        assert prompt_before == prompt_after, (
            "build_system_prompt changed after pinning a skill — "
            "this is the cache-invalidation regression we just fixed. "
            "Pinned bodies must be attached to the last user message, "
            "NOT to the system prompt."
        )


# ─── no-router invariant ─────────────────────────────────────────────────


# Note: there is no automatic per-turn skill router by design.  Skills
# are loaded one-shot by the LLM via manage_skills(action='load') or
# session-pinned by the user via `/skill` (which calls mgr.pin()).
# The opt-in SKILL_PRE_ROUTER and its env-var ladder (env_router_llm /
# _route_keywords) was removed 2026-05.  Eval tooling for "would query X
# trigger skill Y?" is self-contained in skills/create-skill/scripts/test_trigger.py.


# ─── teardown: restore the modules after env-var reload tests ───────────


@pytest.fixture(autouse=True, scope="module")
def _reset_modules_after_module():
    """Safety net: if any test here reloads jyagent.config or jyagent.skills
    (the TestPreRouterEnvVar suite did), make sure we reset them to their
    on-disk state when this module finishes so other tests aren't poisoned
    by leftover env settings."""
    yield
    import jyagent.config as cfg_mod
    import jyagent.skills as skills_mod
    importlib.reload(cfg_mod)
    importlib.reload(skills_mod)
