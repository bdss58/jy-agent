"""Cache-stability invariants for the skills layer.

The MEMORY.md durable rule says: "Mutating Anthropic system_prompt breaks
prompt caching — inject dynamic context as a non-persisted tail message
block instead." Skills used to violate this — `build_prompt_context` was
called with a query every turn, ran the LLM router, then concatenated the
active-skill bodies into the system prompt. Any activation diff would
mutate the system prompt → invalidate the Anthropic prompt cache prefix
→ pay ~12× cost on cache-heavy workloads.

These tests pin the post-refactor design (Design B, progressive disclosure):

  * Stage 1 catalog (build_catalog_block) goes into the system prompt and
    is byte-stable across activation changes — only the on-disk skills/
    directory contents can change it.
  * Stage 2 active bodies (build_active_bodies_block) are emitted as a
    SEPARATE block that the agent attaches to the last user message, NOT
    to the system prompt.
  * There is NO per-turn automatic skill router — skills are activated by
    the LLM (via `manage_skills`) or the user (via `/skill`).  The previous
    opt-in `SKILL_PRE_ROUTER` flag was removed 2026-05.

Any future refactor that re-introduces activation-state into the catalog
(or pulls active bodies back into the system prompt) will fail here.
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
    """build_catalog_block must NOT depend on the active set."""

    def test_catalog_unchanged_after_activate(self, tmp_path):
        skills_dir = _make_skills_dir(tmp_path, [
            ("alpha", "First skill"),
            ("beta", "Second skill"),
        ])
        mgr = SkillManager(skills_dir)
        mgr.discover()

        catalog_before = mgr.build_catalog_block()
        assert mgr.activate("alpha") is True
        catalog_after = mgr.build_catalog_block()

        assert catalog_before == catalog_after, (
            "build_catalog_block() changed after activate() — this would "
            "invalidate the Anthropic prompt-cache prefix on every "
            "activation diff. Active state must NOT leak into the catalog."
        )

    def test_catalog_unchanged_after_deactivate(self, tmp_path):
        skills_dir = _make_skills_dir(tmp_path, [
            ("alpha", "First skill"),
            ("beta", "Second skill"),
        ])
        mgr = SkillManager(skills_dir)
        mgr.discover()
        mgr.activate("alpha")

        catalog_with_active = mgr.build_catalog_block()
        mgr.deactivate("alpha")
        catalog_without_active = mgr.build_catalog_block()

        assert catalog_with_active == catalog_without_active

    def test_catalog_has_no_active_attribute(self, tmp_path):
        """`status="active"` in the XML would couple the catalog to activation
        state and silently re-introduce the cache-invalidation bug."""
        skills_dir = _make_skills_dir(tmp_path, [("alpha", "x")])
        mgr = SkillManager(skills_dir)
        mgr.discover()
        mgr.activate("alpha")
        catalog = mgr.build_catalog_block()
        assert "status=" not in catalog, (
            "Catalog leaks activation state via a `status=` attribute. "
            "Active state belongs in build_active_bodies_block(), not here."
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


# ─── active bodies isolation ─────────────────────────────────────────────


class TestActiveBodiesAreSeparate:
    """build_active_bodies_block must contain ONLY active-skill bodies, and
    must be empty when nothing is active."""

    def test_empty_when_no_skills_active(self, tmp_path):
        skills_dir = _make_skills_dir(tmp_path, [("alpha", "x")])
        mgr = SkillManager(skills_dir)
        mgr.discover()
        assert mgr.build_active_bodies_block() == ""

    def test_contains_only_active_skills(self, tmp_path):
        skills_dir = _make_skills_dir(tmp_path, [
            ("alpha", "first"),
            ("beta", "second"),
        ])
        mgr = SkillManager(skills_dir)
        mgr.discover()
        mgr.activate("alpha")
        bodies = mgr.build_active_bodies_block()

        assert 'name="alpha"' in bodies
        assert 'name="beta"' not in bodies

    def test_contains_no_catalog(self, tmp_path):
        """Bodies block must not duplicate the catalog — catalog goes in
        the system prompt, bodies ride with the user message."""
        skills_dir = _make_skills_dir(tmp_path, [("alpha", "x")])
        mgr = SkillManager(skills_dir)
        mgr.discover()
        mgr.activate("alpha")
        bodies = mgr.build_active_bodies_block()
        assert "<available_skills>" not in bodies


# ─── full agent system prompt invariant ──────────────────────────────────


class TestAgentSystemPromptIsStable:
    """The end-to-end invariant: _build_full_system_prompt must produce
    byte-identical output before and after a skill is activated."""

    def test_system_prompt_byte_identical_across_activation(
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
        from jyagent import agent as agent_mod
        monkeypatch.setattr(
            agent_mod, "build_memory_context",
            lambda query="": "STUB MEMORY CONTEXT",
        )
        # Reset the module-level cache so the stub takes effect.
        monkeypatch.setattr(agent_mod, "_cached_memory_context", None)

        sp_before = agent_mod._build_full_system_prompt(
            "user query one", mgr, force_rebuild=True,
        )
        mgr.activate("alpha")
        sp_after_activate = agent_mod._build_full_system_prompt(
            "user query two", mgr,
        )
        mgr.activate("beta")
        sp_after_two = agent_mod._build_full_system_prompt(
            "user query three", mgr,
        )
        mgr.deactivate("alpha")
        sp_after_deactivate = agent_mod._build_full_system_prompt(
            "user query four", mgr,
        )

        assert sp_before == sp_after_activate == sp_after_two == sp_after_deactivate, (
            "_build_full_system_prompt is NOT stable across activation "
            "changes. This will invalidate the Anthropic prompt-cache "
            "prefix on every skill activation, costing ~12× per turn on "
            "cache-heavy workloads. Active skill bodies must be attached "
            "as a tail message block, not concatenated into the system prompt."
        )


# ─── env var opt-in for pre-router ───────────────────────────────────────
#
# TestPreRouterEnvVar was removed 2026-05 together with the SKILL_PRE_ROUTER
# feature itself.  The explicit eval-only routing API
# (``auto_activate_for_query``) remains and is covered by
# tests/test_skill_router.py — it is independent of any env flag.


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
