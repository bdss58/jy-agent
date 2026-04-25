"""Regression tests for the LLM-based skill router (jyagent/skills.py).

Covers the bugs found on 2025-11-21:
  1. `complete_text` unconditionally injected adaptive thinking, causing
     `validate_anthropic_reasoning` to raise ValueError for any Anthropic model
     below Claude 4.6. The router caught this in a blanket `except Exception`
     and silently fell back to keyword matching — making the router look
     working while actually never executing against the model.

Fix:
  - `complete_text` gained a `reasoning` kwarg with sentinel default.
     Callers can pass `reasoning=None` to explicitly disable reasoning.
  - Router passes `reasoning=None` (it's a cheap routing call).
  - Router logs a visible warning on failure instead of silently swallowing.

These tests use a mock adapter — no real API calls.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from jyagent.llm.core import (
    RuntimeOwner,
    register_adapter,
    _ADAPTERS,
)
from jyagent.llm.types import ModelSpec
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


def _register_mock_adapter(provider: str, response_text: str):
    """Register a mock runtime adapter returning `response_text`."""
    mock_adapter = MagicMock()
    mock_adapter.provider = provider
    mock_adapter.api_name = f"{provider}-mock"
    mock_adapter.complete.return_value = {
        "role": "assistant",
        "content": [{"type": "text", "text": response_text}],
        "provider": provider,
        "model": "mock-model",
        "stop_reason": "stop",
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }
    register_adapter(mock_adapter)
    return mock_adapter


@pytest.fixture
def adapter_cleanup():
    saved = dict(_ADAPTERS)
    yield
    _ADAPTERS.clear()
    _ADAPTERS.update(saved)


# ─── core fix: reasoning override on complete_text ───────────────────────

class TestCompleteTextReasoningOverride:
    """complete_text must accept reasoning=None to skip env-derived reasoning."""

    def test_reasoning_none_disables_auto_derivation(self, adapter_cleanup):
        mock_adapter = _register_mock_adapter("rt_none", "ok")
        with patch("jyagent.llm.core.build_model_spec") as mock_bms, \
             patch("jyagent.llm.core.get_reasoning_config_for_provider") as mock_grc:
            mock_bms.return_value = ModelSpec(provider="rt_none", model="m")
            owner = RuntimeOwner(ModelSpec(provider="rt_none", model="m"))
            result = owner.complete_text("hi", reasoning=None)
        assert result == "ok"
        # Critical: when caller passes reasoning=None, we must NOT call
        # get_reasoning_config_for_provider (that's what was blowing up).
        mock_grc.assert_not_called()
        # Adapter.complete is called positionally (model_spec, context, options);
        # the third positional is our RuntimeOptions with reasoning=None.
        args, _ = mock_adapter.complete.call_args
        assert len(args) >= 3
        assert args[2].reasoning is None

    def test_default_still_auto_derives(self, adapter_cleanup):
        """Backward compat: no kwarg == auto-derive from env, as before."""
        _register_mock_adapter("rt_auto", "ok")
        with patch("jyagent.llm.core.build_model_spec") as mock_bms, \
             patch("jyagent.llm.core.get_reasoning_config_for_provider") as mock_grc:
            mock_bms.return_value = ModelSpec(provider="rt_auto", model="m")
            mock_grc.return_value = None
            owner = RuntimeOwner(ModelSpec(provider="rt_auto", model="m"))
            owner.complete_text("hi")
        mock_grc.assert_called_once()

    def test_reasoning_none_survives_broken_reasoning_config(self, adapter_cleanup):
        """Simulates the real bug: auto-derive would raise, but reasoning=None bypasses it."""
        _register_mock_adapter("rt_broken", "ok")
        with patch("jyagent.llm.core.build_model_spec") as mock_bms, \
             patch("jyagent.llm.core.get_reasoning_config_for_provider",
                   side_effect=ValueError("adaptive thinking not supported")):
            mock_bms.return_value = ModelSpec(provider="rt_broken", model="m")
            owner = RuntimeOwner(ModelSpec(provider="rt_broken", model="m"))
            # With reasoning=None, must not touch the broken config fn.
            result = owner.complete_text("hi", reasoning=None)
            assert result == "ok"


# ─── skill router behavior ───────────────────────────────────────────────

class TestSkillRouterLLM:
    """_route_llm against a mock runtime — this would have caught the original bug."""

    def test_llm_router_returns_parsed_skills(self, tmp_path, adapter_cleanup):
        skills_dir = _make_skills_dir(tmp_path, [
            ("web-search", "Use for web queries and current info"),
            ("browser-automation", "Automate web browser interactions"),
            ("deep-research", "Comprehensive multi-source research"),
        ])
        _register_mock_adapter("rt_llm", '["web-search"]')

        with patch("jyagent.skills.get_skill_router_model_spec") as mock_spec, \
             patch("jyagent.llm.core.build_model_spec") as mock_bms, \
             patch("jyagent.llm.core.get_reasoning_config_for_provider") as mock_grc:
            spec = ModelSpec(provider="rt_llm", model="mock-model")
            mock_spec.return_value = spec
            mock_bms.return_value = spec
            mock_grc.return_value = None

            mgr = SkillManager(skills_dir)
            mgr.discover()
            owner = RuntimeOwner(spec)
            result = mgr._route_llm("what's the latest python release?",
                                    runtime_owner=owner)

        assert result == ["web-search"]
        assert mgr.get_active_skills() == ["web-search"]

    def test_llm_router_disables_reasoning_to_avoid_validation_error(
        self, tmp_path, adapter_cleanup
    ):
        """The router MUST pass reasoning=None so env-derived thinking config
        doesn't break cheap Anthropic models (e.g. Haiku).
        """
        skills_dir = _make_skills_dir(tmp_path, [
            ("web-search", "Use for web queries"),
        ])
        mock_adapter = _register_mock_adapter("rt_reasoning_check", '["web-search"]')

        with patch("jyagent.skills.get_skill_router_model_spec") as mock_spec, \
             patch("jyagent.llm.core.build_model_spec") as mock_bms, \
             patch("jyagent.llm.core.get_reasoning_config_for_provider") as mock_grc:
            spec = ModelSpec(provider="rt_reasoning_check", model="mock-model")
            mock_spec.return_value = spec
            mock_bms.return_value = spec
            # Make auto-derive raise — this is what Haiku was doing in prod.
            mock_grc.side_effect = ValueError("adaptive thinking not supported")

            mgr = SkillManager(skills_dir)
            mgr.discover()
            owner = RuntimeOwner(spec)
            result = mgr._route_llm("python release", runtime_owner=owner)

        # Must succeed despite broken reasoning config, because router passes
        # reasoning=None.
        assert result == ["web-search"]
        # And confirm the adapter received reasoning=None in its options
        # (positional arg 2).
        args, _ = mock_adapter.complete.call_args
        assert args[2].reasoning is None

    def test_llm_router_can_deactivate_skills(self, tmp_path, adapter_cleanup):
        """Router evaluates full catalog — can REMOVE stale active skills."""
        skills_dir = _make_skills_dir(tmp_path, [
            ("web-search", "Use for web queries"),
            ("browser-automation", "Automate browsers"),
        ])
        _register_mock_adapter("rt_deact", '["browser-automation"]')

        with patch("jyagent.skills.get_skill_router_model_spec") as mock_spec, \
             patch("jyagent.llm.core.build_model_spec") as mock_bms, \
             patch("jyagent.llm.core.get_reasoning_config_for_provider") as mock_grc:
            spec = ModelSpec(provider="rt_deact", model="mock-model")
            mock_spec.return_value = spec
            mock_bms.return_value = spec
            mock_grc.return_value = None

            mgr = SkillManager(skills_dir)
            mgr.discover()
            mgr.activate("web-search")  # stale active
            owner = RuntimeOwner(spec)
            result = mgr._route_llm("screenshot google", runtime_owner=owner)

        assert result == ["browser-automation"]
        assert "web-search" not in mgr.get_active_skills()

    def test_llm_router_ignores_unknown_skill_names(self, tmp_path, adapter_cleanup):
        skills_dir = _make_skills_dir(tmp_path, [
            ("web-search", "Use for web queries"),
        ])
        _register_mock_adapter(
            "rt_unknown",
            '["web-search", "nonexistent-skill", "another-fake"]',
        )

        with patch("jyagent.skills.get_skill_router_model_spec") as mock_spec, \
             patch("jyagent.llm.core.build_model_spec") as mock_bms, \
             patch("jyagent.llm.core.get_reasoning_config_for_provider") as mock_grc:
            spec = ModelSpec(provider="rt_unknown", model="mock-model")
            mock_spec.return_value = spec
            mock_bms.return_value = spec
            mock_grc.return_value = None

            mgr = SkillManager(skills_dir)
            mgr.discover()
            owner = RuntimeOwner(spec)
            result = mgr._route_llm("python release", runtime_owner=owner)

        assert result == ["web-search"]

    def test_llm_router_strips_markdown_fences(self, tmp_path, adapter_cleanup):
        skills_dir = _make_skills_dir(tmp_path, [
            ("web-search", "Use for web queries"),
        ])
        _register_mock_adapter("rt_fence", '```json\n["web-search"]\n```')

        with patch("jyagent.skills.get_skill_router_model_spec") as mock_spec, \
             patch("jyagent.llm.core.build_model_spec") as mock_bms, \
             patch("jyagent.llm.core.get_reasoning_config_for_provider") as mock_grc:
            spec = ModelSpec(provider="rt_fence", model="mock-model")
            mock_spec.return_value = spec
            mock_bms.return_value = spec
            mock_grc.return_value = None

            mgr = SkillManager(skills_dir)
            mgr.discover()
            owner = RuntimeOwner(spec)
            result = mgr._route_llm("python release", runtime_owner=owner)

        assert result == ["web-search"]

    def test_llm_router_surfaces_errors_on_stderr(
        self, tmp_path, adapter_cleanup, capsys
    ):
        """Router must log a visible warning when complete_text raises — the
        original bug hid these silently.
        """
        skills_dir = _make_skills_dir(tmp_path, [
            ("web-search", "Use for web queries"),
        ])
        mock_adapter = _register_mock_adapter("rt_err", "ignored")
        mock_adapter.complete.side_effect = RuntimeError("boom")

        with patch("jyagent.skills.get_skill_router_model_spec") as mock_spec, \
             patch("jyagent.llm.core.build_model_spec") as mock_bms, \
             patch("jyagent.llm.core.get_reasoning_config_for_provider") as mock_grc:
            spec = ModelSpec(provider="rt_err", model="mock-model")
            mock_spec.return_value = spec
            mock_bms.return_value = spec
            mock_grc.return_value = None

            mgr = SkillManager(skills_dir)
            mgr.discover()
            owner = RuntimeOwner(spec)
            result = mgr._route_llm("anything", runtime_owner=owner)

        assert result is None
        captured = capsys.readouterr()
        # Must mention router failure + exception class + the error message.
        assert "Skill router failed" in captured.err
        assert "RuntimeError" in captured.err
        assert "boom" in captured.err

    def test_llm_router_returns_none_on_invalid_json(self, tmp_path, adapter_cleanup):
        skills_dir = _make_skills_dir(tmp_path, [
            ("web-search", "Use for web queries"),
        ])
        _register_mock_adapter("rt_badjson", "not json at all")

        with patch("jyagent.skills.get_skill_router_model_spec") as mock_spec, \
             patch("jyagent.llm.core.build_model_spec") as mock_bms, \
             patch("jyagent.llm.core.get_reasoning_config_for_provider") as mock_grc:
            spec = ModelSpec(provider="rt_badjson", model="mock-model")
            mock_spec.return_value = spec
            mock_bms.return_value = spec
            mock_grc.return_value = None

            mgr = SkillManager(skills_dir)
            mgr.discover()
            owner = RuntimeOwner(spec)
            result = mgr._route_llm("anything", runtime_owner=owner)

        assert result is None


# ─── end-to-end: auto_activate_for_query uses LLM router when available ──

class TestAutoActivateForQueryUsesLLM:
    def test_llm_success_path_is_taken(self, tmp_path, adapter_cleanup):
        skills_dir = _make_skills_dir(tmp_path, [
            ("web-search", "Use for web queries"),
            ("browser-automation", "Automate browsers"),
        ])
        _register_mock_adapter("rt_e2e", '["browser-automation"]')

        with patch("jyagent.skills.get_skill_router_model_spec") as mock_spec, \
             patch("jyagent.llm.core.build_model_spec") as mock_bms, \
             patch("jyagent.llm.core.get_reasoning_config_for_provider") as mock_grc:
            spec = ModelSpec(provider="rt_e2e", model="mock-model")
            mock_spec.return_value = spec
            mock_bms.return_value = spec
            mock_grc.return_value = None

            mgr = SkillManager(skills_dir)
            mgr.discover()
            owner = RuntimeOwner(spec)
            result = mgr.auto_activate_for_query(
                "screenshot google",
                runtime_owner=owner,
            )

        assert result == ["browser-automation"]
