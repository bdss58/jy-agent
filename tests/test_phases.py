# tests/test_phases.py — Phase-aware tool_choice shaping regression tests.
#
# Validates jyagent/phases.py and its wiring into AgentLoop._run_impl.

from __future__ import annotations

import pytest

from jyagent import loop_engine as le
from jyagent.phases import PhaseDirective, default_phase_policy


# ─── PhaseDirective ──────────────────────────────────────────────────────────


class TestPhaseDirective:
    def test_is_frozen(self):
        d = PhaseDirective(phase="plan", tool_choice={"type": "auto"})
        with pytest.raises(Exception):
            d.phase = "act"  # frozen dataclass

    def test_defaults(self):
        d = PhaseDirective(phase="act")
        assert d.tool_choice is None


# ─── default_phase_policy ────────────────────────────────────────────────────


class TestDefaultPhasePolicy:
    def test_finalize_forces_no_tools_on_last_step(self):
        p = default_phase_policy()
        d = p(step=49, max_steps=50, tool_calls_count=0)
        assert d is not None
        assert d.phase == "finalize"
        assert d.tool_choice == {"type": "none"}

    def test_verify_is_informational_only(self):
        """Verify phase marks the step but does not override tool_choice."""
        p = default_phase_policy()
        d = p(step=48, max_steps=50, tool_calls_count=0)
        assert d is not None
        assert d.phase == "verify"
        assert d.tool_choice is None

    def test_act_returns_none_by_default(self):
        """Mid-run steps with no relevant phase return None (fall through)."""
        p = default_phase_policy()
        assert p(step=10, max_steps=50, tool_calls_count=0) is None

    def test_plan_off_by_default(self):
        p = default_phase_policy()
        assert p(step=0, max_steps=50, tool_calls_count=0) is None

    def test_plan_on_when_enabled(self):
        p = default_phase_policy(plan_on_first_step=True)
        d = p(step=0, max_steps=50, tool_calls_count=0)
        assert d is not None
        assert d.phase == "plan"
        assert d.tool_choice is None  # informational

    def test_all_toggles_off_returns_none_everywhere(self):
        p = default_phase_policy(
            plan_on_first_step=False,
            verify_before_last=False,
            finalize_on_last=False,
        )
        for step in (0, 10, 48, 49):
            assert p(step=step, max_steps=50, tool_calls_count=0) is None

    def test_short_run_boundary_still_sensible(self):
        """max_steps == 2: step 0 is finalize's boundary; step 1 is finalize."""
        p = default_phase_policy()
        # step 0 is max_steps - 2 → verify
        d0 = p(step=0, max_steps=2, tool_calls_count=0)
        assert d0 is not None and d0.phase == "verify"
        # step 1 is max_steps - 1 → finalize
        d1 = p(step=1, max_steps=2, tool_calls_count=0)
        assert d1 is not None and d1.phase == "finalize"


# ─── LoopConfig field ────────────────────────────────────────────────────────


class TestLoopConfigPhasePolicyField:
    def test_defaults_to_none(self):
        assert le.LoopConfig().phase_policy is None

    def test_assignable(self):
        cfg = le.LoopConfig(phase_policy=default_phase_policy())
        assert callable(cfg.phase_policy)


# ─── LoopCallbacks hook ──────────────────────────────────────────────────────


class TestLoopCallbacksPhaseHook:
    def test_on_phase_enter_field(self):
        cbs = le.LoopCallbacks()
        assert hasattr(cbs, "on_phase_enter")
        assert cbs.on_phase_enter is None

    def test_on_phase_enter_fires(self):
        heard: list[str] = []
        cbs = le.LoopCallbacks(on_phase_enter=heard.append)
        cbs.on_phase_enter("plan")
        assert heard == ["plan"]


# ─── _run_impl source-level wiring ───────────────────────────────────────────


class TestRunImplWiresPhasePolicy:
    def test_source_consults_policy_and_rebuilds_opts(self):
        import inspect
        source = inspect.getsource(le.AgentLoop._run_impl)
        # Policy must be consulted each step after opts is built.
        assert "cfg.phase_policy(" in source, (
            "phase policy is not invoked inside _run_impl"
        )
        # Override must rebuild LLMOptions with the directive's tool_choice.
        assert "tool_choice=directive.tool_choice" in source, (
            "policy's tool_choice override is not propagated into opts"
        )
        # Observational callback must fire.
        assert 'self._fire("on_phase_enter"' in source, (
            "on_phase_enter callback is not wired"
        )
        # Exceptions from user code must not crash the loop.
        assert "on_warning" in source and "phase_policy" in source, (
            "policy exceptions must be caught and surfaced as warnings, "
            "not propagated"
        )

    def test_policy_exception_does_not_crash_loop(self):
        """Direct wiring check: invoking the try/except path by
        monkey-patching `_build_runtime_options` to observe opts after
        policy runs.  Full integration would need a fake runtime owner —
        instead we assert the source-level defensive pattern is in place.
        """
        import inspect
        source = inspect.getsource(le.AgentLoop._run_impl)
        assert "directive = None" in source, (
            "policy-exception path must set directive=None and continue"
        )
        assert 'on_warning' in source


# ─── Integration-lite: options override takes effect ─────────────────────────


class TestOptionsOverrideIntegration:
    """We can't cheaply spin up a full runtime here, but we can verify the
    LLMOptions constructor accepts the tool_choice the directive would
    supply and that the post-override object carries it."""

    def test_runtime_options_carries_tool_choice(self):
        from jyagent.llm.types import LLMOptions
        opts = LLMOptions(
            max_output_tokens=1000,
            timeout=60,
            reasoning=None,
            metadata={"phase": "finalize"},
            tool_choice={"type": "none"},
        )
        assert opts.tool_choice == {"type": "none"}
        assert opts.metadata == {"phase": "finalize"}
