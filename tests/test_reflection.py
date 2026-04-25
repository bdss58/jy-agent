# tests/test_reflection.py — Mid-loop reflection / critic step regression tests.
#
# Validates jyagent/reflection.py and its wiring into AgentLoop._run_impl:
#   * should_reflect triggers for every-N cadence and sub-agent returns
#   * no back-to-back injection when last user msg is already a reflection
#   * build_reflection_prompt includes the marker and the 3-question check
#   * AgentLoop wiring: injects at the correct point, fires on_reflection,
#     increments last_reflection_count, respects disabled default.

from __future__ import annotations

import pytest

from jyagent.runtime.loop import engine as le
from jyagent.runtime.loop import reflection
from jyagent.runtime.loop.reflection import (
    REFLECTION_MARKER,
    SUBAGENT_TOOL_NAMES,
    build_reflection_prompt,
    should_reflect,
)


# ─── Pure-function tests ─────────────────────────────────────────────────────


class TestBuildReflectionPrompt:
    def test_contains_marker_and_all_three_questions(self):
        prompt = build_reflection_prompt("every_n", 5)
        assert prompt.startswith(REFLECTION_MARKER)
        assert "</reflection-prompt>" in prompt
        # Three core questions must all appear.
        for key in ("concrete progress", "minimum", "most efficient path"):
            assert key in prompt.lower()
        assert "5" in prompt  # tool-call count echoed

    def test_every_n_vs_after_subagent_differ_in_header(self):
        a = build_reflection_prompt("every_n", 3)
        b = build_reflection_prompt("after_subagent", 3)
        assert a != b
        assert "Sub-agent" in b
        assert "Progress check" in a

    def test_unknown_reason_falls_back_gracefully(self):
        p = build_reflection_prompt("bogus_reason", 1)
        assert REFLECTION_MARKER in p  # still well-formed


class TestShouldReflect:
    def _make_messages(self, tail_content):
        if tail_content is None:
            return []
        return [{"role": "user", "content": tail_content}]

    def test_disabled_no_triggers(self):
        inject, reason = should_reflect(
            reflect_every_n=0,
            reflect_after_subagent=False,
            tool_calls_total=100,
            tool_calls_at_last_reflection=0,
            batch_tool_names=["dispatch_agent"],
            messages=[],
        )
        assert inject is False
        assert reason == ""

    def test_cadence_triggers_at_threshold(self):
        inject, reason = should_reflect(
            reflect_every_n=3,
            reflect_after_subagent=False,
            tool_calls_total=3,
            tool_calls_at_last_reflection=0,
            batch_tool_names=["read_file"],
            messages=[],
        )
        assert inject is True
        assert reason == "every_n"

    def test_cadence_does_not_trigger_below_threshold(self):
        inject, _ = should_reflect(
            reflect_every_n=5,
            reflect_after_subagent=False,
            tool_calls_total=4,
            tool_calls_at_last_reflection=0,
            batch_tool_names=["read_file"],
            messages=[],
        )
        assert inject is False

    def test_cadence_measures_delta_since_last_reflection(self):
        # 8 total - 5 last = 3 delta, threshold 3 → fire.
        inject, _ = should_reflect(
            reflect_every_n=3,
            reflect_after_subagent=False,
            tool_calls_total=8,
            tool_calls_at_last_reflection=5,
            batch_tool_names=[],
            messages=[],
        )
        assert inject is True

    def test_subagent_trigger_fires(self):
        inject, reason = should_reflect(
            reflect_every_n=0,
            reflect_after_subagent=True,
            tool_calls_total=1,
            tool_calls_at_last_reflection=0,
            batch_tool_names=["dispatch_agent"],
            messages=[],
        )
        assert inject is True
        assert reason == "after_subagent"

    def test_subagent_trigger_preempts_cadence(self):
        """When both triggers would fire, subagent wins (richer context)."""
        inject, reason = should_reflect(
            reflect_every_n=1,
            reflect_after_subagent=True,
            tool_calls_total=10,
            tool_calls_at_last_reflection=0,
            batch_tool_names=["read_file", "dispatch_agent"],
            messages=[],
        )
        assert inject is True
        assert reason == "after_subagent"

    def test_subagent_trigger_suppressed_when_no_dispatch_in_batch(self):
        inject, _ = should_reflect(
            reflect_every_n=0,
            reflect_after_subagent=True,
            tool_calls_total=1,
            tool_calls_at_last_reflection=0,
            batch_tool_names=["read_file", "grep_files"],
            messages=[],
        )
        assert inject is False

    def test_no_back_to_back_injection_string_tail(self):
        """If the last user msg already starts with the marker, skip."""
        msgs = [{"role": "user", "content": f"{REFLECTION_MARKER}\n..."}]
        inject, _ = should_reflect(
            reflect_every_n=1,
            reflect_after_subagent=True,
            tool_calls_total=5,
            tool_calls_at_last_reflection=0,
            batch_tool_names=["dispatch_agent"],
            messages=msgs,
        )
        assert inject is False

    def test_no_back_to_back_injection_list_content_tail(self):
        """Same guard for list-of-blocks content."""
        msgs = [{
            "role": "user",
            "content": [{"type": "text", "text": f"{REFLECTION_MARKER}\nfoo"}],
        }]
        inject, _ = should_reflect(
            reflect_every_n=1,
            reflect_after_subagent=True,
            tool_calls_total=5,
            tool_calls_at_last_reflection=0,
            batch_tool_names=["dispatch_agent"],
            messages=msgs,
        )
        assert inject is False

    def test_subagent_tool_name_set_contains_dispatch_agent(self):
        """Guards against accidental renames in either direction."""
        assert "dispatch_agent" in SUBAGENT_TOOL_NAMES


# ─── AgentLoop wiring tests ──────────────────────────────────────────────────


class TestLoopConfigReflectionFields:
    def test_fields_default_disabled(self):
        cfg = le.LoopConfig()
        assert cfg.reflect_every_n_tool_calls == 0
        assert cfg.reflect_after_subagent is False


class TestLoopCallbacksReflectionHook:
    def test_on_reflection_field_exists(self):
        cbs = le.LoopCallbacks()
        assert hasattr(cbs, "on_reflection")
        assert cbs.on_reflection is None

    def test_on_reflection_field_is_assignable(self):
        fired: list[str] = []
        cbs = le.LoopCallbacks(on_reflection=fired.append)
        cbs.on_reflection("test")
        assert fired == ["test"]


class TestRunImplWiresReflection:
    """Source-level check that the reflection block is wired into _run_impl
    at the correct point (after tool execution, after stuck detection, but
    inside the for-step loop)."""

    def test_source_contains_reflection_import_and_call(self):
        import inspect
        source = inspect.getsource(le.AgentLoop._run_impl)
        assert "from . import reflection" in source, (
            "AgentLoop._run_impl must lazy-import the reflection module "
            "when either reflection config knob is enabled."
        )
        assert "reflection.should_reflect(" in source, (
            "reflection.should_reflect must be consulted each step"
        )
        assert "reflection.build_reflection_prompt(" in source, (
            "build_reflection_prompt must be used to produce the user msg"
        )
        assert 'self._fire("on_reflection"' in source, (
            "on_reflection callback must fire when an injection happens"
        )
        assert "last_reflection_count = tool_calls_count" in source, (
            "cadence counter must advance on every injection"
        )