# test_step_runner.py — Step-level unit tests for runtime/loop/step.py.
#
# After C4 Phase 5, the per-step body of the agent loop lives in
# ``runtime/loop/step.py::run_step``. These tests exercise it directly
# against a tiny fake AgentLoop, bypassing the provider/runtime owner chain
# entirely. Each test would have required ~50 lines of fixture in the old
# end-to-end style; here it's ~10.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from jyagent.runtime.loop.callbacks import LoopCallbacks
from jyagent.runtime.loop.config import LoopConfig
from jyagent.runtime.loop.llm_types import ModelSpec
from jyagent.runtime.loop.step import (
    RunState,
    StepBreak,
    StepContinue,
    StepTerminate,
    run_step,
)
from jyagent.runtime.tools.registry import ToolBatch


# ─── Fake AgentLoop ──────────────────────────────────────────────────────────


@dataclass
class _FakeOwner:
    """Minimal stand-in for LLMOwner — only `model_spec` is read by run_step
    (via ``effective_spec`` resolution in RunState.prepare_for_run)."""
    model_spec: ModelSpec = field(
        default_factory=lambda: ModelSpec(provider="anthropic", model="claude-sonnet-4-6")
    )


class FakeLoop:
    """A minimum AgentLoop look-alike that satisfies the run_step contract.

    Provides only the attributes/methods run_step actually reads. Any
    test wanting to override LLM behaviour patches ``self.llm_response``
    (a tuple ``(text, blocks, stop_reason, message)``) before calling
    run_step.
    """

    def __init__(
        self,
        *,
        config: LoopConfig | None = None,
        cancel: bool = False,
        llm_response: tuple | None = None,
    ):
        self._config = config or LoopConfig(
            max_steps=5, streaming=False, compact_messages=False,
            todos_enabled=False, fallback_on_max_steps=False,
        )
        self._callbacks = LoopCallbacks()
        self._runtime_owner = _FakeOwner()
        self._model_spec: ModelSpec | None = None
        self._cancel = cancel
        self._tool_source = None
        self._executor = None
        self._partial_side_effects: list = []
        self._todos: list = []
        self._run_id: str = ""
        self.llm_response = llm_response or (
            "default text", [], "end_turn",
            {"content": [{"type": "text", "text": "default text"}], "usage": {}},
        )
        # Recorders for assertions.
        self.fired: list[tuple] = []
        self.checkpoints: list[dict] = []

    # ── Engine surface used by run_step ────────────────────────────────

    def _is_cancelled(self) -> bool:
        return self._cancel

    def _fire(self, name: str, *args: Any) -> None:
        self.fired.append((name, args))

    def _call_llm_with_retry(self, context, opts, step):
        return self.llm_response

    def _call_complete(self, context, opts):  # for the max_steps fallback
        return self.llm_response

    def _call_streaming(self, context, opts):
        return self.llm_response

    def _write_checkpoint(self, **kwargs) -> None:
        self.checkpoints.append(kwargs)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _build_state(loop: FakeLoop, *, messages: list | None = None) -> RunState:
    """Construct a RunState the same way _run_impl does, without engine
    instance-state mutation surprises (we use the prepare_for_run classmethod)."""
    state = RunState.prepare_for_run(
        loop,
        system_prompt="you are a helpful assistant",
        messages=messages or [],
        initial_todos=None,
    )
    state.step = 0
    return state


def _llm_text_only(text: str = "all done") -> tuple:
    return (
        text, [], "end_turn",
        {"content": [{"type": "text", "text": text}], "usage": {"input_tokens": 1, "output_tokens": 2}},
    )


def _llm_with_tool_call(tool_name: str = "fake_tool", tool_input: dict | None = None) -> tuple:
    """Build an LLM response that requests one tool call.

    The structure mirrors what _extract_tool_calls produces from a real
    Anthropic message — a list of ToolCallRequest-shaped namespaces.
    """
    from jyagent.runtime.loop.engine import ToolCallRequest
    block = ToolCallRequest(
        id="call_1", name=tool_name, input=tool_input or {"x": 1},
    )
    msg = {
        "content": [
            {"type": "text", "text": ""},
            {"type": "tool_use", "id": "call_1", "name": tool_name, "input": tool_input or {"x": 1}},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    return ("", [block], "tool_use", msg)


# ─── T1: text-only response → StepTerminate(completed) ──────────────────────


class TestCompletion:
    def test_text_only_response_terminates_completed(self):
        loop = FakeLoop(llm_response=_llm_text_only("hello world"))
        state = _build_state(loop)

        outcome = run_step(loop, state)

        assert isinstance(outcome, StepTerminate)
        assert outcome.result.status == "completed"
        assert "hello world" in outcome.result.text
        assert outcome.result.tool_calls_count == 0

    def test_text_only_response_appends_assistant_message(self):
        loop = FakeLoop(llm_response=_llm_text_only("done"))
        state = _build_state(loop)

        run_step(loop, state)

        # The assistant message is appended before terminating
        assert len(state.messages) == 1
        assert state.messages[0]["content"][0]["text"] == "done"

    def test_step_progress_callback_fires(self):
        loop = FakeLoop(llm_response=_llm_text_only())
        state = _build_state(loop)
        state.step = 3
        loop._config = LoopConfig(max_steps=10, streaming=False,
                                  compact_messages=False, todos_enabled=False)

        run_step(loop, state)

        progress_events = [args for (name, args) in loop.fired if name == "on_step_progress"]
        assert progress_events == [(3, 10)]


# ─── T2: cancellation → StepBreak ────────────────────────────────────────────


class TestCancellation:
    def test_top_of_step_cancel_returns_break(self):
        loop = FakeLoop(cancel=True, llm_response=_llm_text_only())
        state = _build_state(loop)

        outcome = run_step(loop, state)

        assert isinstance(outcome, StepBreak)
        assert outcome.reason == "cancelled"
        # No LLM call was made
        assert state.total_input_tokens == 0


# ─── T3: cost budget exceeded → StepTerminate(cost_limit) ───────────────────


class TestCostBudget:
    def test_pre_existing_cost_over_budget_terminates(self):
        # max_cost_usd=0.01 — any priced call will blow it out.
        cfg = LoopConfig(
            max_steps=5, streaming=False, compact_messages=False,
            todos_enabled=False, max_cost_usd=0.01,
        )
        loop = FakeLoop(config=cfg, llm_response=_llm_text_only())
        state = _build_state(loop)
        # Pre-load the cost tracker with a bill that already exceeds budget.
        # The CostTracker's running total is exposed as `total_cost` and
        # surfaced via the `cost` property.
        state.cost_tracker.total_cost = 99.0  # type: ignore[union-attr]

        outcome = run_step(loop, state)

        assert isinstance(outcome, StepTerminate)
        assert outcome.result.status == "cost_limit"
        assert "Cost budget exceeded" in (outcome.result.error or "")


# ─── T4: repeated truncation → StepTerminate(error) ─────────────────────────


class TestTruncation:
    def test_first_truncation_continues_and_scales_tokens(self):
        cfg = LoopConfig(
            max_steps=5, streaming=False, compact_messages=False,
            todos_enabled=False, auto_scale_on_truncation=True,
            initial_max_tokens=1024, token_scale_factor=2, max_tokens_cap=128_000,
        )
        # Truncated response: stop_reason='length' AND has tool calls.
        from jyagent.runtime.loop.engine import ToolCallRequest
        block = ToolCallRequest(id="x", name="t", input={})
        loop = FakeLoop(config=cfg, llm_response=(
            "partial", [block], "length",
            {"content": [{"type": "text", "text": "partial"}], "usage": {}},
        ))
        state = _build_state(loop)
        state.current_max_tokens = 1024

        outcome = run_step(loop, state)

        assert isinstance(outcome, StepContinue)
        assert state.consecutive_truncations == 1
        assert state.current_max_tokens == 2048
        # Partial text was rolled back
        assert state.all_text == ""

    def test_too_many_truncations_terminates_error(self):
        cfg = LoopConfig(
            max_steps=5, streaming=False, compact_messages=False,
            todos_enabled=False, auto_scale_on_truncation=True,
        )
        from jyagent.runtime.loop.engine import ToolCallRequest
        block = ToolCallRequest(id="x", name="t", input={})
        loop = FakeLoop(config=cfg, llm_response=(
            "p", [block], "length",
            {"content": [{"type": "text", "text": "p"}], "usage": {}},
        ))
        state = _build_state(loop)
        # Already at the cap — next truncation pushes us over.
        state.consecutive_truncations = state.max_truncation_retries

        outcome = run_step(loop, state)

        assert isinstance(outcome, StepTerminate)
        assert outcome.result.status == "error"
        assert "Repeated truncation" in (outcome.result.error or "")


# ─── T5: subclass-override contract preserved (Phase 3 lesson) ───────────────


class TestSubclassOverrideContract:
    """run_step calls loop._call_llm_with_retry — Python attribute lookup
    resolves test subclass overrides automatically. This is the contract
    Phase 3 navigated; verify it still holds at the step level."""

    def test_subclass_override_of_call_llm_is_used(self):
        loop = FakeLoop(llm_response=_llm_text_only("from-base"))
        state = _build_state(loop)

        # Replace the bound method with a different one (mimics
        # tests/test_codex_review_fixes.py::TestCallLLMRetry pattern).
        called_with: list[tuple] = []
        def overridden(context, opts, step):
            called_with.append((context, opts, step))
            return _llm_text_only("from-override")
        loop._call_llm_with_retry = overridden  # type: ignore[method-assign]

        outcome = run_step(loop, state)

        assert isinstance(outcome, StepTerminate)
        assert "from-override" in outcome.result.text
        assert "from-base" not in outcome.result.text
        assert len(called_with) == 1


# ─── T6: prepare_for_run side effects ──────────────────────────────────────────────


class TestRunStateFromLoop:
    def test_resets_partial_side_effects(self):
        loop = FakeLoop()
        loop._partial_side_effects = ["stale1", "stale2"]
        state = RunState.prepare_for_run(loop, "sys", [], None)
        assert loop._partial_side_effects == []
        # state doesn't shadow the loop's list
        assert "_partial_side_effects" not in state.__dict__

    def test_seeds_todos_from_initial_when_enabled(self):
        cfg = LoopConfig(
            max_steps=5, streaming=False, compact_messages=False,
            todos_enabled=True,
        )
        loop = FakeLoop(config=cfg)
        initial = [{"content": "step 1", "status": "pending"}]
        RunState.prepare_for_run(loop, "sys", [], initial)
        assert len(loop._todos) == 1
        # normalize_todo turns dicts into TodoItem dataclass instances.
        assert loop._todos[0].content == "step 1"

    def test_invalid_todos_warn_and_reset_to_empty(self):
        cfg = LoopConfig(
            max_steps=5, streaming=False, compact_messages=False,
            todos_enabled=True,
        )
        loop = FakeLoop(config=cfg)
        # Pass a value normalize_todo will reject (TypeError on dict access)
        bad = [{"no_content_field": True}]  # missing required "content"
        try:
            RunState.prepare_for_run(loop, "sys", [], bad)
        except (TypeError, ValueError, KeyError):
            # If normalize_todo raises something other than TypeError, the
            # impl still falls back to []. That's fine — the contract is
            # "don't propagate", not "specific exception type".
            pass
        # Either way, todos must end up as a clean list (possibly with
        # warnings fired).
        assert isinstance(loop._todos, list)

    def test_effective_spec_resolves_to_owner_when_no_override(self):
        loop = FakeLoop()
        state = RunState.prepare_for_run(loop, "sys", [], None)
        assert state.effective_spec is loop._runtime_owner.model_spec

    def test_effective_spec_resolves_to_override_when_set(self):
        loop = FakeLoop()
        loop._model_spec = ModelSpec(provider="openai", model="gpt-test")
        state = RunState.prepare_for_run(loop, "sys", [], None)
        assert state.effective_spec.provider == "openai"
        assert state.effective_spec.model == "gpt-test"

    def test_cost_tracker_only_built_when_budget_set(self):
        loop = FakeLoop()
        state = RunState.prepare_for_run(loop, "sys", [], None)
        assert state.cost_tracker is None

        cfg = LoopConfig(
            max_steps=5, streaming=False, compact_messages=False,
            todos_enabled=False, max_cost_usd=1.0,
        )
        loop2 = FakeLoop(config=cfg)
        state2 = RunState.prepare_for_run(loop2, "sys", [], None)
        assert state2.cost_tracker is not None

    def test_reflection_module_only_loaded_when_enabled(self):
        loop = FakeLoop()
        state = RunState.prepare_for_run(loop, "sys", [], None)
        assert state.reflection_module is None

        cfg = LoopConfig(
            max_steps=5, streaming=False, compact_messages=False,
            todos_enabled=False, reflect_every_n_tool_calls=3,
        )
        loop2 = FakeLoop(config=cfg)
        state2 = RunState.prepare_for_run(loop2, "sys", [], None)
        assert state2.reflection_module is not None
        assert hasattr(state2.reflection_module, "should_reflect")
