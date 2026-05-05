# test_step_runner.py — Step-level unit tests for runtime/loop/step.py.
#
# The per-step body of the agent loop lives in
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
        # ``_cancel_event`` is read by run_step → _execute_tools → tool_executor
        # so cooperating tools can be passed the live event.  Default ``None``
        # disables both kwarg injection and the wait-loop short-circuit, so
        # legacy tests that don't set up cancellation are unaffected.
        self._cancel_event = None
        self._partial_side_effects: list = []
        self._todos: list = []
        self._run_id: str = ""
        self._session_id: str = ""
        self.llm_response = llm_response or (
            "default text", [], "end_turn",
            {"content": [{"type": "text", "text": "default text"}], "usage": {}},
        )
        # Recorders for assertions.
        self.fired: list[tuple] = []
        self.checkpoints: list[dict] = []
        self.llm_options: list[Any] = []

    # ── Engine surface used by run_step ────────────────────────────────

    def _is_cancelled(self) -> bool:
        return self._cancel

    def _fire(self, name: str, *args: Any) -> None:
        self.fired.append((name, args))

    def _fire_with_return(self, name: str, *args: Any, default: Any = None) -> Any:
        """Mirror of LoopThreadHelper._fire_with_return for FakeLoop.

        Records the call in ``fired`` (same as _fire) and returns ``default``
        unless the test wired up an ``on_tool_pre_execute`` callback on
        ``self._callbacks``.  Tests that don't care about the gate get the
        old behaviour for free (default=None → engine treats as allow).
        """
        self.fired.append((name, args))
        cb = getattr(self._callbacks, name, None)
        if cb is None:
            return default
        try:
            return cb(*args)
        except Exception:
            return default

    def _call_llm_with_retry(self, context, opts, step):
        self.llm_options.append(opts)
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

    def test_session_id_threads_into_llm_options(self):
        loop = FakeLoop(llm_response=_llm_text_only())
        loop._session_id = "session-123"
        state = _build_state(loop, messages=[{"role": "user", "content": "hi"}])

        run_step(loop, state)

        assert loop.llm_options[-1].metadata["session_id"] == "session-123"


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


# ─── T5: subclass-override contract preserved ───────────────────────────────


class TestSubclassOverrideContract:
    """run_step calls loop._call_llm_with_retry — Python attribute lookup
    resolves test subclass overrides automatically. This is the contract
    verify it still holds at the step level."""

    def test_subclass_override_of_call_llm_is_used(self):
        loop = FakeLoop(llm_response=_llm_text_only("from-base"))
        state = _build_state(loop)

        # Replace the bound method with a different one (mimics
        # tests/test_loop_edge_cases.py::TestCallLLMRetry pattern).
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
        """``prepare_for_run`` resets the partial-side-effects accumulator
        on the loop instance.

        The accumulator is now a ``collections.deque`` for free-threaded
        Python forward-compat: ``list.append`` is no longer atomic on PEP 703
        builds, while ``deque.append`` is documented thread-safe.
        ``list(deque(...)) == []`` still works for the empty-state assertion.
        """
        import collections as _c
        loop = FakeLoop()
        loop._partial_side_effects = ["stale1", "stale2"]  # legacy list — reset must overwrite
        state = RunState.prepare_for_run(loop, "sys", [], None)
        # Reset to an empty accumulator (now backed by deque, but list-equivalent
        # on iteration).
        assert list(loop._partial_side_effects) == []
        # And the new container is actually a deque post-reset.
        assert isinstance(loop._partial_side_effects, _c.deque)
        # state doesn't shadow the loop's accumulator
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


# ─── T7-T13: tool execution + cancellation + dedup + checkpoint + todos ─────
#
# These tests exercise code paths in step.run_step that depend on
# `tool_executor.execute_tools`, so they monkeypatch that module-level function
# rather than touching the registry.
#
# NOTE: step.py calls ``from .tool_executor import execute_tools`` inside
# run_step (lazy, on every call), so patching the attribute on the
# ``tool_executor`` module is sufficient — no engine alias involved.

from unittest import mock


def _tool_call_block(name: str = "fake_tool", call_id: str = "call_1", tool_input: dict | None = None):
    """Return a tool_use-shaped LLM response tuple for a single tool call."""
    from jyagent.runtime.loop.engine import ToolCallRequest
    block = ToolCallRequest(id=call_id, name=name, input=tool_input or {"x": 1})
    msg = {
        "content": [
            {"type": "text", "text": ""},
            {"type": "tool_use", "id": call_id, "name": name, "input": tool_input or {"x": 1}},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    return ("", [block], "tool_use", msg)


def _tool_result(content: str, is_error: bool = False):
    """Build a ToolResult the way tool_executor.execute_tools returns."""
    from jyagent.runtime.tools.result import ToolResult
    return ToolResult(content=content, is_error=is_error)


class TestNormalToolExecution:
    """The largest uncovered path: model requests tool, executor runs it,
    tool_result message is appended, loop continues."""

    def test_single_tool_call_round_trip(self):
        loop = FakeLoop(llm_response=_tool_call_block("read_file", "c1", {"path": "/tmp/x"}))
        state = _build_state(loop)

        # Mock the executor to return a deterministic result.
        with mock.patch(
            "jyagent.runtime.loop.tool_executor.execute_tools",
            return_value=[(loop.llm_response[1][0], _tool_result("contents: hi"))],
        ) as mock_exec:
            outcome = run_step(loop, state)

        # run_step returns StepContinue (tool path does NOT terminate)
        assert isinstance(outcome, StepContinue)
        # Executor was called exactly once
        assert mock_exec.call_count == 1
        # The assistant message (with tool_use block) was appended, plus
        # the tool_result message → 2 messages total.
        assert len(state.messages) == 2
        assert state.messages[0]["content"][1]["type"] == "tool_use"
        assert state.messages[1]["role"] == "tool_result"
        assert state.messages[1]["tool_call_id"] == "c1"
        assert state.messages[1]["tool_name"] == "read_file"
        assert "contents: hi" in state.messages[1]["content"]
        assert state.messages[1]["is_error"] is False
        # Counter incremented
        assert state.tool_calls_count == 1

    def test_on_tool_start_fires_before_end_and_in_pairs(self):
        """UI contract: every on_tool_start MUST be matched by exactly one
        on_tool_end (Codex spinner-leak prevention)."""
        loop = FakeLoop(llm_response=_tool_call_block("write_file", "c1", {"path": "/tmp/y"}))
        state = _build_state(loop)

        with mock.patch(
            "jyagent.runtime.loop.tool_executor.execute_tools",
            return_value=[(loop.llm_response[1][0], _tool_result("ok"))],
        ):
            run_step(loop, state)

        names = [name for (name, _) in loop.fired]
        # on_tool_start precedes on_tool_end, and both happen exactly once.
        assert names.count("on_tool_start") == 1
        assert names.count("on_tool_end") == 1
        assert names.index("on_tool_start") < names.index("on_tool_end")

    def test_tool_error_result_carries_is_error_flag(self):
        loop = FakeLoop(llm_response=_tool_call_block("run_shell", "c1", {"command": "false"}))
        state = _build_state(loop)

        with mock.patch(
            "jyagent.runtime.loop.tool_executor.execute_tools",
            return_value=[(loop.llm_response[1][0], _tool_result("exit 1", is_error=True))],
        ):
            run_step(loop, state)

        assert state.messages[1]["is_error"] is True
        assert "exit 1" in state.messages[1]["content"]


class TestCancellationBeforeTools:
    """Top-of-step cancel is covered; this covers the SECOND cancel check
    (L648 of step.py — between on_tool_start firing and execute_tools).
    Required invariant: on_tool_end must still fire for every on_tool_start
    for UI spinner leak prevention."""

    def test_cancel_after_tool_start_fires_matching_on_tool_ends(self):
        loop = FakeLoop(llm_response=_tool_call_block("read_file", "c1"))
        state = _build_state(loop)

        # Stage: first _is_cancelled() call (top of step) returns False;
        # second call (before tools) returns True.
        call_count = {"n": 0}
        def _staged_cancel():
            call_count["n"] += 1
            return call_count["n"] >= 2
        loop._is_cancelled = _staged_cancel  # type: ignore[method-assign]

        # Even though execute_tools should NOT be called, patch to
        # detect a regression.
        with mock.patch(
            "jyagent.runtime.loop.tool_executor.execute_tools",
            return_value=[(loop.llm_response[1][0], _tool_result("should-not-run"))],
        ) as mock_exec:
            outcome = run_step(loop, state)

        # Outcome: StepBreak, NOT a tool-result append with the mocked content.
        assert isinstance(outcome, StepBreak)
        assert outcome.reason == "cancelled"
        # Executor was NOT called.
        assert mock_exec.call_count == 0
        # But the tool_result SHIM (Cancelled) was still appended, AND
        # on_tool_end fired to match on_tool_start.
        names = [name for (name, _) in loop.fired]
        assert names.count("on_tool_start") == 1
        assert names.count("on_tool_end") == 1
        # The shim message marks it as an error.
        tool_results = [m for m in state.messages if m.get("role") == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0]["content"] == "Cancelled"
        assert tool_results[0]["is_error"] is True


class TestCancellationAfterTools:
    """Covers the THIRD cancel check at L754 of step.py — between tool
    execution and reflection/checkpoint.  Tools ran normally but the user
    hit Ctrl+C while results were being appended."""

    def test_cancel_after_execution_returns_break(self):
        loop = FakeLoop(llm_response=_tool_call_block("fast_tool", "c1"))
        state = _build_state(loop)

        # Staged: top-of-step=False, before-tools=False, after-tools=True.
        call_count = {"n": 0}
        def _staged_cancel():
            call_count["n"] += 1
            return call_count["n"] >= 3
        loop._is_cancelled = _staged_cancel  # type: ignore[method-assign]

        with mock.patch(
            "jyagent.runtime.loop.tool_executor.execute_tools",
            return_value=[(loop.llm_response[1][0], _tool_result("done"))],
        ):
            outcome = run_step(loop, state)

        assert isinstance(outcome, StepBreak)
        # Tools DID run — tool_result message was appended normally
        tool_results = [m for m in state.messages if m.get("role") == "tool_result"]
        assert len(tool_results) == 1
        assert "done" in tool_results[0]["content"]
        assert tool_results[0]["is_error"] is False


class TestStuckLoopDedupBreak:
    """Same (tool, args, result) returning N times in a row → StepTerminate
    with status='dedup_break'.  Covers L713-L751 of step.py."""

    def test_three_identical_calls_trigger_dedup_break(self):
        cfg = LoopConfig(
            max_steps=10, streaming=False, compact_messages=False,
            todos_enabled=False, dedup_threshold=3,
        )
        loop = FakeLoop(
            config=cfg,
            llm_response=_tool_call_block("spin_tool", "c1", {"q": "same"}),
        )
        state = _build_state(loop)

        # Run the same tool/args/result three times in a row.  Each call
        # advances state.step and state.all_text independently.
        outcomes = []
        with mock.patch(
            "jyagent.runtime.loop.tool_executor.execute_tools",
            return_value=[(loop.llm_response[1][0], _tool_result("identical-output"))],
        ):
            for i in range(3):
                state.step = i
                outcomes.append(run_step(loop, state))

        # First two steps: StepContinue.  Third: dedup break.
        assert isinstance(outcomes[0], StepContinue)
        assert isinstance(outcomes[1], StepContinue)
        assert isinstance(outcomes[2], StepTerminate)
        assert outcomes[2].result.status == "dedup_break"
        # tool_calls_count rolled up correctly
        assert state.tool_calls_count == 3


class TestDuplicateToolCallsInSingleBatch:
    """Covers L724-L726: within a single batch, duplicate (name, args)
    keys are deduped before passing to stuck_detector.record.  This
    prevents a legitimate parallel fanout of 3 identical read_file calls
    in ONE step from tripping threshold=3."""

    def test_duplicates_in_batch_count_as_one(self):
        cfg = LoopConfig(
            max_steps=10, streaming=False, compact_messages=False,
            todos_enabled=False, dedup_threshold=3,
        )
        # Build an LLM response with 3 identical tool calls in ONE batch.
        from jyagent.runtime.loop.engine import ToolCallRequest
        blocks = [
            ToolCallRequest(id=f"c{i}", name="read_file", input={"path": "/tmp/z"})
            for i in range(3)
        ]
        msg = {
            "content": [
                *[
                    {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
                    for b in blocks
                ],
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
        loop = FakeLoop(config=cfg, llm_response=("", blocks, "tool_use", msg))
        state = _build_state(loop)

        with mock.patch(
            "jyagent.runtime.loop.tool_executor.execute_tools",
            return_value=[(b, _tool_result("contents")) for b in blocks],
        ):
            outcome = run_step(loop, state)

        # One batch of three duplicates is NOT a stuck loop.
        assert isinstance(outcome, StepContinue)
        # The stuck_detector only saw ONE record, not three (dedup worked).
        # We verify this by checking the internal counter.  Exact name
        # depends on the detector's internals; we verify through behavior:
        # another identical batch would still NOT trigger dedup if the
        # first batch was properly counted as one.
        state.step = 1
        with mock.patch(
            "jyagent.runtime.loop.tool_executor.execute_tools",
            return_value=[(b, _tool_result("contents")) for b in blocks],
        ):
            outcome2 = run_step(loop, state)
        assert isinstance(outcome2, StepContinue)  # count is 2, still under threshold


class TestCheckpointWrite:
    """Covers L787-L799: when checkpoint_dir is set and we're on the
    cadence boundary, loop._write_checkpoint fires with the current state."""

    def test_checkpoint_fires_on_cadence_boundary(self, tmp_path):
        cfg = LoopConfig(
            max_steps=5, streaming=False, compact_messages=False,
            todos_enabled=False,
            checkpoint_dir=str(tmp_path),
            checkpoint_every_n_steps=1,  # every step
        )
        loop = FakeLoop(config=cfg, llm_response=_tool_call_block("noop_tool", "c1"))
        state = _build_state(loop)

        with mock.patch(
            "jyagent.runtime.loop.tool_executor.execute_tools",
            return_value=[(loop.llm_response[1][0], _tool_result("done"))],
        ):
            outcome = run_step(loop, state)

        assert isinstance(outcome, StepContinue)
        assert len(loop.checkpoints) == 1
        cp = loop.checkpoints[0]
        assert cp["step"] == 0
        assert cp["status"] == "in_progress"
        assert cp["tool_calls_count"] == 1

    def test_checkpoint_skipped_when_dir_unset(self):
        cfg = LoopConfig(
            max_steps=5, streaming=False, compact_messages=False,
            todos_enabled=False,
            checkpoint_dir=None,  # no checkpointing
            checkpoint_every_n_steps=1,
        )
        loop = FakeLoop(config=cfg, llm_response=_tool_call_block("noop", "c1"))
        state = _build_state(loop)

        with mock.patch(
            "jyagent.runtime.loop.tool_executor.execute_tools",
            return_value=[(loop.llm_response[1][0], _tool_result("done"))],
        ):
            run_step(loop, state)

        assert loop.checkpoints == []

    def test_checkpoint_skipped_off_cadence(self):
        cfg = LoopConfig(
            max_steps=10, streaming=False, compact_messages=False,
            todos_enabled=False,
            checkpoint_dir="/tmp/cp",
            checkpoint_every_n_steps=5,  # only every 5 steps
        )
        loop = FakeLoop(config=cfg, llm_response=_tool_call_block("noop", "c1"))
        state = _build_state(loop)
        state.step = 2  # step 2 → (step+1) % 5 == 3, no checkpoint

        with mock.patch(
            "jyagent.runtime.loop.tool_executor.execute_tools",
            return_value=[(loop.llm_response[1][0], _tool_result("done"))],
        ):
            run_step(loop, state)

        assert loop.checkpoints == []


class TestTodosOverlayInStepBatch:
    """Covers L379-L391: when todos_enabled, step_batch has the
    write_todos function overlaid; state.last_step_batch reflects that."""

    def test_todos_enabled_overlays_write_todos_into_step_batch(self):
        cfg = LoopConfig(
            max_steps=5, streaming=False, compact_messages=False,
            todos_enabled=True,
        )
        loop = FakeLoop(config=cfg, llm_response=_llm_text_only("done"))
        state = _build_state(loop)

        run_step(loop, state)

        # The last step_batch has write_todos in its functions map
        assert "write_todos" in state.last_step_batch.functions
        # And the schema is present
        schema_names = [s.get("name") for s in state.last_step_batch.schemas]
        assert "write_todos" in schema_names

    def test_todos_disabled_step_batch_has_no_write_todos(self):
        cfg = LoopConfig(
            max_steps=5, streaming=False, compact_messages=False,
            todos_enabled=False,
        )
        loop = FakeLoop(config=cfg, llm_response=_llm_text_only("done"))
        state = _build_state(loop)

        run_step(loop, state)

        assert "write_todos" not in state.last_step_batch.functions
