# test_tool_pre_execute_gate.py — Tests for the on_tool_pre_execute approval gate.
#
# Covers:
#   * gate returning "deny"  → tool is NOT executed; denial result appended;
#     on_tool_start / on_tool_end stay paired (UI spinner leak prevention).
#   * gate returning "allow" → tool runs normally.
#   * mixed batch (one denied, one allowed) → only allowed dispatches.
#   * gate raising an exception → engine treats it as default (allow).
#
# Pattern follows tests/test_step_runner.py: a tiny FakeLoop + mock.patch on
# tool_executor.execute_tools.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest import mock

from jyagent.runtime.loop.callbacks import LoopCallbacks
from jyagent.runtime.loop.config import LoopConfig
from jyagent.runtime.loop.llm_types import ModelSpec
from jyagent.runtime.loop.step import RunState, StepBreak, StepContinue, run_step
from jyagent.runtime.tools.result import ToolResult


# ─── Test doubles (mirroring tests/test_step_runner.py) ──────────────────────


@dataclass
class _FakeOwner:
    model_spec: ModelSpec = field(
        default_factory=lambda: ModelSpec(provider="anthropic", model="claude-sonnet-4-6")
    )


class FakeLoop:
    """Minimal AgentLoop look-alike supporting both _fire and _fire_with_return."""

    def __init__(self, *, callbacks: LoopCallbacks, llm_response: tuple):
        self._config = LoopConfig(
            max_steps=5, streaming=False, compact_messages=False,
            todos_enabled=False, fallback_on_max_steps=False,
        )
        self._callbacks = callbacks
        self._runtime_owner = _FakeOwner()
        self._model_spec: ModelSpec | None = None
        self._cancel = False
        self._tool_source = None
        self._executor = None
        self._cancel_event = None
        self._partial_side_effects: list = []
        self._todos: list = []
        self._run_id: str = ""
        self._session_id: str = ""
        self.llm_response = llm_response
        self.fired: list[tuple] = []

    def _is_cancelled(self) -> bool:
        return self._cancel

    def _fire(self, name: str, *args: Any) -> None:
        self.fired.append((name, args))
        cb = getattr(self._callbacks, name, None)
        if cb is not None:
            try:
                cb(*args)
            except Exception:
                pass

    def _fire_with_return(self, name: str, *args: Any, default: Any = None) -> Any:
        self.fired.append((name, args))
        cb = getattr(self._callbacks, name, None)
        if cb is None:
            return default
        try:
            return cb(*args)
        except Exception:
            return default

    def _call_llm_with_retry(self, context, opts, step):
        return self.llm_response

    def _call_complete(self, context, opts):
        return self.llm_response

    def _call_streaming(self, context, opts):
        return self.llm_response

    def _write_checkpoint(self, **kwargs) -> None:
        pass


def _build_state(loop, *, messages=None) -> RunState:
    state = RunState.prepare_for_run(
        loop, system_prompt="sys", messages=messages or [], initial_todos=None,
    )
    state.step = 0
    return state


def _one_tool_call(name="writer", call_id="c1", tool_input=None):
    """Build an LLM response that requests a single tool call."""
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


def _two_tool_calls():
    """Build an LLM response that requests two tool calls in one batch."""
    from jyagent.runtime.loop.engine import ToolCallRequest
    blocks = [
        ToolCallRequest(id="c1", name="reader", input={"path": "/x"}),
        ToolCallRequest(id="c2", name="writer", input={"path": "/y"}),
    ]
    msg = {
        "content": [
            {"type": "text", "text": ""},
            {"type": "tool_use", "id": "c1", "name": "reader", "input": {"path": "/x"}},
            {"type": "tool_use", "id": "c2", "name": "writer", "input": {"path": "/y"}},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    return ("", blocks, "tool_use", msg)


# ─── Tests ───────────────────────────────────────────────────────────────────


class TestPreExecuteGate:

    def test_deny_skips_tool_and_pairs_callbacks(self):
        """deny → executor not called; denial tool_result appended; pairs balanced."""
        cbs = LoopCallbacks(
            on_tool_pre_execute=lambda name, inp: "deny",
        )
        loop = FakeLoop(callbacks=cbs, llm_response=_one_tool_call("writer"))
        state = _build_state(loop)

        with mock.patch(
            "jyagent.runtime.loop.tool_executor.execute_tools",
            return_value=[(loop.llm_response[1][0], ToolResult(content="should-not-run"))],
        ) as mock_exec:
            outcome = run_step(loop, state)

        # Executor never invoked.
        assert mock_exec.call_count == 0

        # Outcome: continue (not break/terminate).
        assert isinstance(outcome, StepContinue)

        # Callback pairs balanced.
        names = [n for (n, _) in loop.fired]
        assert names.count("on_tool_start") == 1
        assert names.count("on_tool_pre_execute") == 1
        assert names.count("on_tool_end") == 1
        assert names.index("on_tool_start") < names.index("on_tool_pre_execute")
        assert names.index("on_tool_pre_execute") < names.index("on_tool_end")

        # Tool_result message appended with denial content.
        results = [m for m in state.messages if m.get("role") == "tool_result"]
        assert len(results) == 1
        assert results[0]["is_error"] is True
        assert "Denied" in results[0]["content"]

    def test_allow_runs_tool_normally(self):
        """allow → executor invoked; result appended as normal."""
        cbs = LoopCallbacks(
            on_tool_pre_execute=lambda name, inp: "allow",
        )
        loop = FakeLoop(callbacks=cbs, llm_response=_one_tool_call("reader"))
        state = _build_state(loop)

        with mock.patch(
            "jyagent.runtime.loop.tool_executor.execute_tools",
            return_value=[(loop.llm_response[1][0], ToolResult(content="contents"))],
        ) as mock_exec:
            run_step(loop, state)

        assert mock_exec.call_count == 1
        results = [m for m in state.messages if m.get("role") == "tool_result"]
        assert len(results) == 1
        assert results[0]["is_error"] is False
        assert "contents" in results[0]["content"]

    def test_no_callback_defaults_to_allow(self):
        """When on_tool_pre_execute is None the engine runs the tool normally."""
        cbs = LoopCallbacks()  # no gate
        loop = FakeLoop(callbacks=cbs, llm_response=_one_tool_call("reader"))
        state = _build_state(loop)

        with mock.patch(
            "jyagent.runtime.loop.tool_executor.execute_tools",
            return_value=[(loop.llm_response[1][0], ToolResult(content="ok"))],
        ) as mock_exec:
            run_step(loop, state)

        assert mock_exec.call_count == 1
        results = [m for m in state.messages if m.get("role") == "tool_result"]
        assert len(results) == 1
        assert results[0]["is_error"] is False

    def test_mixed_batch_only_denied_skipped(self):
        """Two-tool batch: deny one, allow the other → executor sees only the allowed one."""
        # Deny by name: writer denied, reader allowed.
        cbs = LoopCallbacks(
            on_tool_pre_execute=lambda name, inp: "deny" if name == "writer" else "allow",
        )
        loop = FakeLoop(callbacks=cbs, llm_response=_two_tool_calls())
        state = _build_state(loop)

        # Mock executor to return one ToolResult for the (single) allowed block.
        # We'll capture the input list to assert what was passed.
        captured = {}
        reader_block = loop.llm_response[1][0]  # the reader

        def _fake_execute(blocks, *args, **kwargs):
            captured["blocks"] = list(blocks)
            return [(reader_block, ToolResult(content="read-ok"))]

        with mock.patch(
            "jyagent.runtime.loop.tool_executor.execute_tools", side_effect=_fake_execute,
        ):
            run_step(loop, state)

        # Executor saw only the allowed (reader) block.
        assert len(captured["blocks"]) == 1
        assert captured["blocks"][0].name == "reader"

        # Two tool_result messages: one denial (writer), one success (reader).
        results = [m for m in state.messages if m.get("role") == "tool_result"]
        assert len(results) == 2
        by_name = {r["tool_name"]: r for r in results}
        assert "Denied" in by_name["writer"]["content"]
        assert by_name["writer"]["is_error"] is True
        assert "read-ok" in by_name["reader"]["content"]
        assert by_name["reader"]["is_error"] is False

    def test_raising_callback_treated_as_allow(self):
        """A buggy gate must not crash the engine — default to allow."""
        def _boom(name, inp):
            raise RuntimeError("gate exploded")

        cbs = LoopCallbacks(on_tool_pre_execute=_boom)
        loop = FakeLoop(callbacks=cbs, llm_response=_one_tool_call("reader"))
        state = _build_state(loop)

        with mock.patch(
            "jyagent.runtime.loop.tool_executor.execute_tools",
            return_value=[(loop.llm_response[1][0], ToolResult(content="still-ran"))],
        ) as mock_exec:
            run_step(loop, state)

        # Default = allow → tool ran.
        assert mock_exec.call_count == 1
        results = [m for m in state.messages if m.get("role") == "tool_result"]
        assert "still-ran" in results[0]["content"]
