# Tests for sub-agent terminal state handling.

import os
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jyagent.registry import get_registry
from jyagent.tools.subagent import dispatch_agent
import jyagent.tools.subagent as subagent


class _DummySpinner:
    def __init__(self, *_args, **_kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass


class _FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text

    def model_dump(self, exclude_none=True):
        return {"type": "text", "text": self.text}


class _FakeToolUseBlock:
    type = "tool_use"

    def __init__(self, block_id, name, tool_input):
        self.id = block_id
        self.name = name
        self.input = tool_input

    def model_dump(self, exclude_none=True):
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }


class _FakeResponse:
    def __init__(self, content, input_tokens=0, output_tokens=0):
        self.content = content
        self.usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class _FakeMessagesAPI:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("No fake responses left for client.messages.create()")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessagesAPI(responses)


def _register_tool(name, fn):
    schema = {
        "name": name,
        "description": "test tool",
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
            },
            "required": ["value"],
        },
    }
    get_registry().register(name, fn, schema)
    return schema


@pytest.fixture(autouse=True)
def _disable_spinner(monkeypatch):
    monkeypatch.setattr(subagent, "_SubagentSpinner", _DummySpinner)


@pytest.fixture(autouse=True)
def _reset_nesting_depth():
    subagent._nesting_depth.value = 0
    yield
    subagent._nesting_depth.value = 0


class TestDispatchAgent:
    def test_completed_child_returns_success(self, monkeypatch):
        client = _FakeClient([
            _FakeResponse([_FakeTextBlock("final answer")], input_tokens=5, output_tokens=7),
        ])
        monkeypatch.setattr(subagent, "_get_client", lambda: client)

        result = dispatch_agent("summarize this", max_steps=2)

        assert result.is_error is False
        assert result.content == "final answer"

    def test_api_failure_returns_hard_error_with_partial_output(self, monkeypatch):
        tool_name = "_test_subagent_api_failure_tool"
        _register_tool(tool_name, lambda value: f"ok:{value}")
        client = _FakeClient([
            _FakeResponse(
                [
                    _FakeTextBlock("working notes"),
                    _FakeToolUseBlock("tool-1", tool_name, {"value": "x"}),
                ],
                input_tokens=3,
                output_tokens=4,
            ),
            RuntimeError("backend unavailable"),
        ])
        monkeypatch.setattr(subagent, "_get_client", lambda: client)

        try:
            result = dispatch_agent("analyze", max_steps=3, tool_whitelist=[tool_name])
        finally:
            get_registry().unregister(tool_name)

        assert result.is_error is True
        assert "Error: Sub-agent API failure at step 2: backend unavailable" in result.content
        assert "Partial output:" in result.content
        assert "working notes" in result.content

    def test_max_steps_returns_hard_error_and_best_effort_answer(self, monkeypatch):
        tool_name = "_test_subagent_max_steps_tool"
        _register_tool(tool_name, lambda value: f"ok:{value}")
        client = _FakeClient([
            _FakeResponse(
                [
                    _FakeTextBlock("step draft"),
                    _FakeToolUseBlock("tool-1", tool_name, {"value": "x"}),
                ],
                input_tokens=2,
                output_tokens=3,
            ),
            _FakeResponse([_FakeTextBlock("best final answer")], input_tokens=4, output_tokens=5),
        ])
        monkeypatch.setattr(subagent, "_get_client", lambda: client)

        try:
            result = dispatch_agent("analyze", max_steps=1, tool_whitelist=[tool_name])
        finally:
            get_registry().unregister(tool_name)

        assert result.is_error is True
        assert "Error: Sub-agent reached max_steps (1)." in result.content
        assert "Best-effort final answer:" in result.content
        assert "best final answer" in result.content
        assert "Partial output:" in result.content
        assert "step draft" in result.content
        assert "tools" in client.messages.calls[0]
        assert "tools" not in client.messages.calls[1]

    def test_timeout_remains_hard_error(self, monkeypatch):
        def _slow_run_subagent(*_args, **_kwargs):
            time.sleep(0.2)
            return {
                "status": "completed",
                "content": "done",
                "steps": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "tool_calls": 0,
            }

        monkeypatch.setattr(subagent, "_run_subagent", _slow_run_subagent)
        monkeypatch.setattr(subagent, "_SUBAGENT_TIMEOUT", 0.05)

        result = dispatch_agent("slow task")

        assert result.is_error is True
        assert "timed out" in result.content

    def test_nesting_depth_rejection_remains_hard_error(self):
        subagent._nesting_depth.value = subagent._MAX_NESTING

        result = dispatch_agent("nested task")

        assert result.is_error is True
        assert "Maximum sub-agent nesting depth" in result.content

    def test_child_tool_error_can_recover_and_finish_successfully(self, monkeypatch):
        tool_name = "_test_subagent_recovering_tool"

        def _boom(value):
            raise RuntimeError("tool exploded")

        _register_tool(tool_name, _boom)
        client = _FakeClient([
            _FakeResponse([_FakeToolUseBlock("tool-1", tool_name, {"value": "x"})]),
            _FakeResponse([_FakeTextBlock("Recovered answer")], input_tokens=1, output_tokens=2),
        ])
        monkeypatch.setattr(subagent, "_get_client", lambda: client)

        try:
            result = dispatch_agent("recover", max_steps=3, tool_whitelist=[tool_name])
        finally:
            get_registry().unregister(tool_name)

        assert result.is_error is False
        assert result.content == "Recovered answer"
        tool_results = client.messages.calls[1]["messages"][-1]["content"]
        assert tool_results[0]["is_error"] is True
        assert "Error calling tool" in tool_results[0]["content"]
