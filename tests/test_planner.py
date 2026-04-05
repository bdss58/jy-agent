# Tests for planner utilities (ToolResult, truncation, error detection)

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jyagent.planner as planner
from jyagent.planner import (
    ToolResult, _is_error_result, _result_content,
    _truncate_tool_result, _compact_working_messages, _execute_tool, plan_next_action,
)
from jyagent.registry import get_registry
from jyagent.runtime import RuntimeOwner, get_adapter
from jyagent.runtime.types import ModelSpec


class TestToolResult:
    def test_success(self):
        r = ToolResult("OK")
        assert r.content == "OK"
        assert r.is_error is False
        assert str(r) == "OK"

    def test_error(self):
        r = ToolResult("Something failed", is_error=True)
        assert r.is_error is True
        assert "failed" in str(r)

    def test_repr(self):
        r = ToolResult("test content")
        assert "ToolResult" in repr(r)


class TestErrorDetection:
    def test_tool_result_error(self):
        assert _is_error_result(ToolResult("fail", is_error=True)) is True

    def test_tool_result_success(self):
        assert _is_error_result(ToolResult("ok")) is False

    def test_string_error(self):
        assert _is_error_result("Error: file not found") is True

    def test_string_success(self):
        assert _is_error_result("Success") is False

    def test_error_calling_tool(self):
        assert _is_error_result("Error calling tool X") is True


class TestResultContent:
    def test_tool_result(self):
        r = ToolResult("content here")
        assert _result_content(r) == "content here"

    def test_string(self):
        assert _result_content("plain string") == "plain string"


class TestTruncation:
    def test_short_not_truncated(self):
        result = _truncate_tool_result("short", max_chars=100)
        assert result == "short"

    def test_long_truncated(self):
        long_text = "x" * 10000
        result = _truncate_tool_result(long_text, max_chars=1000)
        assert len(result) <= 1100  # some slack for the truncation message
        assert "truncated" in result.lower()

    def test_error_never_truncated(self):
        long_error = "Error: " + "x" * 10000
        result = _truncate_tool_result(long_error, is_error=True, max_chars=100)
        assert len(result) == len(long_error)  # preserved fully


class TestCompactWorkingMessages:
    def test_under_threshold(self):
        messages = [
            {"role": "user", "content": "short message"},
            {"role": "assistant", "content": "short reply"},
        ]
        compacted = _compact_working_messages(messages, max_tokens=100000)
        assert len(compacted) == 2  # unchanged

    def test_over_threshold_truncates_tool_results(self):
        """Compact should truncate old tool_result blocks, not drop messages."""
        long_result = "x" * 50000
        messages = [
            {"role": "user", "content": [{"type": "tool_result", "content": long_result}]},
            {"role": "assistant", "content": "processing..."},
            {"role": "user", "content": "what happened?"},
            {"role": "assistant", "content": "done"},
        ]
        
        compacted = _compact_working_messages(messages, max_tokens=100)
        # Should still have same number of messages (doesn't drop)
        assert len(compacted) == 4
        # But the old tool result should be truncated
        first_content = compacted[0]["content"]
        tool_result = first_content[0]["content"]
        assert len(tool_result) < len(long_result)


class TestToolInputValidation:
    def test_rejects_non_object_input(self):
        reg = get_registry()

        def _tool(path: str) -> str:
            return path

        schema = {
            "name": "_test_non_object",
            "description": "test",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        }
        reg.register("_test_non_object", _tool, schema)
        try:
            result = _execute_tool("_test_non_object", "not-a-dict", {"_test_non_object": _tool})
            assert result.is_error is True
            assert "expected object input" in result.content
        finally:
            reg.unregister("_test_non_object")

    def test_rejects_invalid_type(self):
        reg = get_registry()

        def _tool(timeout: int) -> str:
            return f"timeout={timeout}"

        schema = {
            "name": "_test_invalid_type",
            "description": "test",
            "input_schema": {
                "type": "object",
                "properties": {
                    "timeout": {"type": "integer"},
                },
                "required": ["timeout"],
            },
        }
        reg.register("_test_invalid_type", _tool, schema)
        try:
            result = _execute_tool("_test_invalid_type", {"timeout": "fast"}, {"_test_invalid_type": _tool})
            assert result.is_error is True
            assert "input.timeout must be integer" in result.content
        finally:
            reg.unregister("_test_invalid_type")

    def test_rejects_enum_violation(self):
        reg = get_registry()

        def _tool(action: str) -> str:
            return action

        schema = {
            "name": "_test_enum",
            "description": "test",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["connect", "status"]},
                },
                "required": ["action"],
            },
        }
        reg.register("_test_enum", _tool, schema)
        try:
            result = _execute_tool("_test_enum", {"action": "launch"}, {"_test_enum": _tool})
            assert result.is_error is True
            assert "must be one of" in result.content
        finally:
            reg.unregister("_test_enum")

    def test_rejects_numeric_constraint_violation(self):
        reg = get_registry()

        def _tool(limit: int) -> str:
            return str(limit)

        schema = {
            "name": "_test_maximum",
            "description": "test",
            "input_schema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "maximum": 10},
                },
                "required": ["limit"],
            },
        }
        reg.register("_test_maximum", _tool, schema)
        try:
            result = _execute_tool("_test_maximum", {"limit": 11}, {"_test_maximum": _tool})
            assert result.is_error is True
            assert "input.limit must be <= 10" in result.content
        finally:
            reg.unregister("_test_maximum")


class TestPlannerFallback:
    def test_max_steps_fallback_uses_stream_backed_complete(self, monkeypatch):
        tool_name = "_test_planner_fallback_tool"

        class _DummyStats:
            def __init__(self):
                self.recorded_usage = []

            def new_turn(self):
                pass

            def record_usage(self, usage, provider="", model=""):
                self.recorded_usage.append((usage, provider, model))

            def record_tool_call(self):
                pass

        class _FakeRuntimeStream:
            def __init__(self, final_message):
                self._final_message = final_message
                self.iterated = False
                self.closed = False

            def __iter__(self):
                self.iterated = True
                yield {"type": "text_delta", "text": "fallback"}

            def get_final_message(self):
                return self._final_message

            def close(self):
                self.closed = True

        def _tool():
            return "tool output"

        schema = {
            "name": tool_name,
            "description": "planner fallback tool",
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        }
        reg = get_registry()
        reg.register(tool_name, _tool, schema)

        dummy_stats = _DummyStats()
        written = []
        final_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "fallback answer"}],
            "provider": "openai",
            "model": "gpt-5-mini",
            "usage": {"input_tokens": 2, "output_tokens": 3},
            "stop_reason": "stop",
        }
        fake_stream = _FakeRuntimeStream(final_message)
        adapter = get_adapter("openai")

        def _fake_stream_with_retry(runtime_owner, context, max_output_tokens, step, all_text, working_messages):
            return (
                "",
                [planner._ToolCallRequest(id="call_1", name=tool_name, input={})],
                "tool_use",
                {
                    "role": "assistant",
                    "content": [{"type": "tool_call", "id": "call_1", "name": tool_name, "arguments": {}}],
                    "provider": "openai",
                    "model": "gpt-5-mini",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "stop_reason": "tool_use",
                },
            )

        monkeypatch.setattr(planner, "_stream_with_retry", _fake_stream_with_retry)
        monkeypatch.setattr(planner, "get_stats", lambda: dummy_stats)
        monkeypatch.setattr(planner, "_stream_write", written.append)
        monkeypatch.setattr(
            adapter,
            "stream",
            lambda model_spec, context, options=None: fake_stream,
        )

        try:
            result, final_text, working_messages = plan_next_action(
                RuntimeOwner(ModelSpec("openai", "gpt-5-mini")),
                [],
                "system prompt",
                max_steps=1,
            )
        finally:
            reg.unregister(tool_name)

        assert result == "fallback answer"
        assert final_text == "fallback answer"
        assert working_messages[-1]["content"] == [{"type": "text", "text": "fallback answer"}]
        assert written == ["fallback answer"]
        assert fake_stream.iterated is True
        assert fake_stream.closed is True
        assert dummy_stats.recorded_usage == [
            ({"input_tokens": 2, "output_tokens": 3}, "openai", "gpt-5-mini")
        ]


class TestPlannerWarnings:
    def test_runtime_warnings_are_printed_without_aborting_turn(self, monkeypatch, capsys):
        class _DummyStats:
            def __init__(self):
                self.recorded_usage = []

            def new_turn(self):
                pass

            def record_usage(self, usage, provider="", model=""):
                self.recorded_usage.append((usage, provider, model))

            def record_tool_call(self):
                pass

        class _FakeRuntimeStream:
            def __init__(self, final_message):
                self._final_message = final_message
                self.closed = False

            def __iter__(self):
                yield {"type": "text_delta", "text": "warning-safe answer"}

            def get_final_message(self):
                return self._final_message

            def close(self):
                self.closed = True

        dummy_stats = _DummyStats()
        final_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "warning-safe answer"}],
            "provider": "openai",
            "model": "gpt-5-mini",
            "usage": {"input_tokens": 4, "output_tokens": 5},
            "stop_reason": "stop",
            "runtime_warnings": [
                "Recovered OpenAI stream after missing terminal event via responses.retrieve()."
            ],
        }
        fake_stream = _FakeRuntimeStream(final_message)
        adapter = get_adapter("openai")

        monkeypatch.setattr(planner, "get_stats", lambda: dummy_stats)
        monkeypatch.setattr(
            adapter,
            "stream",
            lambda model_spec, context, options=None: fake_stream,
        )

        result, final_text, working_messages = plan_next_action(
            RuntimeOwner(ModelSpec("openai", "gpt-5-mini")),
            [],
            "system prompt",
            max_steps=1,
        )

        output = capsys.readouterr().out

        assert "Recovered OpenAI stream after missing terminal event via responses.retrieve()." in output
        assert result == "warning-safe answer"
        assert final_text == "warning-safe answer"
        assert working_messages[-1]["runtime_warnings"] == final_message["runtime_warnings"]
        assert fake_stream.closed is True
        assert dummy_stats.recorded_usage == [
            ({"input_tokens": 4, "output_tokens": 5}, "openai", "gpt-5-mini")
        ]
