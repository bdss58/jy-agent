# Tests for planner utilities (ToolResult, truncation, error detection)

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jyagent.planner import (
    ToolResult, _is_error_result, _result_content,
    _truncate_tool_result, _compact_working_messages, _execute_tool,
)
from jyagent.registry import get_registry


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
