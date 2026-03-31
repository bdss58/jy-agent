# Tests for planner utilities (ToolResult, truncation, error detection)

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jyagent.planner import (
    ToolResult, _is_error_result, _result_content,
    _truncate_tool_result, _compact_working_messages,
)


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
