# tests/test_harness_quick_wins.py — Tests for QW-1, QW-2, QW-3.
#
# These tests verify the harness engineering quick wins:
#   QW-1: Remediation messages for tool errors
#   QW-2: Cost budget enforcement
#   QW-3: Duplicate tool-call detection (infinite loop breaker)

import json
import pytest

# ─── QW-1: Remediation messages ─────────────────────────────────────────────

from jyagent.toolresult import ToolResult
from jyagent.remediation import enrich_error


class TestRemediation:
    """QW-1: enrich_error appends remediation hints to known error patterns."""

    def test_non_error_unchanged(self):
        """Non-error results are never modified."""
        r = ToolResult("All good", is_error=False)
        enriched = enrich_error(r, "read_file")
        assert enriched is r
        assert "REMEDIATION" not in enriched.content

    def test_file_not_found(self):
        r = ToolResult("FileNotFoundError: /tmp/missing.txt", is_error=True)
        enriched = enrich_error(r, "read_file")
        assert "REMEDIATION" in enriched.content
        assert "glob_files" in enriched.content
        assert enriched.is_error

    def test_permission_denied(self):
        r = ToolResult("PermissionError: Permission denied: /etc/shadow", is_error=True)
        enriched = enrich_error(r, "write_file")
        assert "REMEDIATION" in enriched.content
        assert "Permission" in enriched.content

    def test_old_text_not_found(self):
        r = ToolResult("old_text not found in file /tmp/test.py", is_error=True)
        enriched = enrich_error(r, "edit_file")
        assert "REMEDIATION" in enriched.content
        assert "read_file" in enriched.content

    def test_ssl_error(self):
        r = ToolResult("ssl.SSLCertVerificationError: CERTIFICATE_VERIFY_FAILED", is_error=True)
        enriched = enrich_error(r, "web_fetch")
        assert "REMEDIATION" in enriched.content
        assert "SSL" in enriched.content or "CA cert" in enriched.content

    def test_command_not_found(self):
        r = ToolResult("bash: foo: command not found", is_error=True)
        enriched = enrich_error(r, "run_shell")
        assert "REMEDIATION" in enriched.content
        assert "which" in enriched.content

    def test_timeout_error(self):
        r = ToolResult("TimeoutError: Read timed out after 60s", is_error=True)
        enriched = enrich_error(r, "run_shell")
        assert "REMEDIATION" in enriched.content
        assert "timeout" in enriched.content.lower()

    def test_unknown_error_no_remediation(self):
        """Errors not matching any pattern are returned unchanged."""
        r = ToolResult("Error: something completely novel happened", is_error=True)
        enriched = enrich_error(r, "some_tool")
        assert enriched.content == r.content
        assert "REMEDIATION" not in enriched.content

    def test_idempotent(self):
        """Applying enrich_error twice doesn't double the hint."""
        r = ToolResult("FileNotFoundError: /tmp/x.txt", is_error=True)
        enriched1 = enrich_error(r, "read_file")
        enriched2 = enrich_error(enriched1, "read_file")
        assert enriched1.content == enriched2.content

    def test_json_error(self):
        r = ToolResult("json.decoder.JSONDecodeError: Expecting value", is_error=True)
        enriched = enrich_error(r, "run_shell")
        assert "REMEDIATION" in enriched.content
        assert "JSON" in enriched.content

    def test_mcp_not_connected(self):
        r = ToolResult("Error: MCP server 'chrome' not connected", is_error=True)
        enriched = enrich_error(r, "mcp__chrome__navigate")
        assert "REMEDIATION" in enriched.content
        assert "connect" in enriched.content.lower()

    def test_404_error(self):
        r = ToolResult("HTTP Error: 404 Not Found for https://example.com/missing", is_error=True)
        enriched = enrich_error(r, "web_fetch")
        assert "REMEDIATION" in enriched.content
        assert "404" in enriched.content

    def test_module_not_found(self):
        r = ToolResult("ModuleNotFoundError: No module named 'pandas'", is_error=True)
        enriched = enrich_error(r, "run_shell")
        assert "REMEDIATION" in enriched.content
        assert "pip install" in enriched.content


# ─── QW-2: Cost budget ──────────────────────────────────────────────────────

from jyagent.loop_engine import _CostTracker


class TestCostTracker:
    """QW-2: _CostTracker accumulates cost and detects budget exceeded."""

    def test_unknown_pricing(self):
        """Unknown provider/model → known_cost returns None."""
        ct = _CostTracker()
        ct.record({"input_tokens": 1000, "output_tokens": 500}, "unknown_provider", "unknown_model")
        assert ct.known_cost is None

    def test_known_pricing_accumulates(self):
        """Known provider/model → cost accumulates."""
        ct = _CostTracker()
        # Use an existing model from session_stats
        ct.record(
            {"input_tokens": 1_000_000, "output_tokens": 0},
            "anthropic", "claude-opus-4-6",
        )
        # 1M input tokens at $5/M = $5.00
        assert ct.known_cost is not None
        assert abs(ct.known_cost - 5.0) < 0.01

    def test_zero_usage(self):
        """Empty usage → cost stays at 0."""
        ct = _CostTracker()
        ct.record({"input_tokens": 0, "output_tokens": 0}, "anthropic", "claude-opus-4-6")
        assert ct.known_cost == 0.0

    def test_multiple_records(self):
        """Multiple calls accumulate."""
        ct = _CostTracker()
        ct.record(
            {"input_tokens": 1_000_000, "output_tokens": 0},
            "anthropic", "claude-opus-4-6",
        )
        ct.record(
            {"input_tokens": 0, "output_tokens": 1_000_000},
            "anthropic", "claude-opus-4-6",
        )
        # $5 input + $25 output = $30
        assert ct.known_cost is not None
        assert abs(ct.known_cost - 30.0) < 0.01


# ─── QW-3: Duplicate-call detection ─────────────────────────────────────────

from jyagent.loop_engine import _DedupTracker, ToolCallRequest


class TestDedupTracker:
    """QW-3: _DedupTracker detects repeated identical tool calls."""

    def _call(self, name: str, args: dict) -> ToolCallRequest:
        return ToolCallRequest(id=f"call_{name}", name=name, input=args)

    def test_no_dup_below_threshold(self):
        """Below threshold (3) → no feedback."""
        dt = _DedupTracker(threshold=3)
        assert dt.record([self._call("read_file", {"path": "/tmp/x"})]) is None
        assert dt.record([self._call("read_file", {"path": "/tmp/x"})]) is None

    def test_dup_at_threshold(self):
        """At threshold → feedback returned."""
        dt = _DedupTracker(threshold=3)
        dt.record([self._call("read_file", {"path": "/tmp/x"})])
        dt.record([self._call("read_file", {"path": "/tmp/x"})])
        feedback = dt.record([self._call("read_file", {"path": "/tmp/x"})])
        assert feedback is not None
        assert "LOOP DETECTED" in feedback
        assert "read_file" in feedback

    def test_different_args_not_dup(self):
        """Different args → separate tracking, no false positive."""
        dt = _DedupTracker(threshold=3)
        dt.record([self._call("read_file", {"path": "/tmp/a"})])
        dt.record([self._call("read_file", {"path": "/tmp/b"})])
        feedback = dt.record([self._call("read_file", {"path": "/tmp/c"})])
        assert feedback is None

    def test_different_tools_not_dup(self):
        """Different tool names → separate tracking."""
        dt = _DedupTracker(threshold=3)
        dt.record([self._call("read_file", {"path": "/tmp/x"})])
        dt.record([self._call("write_file", {"path": "/tmp/x"})])
        feedback = dt.record([self._call("edit_file", {"path": "/tmp/x"})])
        assert feedback is None

    def test_batch_with_one_dup(self):
        """Only the duplicated call triggers, not the whole batch."""
        dt = _DedupTracker(threshold=2)
        dt.record([self._call("read_file", {"path": "/tmp/x"})])
        feedback = dt.record([
            self._call("read_file", {"path": "/tmp/x"}),  # dup!
            self._call("write_file", {"path": "/tmp/y"}),  # not dup
        ])
        assert feedback is not None
        assert "read_file" in feedback

    def test_custom_threshold(self):
        """Custom threshold works."""
        dt = _DedupTracker(threshold=5)
        for _ in range(4):
            assert dt.record([self._call("foo", {"a": 1})]) is None
        feedback = dt.record([self._call("foo", {"a": 1})])
        assert feedback is not None

    def test_key_stability(self):
        """Args with different key order produce the same key (sorted)."""
        dt = _DedupTracker(threshold=2)
        dt.record([self._call("tool", {"b": 2, "a": 1})])
        feedback = dt.record([self._call("tool", {"a": 1, "b": 2})])
        assert feedback is not None


# ─── Integration: LoopConfig new fields ──────────────────────────────────────

from jyagent.loop_engine import LoopConfig


class TestLoopConfigHarness:
    """Verify new LoopConfig fields have correct defaults."""

    def test_max_cost_default_none(self):
        cfg = LoopConfig()
        assert cfg.max_cost_usd is None

    def test_dedup_threshold_default(self):
        cfg = LoopConfig()
        assert cfg.dedup_threshold == 3

    def test_max_cost_custom(self):
        cfg = LoopConfig(max_cost_usd=0.50)
        assert cfg.max_cost_usd == 0.50

    def test_dedup_threshold_custom(self):
        cfg = LoopConfig(dedup_threshold=5)
        assert cfg.dedup_threshold == 5
