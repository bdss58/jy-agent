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


# --- QW-3: Response-aware stuck-loop detection ---

from jyagent.loop_engine import _StuckLoopDetector, ToolCallRequest


class TestStuckLoopDetector:
    """QW-3: _StuckLoopDetector detects stuck loops via response comparison."""

    def test_identical_response_triggers(self):
        d = _StuckLoopDetector(threshold=3)
        assert d.record("read_file", {"path": "/tmp/x"}, "hello") is None
        assert d.record("read_file", {"path": "/tmp/x"}, "hello") is None
        fb = d.record("read_file", {"path": "/tmp/x"}, "hello")
        assert fb is not None
        assert "STUCK LOOP" in fb
        assert "read_file" in fb

    def test_changing_response_resets(self):
        d = _StuckLoopDetector(threshold=3)
        d.record("cb", {"pid": 1}, '{"s":"run","e":30}')
        d.record("cb", {"pid": 1}, '{"s":"run","e":60}')
        d.record("cb", {"pid": 1}, '{"s":"run","e":90}')
        assert d.record("cb", {"pid": 1}, '{"s":"done","e":95}') is None

    def test_polling_never_stuck_with_changing_elapsed(self):
        d = _StuckLoopDetector(threshold=2)
        for i in range(20):
            r = f'{{"elapsed":{30 + i * 15}}}'
            assert d.record("check_background", {"pid": 42}, r) is None

    def test_different_args_separate(self):
        d = _StuckLoopDetector(threshold=3)
        d.record("rf", {"p": "/a"}, "a")
        d.record("rf", {"p": "/b"}, "b")
        assert d.record("rf", {"p": "/c"}, "c") is None

    def test_different_tools_separate(self):
        d = _StuckLoopDetector(threshold=3)
        d.record("rf", {"p": "/x"}, "d")
        d.record("wf", {"p": "/x"}, "d")
        assert d.record("ef", {"p": "/x"}, "d") is None

    def test_custom_threshold(self):
        d = _StuckLoopDetector(threshold=5)
        for _ in range(4):
            assert d.record("foo", {"a": 1}, "same") is None
        assert d.record("foo", {"a": 1}, "same") is not None

    def test_key_stability(self):
        d = _StuckLoopDetector(threshold=2)
        d.record("t", {"b": 2, "a": 1}, "r")
        assert d.record("t", {"a": 1, "b": 2}, "r") is not None

    def test_response_change_then_revert(self):
        d = _StuckLoopDetector(threshold=3)
        d.record("rf", {"p": "/x"}, "v1")
        d.record("rf", {"p": "/x"}, "v1")
        d.record("rf", {"p": "/x"}, "v2")  # reset
        d.record("rf", {"p": "/x"}, "v2")
        fb = d.record("rf", {"p": "/x"}, "v2")  # 3rd v2
        assert fb is not None

    def test_polling_completes_before_stuck(self):
        d = _StuckLoopDetector(threshold=3)
        d.record("cb", {"pid": 1}, '{"e":10}')
        d.record("cb", {"pid": 1}, '{"e":20}')
        d.record("cb", {"pid": 1}, '{"e":30}')
        assert d.record("cb", {"pid": 1}, '{"s":"done"}') is None

    def test_mcp_tool_naturally_handled(self):
        d = _StuckLoopDetector(threshold=2)
        d.record("mcp__chrome__take_snapshot", {}, "<div>state1</div>")
        assert d.record("mcp__chrome__take_snapshot", {}, "<div>state2</div>") is None

    def test_mcp_tool_stuck_when_same_response(self):
        d = _StuckLoopDetector(threshold=2)
        d.record("mcp__chrome__take_snapshot", {}, "<div>same</div>")
        fb = d.record("mcp__chrome__take_snapshot", {}, "<div>same</div>")
        assert fb is not None

    def test_sleep_with_identical_empty_response(self):
        """sleep returning empty string repeatedly IS stuck (correct behavior)."""
        d = _StuckLoopDetector(threshold=3)
        d.record("run_shell", {"command": "sleep 60"}, "")
        d.record("run_shell", {"command": "sleep 60"}, "")
        fb = d.record("run_shell", {"command": "sleep 60"}, "")
        assert fb is not None

    def test_non_dict_args_handled(self):
        """None args don't crash."""
        d = _StuckLoopDetector(threshold=2)
        d.record("tool", None, "resp")
        fb = d.record("tool", None, "resp")
        assert fb is not None

    def test_first_observation_no_trigger(self):
        d = _StuckLoopDetector(threshold=2)
        assert d.record("tool", {"a": 1}, "resp") is None

    def test_counter_beyond_threshold(self):
        """Counter keeps incrementing beyond threshold."""
        d = _StuckLoopDetector(threshold=2)
        d.record("t", {}, "x")
        fb2 = d.record("t", {}, "x")
        assert fb2 is not None
        fb3 = d.record("t", {}, "x")
        assert fb3 is not None
        assert "3 times" in fb3

    def test_hash_response_deterministic(self):
        """Same content always produces the same hash."""
        h1 = _StuckLoopDetector._hash_response("hello world")
        h2 = _StuckLoopDetector._hash_response("hello world")
        assert h1 == h2

    def test_no_dedup_exempt_needed(self):
        """check_background works without any exemption metadata."""
        d = _StuckLoopDetector(threshold=3)
        # Simulate realistic polling — elapsed changes each time
        for i in range(10):
            resp = f'{{"pid":99,"status":"running","elapsed_seconds":{10.0 + i * 15.5},"output":"line {i}"}}'
            fb = d.record("check_background", {"pid": 99, "tail": 10}, resp)
            assert fb is None
        # Final done response
        fb = d.record("check_background", {"pid": 99, "tail": 10},
                       '{"pid":99,"status":"done","exit_code":0,"elapsed_seconds":200.0}')
        assert fb is None


# --- Integration: LoopConfig ---

from jyagent.loop_engine import LoopConfig


class TestLoopConfigHarness:
    """Verify LoopConfig fields have correct defaults."""

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
