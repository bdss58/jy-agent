"""Regression tests for Codex review 2026-04-25 Tier-A fixes.

Covers:
    A2 — `_tool_dispatch_executor` grows to honour `LoopConfig.max_tool_workers`
    A3 — tracing finalize errors are logged, not raised
    A4 — `run_id` containing `..` cannot escape `checkpoint_dir`
"""
from __future__ import annotations

import logging
import os
import types

import pytest

from jyagent.runtime.loop import checkpoint
from jyagent.runtime.loop import engine as loop_engine
from jyagent.runtime.loop.config import LoopConfig


# ─── A4: path sanitisation ──────────────────────────────────────────────────


class TestCheckpointRunIdSanitisation:
    def test_sanitize_replaces_path_separators(self):
        assert checkpoint._sanitize_run_id("a/b") == "a_b"
        assert checkpoint._sanitize_run_id("a" + os.sep + "b") == "a_b"

    def test_sanitize_strips_leading_dots(self):
        assert checkpoint._sanitize_run_id("..") == "_"
        assert checkpoint._sanitize_run_id("../../etc/passwd") == "_.._etc_passwd"
        assert checkpoint._sanitize_run_id(".hidden") == "hidden"

    def test_sanitize_allows_safe_chars(self):
        assert checkpoint._sanitize_run_id("run-2026_04-25.v1") == "run-2026_04-25.v1"

    def test_sanitize_empty_becomes_underscore(self):
        assert checkpoint._sanitize_run_id("") == "_"
        assert checkpoint._sanitize_run_id(None) == "_"  # type: ignore[arg-type]

    def test_checkpoint_path_blocks_parent_escape(self, tmp_path):
        # Parent-dir escape: the resulting path must stay inside tmp_path.
        path = checkpoint.checkpoint_path(str(tmp_path), "..", 1)
        # Normalise and assert containment.
        resolved = os.path.realpath(path)
        base = os.path.realpath(str(tmp_path))
        assert resolved.startswith(base + os.sep), (
            f"run_id='..' escaped checkpoint_dir: {resolved} not under {base}"
        )

    def test_checkpoint_path_unusual_chars_neutralised(self, tmp_path):
        path = checkpoint.checkpoint_path(str(tmp_path), "a/b;rm -rf /", 1)
        assert ";" not in path
        assert " " not in path


# ─── A3: tracing finalize errors are non-fatal ──────────────────────────────


class _ExplodingTrace:
    """Stand-in for RunTrace whose finalize+flush raise."""

    def __init__(self):
        self.finish_called = False
        self.flush_called = False

    def finish(self, **kwargs):
        self.finish_called = True
        raise PermissionError("simulated read-only fs")

    def flush(self):
        self.flush_called = True
        raise PermissionError("should not be called after finish() raises")


class TestTraceFinalizeNonFatal:
    def test_tracing_failure_does_not_raise(self, caplog):
        """_finalize_run must return a LoopResult even if trace.finish() raises."""
        caplog.set_level(logging.WARNING, logger=loop_engine.__name__)
        trace = _ExplodingTrace()
        result = loop_engine._finalize_run(
            status="completed",
            text="hi",
            final_text="hi",
            messages=[],
            steps=1,
            total_input_tokens=10,
            total_output_tokens=5,
            tool_calls_count=0,
            trace=trace,
        )
        assert result.status == "completed"
        assert result.text == "hi"
        assert trace.finish_called is True
        # At least one warning must have been logged about the trace failure.
        warning_messages = [r.getMessage() for r in caplog.records]
        assert any("trace finalize failed" in m for m in warning_messages), warning_messages

    def test_tracing_disabled_is_quiet(self, caplog):
        """No trace → no warning, no exception."""
        caplog.set_level(logging.WARNING, logger=loop_engine.__name__)
        result = loop_engine._finalize_run(
            status="completed",
            text="hi",
            final_text="hi",
            messages=[],
            steps=1,
            total_input_tokens=0,
            total_output_tokens=0,
            tool_calls_count=0,
            trace=None,
        )
        assert result.status == "completed"
        assert not caplog.records


# ─── A2: dispatch executor honours max_tool_workers ─────────────────────────


class TestDispatchExecutorGrowsWithConfig:
    def test_get_executor_grows_on_demand(self, monkeypatch):
        """Requesting more workers than current cap grows the pool."""
        # Snapshot + reset module state so the test is independent.
        original_executor = loop_engine._tool_dispatch_executor
        original_cap = loop_engine._tool_dispatch_cap
        try:
            exe_small = loop_engine._get_tool_dispatch_executor(8)
            cap_small = loop_engine._tool_dispatch_cap
            assert cap_small >= 8
            assert exe_small._max_workers >= 8

            exe_big = loop_engine._get_tool_dispatch_executor(16)
            cap_big = loop_engine._tool_dispatch_cap
            assert cap_big >= 16
            assert exe_big._max_workers >= 16
            # Growth must have replaced the executor.
            assert exe_big is not exe_small
        finally:
            loop_engine._tool_dispatch_executor = original_executor
            loop_engine._tool_dispatch_cap = original_cap

    def test_get_executor_reuses_when_already_big_enough(self):
        """Asking for a smaller size than current cap returns the same pool."""
        a = loop_engine._get_tool_dispatch_executor(64)
        b = loop_engine._get_tool_dispatch_executor(4)
        assert a is b

    def test_get_executor_floor_is_8(self):
        """Tiny requests still get at least 8 workers."""
        exe = loop_engine._get_tool_dispatch_executor(1)
        assert loop_engine._tool_dispatch_cap >= 8
        assert exe._max_workers >= 8

    def test_agent_loop_init_sizes_executor_from_config(self, monkeypatch):
        """AgentLoop(__init__) must pass cfg.max_tool_workers into the grow helper."""
        captured: dict = {}
        original = loop_engine._get_tool_dispatch_executor

        def spy(min_workers: int = 8):
            captured["min_workers"] = min_workers
            return original(min_workers)

        monkeypatch.setattr(loop_engine, "_get_tool_dispatch_executor", spy)

        # Minimal stub for LLMOwner — AgentLoop only stores it.
        owner = types.SimpleNamespace()
        cfg = LoopConfig(max_tool_workers=12)
        loop_engine.AgentLoop(owner, cfg)  # type: ignore[arg-type]

        assert captured["min_workers"] == 12
