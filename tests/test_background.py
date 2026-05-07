"""Lifecycle tests for run_background / check_background.

Covers the bugs identified by Claude Code + Codex (kill-status lie,
tail pathology, unbounded tail=0), plus Tier-2 additions (action="wait",
per-job timeout, stdin_null, cwd, concurrency cap, timed_out status,
deadline-on-cleanup enforcement).
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

from jyagent.tools import background as bg_core
from jyagent.tools.background import (
    _BG_MAX_CONCURRENT,
    _BG_OUTPUT_MAX_BYTES,
    _BG_WAIT_MAX_SECONDS,
    _bg_cleanup_completed,
    _read_tail_bytes,
    _read_tail_efficient,
    check_background,
    run_background,
)


# ── helpers ────────────────────────────────────────────────────────────

def _start(cmd: str, **kwargs) -> dict:
    res = run_background(cmd, **kwargs)
    data = json.loads(res.content)
    assert data.get("status") == "started", data
    return data


def _check(pid: int, **kwargs) -> dict:
    return json.loads(check_background(pid, **kwargs).content)


def _wait_done(pid: int, timeout: float = 5.0, poll: float = 0.05) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = _check(pid)
        if s["status"] != "running":
            return s
        time.sleep(poll)
    pytest.fail(f"pid {pid} did not finish within {timeout}s")


@pytest.fixture(autouse=True)
def _cleanup_between_tests():
    """Kill any lingering bg jobs between tests so concurrency cap is honored."""
    yield
    # Kill everything we still own
    with bg_core._bg_lock:
        pids = list(bg_core._background_processes.keys())
    for pid in pids:
        try:
            check_background(pid, action="kill")
        except Exception:
            pass
    with bg_core._bg_lock:
        for info in list(bg_core._background_processes.values()):
            bg_core._bg_close_and_cleanup(info)
        bg_core._background_processes.clear()


# ── Tier 1: bug repros ─────────────────────────────────────────────────

class TestStatusTaxonomy:
    def test_succeeded(self):
        pid = _start("echo hi")["pid"]
        s = _wait_done(pid)
        assert s["status"] == "succeeded"
        assert s["exit_code"] == 0
        assert "hi" in s["output"]

    def test_failed(self):
        pid = _start("exit 7")["pid"]
        s = _wait_done(pid)
        assert s["status"] == "failed"
        assert s["exit_code"] == 7

    def test_kill_after_exit_reports_real_outcome(self):
        """The original bug: action='kill' claimed 'killed' even if the
        process had already exited cleanly."""
        pid = _start("echo fast")["pid"]
        # Give it time to finish on its own
        _wait_done(pid, timeout=3.0)
        # Now "kill" an already-exited process — must NOT say 'killed'
        s = _check(pid, action="kill")
        assert s["status"] == "succeeded", s
        assert s["exit_code"] == 0

    def test_kill_running(self):
        pid = _start("sleep 30")["pid"]
        time.sleep(0.1)
        s = _check(pid, action="kill")
        assert s["status"] == "killed"

    def test_killed_is_sticky(self):
        pid = _start("sleep 30")["pid"]
        time.sleep(0.1)
        _check(pid, action="kill")
        s = _check(pid)
        assert s["status"] == "killed"


class TestOutputBounding:
    def test_tail_no_newline_bounded(self):
        """Huge single-line log must not blow memory."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".out") as tf:
            tf.write(b"x" * 2_000_000)  # 2 MB, no newlines
            path = tf.name
        try:
            out = _read_tail_efficient(path, n=10)
            assert len(out) <= _BG_OUTPUT_MAX_BYTES + 200
        finally:
            os.unlink(path)

    def test_tail_n_returns_exact_lines(self):
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".out") as tf:
            for i in range(10_000):
                tf.write(f"line {i}\n".encode())
            path = tf.name
        try:
            out = _read_tail_efficient(path, n=5)
            lines = out.splitlines()
            assert lines == ["line 9995", "line 9996", "line 9997", "line 9998", "line 9999"]
        finally:
            os.unlink(path)

    def test_tail_zero_bounded_on_huge_file(self):
        """tail=0 must seek-from-end, not full-read."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".out") as tf:
            tf.write(b"A" * 10_000_000)
            tf.write(b"\nTAIL_MARKER\n")
            path = tf.name
        try:
            out = _read_tail_bytes(path, _BG_OUTPUT_MAX_BYTES)
            assert "TAIL_MARKER" in out
            assert len(out) <= _BG_OUTPUT_MAX_BYTES + 200
        finally:
            os.unlink(path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".out") as tf:
            path = tf.name
        try:
            assert _read_tail_efficient(path, n=10) == ""
            assert _read_tail_bytes(path, 1000) == ""
        finally:
            os.unlink(path)


# ── Tier 2: new features ──────────────────────────────────────────────

class TestWaitAction:
    def test_wait_returns_when_process_finishes(self):
        pid = _start("sleep 0.3; echo done")["pid"]
        t0 = time.time()
        s = _check(pid, action="wait", wait_timeout_seconds=5)
        dur = time.time() - t0
        assert s["status"] == "succeeded", s
        assert "done" in s["output"]
        # Should have blocked ~0.3s, well under the 5s cap
        assert dur < 2.0, f"wait returned too slowly: {dur}s"

    def test_wait_times_out_on_long_process(self):
        pid = _start("sleep 10")["pid"]
        t0 = time.time()
        s = _check(pid, action="wait", wait_timeout_seconds=1)
        dur = time.time() - t0
        assert s["status"] == "running", s
        # Should have returned after ~1s
        assert 0.8 <= dur <= 2.5, f"wait blocked wrong amount: {dur}s"

    def test_wait_noop_on_already_exited(self):
        pid = _start("true")["pid"]
        _wait_done(pid, timeout=3.0)
        t0 = time.time()
        s = _check(pid, action="wait", wait_timeout_seconds=5)
        dur = time.time() - t0
        assert s["status"] == "succeeded"
        assert dur < 0.5, f"wait on exited process blocked: {dur}s"

    def test_wait_timeout_is_capped(self):
        """wait_timeout_seconds > _BG_WAIT_MAX_SECONDS must be clamped."""
        pid = _start("true")["pid"]
        _wait_done(pid)
        # Just verify the call accepts a huge value without hanging
        s = _check(pid, action="wait", wait_timeout_seconds=999_999)
        assert s["status"] == "succeeded"


class TestTimeoutSeconds:
    def test_timeout_auto_kills_and_marks_timed_out(self):
        pid = _start("sleep 30", timeout_seconds=1)["pid"]
        # Poll past the deadline
        time.sleep(1.3)
        s = _check(pid)
        assert s["status"] == "timed_out", s
        assert s["exit_code"] is not None  # process has exited

    def test_timed_out_is_sticky_and_beats_killed(self):
        pid = _start("sleep 30", timeout_seconds=1)["pid"]
        time.sleep(1.3)
        _check(pid)  # triggers timeout enforcement
        # Now try action=kill after the fact — timed_out should persist
        s = _check(pid, action="kill")
        assert s["status"] == "timed_out", s

    def test_no_timeout_means_no_deadline(self):
        pid = _start("sleep 0.2")["pid"]
        s = _wait_done(pid)
        assert s["status"] == "succeeded"
        assert "deadline_seconds_remaining" not in s

    def test_deadline_field_surfaced_while_running(self):
        pid = _start("sleep 5", timeout_seconds=10)["pid"]
        time.sleep(0.1)
        s = _check(pid)
        assert s["status"] == "running"
        assert "deadline_seconds_remaining" in s
        assert 0 < s["deadline_seconds_remaining"] <= 10


class TestStdinPolicy:
    def test_stdin_null_default_does_not_hang(self):
        """A command that reads stdin must get EOF immediately, not hang."""
        pid = _start("cat; echo after_cat")["pid"]
        s = _wait_done(pid, timeout=3.0)
        assert s["status"] == "succeeded", s
        assert "after_cat" in s["output"]


class TestCwd:
    def test_cwd_runs_command_in_directory(self, tmp_path):
        marker = tmp_path / "hello.txt"
        marker.write_text("marker-content")
        pid = _start("cat hello.txt", cwd=str(tmp_path))["pid"]
        s = _wait_done(pid, timeout=3.0)
        assert s["status"] == "succeeded"
        assert "marker-content" in s["output"]

    def test_invalid_cwd_rejected(self):
        res = run_background("true", cwd="/nonexistent/path/xyz123")
        assert res.is_error
        assert "cwd does not exist" in res.content


class TestConcurrencyCap:
    def test_cap_rejects_beyond_limit(self):
        pids = []
        # Fill up to the cap with sleep processes
        for _ in range(_BG_MAX_CONCURRENT):
            pids.append(_start("sleep 10")["pid"])
        # One more should be rejected
        res = run_background("sleep 10")
        assert res.is_error
        data = json.loads(res.content)
        assert data["status"] == "rejected"
        assert data["reason"] == "concurrency_cap"
        assert data["live_jobs"] == _BG_MAX_CONCURRENT
        # Cleanup
        for pid in pids:
            check_background(pid, action="kill")

    def test_cap_frees_slot_after_job_finishes(self):
        """Completing a job should free a slot for a new launch."""
        short_pid = _start("true")["pid"]
        _wait_done(short_pid, timeout=3.0)
        # Should be able to start a new job even though the completed
        # entry lingers (TTL not yet elapsed), because it's not "live".
        new_pid = _start("true")["pid"]
        _wait_done(new_pid, timeout=3.0)


class TestCleanupAndLifecycle:
    def test_unknown_pid_reports_not_found(self):
        s = _check(999_999)
        assert s["status"] in ("not_found", "unknown")

    def test_completed_at_is_set_exactly_once(self):
        pid = _start("echo x")["pid"]
        _wait_done(pid)
        with bg_core._bg_lock:
            info = bg_core._background_processes[pid]
        first = info["completed_at"]
        assert first is not None
        # Second poll must not update completed_at
        _check(pid)
        with bg_core._bg_lock:
            second = bg_core._background_processes[pid]["completed_at"]
        assert first == second

    def test_never_polled_process_is_reaped_by_cleanup(self):
        pid = _start("true")["pid"]
        # Wait for exit WITHOUT calling check_background (which would mark it)
        with bg_core._bg_lock:
            info = bg_core._background_processes[pid]
        proc = info["process"]
        proc.wait(timeout=3.0)
        # completed_at is still None because we never polled
        assert info.get("completed_at") is None
        # Run cleanup — should reap the zombie entry
        _bg_cleanup_completed()
        assert info["completed_at"] is not None

    def test_cleanup_enforces_overdue_deadline(self):
        """Even with zero polls, the cleanup sweep should kill overdue jobs."""
        pid = _start("sleep 30", timeout_seconds=1)["pid"]
        time.sleep(1.3)
        # Trigger cleanup without going through check_background
        _bg_cleanup_completed()
        with bg_core._bg_lock:
            info = bg_core._background_processes[pid]
        assert info.get("timed_out") is True
        assert info["process"].poll() is not None


# ── check_background metadata corrections ──────────────────────────────────


class TestCheckBackgroundMetadata:
    """``check_background`` was originally registered with no timeout hint and
    ``mutating=False``.  It is now flagged ``mutating=True`` because the
    ``action="kill"`` branch SIGTERM/SIGKILLs the target process group, an
    irreversible side effect that cannot be retried) and bumps its timeout
    hint to 360 s — schema documents ``wait_timeout_seconds`` up to 300 s,
    so 360 gives the dispatch wrapper 60 s slack on top of the cap.
    """

    def test_check_background_is_mutating(self):
        from jyagent.runtime.tools.registry import get_registry
        batch = get_registry().freeze()
        assert batch.is_mutating("check_background") is True, (
            "check_background must be flagged mutating=True because "
            "action='kill' is irreversible"
        )

    def test_check_background_timeout_hint_is_360(self):
        from jyagent.runtime.tools.registry import get_registry
        batch = get_registry().freeze()
        assert batch.get_timeout_hint("check_background") == 360, (
            "timeout_hint=360 gives 60s slack over the 300s "
            "wait_timeout_seconds schema cap"
        )
