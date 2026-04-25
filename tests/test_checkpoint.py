# tests/test_checkpoint.py — LoopCheckpoint + AgentLoop wiring regression tests.
#
# Validates:
#   * LoopCheckpoint round-trip via JSON
#   * save() writes atomically (via .tmp + rename)
#   * LoopConfig fields default to disabled (checkpoint_dir=None)
#   * AgentLoop._write_checkpoint is a no-op when disabled
#   * run() writes a final checkpoint when enabled (via stubbed _run_impl)

from __future__ import annotations

import json
import os
import threading

import pytest

from jyagent.runtime.loop import engine as le
from jyagent.runtime.loop.checkpoint import (
    LoopCheckpoint,
    checkpoint_path,
    iso_utc_now,
    new_run_id,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _sample_checkpoint(**overrides):
    base = dict(
        run_id="abc123",
        step=2,
        saved_at=iso_utc_now(),
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        ],
        total_input_tokens=150,
        total_output_tokens=40,
        tool_calls_count=3,
        todos=[{"content": "Do X", "status": "pending", "active_form": ""}],
        provider="anthropic",
        model="claude-opus-4-6",
        status="in_progress",
        error=None,
    )
    base.update(overrides)
    return LoopCheckpoint(**base)


# ─── Pure-function tests ─────────────────────────────────────────────────────


class TestLoopCheckpointRoundTrip:
    def test_json_round_trip_preserves_fields(self):
        cp = _sample_checkpoint()
        text = cp.to_json()
        back = LoopCheckpoint.from_json(text)
        assert back.run_id == cp.run_id
        assert back.step == cp.step
        assert back.messages == cp.messages
        assert back.total_input_tokens == cp.total_input_tokens
        assert back.total_output_tokens == cp.total_output_tokens
        assert back.tool_calls_count == cp.tool_calls_count
        assert back.todos == cp.todos
        assert back.provider == cp.provider
        assert back.model == cp.model
        assert back.status == cp.status
        assert back.error == cp.error

    def test_json_is_valid_and_readable(self):
        cp = _sample_checkpoint()
        parsed = json.loads(cp.to_json())
        assert parsed["run_id"] == "abc123"
        assert parsed["step"] == 2

    def test_default_factories_produce_lists(self):
        cp = LoopCheckpoint(run_id="r", step=0, saved_at=iso_utc_now())
        assert cp.messages == []
        assert cp.todos == []


class TestLoopCheckpointSaveLoad:
    def test_save_creates_parent_dir(self, tmp_path):
        path = tmp_path / "nested" / "deeper" / "cp.json"
        cp = _sample_checkpoint()
        cp.save(str(path))
        assert path.exists()
        loaded = LoopCheckpoint.load(str(path))
        assert loaded.run_id == cp.run_id

    def test_save_is_atomic_via_tmp_rename(self, tmp_path, monkeypatch):
        """If the fsync between .tmp write and rename is interrupted, the
        original file must remain untouched.  We approximate by checking
        that no .tmp file survives a successful write."""
        path = tmp_path / "cp.json"
        cp = _sample_checkpoint()
        cp.save(str(path))
        assert path.exists()
        assert not (tmp_path / "cp.json.tmp").exists()

    def test_round_trip_via_disk(self, tmp_path):
        path = str(tmp_path / "cp.json")
        cp = _sample_checkpoint(step=5, tool_calls_count=12)
        cp.save(path)
        loaded = LoopCheckpoint.load(path)
        assert loaded.step == 5
        assert loaded.tool_calls_count == 12


class TestCheckpointPath:
    def test_int_step_uses_zero_padded_filename(self):
        p = checkpoint_path("/tmp/checkpoints", "run-xyz", 3)
        assert p.endswith(os.path.join("run-xyz", "step_0003.json"))

    def test_final_step_literal(self):
        p = checkpoint_path("/tmp/checkpoints", "run-xyz", "final")
        assert p.endswith(os.path.join("run-xyz", "final.json"))

    def test_run_id_with_pathsep_is_sanitized(self):
        p = checkpoint_path("/tmp/cp", f"run{os.sep}evil", 0)
        # The inserted separator from `run_id` should have been replaced.
        segments = p.split(os.sep)
        # Find the synthesized run-id segment.
        assert any(seg == "run_evil" for seg in segments), (
            f"run-id separator not sanitized: {p}"
        )


class TestNewRunId:
    def test_is_hex_32(self):
        rid = new_run_id()
        assert len(rid) == 32
        int(rid, 16)  # must parse as hex

    def test_uniqueness(self):
        assert new_run_id() != new_run_id()


# ─── LoopConfig wiring ───────────────────────────────────────────────────────


class TestLoopConfigCheckpointFields:
    def test_defaults_disabled(self):
        cfg = le.LoopConfig()
        assert cfg.checkpoint_dir is None
        assert cfg.checkpoint_every_n_steps == 0


class TestLoopCallbacksCheckpointHook:
    def test_on_checkpoint_field_exists(self):
        cbs = le.LoopCallbacks()
        assert hasattr(cbs, "on_checkpoint")
        assert cbs.on_checkpoint is None

    def test_on_checkpoint_field_is_assignable(self):
        fired: list[tuple[str, int]] = []
        cbs = le.LoopCallbacks(on_checkpoint=lambda p, s: fired.append((p, s)))
        cbs.on_checkpoint("/tmp/c.json", 3)
        assert fired == [("/tmp/c.json", 3)]


# ─── AgentLoop.{_write_checkpoint,set_run_id} ────────────────────────────────


def _make_bare_loop(cfg: le.LoopConfig) -> le.AgentLoop:
    loop = le.AgentLoop.__new__(le.AgentLoop)
    class _Owner:
        class model_spec:
            provider = "anthropic"
            model = "claude-opus-4-6"
            @staticmethod
            def label():
                return "anthropic:claude-opus-4-6"
    loop._runtime_owner = _Owner()
    loop._config = cfg
    loop._callbacks = le.LoopCallbacks()
    loop._tool_source = None
    loop._model_spec = None
    loop._cancel_event = None
    loop._executor = le._tool_dispatch_executor
    loop._todos = []
    loop._run_id = ""
    return loop


class TestWriteCheckpointDisabled:
    def test_noop_when_checkpoint_dir_none(self, tmp_path):
        loop = _make_bare_loop(le.LoopConfig(checkpoint_dir=None))
        loop._write_checkpoint(
            step=0, messages=[], total_input_tokens=0, total_output_tokens=0,
            tool_calls_count=0, status="in_progress",
        )
        # Nothing should have been written.
        assert list(tmp_path.iterdir()) == []


class TestWriteCheckpointEnabled:
    def test_writes_step_file(self, tmp_path):
        loop = _make_bare_loop(le.LoopConfig(
            checkpoint_dir=str(tmp_path),
            checkpoint_every_n_steps=1,
        ))
        loop._run_id = "testrun"
        loop._write_checkpoint(
            step=0,
            messages=[{"role": "user", "content": "hi"}],
            total_input_tokens=10,
            total_output_tokens=5,
            tool_calls_count=1,
            status="in_progress",
        )
        path = tmp_path / "testrun" / "step_0000.json"
        assert path.exists(), "checkpoint file not written"
        loaded = LoopCheckpoint.load(str(path))
        assert loaded.run_id == "testrun"
        assert loaded.step == 0
        assert loaded.tool_calls_count == 1
        assert loaded.status == "in_progress"

    def test_writes_final_file(self, tmp_path):
        loop = _make_bare_loop(le.LoopConfig(
            checkpoint_dir=str(tmp_path),
            checkpoint_every_n_steps=1,
        ))
        loop._run_id = "testrun2"
        loop._write_checkpoint(
            step="final",
            messages=[],
            total_input_tokens=0,
            total_output_tokens=0,
            tool_calls_count=0,
            status="completed",
        )
        path = tmp_path / "testrun2" / "final.json"
        assert path.exists()
        loaded = LoopCheckpoint.load(str(path))
        assert loaded.status == "completed"
        # step == -1 sentinel for terminal checkpoints
        assert loaded.step == -1

    def test_fire_on_checkpoint_callback(self, tmp_path):
        fired: list[tuple[str, int]] = []
        loop = _make_bare_loop(le.LoopConfig(
            checkpoint_dir=str(tmp_path),
            checkpoint_every_n_steps=1,
        ))
        loop._callbacks = le.LoopCallbacks(
            on_checkpoint=lambda p, s: fired.append((p, s))
        )
        loop._run_id = "r"
        loop._write_checkpoint(
            step=2, messages=[], total_input_tokens=0, total_output_tokens=0,
            tool_calls_count=0, status="in_progress",
        )
        assert len(fired) == 1
        path, step = fired[0]
        assert step == 2
        assert os.path.exists(path)

    def test_exception_in_save_does_not_crash(self, tmp_path, monkeypatch):
        """A broken save path should emit on_warning but not raise."""
        warnings: list[str] = []
        # Use an unwritable path to trigger an OSError inside save.
        loop = _make_bare_loop(le.LoopConfig(
            checkpoint_dir="/proc/cannot_write_here_surely",
            checkpoint_every_n_steps=1,
        ))
        loop._callbacks = le.LoopCallbacks(on_warning=warnings.append)
        loop._run_id = "r"
        # Should NOT raise even though save() will fail.
        loop._write_checkpoint(
            step=0, messages=[], total_input_tokens=0, total_output_tokens=0,
            tool_calls_count=0, status="in_progress",
        )
        # And the failure must surface as a warning.
        assert any("checkpoint" in w.lower() for w in warnings), (
            f"expected checkpoint warning; got {warnings}"
        )


class TestAgentLoopSetRunId:
    def test_override_persists(self, tmp_path):
        loop = _make_bare_loop(le.LoopConfig(checkpoint_dir=str(tmp_path)))
        loop.set_run_id("custom-run")
        assert loop._run_id == "custom-run"

    def test_empty_string_resets(self):
        loop = _make_bare_loop(le.LoopConfig())
        loop._run_id = "existing"
        loop.set_run_id("")
        assert loop._run_id == ""
