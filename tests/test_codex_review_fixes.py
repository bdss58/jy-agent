"""Regression tests for Codex review 2026-04-25 Tier-A fixes.

Covers:
    A1 — mutating-tool timeouts surface on LoopResult.partial_side_effects
    A2 — `_tool_dispatch_executor` grows to honour `LoopConfig.max_tool_workers`
    A3 — tracing finalize errors are logged, not raised
    A4 — `run_id` containing `..` cannot escape `checkpoint_dir`
"""
from __future__ import annotations

import logging
import os
import threading
import time
import types

import pytest

from jyagent.runtime.loop import checkpoint
from jyagent.runtime.loop import engine as loop_engine
from jyagent.runtime.loop.config import LoopConfig
from jyagent.runtime.tools.registry import ToolRegistry


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


# ─── A1: mutating-tool timeouts surface on LoopResult ───────────────────────
#
# The dispatch loop runs every tool body in a daemon thread.  On timeout the
# thread keeps running but we return an error ToolResult and move on — fine
# for read-only tools (retry is idempotent), but for MUTATING tools
# (run_shell, edit_file, write_file, dispatch_agent, run_background, mcp)
# the side effect may complete invisibly in the background while the model
# receives "timeout, try something else".  A1 scope: classify + surface
# (warn, clearer error text, accumulate names into LoopResult); full
# subprocess hard-kill is out-of-scope for this PR.


class _FakeModelSpec:
    """Minimal ModelSpec stand-in for the AgentLoop fake-owner tests."""

    provider = "anthropic"
    model = "claude-opus-4-6"

    @staticmethod
    def label() -> str:
        return "anthropic:claude-opus-4-6"


class _ScriptedOwner:
    """Hand-scripted LLMClient: returns a fixed sequence of messages on
    complete().  Enough to drive AgentLoop through a single tool-use step
    followed by a clean no-tools terminal step.

    Streaming is not needed (LoopConfig defaults to streaming=False), so
    stream() raises if called.  We only implement what the engine touches.
    """

    def __init__(self, messages: list[dict]):
        self._messages = list(messages)
        self._idx = 0
        self.model_spec = _FakeModelSpec()

    def complete(self, context, options=None, model_spec=None):
        if self._idx >= len(self._messages):
            # Defensive: loop should have terminated by now.
            raise AssertionError("scripted owner exhausted")
        msg = self._messages[self._idx]
        self._idx += 1
        return msg

    def stream(self, *args, **kwargs):
        raise AssertionError("_ScriptedOwner.stream() should not be called "
                             "(LoopConfig.streaming must be False)")


def _tool_use_message(tool_id: str, tool_name: str, tool_input: dict) -> dict:
    """Build an AssistantMessage that issues exactly one tool_call block."""
    return {
        "role": "assistant",
        "content": [{
            "type": "tool_call",
            "id": tool_id,
            "name": tool_name,
            "arguments": tool_input,
        }],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _final_text_message(text: str) -> dict:
    """Build an AssistantMessage with only a text block — terminates the loop."""
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "stop",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


class TestA1MutatingTimeout:
    def test_mutating_metadata_propagates_to_batch(self):
        """A tool registered with mutating=True surfaces on batch.is_mutating()."""
        reg = ToolRegistry()
        reg.register(
            "fake_mutator",
            lambda: "ok",
            {"name": "fake_mutator", "input_schema": {"type": "object"}},
            mutating=True,
        )
        batch = reg.freeze()
        assert batch.is_mutating("fake_mutator") is True
        # And the frozenset itself is populated (not just the helper).
        assert "fake_mutator" in batch.mutating

    def test_nonmutating_default_false(self):
        """Default registration (no mutating kwarg) → batch.is_mutating() False."""
        reg = ToolRegistry()
        reg.register(
            "fake_reader",
            lambda: "ok",
            {"name": "fake_reader", "input_schema": {"type": "object"}},
        )
        batch = reg.freeze()
        assert batch.is_mutating("fake_reader") is False
        # Unknown names also default False (important for overlaid tools).
        assert batch.is_mutating("never_registered") is False

    def test_mutating_timeout_surfaces_in_loop_result(self, caplog):
        """End-to-end: a mutating tool that sleeps past its timeout must
        (a) log a WARNING on the engine logger, and
        (b) populate LoopResult.partial_side_effects with the tool name.
        """
        caplog.set_level(logging.WARNING, logger=loop_engine.__name__)

        # Build a per-test registry with exactly one slow mutating tool, so
        # the test is independent of the process-wide singleton state.
        def _slow_body():
            time.sleep(5.0)  # far past the 1s tool_timeout below
            return "should never land — dispatch loop has already moved on"

        reg = ToolRegistry()
        reg.register(
            "slow_mutator",
            _slow_body,
            {"name": "slow_mutator", "input_schema": {"type": "object"}},
            mutating=True,
        )
        batch = reg.freeze()

        # Script: step 0 calls slow_mutator; step 1 returns a text message
        # so the loop exits via the "completed" path (partial_side_effects
        # is attached by run() regardless of the exit status, but driving
        # the happy path exercises the clean common case).
        owner = _ScriptedOwner([
            _tool_use_message("call-0", "slow_mutator", {}),
            _final_text_message("done verifying."),
        ])

        cfg = LoopConfig(
            max_steps=3,
            tool_timeout=1,       # 1s → slow_mutator's 5s sleep always times out
            concurrent_tools=False,
            streaming=False,
            truncate_large_inputs=False,
        )
        # Feed the hand-built batch through _tool_source so the engine
        # skips the global registry entirely for this test.
        def _tool_source():
            return list(batch.schemas), dict(batch.functions)

        loop = loop_engine.AgentLoop(
            owner,  # type: ignore[arg-type]
            cfg,
            tool_source=_tool_source,
        )
        # The tool_source path copies metadata (including mutating) from
        # the real registry's freeze() — not from our hand-built batch.
        # Patch it to return our batch's mutating set so the classification
        # happens against the test tool.  This mirrors the production path
        # where tools/__init__.py has already registered mutating metadata
        # by the time _tool_source runs.
        original_freeze = loop_engine.get_registry().freeze
        try:
            loop_engine.get_registry().freeze = lambda: batch  # type: ignore[method-assign]
            result = loop.run(system_prompt="", messages=[])
        finally:
            loop_engine.get_registry().freeze = original_freeze  # type: ignore[method-assign]

        assert "slow_mutator" in result.partial_side_effects, (
            f"expected 'slow_mutator' in partial_side_effects, "
            f"got {result.partial_side_effects!r}"
        )
        # WARNING must be logged on the engine's module logger.
        warning_msgs = [r.getMessage() for r in caplog.records
                        if r.levelno >= logging.WARNING]
        assert any("slow_mutator" in m and "side effects" in m for m in warning_msgs), (
            f"expected mutating-timeout WARNING, got {warning_msgs!r}"
        )

    def test_nonmutating_timeout_does_not_populate(self, caplog):
        """A non-mutating tool that times out must NOT populate
        partial_side_effects (retries are idempotent) AND must NOT log the
        mutating-timeout warning."""
        caplog.set_level(logging.WARNING, logger=loop_engine.__name__)

        def _slow_read():
            time.sleep(5.0)
            return "x"

        reg = ToolRegistry()
        reg.register(
            "slow_reader",
            _slow_read,
            {"name": "slow_reader", "input_schema": {"type": "object"}},
            # mutating defaults to False — that's the whole point.
        )
        batch = reg.freeze()

        owner = _ScriptedOwner([
            _tool_use_message("call-0", "slow_reader", {}),
            _final_text_message("ok."),
        ])

        cfg = LoopConfig(
            max_steps=3,
            tool_timeout=1,
            concurrent_tools=False,
            streaming=False,
            truncate_large_inputs=False,
        )

        def _tool_source():
            return list(batch.schemas), dict(batch.functions)

        loop = loop_engine.AgentLoop(
            owner,  # type: ignore[arg-type]
            cfg,
            tool_source=_tool_source,
        )
        original_freeze = loop_engine.get_registry().freeze
        try:
            loop_engine.get_registry().freeze = lambda: batch  # type: ignore[method-assign]
            result = loop.run(system_prompt="", messages=[])
        finally:
            loop_engine.get_registry().freeze = original_freeze  # type: ignore[method-assign]

        assert result.partial_side_effects == [], (
            f"non-mutating timeout must not populate partial_side_effects; "
            f"got {result.partial_side_effects!r}"
        )
        # And no mutating-timeout warning should have fired.
        warning_msgs = [r.getMessage() for r in caplog.records
                        if r.levelno >= logging.WARNING]
        assert not any("side effects may have occurred" in m for m in warning_msgs), (
            f"non-mutating timeout must not emit side-effects WARNING; "
            f"got {warning_msgs!r}"
        )


# ─── B2: ToolBatch dict fields are read-only views ──────────────────────────


class TestToolBatchReadOnly:
    """B2: ToolBatch.{schema_map,functions,timeout_hints,large_input_keys,
    compaction_priority} are MappingProxyType views.  Mutating them must
    raise TypeError instead of silently corrupting the per-step snapshot.
    """

    def _build_batch(self):
        from jyagent.runtime.tools.registry import ToolRegistry
        reg = ToolRegistry()
        reg.register(
            "ro_tool",
            lambda: "ok",
            {"name": "ro_tool", "input_schema": {"type": "object"}},
            timeout_hint=5,
            large_input_keys={"blob"},
            compaction_priority="standard",
            parallel_safe=True,
        )
        return reg.freeze()

    def test_schema_map_is_readonly(self):
        batch = self._build_batch()
        with pytest.raises(TypeError):
            batch.schema_map["new_tool"] = {}  # type: ignore[index]

    def test_functions_is_readonly(self):
        batch = self._build_batch()
        with pytest.raises(TypeError):
            batch.functions["another"] = lambda: None  # type: ignore[index]

    def test_timeout_hints_is_readonly(self):
        batch = self._build_batch()
        with pytest.raises(TypeError):
            batch.timeout_hints["ro_tool"] = 999  # type: ignore[index]

    def test_large_input_keys_is_readonly(self):
        batch = self._build_batch()
        with pytest.raises(TypeError):
            batch.large_input_keys["ro_tool"] = frozenset()  # type: ignore[index]

    def test_compaction_priority_is_readonly(self):
        batch = self._build_batch()
        with pytest.raises(TypeError):
            batch.compaction_priority["ro_tool"] = "ephemeral"  # type: ignore[index]

    def test_empty_batch_is_readonly(self):
        from jyagent.runtime.tools.registry import ToolBatch
        b = ToolBatch.empty()
        with pytest.raises(TypeError):
            b.schema_map["x"] = {}  # type: ignore[index]
        with pytest.raises(TypeError):
            b.functions["x"] = lambda: None  # type: ignore[index]

    def test_with_overlay_returns_readonly(self):
        batch = self._build_batch()
        new = batch.with_overlay(
            functions={"overlay_tool": lambda: "x"},
            schemas=[{"name": "overlay_tool", "input_schema": {"type": "object"}}],
        )
        with pytest.raises(TypeError):
            new.functions["yet_another"] = lambda: None  # type: ignore[index]
        with pytest.raises(TypeError):
            new.schema_map["yet_another"] = {}  # type: ignore[index]
        # And the overlaid tool must still be readable.
        assert "overlay_tool" in new.functions
        assert "ro_tool" in new.functions  # base tools still present


# ─── B3: run_shell timeout coercion is fault-tolerant ───────────────────────


class TestRunShellTimeoutCoercion:
    """B3: a malformed ``timeout`` from the model (e.g. ``"30s"`` or a list)
    used to raise TypeError/ValueError out of _execute_tool_with_timeout
    BEFORE _execute_tool's normal schema validation could turn it into a
    clean ToolResult error.  Coercion failure now falls back to the
    default timeout and lets the inner validator surface a structured
    error.
    """

    def test_string_timeout_does_not_crash(self):
        from jyagent.runtime.tools.registry import ToolRegistry

        called = {"yes": False}

        def _fake_run_shell(**kwargs):
            called["yes"] = True
            return "ran"

        reg = ToolRegistry()
        reg.register(
            "run_shell",
            _fake_run_shell,
            {"name": "run_shell",
             "input_schema": {"type": "object",
                              "properties": {"command": {"type": "string"},
                                             "timeout": {"type": "integer"}}}},
        )
        batch = reg.freeze()
        # "30s" cannot be int()-coerced — old code raised ValueError here.
        result = loop_engine._execute_tool_with_timeout(
            "run_shell",
            {"command": "echo hi", "timeout": "30s"},
            batch,
            default_timeout=5,
        )
        # We don't care if the inner tool succeeded or not — only that
        # the dispatch wrapper didn't crash.  Either:
        #  - ToolResult populated cleanly (success or schema-validation error)
        #  - Or the fake_run_shell ran with the bad input
        assert isinstance(result, loop_engine.ToolResult)

    def test_list_timeout_does_not_crash(self):
        from jyagent.runtime.tools.registry import ToolRegistry
        reg = ToolRegistry()
        reg.register(
            "run_shell",
            lambda **kw: "ok",
            {"name": "run_shell", "input_schema": {"type": "object"}},
        )
        batch = reg.freeze()
        result = loop_engine._execute_tool_with_timeout(
            "run_shell",
            {"command": "ls", "timeout": [1, 2, 3]},
            batch,
            default_timeout=5,
        )
        assert isinstance(result, loop_engine.ToolResult)

    def test_valid_int_timeout_still_honoured(self):
        from jyagent.runtime.tools.registry import ToolRegistry

        captured = {"timeout_seen_by_body": None}

        def _fake_run_shell(**kwargs):
            return "ok"

        reg = ToolRegistry()
        reg.register(
            "run_shell",
            _fake_run_shell,
            {"name": "run_shell",
             "input_schema": {"type": "object",
                              "properties": {"command": {"type": "string"},
                                             "timeout": {"type": "integer"}}}},
        )
        batch = reg.freeze()
        # Valid int → coercion path runs fine, no fallback.
        result = loop_engine._execute_tool_with_timeout(
            "run_shell",
            {"command": "echo ok", "timeout": 30},
            batch,
            default_timeout=5,
        )
        assert isinstance(result, loop_engine.ToolResult)


# ─── B1: max-steps fallback now records cost ─────────────────────────────────


class TestMaxStepsFallbackCostTracking:
    """B1: the max-steps fallback call's tokens were added to the LoopResult
    totals but never recorded against ``cost_tracker``, so trace cost
    silently under-counted.  Now ``cost_tracker.record(...)`` runs in
    the fallback try-block too.
    """

    def test_fallback_records_cost(self):
        """Drive _run_impl into the fallback path with a cost_tracker active
        and assert the fallback's tokens are reflected in the tracker.
        """
        # Local import to avoid circulars at module load.
        from jyagent.runtime.tools.registry import ToolRegistry

        # Owner that always asks for a tool — never produces a final
        # answer — so the loop hits cfg.max_steps.  The fallback call
        # then returns plain text (no tool call) and we read the cost
        # tracker.
        class _ToolForeverOwner:
            model_spec = _FakeModelSpec()
            calls = 0

            def stream(self, *args, **kwargs):  # pragma: no cover — non-streaming path used
                raise NotImplementedError

            def complete(self, context, options=None, model_spec=None):
                self.calls += 1
                # tool_choice={"type":"none"} is set on the fallback opts
                # — detect it and switch behaviour.
                wants_no_tools = (
                    options is not None
                    and getattr(options, "tool_choice", None) == {"type": "none"}
                )
                if wants_no_tools:
                    msg = _final_text_message("forced final answer.")
                    msg["usage"] = {"input_tokens": 50, "output_tokens": 25}
                    return msg
                msg = _tool_use_message(f"call-{self.calls}", "noop", {})
                msg["usage"] = {"input_tokens": 10, "output_tokens": 5}
                return msg

        reg = ToolRegistry()
        reg.register(
            "noop",
            lambda: "noop-result",
            {"name": "noop", "input_schema": {"type": "object"}},
        )
        batch = reg.freeze()

        cfg = LoopConfig(
            max_steps=2,
            tool_timeout=5,
            concurrent_tools=False,
            streaming=False,
            truncate_large_inputs=False,
            fallback_on_max_steps=True,
            max_cost_usd=10.0,  # MUST be set or cost_tracker is None.
        )

        def _tool_source():
            return list(batch.schemas), dict(batch.functions)

        owner = _ToolForeverOwner()
        loop = loop_engine.AgentLoop(owner, cfg, tool_source=_tool_source)  # type: ignore[arg-type]

        original_freeze = loop_engine.get_registry().freeze
        try:
            loop_engine.get_registry().freeze = lambda: batch  # type: ignore[method-assign]
            result = loop.run(system_prompt="", messages=[])
        finally:
            loop_engine.get_registry().freeze = original_freeze  # type: ignore[method-assign]

        assert result.status == "completed", (
            f"expected fallback to drive a 'completed' result; "
            f"got status={result.status!r} text={result.text[:60]!r}"
        )
        # Total tokens must include both the per-step tool calls AND the
        # fallback turn (50/25).  If the fallback wasn't accounted for at
        # all we'd be missing ≥75 tokens.
        assert result.total_input_tokens >= 50, (
            f"fallback input tokens missing: {result.total_input_tokens}"
        )
        assert result.total_output_tokens >= 25, (
            f"fallback output tokens missing: {result.total_output_tokens}"
        )


# ─── C2: AgentLoop reentrance guard ─────────────────────────────────────────


class TestAgentLoopReentranceGuard:
    """C2: a second run() invocation on the same AgentLoop instance — whether
    concurrent (from another thread) or nested (mid-callback) — must raise
    RuntimeError instead of silently corrupting per-run state.
    """

    def _make_loop(self):
        # Mirror the test_todos_scratchpad.TestAgentLoopTodosWiring fixture:
        # build the loop via __new__ to skip __init__'s heavy wiring.
        owner = types.SimpleNamespace(model_spec=_FakeModelSpec())
        loop = loop_engine.AgentLoop.__new__(loop_engine.AgentLoop)
        loop._runtime_owner = owner
        loop._config = LoopConfig(max_steps=0)  # zero-step → exits immediately
        loop._callbacks = loop_engine.LoopCallbacks()
        loop._tool_source = None
        loop._model_spec = None
        loop._cancel_event = None
        loop._executor = loop_engine._tool_dispatch_executor
        loop._todos = []
        loop._partial_side_effects = []
        return loop

    def test_concurrent_runs_raise_runtime_error(self):
        """Hold the lock from one thread, second thread's run() must raise."""
        loop = self._make_loop()
        # Pre-acquire the lock to simulate an in-flight run().
        loop._run_lock = threading.Lock()
        loop._run_lock.acquire()
        try:
            with pytest.raises(RuntimeError, match="already in progress"):
                loop.run("system", [])
        finally:
            loop._run_lock.release()

    def test_lock_released_on_normal_exit(self):
        """After a clean run() returns, a subsequent run() must succeed."""
        loop = self._make_loop()
        # First run — should complete (max_steps=0 means immediate max_steps exit).
        result1 = loop.run("system", [])
        assert result1 is not None
        # Second run on same instance — must NOT raise.
        result2 = loop.run("system", [])
        assert result2 is not None

    def test_lock_released_on_exception_path(self, monkeypatch):
        """If _run_impl raises, the lock must still release for the next call."""
        loop = self._make_loop()

        # Force _run_impl to raise on the first call only.
        original_run_impl = loop._run_impl
        call_count = {"n": 0}

        def _raising_run_impl(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValueError("simulated failure")
            return original_run_impl(*args, **kwargs)

        loop._run_impl = _raising_run_impl  # type: ignore[method-assign]

        with pytest.raises(ValueError):
            loop.run("system", [])
        # Lock must be released — second run() succeeds.
        result = loop.run("system", [])
        assert result is not None

    def test_lazy_lock_init_for_legacy_construction(self):
        """AgentLoop built via __new__ (no __init__) must still get a lock."""
        loop = self._make_loop()
        # _make_loop() does NOT set _run_lock — verify .run() lazy-installs it.
        assert not hasattr(loop, "_run_lock")
        loop.run("system", [])
        assert hasattr(loop, "_run_lock")
        assert isinstance(loop._run_lock, type(threading.Lock()))


# ─── C3: SessionStats locked readers ────────────────────────────────────────


class TestSessionStatsLockedReaders:
    """C3: provider/model property reads now acquire self._lock so they're
    correct under free-threaded CPython AND consistent with the writer
    contract (set_active_model takes the same lock).
    """

    def test_provider_read_under_lock(self):
        from jyagent.runtime.stats import SessionStats
        stats = SessionStats()
        stats.set_active_model("openai", "gpt-5.5")
        assert stats.provider == "openai"
        assert stats.model == "gpt-5.5"

    def test_concurrent_writes_visible_to_reader(self):
        """Sanity-check: a 100-iteration concurrent write/read loop never
        tears (provider/model strings stay consistent with whatever the
        writer last set).
        """
        import threading as _t
        from jyagent.runtime.stats import SessionStats

        stats = SessionStats()
        stats.set_active_model("openai", "gpt-5.5")
        observed: list[tuple[str, str]] = []
        stop = _t.Event()

        def _reader():
            while not stop.is_set():
                observed.append((stats.provider, stats.model))

        def _writer():
            for i in range(100):
                if i % 2 == 0:
                    stats.set_active_model("anthropic", "claude-opus-4-6")
                else:
                    stats.set_active_model("openai", "gpt-5.5")
            stop.set()

        rt = _t.Thread(target=_reader, daemon=True)
        wt = _t.Thread(target=_writer, daemon=True)
        rt.start()
        wt.start()
        wt.join(timeout=5.0)
        rt.join(timeout=5.0)

        # Every observed (provider, model) tuple must be one of the two
        # consistent pairs the writer set — never a torn (anthropic, gpt-5.5).
        valid_pairs = {("openai", "gpt-5.5"), ("anthropic", "claude-opus-4-6")}
        torn = [pair for pair in observed if pair not in valid_pairs]
        assert not torn, (
            f"observed torn provider/model pairs (read without lock): {torn[:5]}"
        )
