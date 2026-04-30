"""Regression tests for runtime safety fixes.

Covers:
    - Mutating-tool timeouts surface on LoopResult.partial_side_effects.
    - `_tool_dispatch_executor` grows to honour `LoopConfig.max_tool_workers`.
    - Tracing finalize errors are logged, not raised.
    - `run_id` containing `..` cannot escape `checkpoint_dir`.
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


# ─── Path sanitisation ──────────────────────────────────────────────────────


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


# ─── Tracing finalize errors are non-fatal ──────────────────────────────────


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


# ─── Dispatch executor honours max_tool_workers ─────────────────────────────


class TestDispatchExecutorGrowsWithConfig:
    def test_get_executor_grows_on_demand(self, monkeypatch):
        """Requesting more workers than current cap grows the pool."""
        # Snapshot + reset module state so the test is independent.
        # The canonical home for this state is now
        # runtime/loop/tool_executor.py.  Restoring by writing through
        # ``loop_engine._tool_dispatch_executor`` would create a STATIC
        # attribute that shadows the PEP-562 ``__getattr__`` passthrough,
        # breaking later tests (e.g. test_backcompat_alias_points_to_dispatch)
        # that expect the back-compat names to mirror the live pool.
        from jyagent.runtime.loop import tool_executor as _te
        original_executor = _te.tool_dispatch_executor
        original_cap = _te.tool_dispatch_cap
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
            _te.tool_dispatch_executor = original_executor
            _te.tool_dispatch_cap = original_cap

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


# ─── Mutating-tool timeouts surface on LoopResult ───────────────────────────
#
# The dispatch loop runs every tool body in a daemon thread.  On timeout the
# thread keeps running but we return an error ToolResult and move on — fine
# for read-only tools (retry is idempotent), but for MUTATING tools
# (run_shell, edit_file, write_file, dispatch_agent, run_background, mcp)
# the side effect may complete invisibly in the background while the model
# receives "timeout, try something else".  Scope: classify + surface
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


# ─── ToolBatch dict fields are read-only views ──────────────────────────────


class TestToolBatchReadOnly:
    """ToolBatch.{schema_map,functions,timeout_hints,large_input_keys,
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


# ─── run_shell timeout coercion is fault-tolerant ───────────────────────────


class TestRunShellTimeoutCoercion:
    """A malformed ``timeout`` from the model (e.g. ``"30s"`` or a list)
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


# ─── Max-steps fallback now records cost ────────────────────────────────────


class TestMaxStepsFallbackCostTracking:
    """The max-steps fallback call's tokens were added to the LoopResult
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

# ─── Max-steps fallback preserves prompt cache (system_prompt stable) ──────


class TestMaxStepsFallbackPromptCache:
    """The fallback call MUST NOT mutate ``system_prompt``.

    Mutating system_prompt mid-run breaks Anthropic prompt caching (~12×
    cost penalty on the cached portion).  The directive must be injected
    as a tail user message instead — durable rule from MEMORY.md.

    This is a regression test for the fix that replaced
    ``system_prompt + "\\n\\n[SYSTEM: ...]"`` with a tail user-message
    directive in the fallback path.
    """

    def test_fallback_does_not_mutate_system_prompt(self):
        from jyagent.runtime.tools.registry import ToolRegistry

        ORIGINAL_SYSTEM = "You are a helpful agent. Original system prompt."
        captured: dict = {}

        class _ToolForeverOwner:
            model_spec = _FakeModelSpec()

            def stream(self, *args, **kwargs):
                raise NotImplementedError

            def complete(self, context, options=None, model_spec=None):
                wants_no_tools = (
                    options is not None
                    and getattr(options, "tool_choice", None) == {"type": "none"}
                )
                if wants_no_tools:
                    # This IS the fallback call.  Capture what it received.
                    captured["fallback_system_prompt"] = context.get("system_prompt")
                    captured["fallback_messages"] = list(context.get("messages", []))
                    msg = _final_text_message("ok.")
                    msg["usage"] = {"input_tokens": 1, "output_tokens": 1}
                    return msg
                # Normal step: ask for a tool to drive into max_steps.
                tool_msg = {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_call",
                        "id": "t1",
                        "name": "noop",
                        "arguments": {},
                    }],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 5, "output_tokens": 5},
                }
                return tool_msg

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
        )

        def _tool_source():
            return list(batch.schemas), dict(batch.functions)

        owner = _ToolForeverOwner()
        loop = loop_engine.AgentLoop(owner, cfg, tool_source=_tool_source)  # type: ignore[arg-type]

        original_freeze = loop_engine.get_registry().freeze
        try:
            loop_engine.get_registry().freeze = lambda: batch  # type: ignore[method-assign]
            result = loop.run(system_prompt=ORIGINAL_SYSTEM, messages=[])
        finally:
            loop_engine.get_registry().freeze = original_freeze  # type: ignore[method-assign]

        # 1. The fallback call must have received the ORIGINAL system_prompt
        #    byte-identically — this is what keeps the Anthropic prompt cache
        #    warm across the fallback turn.
        assert captured.get("fallback_system_prompt") == ORIGINAL_SYSTEM, (
            "fallback mutated system_prompt — broke Anthropic prompt cache. "
            f"got: {captured.get('fallback_system_prompt')!r}"
        )

        # 2. The directive must appear as a tail user message in the
        #    fallback's message list (so the model still sees the
        #    instruction; it just lives in a non-cached suffix).
        fb_messages = captured.get("fallback_messages") or []
        assert fb_messages, "fallback received empty messages list"
        tail = fb_messages[-1]
        assert tail.get("role") == "user", (
            f"expected tail user-role directive; got role={tail.get('role')!r}"
        )
        tail_text = ""
        for block in tail.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                tail_text += block.get("text", "")
        assert "WITHOUT using any tools" in tail_text, (
            f"tail user message missing finalize directive; got: {tail_text!r}"
        )

        # 3. The persisted transcript on LoopResult must include both the
        #    directive and the fallback assistant reply (symmetric history).
        assert result.status == "completed"
        assert any(
            m.get("role") == "user"
            and any(
                isinstance(b, dict)
                and b.get("type") == "text"
                and "WITHOUT using any tools" in b.get("text", "")
                for b in m.get("content", [])
            )
            for m in result.messages
        ), "finalize directive not persisted in result.messages"



# ─── AgentLoop reentrance guard ─────────────────────────────────────────────


class TestAgentLoopReentranceGuard:
    """A second run() invocation on the same AgentLoop instance — whether
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


# ─── SessionStats locked readers ────────────────────────────────────────────


class TestSessionStatsLockedReaders:
    """Provider/model property reads now acquire self._lock so they're
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


# ─── Cancellation latency in LLM calls ──────────────────────────────────────


class TestC1CancellationLatency:
    """A cancel_event set during a slow LLM complete() or stream()
    call must unblock the call within ~200 ms instead of waiting for
    the provider timeout.
    """

    def _make_loop_with_owner(self, owner, cancel_event):
        """Build an AgentLoop via __new__ just like the todos fixture,
        but wire the owner + cancel_event so _call_streaming /
        _call_complete are the only thing under test."""
        loop = loop_engine.AgentLoop.__new__(loop_engine.AgentLoop)
        loop._runtime_owner = owner
        loop._config = LoopConfig(max_steps=1, streaming=False)
        loop._callbacks = loop_engine.LoopCallbacks()
        loop._tool_source = None
        loop._model_spec = None
        loop._cancel_event = cancel_event
        loop._executor = loop_engine._tool_dispatch_executor
        loop._todos = []
        loop._partial_side_effects = []
        return loop

    def test_complete_cancels_within_200ms(self):
        """_call_complete must raise KeyboardInterrupt within ~200 ms
        of cancel_event.set() instead of waiting for the 10 s fake
        provider call to finish.
        """
        class _SlowOwner:
            model_spec = _FakeModelSpec()
            entered = threading.Event()

            def complete(self, *args, **kwargs):
                self.entered.set()
                time.sleep(10.0)  # simulates a stuck provider
                return _final_text_message("should never return")

            def stream(self, *args, **kwargs):
                raise AssertionError("stream should not be called")

        owner = _SlowOwner()
        cancel_event = threading.Event()
        loop = self._make_loop_with_owner(owner, cancel_event)

        # Kick off the call in a worker; trigger cancel from main.
        result_holder: dict = {}

        def _call_target():
            try:
                loop._call_complete({}, loop_engine.LLMOptions())
                result_holder["outcome"] = "returned"
            except KeyboardInterrupt as e:
                result_holder["outcome"] = "cancelled"
                result_holder["exc"] = e
            except BaseException as e:  # noqa: BLE001
                result_holder["outcome"] = "other_error"
                result_holder["exc"] = e

        t = threading.Thread(target=_call_target, daemon=True)
        t0 = time.monotonic()
        t.start()
        # Wait until the worker has actually entered the fake provider call,
        # so the cancel fires DURING the sleep not before it.
        assert owner.entered.wait(2.0), "slow owner never entered complete()"
        cancel_event.set()
        t.join(timeout=2.0)
        elapsed = time.monotonic() - t0

        assert not t.is_alive(), "call thread did not unblock after cancel"
        assert result_holder.get("outcome") == "cancelled", (
            f"expected KeyboardInterrupt, got {result_holder!r}"
        )
        # Must unblock quickly — 1 s is generous (poll is 100 ms).
        assert elapsed < 1.5, (
            f"cancel latency too high: {elapsed:.2f}s (poll is 100ms)"
        )

    def test_complete_no_cancel_event_uses_fast_path(self):
        """When cancel_event is None, _call_complete must run in the
        current thread (no daemon worker), so the call returns its
        result directly.
        """
        class _FastOwner:
            model_spec = _FakeModelSpec()

            def complete(self, *args, **kwargs):
                msg = _final_text_message("fast answer")
                msg["usage"] = {"input_tokens": 5, "output_tokens": 3}
                return msg

            def stream(self, *args, **kwargs):
                raise AssertionError("stream should not be called")

        owner = _FastOwner()
        loop = self._make_loop_with_owner(owner, cancel_event=None)
        step_text, tool_calls, stop, msg = loop._call_complete({}, loop_engine.LLMOptions())
        assert step_text == "fast answer"
        assert stop == "stop"
        assert tool_calls == []

    def test_complete_precheck_fires_before_call(self):
        """If cancel_event is already set when _call_complete is invoked,
        the provider call must NOT be issued."""
        class _WouldBeCalledOwner:
            model_spec = _FakeModelSpec()
            was_called = False

            def complete(self, *args, **kwargs):
                self.was_called = True
                return _final_text_message("x")

            def stream(self, *args, **kwargs):
                raise AssertionError("stream should not be called")

        owner = _WouldBeCalledOwner()
        cancel_event = threading.Event()
        cancel_event.set()  # pre-set
        loop = self._make_loop_with_owner(owner, cancel_event)

        with pytest.raises(KeyboardInterrupt, match="before complete"):
            loop._call_complete({}, loop_engine.LLMOptions())
        assert not owner.was_called, "complete() was issued despite pre-set cancel"

    def test_streaming_watcher_closes_stream_on_cancel(self):
        """_call_streaming must spawn a watcher that calls stream.close()
        when cancel_event fires.  We verify the close() call, not the
        end-to-end stream path (that's exercised by existing streaming
        tests in the suite)."""

        close_called = threading.Event()
        entered = threading.Event()

        class _StuckStream:
            """Stream whose __iter__ blocks until close() is called
            (simulating a network-stuck provider)."""
            _stopped = threading.Event()

            def __iter__(self):
                entered.set()
                # Block until close() unblocks us.
                self._stopped.wait(timeout=5.0)
                # Iterator exits cleanly — loop sees no events.
                return iter([])

            def close(self):
                close_called.set()
                self._stopped.set()

            def get_final_message(self):
                return _final_text_message("partial")

        class _StreamOwner:
            model_spec = _FakeModelSpec()

            def complete(self, *args, **kwargs):
                raise AssertionError("complete should not be called")

            def stream(self, *args, **kwargs):
                return _StuckStream()

        cancel_event = threading.Event()
        owner = _StreamOwner()
        loop = self._make_loop_with_owner(owner, cancel_event)
        loop._config = LoopConfig(max_steps=1, streaming=True)

        def _call_target():
            try:
                loop._call_streaming({}, loop_engine.LLMOptions())
            except BaseException:  # noqa: BLE001 — any exit is fine for this test
                pass

        t = threading.Thread(target=_call_target, daemon=True)
        t.start()
        assert entered.wait(2.0), "streaming never entered __iter__"
        t0 = time.monotonic()
        cancel_event.set()
        assert close_called.wait(1.0), "watcher never called stream.close()"
        elapsed = time.monotonic() - t0
        # Watcher poll is 50 ms; close should fire within ~200 ms.
        assert elapsed < 0.5, f"watcher close latency too high: {elapsed:.2f}s"
        t.join(timeout=2.0)
        assert not t.is_alive(), "streaming call did not unblock after close"


# ─── CostTracker extracted to runtime/loop/cost.py ──────────────────────────


class TestC4Phase1CostExtraction:
    """_CostTracker moved to runtime/loop/cost.py
    under the name CostTracker.  Engine re-exports as _CostTracker for
    internal back-compat.  Both import paths must work.
    """

    def test_cost_tracker_importable_from_new_home(self):
        from jyagent.runtime.loop.cost import CostTracker
        ct = CostTracker()
        assert ct.total_cost == 0.0
        assert ct.unpriced_calls == 0
        assert ct.cost == 0.0
        assert ct.has_unpriced_usage is False

    def test_engine_reexport_still_works(self):
        """Engine's private `_CostTracker` alias must still point at the
        same class for any internal call site that didn't migrate."""
        from jyagent.runtime.loop import engine as _engine
        from jyagent.runtime.loop.cost import CostTracker
        assert _engine._CostTracker is CostTracker

    def test_cost_tracker_records_priced_call(self):
        from jyagent.runtime.loop.cost import CostTracker
        ct = CostTracker()
        # Anthropic pricing entry exists; cost_usd > 0 for nonzero usage.
        ct.record(
            {"input_tokens": 1000, "output_tokens": 500},
            "anthropic",
            "claude-opus-4-6",
        )
        assert ct.cost > 0.0
        assert ct.unpriced_calls == 0

    def test_cost_tracker_flags_unpriced(self):
        from jyagent.runtime.loop.cost import CostTracker
        ct = CostTracker()
        ct.record(
            {"input_tokens": 1000, "output_tokens": 500},
            "nonexistent-provider",
            "nonexistent-model",
        )
        assert ct.has_unpriced_usage is True
        assert ct.cost == 0.0  # lower bound


# ─── Tool executor extracted to runtime/loop/tool_executor.py ────────────────


class TestC4Phase2ToolExecutorExtraction:
    """The tool-execution stack (execute_tool, execute_tool_with_timeout,
    execute_tools, dispatch-pool state) now lives in runtime/loop/tool_executor.py.
    Engine aliases the functions with underscore-prefixed names for internal
    callers and exposes the mutable pool state via a PEP-562 ``__getattr__``
    passthrough so existing tests that read the old names keep working
    against the LIVE pool (not a stale snapshot).
    """

    def test_execute_tool_importable_from_new_home(self):
        from jyagent.runtime.loop.tool_executor import (
            execute_tool,
            execute_tool_with_timeout,
            execute_tools,
            get_tool_dispatch_executor,
        )
        # All must be callables (smoke check).
        assert callable(execute_tool)
        assert callable(execute_tool_with_timeout)
        assert callable(execute_tools)
        assert callable(get_tool_dispatch_executor)

    def test_engine_reexport_still_works(self):
        """Engine's underscore-prefixed aliases must be the same objects
        as the tool_executor public names (identity, not equality)."""
        from jyagent.runtime.loop import engine as _engine
        from jyagent.runtime.loop import tool_executor as _te
        assert _engine._execute_tool is _te.execute_tool
        assert _engine._execute_tool_with_timeout is _te.execute_tool_with_timeout
        assert _engine._execute_tools is _te.execute_tools
        assert _engine._get_tool_dispatch_executor is _te.get_tool_dispatch_executor

    def test_module_globals_track_live_value(self):
        """Critical: engine._tool_dispatch_executor must mirror the LIVE pool
        in tool_executor.py, even after a grow.  This is the PEP-562 passthrough
        test — a naive ``from .tool_executor import _tool_dispatch_executor``
        would snapshot at import time and go stale on the first grow."""
        from jyagent.runtime.loop import engine as _engine
        from jyagent.runtime.loop import tool_executor as _te
        # Grow to a large size, then confirm engine's back-compat name
        # returns the NEW pool object (not a stale snapshot).
        pre_grow = _engine._tool_dispatch_executor
        _engine._get_tool_dispatch_executor(256)
        post_grow_engine = _engine._tool_dispatch_executor
        post_grow_te = _te.tool_dispatch_executor
        assert post_grow_engine is post_grow_te, (
            f"engine view ({id(post_grow_engine)}) diverged from "
            f"tool_executor view ({id(post_grow_te)}) after grow — "
            "PEP-562 passthrough broken"
        )
        # The post-grow object MUST differ from pre-grow (otherwise the
        # test isn't actually exercising the rebind case).
        assert post_grow_engine is not pre_grow, (
            "grow didn't rebind — pool already >= 256 before the call? "
            "retry the test in isolation"
        )

    def test_tool_dispatch_cap_tracks_live_value(self):
        """Same PEP-562 check but for the integer cap (a non-object type)."""
        from jyagent.runtime.loop import engine as _engine
        from jyagent.runtime.loop import tool_executor as _te
        _engine._get_tool_dispatch_executor(300)
        assert _engine._tool_dispatch_cap == _te.tool_dispatch_cap
        assert _engine._tool_dispatch_cap >= 300

    def test_tool_executor_alias_points_at_dispatch(self):
        """engine._tool_executor (historical shorthand) must equal
        engine._tool_dispatch_executor — both names forward to the same
        live object in tool_executor.py."""
        from jyagent.runtime.loop import engine as _engine
        assert _engine._tool_executor is _engine._tool_dispatch_executor

    def test_engine_getattr_raises_on_unknown(self):
        """The __getattr__ shim must still raise AttributeError for names
        it doesn't own — no silent fallback."""
        from jyagent.runtime.loop import engine as _engine
        with pytest.raises(AttributeError, match="no attribute"):
            _ = _engine._definitely_does_not_exist  # noqa: SLF001


# ─── LLM call + retry extracted to runtime/loop/llm_runner.py ────────────────


class TestC4Phase3LLMRunnerExtraction:
    """The LLM call machinery (``call_complete`` /
    ``call_streaming`` / ``call_with_retry``) plus its helpers
    (``extract_text``, ``extract_tool_calls``, ``is_transient_error``,
    ``build_runtime_options``) now lives in
    ``jyagent.runtime.loop.llm_runner``.  Engine keeps four back-compat
    aliases so tests and internal callers that import the
    underscore-prefixed names continue to work.

    Unlike the tool executor module, there is no mutable module state to worry
    about — only
    functions and a class — so a plain ``from X import Y`` snapshot is
    fine and identity assertions suffice.  AgentLoop's ``_call_complete``
    and ``_call_streaming`` methods are thin delegates onto
    ``LLMRunner``; ``_call_llm_with_retry`` keeps its own retry loop in
    AgentLoop so tests that override ``_call_*`` on a subclass to inject
    failures still affect the retry behaviour.
    """

    def test_llm_runner_class_is_importable(self):
        from jyagent.runtime.loop.llm_runner import LLMRunner
        # Minimal surface contract.
        for attr in ("call_complete", "call_streaming", "call_with_retry"):
            assert callable(getattr(LLMRunner, attr)), attr

    def test_helper_functions_exist_with_public_names(self):
        """The four helper functions must exist under their public
        (non-underscored) names in llm_runner."""
        from jyagent.runtime.loop import llm_runner
        for name in (
            "extract_text",
            "extract_tool_calls",
            "is_transient_error",
            "build_runtime_options",
        ):
            assert hasattr(llm_runner, name), name
            assert callable(getattr(llm_runner, name)), name

    def test_engine_aliases_point_at_llm_runner(self):
        """engine._{extract_text,extract_tool_calls,is_transient_error,
        build_runtime_options} must be identical objects to the
        llm_runner public names.  Any divergence here means two copies of
        the function live in the tree."""
        from jyagent.runtime.loop import engine as _engine
        from jyagent.runtime.loop import llm_runner as _lr
        assert _engine._extract_text is _lr.extract_text
        assert _engine._extract_tool_calls is _lr.extract_tool_calls
        assert _engine._is_transient_error is _lr.is_transient_error
        assert _engine._build_runtime_options is _lr.build_runtime_options

    def test_agent_loop_get_llm_runner_caches(self):
        """AgentLoop._get_llm_runner() must build once and cache — the
        same instance is returned across multiple calls."""
        from unittest.mock import MagicMock
        from jyagent.runtime.loop.engine import AgentLoop, LoopConfig
        from jyagent.runtime.loop.llm_runner import LLMRunner

        owner = MagicMock()
        owner.model_spec = MagicMock(provider="test", model="test-model")
        loop = AgentLoop(
            runtime_owner=owner,
            config=LoopConfig(max_steps=1, streaming=False),
        )
        r1 = loop._get_llm_runner()
        r2 = loop._get_llm_runner()
        assert isinstance(r1, LLMRunner)
        assert r1 is r2

    def test_is_transient_error_retries_424_from_anthropic(self):
        """424 is a common proxy/gateway envelope code (e.g. domestic
        Anthropic relays wrapping upstream transients). We treat it as
        transient alongside 429/5xx. 400 stays non-transient because it
        almost always indicates a client-side error (bad request, quota,
        billing, invalid schema) where retry just burns budget."""
        import anthropic
        import httpx
        from jyagent.runtime.loop.llm_runner import is_transient_error

        def _mk(status: int) -> anthropic.APIStatusError:
            req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            resp = httpx.Response(status, request=req)
            return anthropic.APIStatusError(message="x", response=resp, body=None)

        # 424 (new) + existing whitelist members must all retry.
        for code in (424, 429, 500, 502, 503, 529):
            assert is_transient_error(_mk(code)) is True, f"expected {code} transient"

        # 400 deliberately stays non-transient — retrying a billing/quota
        # error just wastes the retry budget.
        assert is_transient_error(_mk(400)) is False
        # Other 4xx should also stay non-transient.
        for code in (401, 403, 404, 422):
            assert is_transient_error(_mk(code)) is False, f"{code} should NOT retry"

    def test_call_complete_delegates_to_runner(self):
        """_call_complete is a thin delegate onto LLMRunner.call_complete —
        patching the runner's method must be visible through the engine
        entry point."""
        from unittest.mock import MagicMock, patch
        from jyagent.runtime.loop.engine import AgentLoop, LoopConfig

        owner = MagicMock()
        owner.model_spec = MagicMock(provider="test", model="test-model")
        loop = AgentLoop(
            runtime_owner=owner,
            config=LoopConfig(max_steps=1, streaming=False),
        )
        runner = loop._get_llm_runner()
        sentinel = ("text", [], "stop", {"role": "assistant"})
        with patch.object(runner, "call_complete", return_value=sentinel) as mock_cc:
            result = loop._call_complete({}, None)
            mock_cc.assert_called_once_with({}, None)
            assert result is sentinel

    def test_call_streaming_delegates_to_runner(self):
        """Same identity/forwarding check for the streaming path."""
        from unittest.mock import MagicMock, patch
        from jyagent.runtime.loop.engine import AgentLoop, LoopConfig

        owner = MagicMock()
        owner.model_spec = MagicMock(provider="test", model="test-model")
        loop = AgentLoop(
            runtime_owner=owner,
            config=LoopConfig(max_steps=1, streaming=True),
        )
        runner = loop._get_llm_runner()
        sentinel = ("text", [], "stop", {"role": "assistant"})
        with patch.object(runner, "call_streaming", return_value=sentinel) as mock_cs:
            result = loop._call_streaming({}, None)
            mock_cs.assert_called_once_with({}, None)
            assert result is sentinel

    def test_retry_loop_dispatches_through_self_call_methods(self):
        """The retry loop in AgentLoop._call_llm_with_retry MUST invoke
        ``self._call_streaming`` / ``self._call_complete`` (not the
        runner methods directly).  This preserves the long-standing
        contract where subclasses / monkeypatches on those names inject
        failures visible to the retry machinery.
        """
        from unittest.mock import MagicMock
        from jyagent.runtime.loop.engine import AgentLoop, LoopConfig

        class _Transient(Exception):
            """Treated as transient by is_transient_error via duck-type."""
            pass

        # is_transient_error keys off APIConnectionError / APIStatusError /
        # 5xx status — we need something it'll accept.  Easier route:
        # patch is_transient_error to always return True for this test.
        from jyagent.runtime.loop import llm_runner

        owner = MagicMock()
        owner.model_spec = MagicMock(provider="test", model="test-model")

        streaming_calls = 0
        complete_calls = 0
        final = ("ok", [], "stop", {"role": "assistant"})

        class _Loop(AgentLoop):
            def _is_cancelled(self) -> bool:
                return False

            def _call_streaming(self, context, options):
                nonlocal streaming_calls
                streaming_calls += 1
                if streaming_calls < 2:
                    raise _Transient("boom")
                return final

            def _call_complete(self, context, options):
                nonlocal complete_calls
                complete_calls += 1
                if complete_calls < 2:
                    raise _Transient("boom")
                return final

        # Non-streaming path.
        loop_c = _Loop(
            runtime_owner=owner,
            config=LoopConfig(max_steps=1, streaming=False, retry_attempts=2,
                              retry_base_delay=0.0),
        )
        orig = llm_runner.is_transient_error
        try:
            llm_runner.is_transient_error = lambda e: isinstance(e, _Transient)
            # Also patch the engine-side alias since the retry loop lives
            # in engine.py and imports is_transient_error via a local
            # alias at module-load time.
            import jyagent.runtime.loop.engine as _engine
            _engine._is_transient_error = llm_runner.is_transient_error
            result = loop_c._call_llm_with_retry({}, None, step=0)
        finally:
            llm_runner.is_transient_error = orig
            import jyagent.runtime.loop.engine as _engine
            _engine._is_transient_error = orig
        assert result == final
        assert complete_calls == 2, "retry should invoke self._call_complete twice"

        # Streaming path.
        loop_s = _Loop(
            runtime_owner=owner,
            config=LoopConfig(max_steps=1, streaming=True, retry_attempts=2,
                              retry_base_delay=0.0),
        )
        try:
            llm_runner.is_transient_error = lambda e: isinstance(e, _Transient)
            import jyagent.runtime.loop.engine as _engine
            _engine._is_transient_error = llm_runner.is_transient_error
            result = loop_s._call_llm_with_retry({}, None, step=0)
        finally:
            llm_runner.is_transient_error = orig
            import jyagent.runtime.loop.engine as _engine
            _engine._is_transient_error = orig
        assert result == final
        assert streaming_calls == 2, "retry should invoke self._call_streaming twice"


# ─── Compaction helpers extracted to runtime/loop/compaction.py ─────────────


class TestC4Phase4CompactionExtraction:
    """The three compaction helpers (``truncate_result``,
    ``compact_messages``, ``truncate_tool_call_blocks``) moved to
    ``jyagent.runtime.loop.compaction``.  Engine keeps three
    underscore-prefixed back-compat aliases so the many existing test
    imports and internal call sites continue to work unchanged.

    All three are **pure functions** — no closure state, no mutable
    module attributes — so identity checks are sufficient proof that
    the engine alias and the compaction-module original are the same
    object.  (The tool executor PEP-562 shim is not needed here because there
    is nothing to rebind.)
    """

    def test_compaction_module_exposes_public_names(self):
        from jyagent.runtime.loop import compaction
        for name in ("truncate_result", "compact_messages", "truncate_tool_call_blocks"):
            assert hasattr(compaction, name), name
            assert callable(getattr(compaction, name)), name

    def test_engine_aliases_point_at_compaction(self):
        """engine._{truncate_result,compact_messages,truncate_tool_call_blocks}
        must be the same object as the compaction-module original.
        Any divergence means two copies of the function exist in the
        tree."""
        from jyagent.runtime.loop import engine as _engine
        from jyagent.runtime.loop import compaction as _c
        assert _engine._truncate_result is _c.truncate_result
        assert _engine._compact_messages is _c.compact_messages
        assert _engine._truncate_tool_call_blocks is _c.truncate_tool_call_blocks

    def test_truncate_result_head_tail_split(self):
        """truncate_result keeps first 85% + last 10% of max_chars when
        over budget; returns unchanged when under budget or on error."""
        from jyagent.runtime.loop.compaction import truncate_result

        # Under budget — pass through.
        assert truncate_result("short", 100) == "short"

        # Error results are NEVER truncated — users need the full trace.
        big_error = "x" * 10_000
        assert truncate_result(big_error, 100, is_error=True) == big_error

        # Over budget + non-error — truncated with marker.
        big = "A" * 1000 + "B" * 1000  # 2000 chars, max 100
        out = truncate_result(big, 100, is_error=False)
        assert len(out) < len(big)
        assert "truncated" in out
        assert "total: 2000 chars" in out
        # Head preserved, tail preserved.
        assert out.startswith("A")
        assert out.endswith("B")

    def test_compact_messages_passthrough_when_under_budget(self):
        """When estimated tokens <= max_tokens, compact_messages returns
        the ORIGINAL list (identity) — no deep copy, no mutation."""
        from jyagent.runtime.loop.compaction import compact_messages
        from jyagent.runtime.tools.registry import ToolBatch

        msgs = [{"role": "user", "content": "hi"}]
        # Very large max_tokens so we're certainly under budget.
        out = compact_messages(msgs, max_tokens=10**9, compact_chars=1000, batch=ToolBatch.empty())
        assert out is msgs  # identity, not equality — proves the fast path

    def test_truncate_tool_call_blocks_leaves_unrelated_blocks_alone(self):
        """truncate_tool_call_blocks only touches ``tool_call`` blocks
        whose tool has ``large_input_keys`` — text blocks pass through
        unchanged (identity preserved)."""
        from jyagent.runtime.loop.compaction import truncate_tool_call_blocks
        from jyagent.runtime.tools.registry import ToolBatch

        text_block = {"type": "text", "text": "hello"}
        tool_block = {
            "type": "tool_call",
            "id": "call_1",
            "name": "unknown_tool",  # not in empty batch → no large_keys
            "arguments": {"code": "x" * 100_000},
        }
        out = truncate_tool_call_blocks([text_block, tool_block], ToolBatch.empty())
        # Empty batch → no known large_input_keys → no truncation.
        assert out[0] is text_block
        assert out[1] is tool_block  # same object reference


# ─── Import-time cleanup ────────────────────────────────────────────────────
#
# These tests cover eager engine loading on `import runtime` and accidental
# module-level pool creation.  They run in subprocess so they get a fresh
# `sys.modules` — the in-process pytest run has already imported the engine
# for hundreds of other tests.

class TestC4ImportTimeCleanup:
    """Verify `import jyagent.runtime` is cheap and side-effect-free.

    All checks run in subprocess so the pytest-process module cache doesn't
    mask the laziness.
    """

    def _run_in_subprocess(self, code: str, timeout: int = 15) -> str:
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"subprocess exited {result.returncode}\n"
                f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
            )
        return result.stdout

    def test_runtime_import_does_not_load_engine(self):
        """`import jyagent.runtime` must NOT load engine.py.

        Engine pulls in tool_executor/llm_runner/compaction/step — a hard
        regression if it loads on plain `import jyagent.runtime`.
        """
        out = self._run_in_subprocess("""
import sys
import jyagent.runtime
heavy = [
    'jyagent.runtime.loop.engine',
    'jyagent.runtime.loop.tool_executor',
    'jyagent.runtime.loop.llm_runner',
    'jyagent.runtime.loop.compaction',
    'jyagent.runtime.loop.step',
]
loaded = [m for m in heavy if m in sys.modules]
assert not loaded, f'unexpectedly eager: {loaded}'
print('OK')
""")
        assert "OK" in out

    def test_runtime_loop_import_does_not_load_engine(self):
        """`import jyagent.runtime.loop` must NOT load engine.py either."""
        out = self._run_in_subprocess("""
import sys
import jyagent.runtime.loop
assert 'jyagent.runtime.loop.engine' not in sys.modules, 'engine eagerly loaded'
# But cheap leaves SHOULD be loaded — they're explicit eager imports.
assert 'jyagent.runtime.loop.callbacks' in sys.modules
assert 'jyagent.runtime.loop.config' in sys.modules
print('OK')
""")
        assert "OK" in out

    def test_runtime_import_creates_no_thread_pool(self):
        """`import jyagent.runtime` must NOT create the dispatch pool.

        Pool creation registers an atexit hook and spins a daemon thread —
        unwanted for callers that never construct an `AgentLoop`.
        """
        out = self._run_in_subprocess("""
import jyagent.runtime
# tool_executor itself isn't loaded yet; engine keeps it lazy.
# import it explicitly and check the pool is None.
from jyagent.runtime.loop import tool_executor as te
assert te.tool_dispatch_executor is None, 'pool eagerly initialised at import'
assert te.tool_dispatch_cap == 0, f'expected cap=0 got {te.tool_dispatch_cap}'
print('OK')
""")
        assert "OK" in out

    def test_lazy_agentloop_attribute_works(self):
        """`from jyagent.runtime import AgentLoop` lazy-loads engine."""
        out = self._run_in_subprocess("""
import sys
import jyagent.runtime
assert 'jyagent.runtime.loop.engine' not in sys.modules
# Lazy load via attribute access:
AL = jyagent.runtime.AgentLoop
assert 'jyagent.runtime.loop.engine' in sys.modules, 'lazy load did not fire'
# Cached on the module dict for future O(1) lookups:
assert 'AgentLoop' in vars(jyagent.runtime)
# Same object on second access:
assert jyagent.runtime.AgentLoop is AL
print('OK')
""")
        assert "OK" in out

    def test_lazy_from_import_pattern_works(self):
        """`from jyagent.runtime import AgentLoop` (statement form)
        triggers __getattr__ exactly like attribute access."""
        out = self._run_in_subprocess("""
import sys
# Engine not loaded yet by trick: import the package WITHOUT touching AgentLoop.
import jyagent.runtime
assert 'jyagent.runtime.loop.engine' not in sys.modules
# Now do the from-import — Python falls back to __getattr__ for missing names.
from jyagent.runtime import AgentLoop
assert AgentLoop.__name__ == 'AgentLoop'
assert 'jyagent.runtime.loop.engine' in sys.modules
print('OK')
""")
        assert "OK" in out

    def test_unknown_attribute_raises_attribute_error(self):
        """PEP-562 contract: unknown attributes still raise AttributeError."""
        out = self._run_in_subprocess("""
import jyagent.runtime
try:
    _ = jyagent.runtime.DefinitelyNotARealClass
    raise SystemExit('FAIL: should have raised AttributeError')
except AttributeError as e:
    assert 'DefinitelyNotARealClass' in str(e)
print('OK')
""")
        assert "OK" in out

    def test_loop_phase_submodules_lazy_accessible(self):
        """Phase modules (phases, reflection, ...) remain accessible via the
        `from jyagent.runtime.loop import phases` pattern even though they
        are no longer eagerly imported at package init."""
        out = self._run_in_subprocess("""
from jyagent.runtime.loop import phases, reflection, checkpoint, todos, verification, remediation, tracing
# All should be modules:
import types
for m in (phases, reflection, checkpoint, todos, verification, remediation, tracing):
    assert isinstance(m, types.ModuleType), f'{m} is not a module'
print('OK')
""")
        assert "OK" in out
    def test_execute_tools_with_executor_none_lazy_inits_pool(self):
        """Direct callers of
        `tool_executor.execute_tools(...)` with `executor=None` MUST work
        even when no AgentLoop has been constructed yet.

        Before the lazy-init fallback (`pool = executor or
        get_tool_dispatch_executor(max_workers)`), the module global was
        None at import time so `pool.submit(...)` would raise
        AttributeError on a parallel batch.

        Subprocess-isolated so we get a fresh sys.modules with no other
        test having already grown the pool.
        """
        out = self._run_in_subprocess('''
from jyagent.runtime.loop import tool_executor as te
from jyagent.runtime.tools.registry import ToolRegistry

# Verify the pool is None at this point (no AgentLoop ever constructed)
assert te.tool_dispatch_executor is None

# Build a parallel-safe batch with two tools
def _t(label="x"):
    return f"{label}-done"

reg = ToolRegistry()
reg.register("a", _t, {"name": "a", "input_schema": {"type": "object"}}, parallel_safe=True)
reg.register("b", _t, {"name": "b", "input_schema": {"type": "object"}}, parallel_safe=True)
batch = reg.freeze()

from jyagent.runtime.loop.engine import ToolCallRequest
blocks = [
    ToolCallRequest(id="1", name="a", input={"label": "a"}),
    ToolCallRequest(id="2", name="b", input={"label": "b"}),
]

# Critical path: executor=None forces the fallback that used to read None
results = te.execute_tools(
    blocks, batch,
    concurrent_mode=True, max_workers=4, timeout=10,
    executor=None,
)

assert len(results) == 2
for block, r in results:
    assert not r.is_error, f"{block.name} errored: {r.content}"
    assert "done" in r.content

# And the pool was lazily materialised
assert te.tool_dispatch_executor is not None
print("OK")
''', timeout=20)
        assert "OK" in out
