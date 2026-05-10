"""Regression tests for runtime/loop/ safety fixes.

Bundles four fixes from the 2026-05 review:

1. **Stale executor reference** (Critical). ``AgentLoop.__init__`` used to
   cache ``self._executor = get_tool_dispatch_executor(...)``.  A second
   loop with a larger ``max_tool_workers`` triggered the shared pool to
   grow, which shut the old pool down — leaving the first loop with a
   dead reference that crashed on next ``.submit()``.  The fix turns
   ``_executor`` into a property that re-resolves the live shared pool.

2. **Terminal checkpoint missing cache/api fields** (High). The
   periodic checkpoint passed cache-token and ``api_calls`` counters,
   but the terminal checkpoint at the bottom of ``run()`` omitted them
   so ``final.json`` silently zeroed those fields despite ``LoopResult``
   having the correct values.

3. **Buffered streaming flushed on error** (High). In
   ``LLMRunner.call_streaming`` buffered mode flushed the accumulated
   text on every ``done`` event — including a ``done`` whose terminal
   message carried ``stop_reason='error'``, leaking failed-attempt text
   onto the live ``on_text_delta`` channel before retry.

4. **Fallback exception swallowing with poisoned counters** (High).
   The ``max_steps`` fallback path mutated ``state.*`` and
   ``cost_tracker`` BEFORE the can-fail truncation step, then caught
   broad ``Exception: pass`` so a late failure produced a max_steps
   result with poisoned counters and no error trace.  The fix
   reorders the body so all mutation happens after every can-fail
   call has succeeded, and surfaces failures via ``on_warning``.
"""
from __future__ import annotations

import concurrent.futures
import json
import threading
from pathlib import Path

import pytest

from jyagent.runtime.loop import engine as le
from jyagent.runtime.loop import tool_pool as le_te
from jyagent.runtime.loop import llm_runner as le_lr


# ─────────────────────────────────────────────────────────────────────────
# Fix #1: stale executor reference
# ─────────────────────────────────────────────────────────────────────────


class TestExecutorPropertyReResolves:
    """``loop._executor`` must always return the *current* shared pool,
    not a snapshot taken at construction time.  Without this, a second
    loop with a larger ``max_tool_workers`` shutting down + replacing
    the shared pool leaves the first loop with a dead executor."""

    def _bare_loop(self, *, max_tool_workers: int) -> le.AgentLoop:
        loop = le.AgentLoop.__new__(le.AgentLoop)
        loop._config = le.LoopConfig(max_tool_workers=max_tool_workers)
        return loop

    def test_property_re_resolves_after_pool_growth(self):
        """A grown pool must be visible to a previously-constructed loop."""
        # Force a growth event regardless of what other tests already did
        # to the shared pool: snapshot the current cap, then request
        # ``cap + 8`` to guarantee the growth path runs in this test.
        baseline = le_te.get_tool_dispatch_executor(8)
        baseline_cap = le_te.tool_dispatch_cap
        loop = self._bare_loop(max_tool_workers=8)
        first = loop._executor
        # First read returns the baseline pool.
        assert first is baseline

        # Trigger growth — the old pool gets shutdown(wait=False).
        target = baseline_cap + 8
        grown = le_te.get_tool_dispatch_executor(target)
        assert grown is not first, "growth must produce a fresh pool"
        assert le_te.tool_dispatch_cap >= target

        # The crucial assertion: the loop's property now sees the live
        # pool, not the dead one it would have cached at __init__ time
        # under the previous design.
        assert loop._executor is grown
        # And the live pool must accept new submissions (it's not shut down).
        fut = loop._executor.submit(lambda: 42)
        assert fut.result(timeout=2.0) == 42

    def test_test_override_setter_honored(self):
        """Setting ``loop._executor = X`` must be honored — escape hatch
        for tests that pre-build a fake executor.  Exercised by several
        legacy tests under tests/test_*.py (see git blame).
        """
        loop = self._bare_loop(max_tool_workers=8)
        sentinel = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            loop._executor = sentinel
            assert loop._executor is sentinel
        finally:
            sentinel.shutdown(wait=False)

    def test_default_path_does_not_use_override(self):
        """A loop that never sets ``_executor`` falls through to the
        live shared pool — the override is opt-in."""
        loop = self._bare_loop(max_tool_workers=8)
        assert loop._executor is le_te.get_tool_dispatch_executor(8)


# ─────────────────────────────────────────────────────────────────────────
# Fix #2: terminal checkpoint forwards cache/api fields
# ─────────────────────────────────────────────────────────────────────────


class TestTerminalCheckpointForwardsAccountingFields:
    """The terminal ('final') checkpoint must record cache-token and
    api_calls counters so ``final.json`` matches ``LoopResult``."""

    def test_final_json_records_cache_and_api_counters(self, tmp_path: Path):
        cfg = le.LoopConfig(checkpoint_dir=str(tmp_path))
        loop = le.AgentLoop.__new__(le.AgentLoop)
        loop._config = cfg
        loop._callbacks = le.LoopCallbacks()
        loop._run_id = "rrun"
        loop._todos = []
        loop._model_spec = None

        class _Spec:
            provider = "anthropic"
            model = "claude-test"
        class _Owner:
            model_spec = _Spec()
        loop._runtime_owner = _Owner()

        loop._write_checkpoint(
            step="final",
            messages=[{"role": "user", "content": "hi"}],
            total_input_tokens=10,
            total_output_tokens=20,
            tool_calls_count=2,
            total_cache_creation_tokens=33,
            total_cache_read_tokens=44,
            api_calls=7,
            status="completed",
        )

        # find the file (writer creates ``checkpoint_dir/<run_id>/final.json``)
        run_dir = tmp_path / "rrun"
        assert run_dir.is_dir(), list(tmp_path.iterdir())
        files = list(run_dir.iterdir())
        assert len(files) == 1, files
        data = json.loads(files[0].read_text())
        assert data["total_cache_creation_tokens"] == 33
        assert data["total_cache_read_tokens"] == 44
        assert data["api_calls"] == 7
        # Sanity: status preserved.
        assert data["status"] == "completed"


# ─────────────────────────────────────────────────────────────────────────
# Fix #3: buffered streaming defers flush past the error gate
# ─────────────────────────────────────────────────────────────────────────


class _FakeRuntimeStream:
    """Replays a fixed event list, mirroring jyagent's adapter contract."""

    def __init__(self, events):
        self._events = list(events)
        self._final = None
        for e in events:
            if e.get("type") in ("done", "error"):
                self._final = e.get("message")

    def __enter__(self): return self

    def __exit__(self, *a): self.close()

    def __iter__(self):
        yield from self._events

    def get_final_message(self):
        return self._final or {"role": "assistant", "content": [], "stop_reason": "stop"}

    def close(self): pass


class _FakeRuntimeOwner:
    def __init__(self, event_seqs):
        self._seqs = list(event_seqs)
        self._calls = 0

        class _spec:
            provider = "anthropic"
            model = "claude-x"
            @staticmethod
            def label(): return "anthropic:claude-x"

        self.model_spec = _spec()

    def stream(self, context, options=None, model_spec=None):
        events = self._seqs[self._calls]
        self._calls += 1
        return _FakeRuntimeStream(events)

    def complete(self, *a, **kw):
        raise AssertionError("unused in this test")


def _make_loop_for_streaming(owner, *, buffered: bool, callbacks=None):
    loop = le.AgentLoop.__new__(le.AgentLoop)
    loop._runtime_owner = owner
    loop._config = le.LoopConfig(
        streaming=True,
        buffered_streaming=buffered,
        retry_attempts=1,        # don't actually retry — we want the raw raise
        retry_base_delay=0.001,
    )
    loop._callbacks = callbacks or le.LoopCallbacks()
    loop._tool_source = None
    loop._model_spec = None
    loop._cancel_event = None
    return loop


class TestBufferedFlushDeferredPastErrorGate:
    """Buffered mode must NOT flush text from a ``done`` event whose
    terminal message carries ``stop_reason='error'`` — that text belongs
    on the retry's ``on_stream_retry`` callback, not the live UI stream."""

    def test_error_done_does_not_flush_buffered_text(self):
        # Terminal "done" with an error stop_reason — the previous code
        # flushed the partial text BEFORE checking stop_reason.
        err_msg = {
            "role": "assistant",
            "content": [],
            "stop_reason": "error",
            "error_message": "503 from upstream",
            "usage": {},
        }
        events = [
            {"type": "text_delta", "text": "partial answer"},
            {"type": "done", "message": err_msg},
        ]
        owner = _FakeRuntimeOwner([events])

        emitted = []
        cbs = le.LoopCallbacks(on_text_delta=lambda t: emitted.append(t))
        loop = _make_loop_for_streaming(owner, buffered=True, callbacks=cbs)

        # The call must raise (it's an error stream) — that's expected.
        with pytest.raises(RuntimeError) as excinfo:
            loop._call_streaming(context={}, options=None)
        # Partial text must be on the exception (for retry's
        # on_stream_retry), NOT on the live delta stream.
        assert excinfo.value.partial_stream_text == "partial answer"
        assert emitted == [], (
            f"buffered mode leaked failed-attempt text to on_text_delta: {emitted}"
        )

    def test_clean_done_still_flushes(self):
        """Sanity: the success path must still flush — the fix only
        defers the flush past the error check, it must not break the
        clean-completion case."""
        done_msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi there"}],
            "stop_reason": "stop",
            "usage": {},
        }
        events = [
            {"type": "text_delta", "text": "Hi "},
            {"type": "text_delta", "text": "there"},
            {"type": "done", "message": done_msg},
        ]
        owner = _FakeRuntimeOwner([events])
        emitted = []
        cbs = le.LoopCallbacks(on_text_delta=lambda t: emitted.append(t))
        loop = _make_loop_for_streaming(owner, buffered=True, callbacks=cbs)

        text, _, stop, _ = loop._call_streaming(context={}, options=None)
        assert text == "Hi there"
        assert stop == "stop"
        # Buffered mode → exactly one flush of the full text.
        assert emitted == ["Hi there"]


# ─────────────────────────────────────────────────────────────────────────
# Fix #4: max_steps fallback failure leaves clean state + warning
# ─────────────────────────────────────────────────────────────────────────


class TestFallbackFailureDoesNotPoisonCounters:
    """When the max_steps fallback raises (after retries are exhausted),
    the loop must:
      1. NOT have mutated ``state.*`` or ``cost_tracker``.
      2. Fire ``on_warning`` so outer layers know the fallback crashed
         (the previous bare ``except Exception: pass`` silently turned
         a fallback crash into a clean-looking max_steps result).
    """

    def test_warning_fires_when_fallback_call_raises(self):
        """End-to-end: configure a loop whose first step exhausts
        max_steps and whose fallback ``_call_complete`` raises a
        non-transient error.  The on_warning callback must record
        the failure, and the LoopResult must report status='max_steps'
        with un-poisoned counters."""
        # We exercise the fallback path via a heavily-mocked AgentLoop
        # subclass to avoid wiring up a real provider.
        warnings: list[str] = []
        cbs = le.LoopCallbacks(on_warning=lambda m: warnings.append(m))

        from jyagent.runtime.loop.step import RunState
        from jyagent.runtime.loop.cost import CostTracker
        from jyagent.runtime.tools.registry import ToolBatch

        class _FBLoop(le.AgentLoop):
            def _call_complete(self, context, options):
                raise RuntimeError("simulated fallback crash")
            def _call_streaming(self, context, options):
                raise RuntimeError("simulated fallback crash")

        class _Spec:
            provider = "anthropic"
            model = "claude-test"
            @staticmethod
            def label(): return "anthropic:claude-test"
        class _Owner:
            model_spec = _Spec()
            def stream(self, *a, **kw): raise AssertionError("unused")
            def complete(self, *a, **kw): raise AssertionError("unused")

        loop = _FBLoop.__new__(_FBLoop)
        loop._runtime_owner = _Owner()
        loop._config = le.LoopConfig(
            max_steps=0,                  # force the max_steps branch immediately
            fallback_on_max_steps=True,
            streaming=False,
            checkpoint_dir=None,
        )
        loop._callbacks = cbs
        loop._tool_source = None
        loop._model_spec = None
        loop._cancel_event = None
        loop._session_id = ""
        loop._todos = []
        loop._partial_side_effects: list = []
        loop._run_id = ""
        loop._run_lock = threading.Lock()

        # Drive _run_impl directly with a fresh state.
        result = loop._run_impl(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
        )

        # 1. The fallback warning was surfaced.
        assert any("fallback" in w.lower() for w in warnings), (
            f"on_warning was not fired with a fallback-failure message; got: {warnings}"
        )

        # 2. Status fell back to max_steps (not "completed", which is the
        # success path of fallback) — and counters are un-poisoned (zero
        # on every accounting field, since no LLM call ever succeeded).
        assert result.status == "max_steps", result.status
        assert result.total_input_tokens == 0
        assert result.total_output_tokens == 0
        assert result.total_cache_creation_tokens == 0
        assert result.total_cache_read_tokens == 0
        assert result.api_calls == 0
