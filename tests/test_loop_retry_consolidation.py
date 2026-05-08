"""Regression tests for the retry-loop consolidation.

Before this refactor, ``AgentLoop._call_llm_with_retry`` and
``LLMRunner.call_with_retry`` each carried a ~50-line copy of the same
retry loop body.  The drift risk was high — codex's review specifically
flagged it.  Both call sites now route through a shared
``_retry_llm_call`` helper, with the only difference being the
streaming/complete dispatch callables and the ``is_transient`` predicate
each shim passes.

These tests pin the contract that matters in practice:

1. Both shims share a single helper (no body duplication).
2. The engine shim still routes through ``self._call_streaming`` /
   ``self._call_complete`` so subclass overrides + monkeypatches inject
   transient failures correctly.
3. Tests that patch ``engine.is_transient_error`` still see their
   override applied (the engine shim passes its own module's symbol).
4. Tests that patch ``llm_runner.is_transient_error`` still see their
   override applied (the LLMRunner shim's default lookup honors it).
5. The cancel-aware backoff still raises ``KeyboardInterrupt`` when the
   sleep wakes early on cancel.
"""
from __future__ import annotations

import inspect
import threading
from unittest.mock import MagicMock

import pytest

from jyagent.runtime.loop import engine as le
from jyagent.runtime.loop import llm_runner as le_lr


# ─────────────────────────────────────────────────────────────────────────
# Structural: there is exactly one retry-loop body
# ─────────────────────────────────────────────────────────────────────────


class TestRetryLoopBodyIsSharedHelper:
    def test_engine_shim_calls_helper(self):
        """``AgentLoop._call_llm_with_retry`` must delegate to
        ``_retry_llm_call``; if the body grows back to a full retry loop
        the duplication is back."""
        src = inspect.getsource(le.AgentLoop._call_llm_with_retry)
        assert "_retry_llm_call(" in src, src
        # The shim should be small — no ``for attempt in range`` or
        # ``random.uniform`` or backoff code.  Cap at 40 lines so a
        # future contributor doesn't quietly re-inline the loop.
        assert src.count("\n") < 40, (
            f"_call_llm_with_retry shim grew to {src.count(chr(10))} lines — "
            f"the retry body should live in _retry_llm_call, not here.\n{src}"
        )

    def test_llmrunner_shim_calls_helper(self):
        src = inspect.getsource(le_lr.LLMRunner.call_with_retry)
        assert "_retry_llm_call(" in src, src
        assert src.count("\n") < 40, (
            f"LLMRunner.call_with_retry shim grew to {src.count(chr(10))} "
            f"lines — the retry body should live in _retry_llm_call, "
            f"not here.\n{src}"
        )


# ─────────────────────────────────────────────────────────────────────────
# Behavioral: the engine shim still dispatches through self._call_*
# (preserved contract; this duplicates an existing test in
#  test_loop_edge_cases.py but is kept here as the canonical regression
#  target for the consolidation specifically)
# ─────────────────────────────────────────────────────────────────────────


class TestEngineShimDispatchesThroughSelfCallMethods:
    def test_subclass_override_of_call_streaming_is_invoked(self):
        class _Transient(Exception):
            pass

        owner = MagicMock()
        owner.model_spec = MagicMock(provider="test", model="test-model")

        streaming_calls = 0
        final = ("ok", [], "stop", {"role": "assistant", "content": []})

        class _Loop(le.AgentLoop):
            def _is_cancelled(self) -> bool:
                return False

            def _call_streaming(self, context, options):
                nonlocal streaming_calls
                streaming_calls += 1
                if streaming_calls < 2:
                    raise _Transient("simulated transient")
                return final

        loop = _Loop(
            runtime_owner=owner,
            config=le.LoopConfig(
                max_steps=1, streaming=True,
                retry_attempts=2, retry_base_delay=0.0,
            ),
        )

        # Patch the engine-side alias — the shim resolves
        # ``is_transient_error`` from engine's module scope at call time.
        orig_engine = le.is_transient_error
        try:
            le.is_transient_error = lambda e: isinstance(e, _Transient)
            result = loop._call_llm_with_retry({}, None, step=0)
        finally:
            le.is_transient_error = orig_engine
        assert result == final
        assert streaming_calls == 2, (
            f"retry should have invoked subclass _call_streaming twice, "
            f"got {streaming_calls}"
        )


# ─────────────────────────────────────────────────────────────────────────
# Behavioral: both is_transient_error patch surfaces still work
# ─────────────────────────────────────────────────────────────────────────


class _MiniLoop:
    """Bare-minimum object the engine shim exercises; saves the cost of
    constructing a real AgentLoop for tests that only care about the
    retry-helper plumbing."""

    def __init__(self, *, streaming: bool, fail_until: int, transient_exc: type[BaseException], retry_on_all_errors: bool = True):
        self._config = le.LoopConfig(
            streaming=streaming,
            retry_attempts=3, retry_base_delay=0.0,
            retry_on_all_errors=retry_on_all_errors,
        )
        self._fail_until = fail_until
        self._transient_exc = transient_exc
        self._calls = 0
        self.fired: list[tuple] = []

    def _is_cancelled(self) -> bool:
        return False

    def _cancellable_sleep(self, secs: float) -> bool:
        return False  # never cancelled

    def _fire(self, name, *args) -> None:
        self.fired.append((name, args))

    def _call_streaming(self, context, options):
        self._calls += 1
        if self._calls <= self._fail_until:
            raise self._transient_exc("boom")
        return ("ok", [], "stop", {"role": "assistant", "content": []})

    _call_complete = _call_streaming  # share for both modes


class TestIsTransientPredicatePatchSurfaces:
    def test_engine_module_patch_is_honored(self):
        """Patching ``engine.is_transient_error`` flips the retry decision
        because the engine shim threads it through ``is_transient=``."""
        class _Boom(Exception):
            pass

        loop = _MiniLoop(streaming=True, fail_until=1, transient_exc=_Boom)
        orig = le.is_transient_error
        try:
            le.is_transient_error = lambda e: isinstance(e, _Boom)
            result = le.AgentLoop._call_llm_with_retry(loop, {}, None, 0)
        finally:
            le.is_transient_error = orig
        assert result[2] == "stop"
        # on_stream_retry fired with reason='transient_error' (not 'error')
        # — proves the engine alias was consulted.
        retry_signals = [args for name, args in loop.fired if name == "on_stream_retry"]
        assert retry_signals, loop.fired
        assert retry_signals[0][0] == "transient_error", retry_signals

    def test_llm_runner_module_patch_is_honored(self):
        """Patching ``llm_runner.is_transient_error`` flips the retry
        decision for the LLMRunner.call_with_retry shim, which uses the
        helper's default lookup."""
        class _Boom(Exception):
            pass

        # Build a runner directly — easier than constructing an AgentLoop.
        runner = le_lr.LLMRunner.__new__(le_lr.LLMRunner)
        runner.config = le.LoopConfig(
            streaming=False, retry_attempts=3, retry_base_delay=0.0,
        )
        runner.callbacks = le.LoopCallbacks(
            on_retry=lambda *a: None,
            on_stream_retry=lambda *a: None,
        )
        runner.cancel_event = None

        calls = {"n": 0}

        def fake_complete(ctx, opts):
            calls["n"] += 1
            if calls["n"] < 2:
                raise _Boom("boom")
            return ("ok", [], "stop", {"role": "assistant", "content": []})

        runner.call_complete = fake_complete  # type: ignore[method-assign]
        runner.call_streaming = fake_complete  # unused in non-streaming mode

        orig = le_lr.is_transient_error
        try:
            le_lr.is_transient_error = lambda e: isinstance(e, _Boom)
            result = runner.call_with_retry({}, None)
        finally:
            le_lr.is_transient_error = orig
        assert result[2] == "stop"
        assert calls["n"] == 2

    def test_non_transient_does_not_retry(self):
        """A non-transient exception (predicate returns False) propagates
        on the first attempt; no on_retry / on_stream_retry callbacks.

        Requires ``retry_on_all_errors=False`` since the production
        default of ``True`` would retry every exception regardless of
        the transient predicate.
        """
        class _Permanent(Exception):
            pass

        loop = _MiniLoop(
            streaming=True, fail_until=999, transient_exc=_Permanent,
            retry_on_all_errors=False,
        )
        # Default predicate (real is_transient_error) doesn't recognize
        # _Permanent — should NOT retry.
        with pytest.raises(_Permanent):
            le.AgentLoop._call_llm_with_retry(loop, {}, None, 0)
        retry_signals = [args for name, args in loop.fired if name == "on_stream_retry"]
        assert retry_signals == [], loop.fired
        on_retry = [args for name, args in loop.fired if name == "on_retry"]
        assert on_retry == [], loop.fired


# ─────────────────────────────────────────────────────────────────────────
# Behavioral: cancel during retry backoff escalates to KeyboardInterrupt
# ─────────────────────────────────────────────────────────────────────────


class TestCancelDuringBackoffEscalates:
    def test_cancellable_sleep_returning_true_raises_keyboardinterrupt(self):
        """If ``cancellable_sleep(delay)`` returns True (cancel fired
        during the sleep) the helper must escalate to
        ``KeyboardInterrupt``, not silently continue to the next retry."""
        class _Transient(Exception):
            pass

        attempts = {"n": 0}

        def call_streaming(ctx, opts):
            attempts["n"] += 1
            raise _Transient("first-call-fails")

        cancellable_sleep_calls = {"n": 0}

        def cancellable_sleep(delay):
            cancellable_sleep_calls["n"] += 1
            return True  # cancel fired during sleep

        with pytest.raises(KeyboardInterrupt):
            le_lr._retry_llm_call(
                config=le.LoopConfig(
                    streaming=True, retry_attempts=3, retry_base_delay=0.0,
                ),
                context={},
                options=None,
                call_streaming=call_streaming,
                call_complete=lambda c, o: None,
                fire=lambda *a: None,
                is_cancelled=lambda: False,
                cancellable_sleep=cancellable_sleep,
                is_transient=lambda e: isinstance(e, _Transient),
            )
        # First call failed, sleep returned True → escalate.  No second
        # attempt should have run.
        assert attempts["n"] == 1, attempts
        assert cancellable_sleep_calls["n"] == 1
