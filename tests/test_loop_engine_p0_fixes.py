# tests/test_loop_engine_p0_fixes.py — Regression tests for P0 loop-engine bugs.
#
# Validates three correctness fixes identified by the 2026-04 cross-review
# (Codex + Claude Code):
#
#   P0 #1 — Nested same-pool deadlock in _execute_tools / _execute_tool_with_timeout.
#   P0 #3 — _CostTracker must use the effective model spec (sub-agent override).
#   P0 #4 — fallback_on_max_steps must fire on max-step exit, regardless of
#            incidental text emitted from prior tool-use steps.

from __future__ import annotations

import concurrent.futures
import time
import threading
from dataclasses import dataclass

import pytest

from jyagent.runtime.loop import engine as le
from jyagent.runtime.loop.engine import (
    _execute_tool_with_timeout,
    _execute_tools,
    ToolCallRequest,
)
from jyagent.runtime.tools.result import ToolResult
from jyagent.runtime.tools.registry import ToolBatch, ToolRegistry


# Combined-source helper for structural tests that grep the loop body.
# After the C4 Phase 5 split, the per-step body lives in
# ``runtime.loop.step.run_step``; the engine's ``AgentLoop._run_impl``
# now contains only setup + the for-loop dispatcher + post-loop terminal
# handlers + the outer try/except. Tests that grep for patterns must
# therefore consult both sources.
def _loop_combined_source():
    import inspect
    from jyagent.runtime.loop import engine as _le, step as _step
    return (
        inspect.getsource(_le.AgentLoop._run_impl)
        + "\n# ─── step.run_step ───\n"
        + inspect.getsource(_step.run_step)
    )


# ─── Shared fake registry ────────────────────────────────────────────────────


# ─── Shared per-test ToolBatch builder ───────────────────────────────────────


def _make_batch(
    *,
    functions: dict | None = None,
    schemas: list | None = None,
    parallel_safe: bool = True,
    timeout_hint: int | None = None,
) -> ToolBatch:
    """Construct a real ``ToolBatch`` from a dict of test functions.

    Replaces the old ``_FakeRegistry`` duck-typed stub.  After the
    runtime-tool-batch-snapshot refactor (2026-04), all dispatch helpers
    consume an immutable ``ToolBatch`` rather than a registry — building
    one from a real ``ToolRegistry.freeze()`` keeps tests honest about
    the production code path (no mock surface drift).
    """
    funcs = functions or {}
    reg = ToolRegistry()
    for name, fn in funcs.items():
        # Each test function gets an empty schema by default; tests that
        # need schema validation pass their own list via ``schemas``.
        schema = next(
            (s for s in (schemas or []) if s.get("name") == name),
            {"name": name, "input_schema": {"type": "object"}},
        )
        reg.register(
            name, fn, schema,
            parallel_safe=parallel_safe,
            timeout_hint=timeout_hint,
        )
    return reg.freeze()


def _sleep_tool(ms: int):
    """Factory: returns a tool function that sleeps and returns its label."""

    def _fn(label: str = "x"):
        time.sleep(ms / 1000.0)
        return f"{label}-done"

    return _fn


# ─── P0 #1 — Nested-future deadlock ──────────────────────────────────────────


class TestNoNestedPoolDeadlock:
    """Parallel batches must not deadlock on the shared dispatch pool.

    Bug: with a single shared pool of N workers, a parallel batch of N tools
    each submitting an inner body future would exhaust the pool and block
    forever.  The current design puts tool bodies on daemon threads, not a
    second pool — so there's no inner-future contention to deadlock on.
    """

    def test_backcompat_alias_points_to_dispatch(self):
        # Read both names via the live module attribute path.
        # C4 Phase 2 (2026-04-25): the pool is now canonical in
        # runtime/loop/tool_executor.py and engine.py forwards via PEP-562
        # ``__getattr__``.  The earlier-imported ``_tool_dispatch_executor``
        # at the top of this file is an import-time snapshot that goes stale
        # once any test grows the pool (previous tests in TestDispatchExecutorGrowsWithConfig
        # do exactly that).  Do a fresh read so this back-compat invariant
        # check tests the POST-MOVE semantics (both names surface the
        # current live pool) instead of the pre-move invariant (both names
        # were module-load aliases to the same frozen object).
        assert le._tool_executor is le._tool_dispatch_executor

    def test_parallel_batch_larger_than_dispatch_pool_completes(self):
        """A parallel batch of tools equal to the dispatch-pool width must
        complete.  With the old one-pool design this was a hang; with two
        pools it completes in roughly sleep_ms (not sleep_ms × N).
        """
        # Match dispatch width — this is the failure mode of the old design.
        # P3-2 (2026-04-27): pool is lazy-init; trigger it before reading
        # `_max_workers` so we don't get None.  `le._tool_dispatch_executor`
        # routes through the PEP-562 __getattr__ to the LIVE value.
        le._get_tool_dispatch_executor(8)
        n = le._tool_dispatch_executor._max_workers  # type: ignore[attr-defined]
        sleep_ms = 150
        functions = {f"t{i}": _sleep_tool(sleep_ms) for i in range(n)}
        batch = _make_batch(functions=functions, parallel_safe=True)
        blocks = [
            ToolCallRequest(id=f"id{i}", name=f"t{i}", input={"label": f"t{i}"})
            for i in range(n)
        ]

        # Generous outer timeout — deadlock would hit this; parallel execution
        # finishes in << 2s.
        t0 = time.perf_counter()

        def _run():
            return _execute_tools(
                blocks=blocks,
                batch=batch,
                concurrent_mode=True,
                max_workers=n,
                timeout=10,
            )

        # Run on a caller thread with a hard overall timeout so a regression
        # (re-introduced deadlock) fails cleanly instead of hanging pytest.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as caller:
            fut = caller.submit(_run)
            try:
                results = fut.result(timeout=5.0)
            except concurrent.futures.TimeoutError:
                pytest.fail(
                    f"_execute_tools hung for >5s on a {n}-wide parallel batch — "
                    "regression of the nested-pool deadlock."
                )

        elapsed = time.perf_counter() - t0
        # Sanity: all succeeded, results are ordered and non-error.
        assert len(results) == n
        for idx, (block, result) in enumerate(results):
            assert block.name == f"t{idx}"
            assert not result.is_error
            assert result.content.endswith("-done")
        # Parallelism check — must be much faster than sequential n × sleep_ms.
        sequential = (sleep_ms / 1000.0) * n
        assert elapsed < sequential / 2, (
            f"Batch took {elapsed:.2f}s for n={n} @ {sleep_ms}ms/tool; "
            f"sequential would be ~{sequential:.2f}s. Parallelism lost."
        )

    def test_timeout_wrapper_completes_successfully_on_fast_tool(self):
        """A fast tool completes via the daemon-thread timeout wrapper with
        no pool involvement (post-2026-04 fix replaced the inner pool with
        per-call daemon threads)."""
        batch = _make_batch(functions={"t": _sleep_tool(10)})
        result = _execute_tool_with_timeout("t", {"label": "t"}, batch, 5)
        assert not result.is_error
        assert result.content == "t-done"


# ─── P0 #3 — Cost tracker uses effective model spec ──────────────────────────


class TestCostTrackerUsesEffectiveSpec:
    """The loop must record cost against self._model_spec (sub-agent override)
    when set, not the runtime-owner's default spec.

    We validate by inspecting the compiled source of AgentLoop.run — this is
    both faster and more robust than booting a full runtime.
    """

    def test_cost_record_uses_effective_spec_variable(self):
        import inspect
        from jyagent.runtime.loop import step
        # After C4 Phase 5, effective_spec is hoisted in RunState.prepare_for_run()
        # (the run-state factory called at the top of _run_impl). The
        # cost_tracker.record(...) call lives in step.run_step.
        prepare_src = inspect.getsource(step.RunState.prepare_for_run)
        run_step_src = inspect.getsource(step.run_step)

        assert "effective_spec = loop._model_spec or loop._runtime_owner.model_spec" in prepare_src, (
            "RunState.prepare_for_run must resolve effective_spec once at the top of "
            "the run-state factory (formerly at the top of _run_impl)."
        )
        # cost_tracker.record(...) must consume effective_spec.
        idx = run_step_src.find("cost_tracker.record(")
        assert idx != -1, "cost_tracker.record(...) call is missing in run_step"
        snippet = run_step_src[idx : idx + 400]
        assert "effective_spec.provider" in snippet
        assert "effective_spec.model" in snippet
        assert "loop._runtime_owner.model_spec.provider" not in snippet, (
            "cost_tracker.record must NOT hardcode the owner spec — that loses "
            "sub-agent model overrides."
        )
        # Also confirm the engine's max_steps fallback path records cost
        # against effective_spec (B1 fix from codex review 2026-04-25).
        engine_src = inspect.getsource(le.AgentLoop._run_impl)
        fallback_idx = engine_src.find("fallback_on_max_steps")
        assert fallback_idx != -1, "max_steps fallback block missing"
        fallback_snippet = engine_src[fallback_idx : fallback_idx + 3000]
        assert "cost_tracker.record(" in fallback_snippet, (
            "max_steps fallback must record cost against effective_spec "
            "(B1 fix)."
        )
        assert "effective_spec.provider" in fallback_snippet


# ─── P0 #4 — Max-steps fallback always fires when enabled ────────────────────


class TestMaxStepsFallbackCondition:
    """fallback_on_max_steps must trigger whenever max_steps is reached AND
    the config enables it — independent of incidental `final_text` from prior
    tool-use steps.

    The old gate `if cfg.fallback_on_max_steps and not final_text:` silently
    skipped the fallback whenever any prior step emitted even a single token
    of pre-tool prose.  This test pins the post-fix behaviour.
    """

    def test_fallback_condition_is_config_only(self):
        import inspect
        source = inspect.getsource(le.AgentLoop._run_impl)
        # The gate must NOT reference final_text any more.
        # Find the "Max steps reached" comment block and inspect what follows.
        idx = source.find("# Max steps reached")
        assert idx != -1, "Expected 'Max steps reached' anchor comment"
        # Wide enough to clear any in-place rationale comments and reach the
        # actual `if cfg.fallback_on_max_steps:` gate that follows.
        # Wider window because the max-steps path now includes a
        # verification-dangle cleanup block between the anchor and the gate.
        snippet = source[idx : idx + 2000]
        # The gate must be the config flag alone.
        assert "if cfg.fallback_on_max_steps:" in snippet, (
            "Fallback gate must be `if cfg.fallback_on_max_steps:` — the old "
            "`and not final_text` condition nearly always evaluated false."
        )
        assert "fallback_on_max_steps and not final_text" not in snippet, (
            "`and not final_text` regression — remove the condition."
        )



# ─── P0 — Cancel-aware retry backoff ─────────────────────────────────────────


class TestCancellableRetrySleep:
    """Retry backoff must wake on cancellation so Ctrl-C is responsive.

    Old design: `time.sleep(delay)` blocks the full exponential-backoff window,
    so a user hitting Ctrl-C during a 4s retry wait burns that 4s before the
    next cancel check.  Fix: wait on the cancel_event instead of sleeping.
    """

    def _make_loop(self, cancel_event):
        # Build a minimal AgentLoop with a fake runtime_owner — only the
        # cancel helpers are exercised, not the full loop.
        class _Owner:
            class model_spec:
                provider = "anthropic"
                model = "claude-opus-4-6"

                @staticmethod
                def label():
                    return "anthropic:claude-opus-4-6"

        loop = le.AgentLoop.__new__(le.AgentLoop)
        loop._runtime_owner = _Owner()
        loop._config = le.LoopConfig()
        loop._callbacks = le.LoopCallbacks()
        loop._tool_source = None
        loop._model_spec = None
        loop._cancel_event = cancel_event
        loop._executor = le._tool_dispatch_executor
        return loop

    def test_sleep_without_cancel_event_blocks(self):
        """No cancel_event → falls back to plain sleep (returns False)."""
        loop = self._make_loop(cancel_event=None)
        t0 = time.perf_counter()
        result = loop._cancellable_sleep(0.05)
        elapsed = time.perf_counter() - t0
        assert result is False
        assert 0.04 <= elapsed <= 0.3  # generous upper bound for CI jitter

    def test_sleep_returns_early_when_cancelled_mid_wait(self):
        """cancel_event set during sleep → returns True well before deadline."""
        ev = threading.Event()
        loop = self._make_loop(cancel_event=ev)

        # Fire cancel after 50ms; request a 5s sleep.
        def _trip():
            time.sleep(0.05)
            ev.set()

        threading.Thread(target=_trip, daemon=True).start()
        t0 = time.perf_counter()
        result = loop._cancellable_sleep(5.0)
        elapsed = time.perf_counter() - t0
        assert result is True, "should report cancellation"
        assert elapsed < 1.0, (
            f"cancellable sleep should wake within ~50ms; took {elapsed:.2f}s"
        )

    def test_sleep_returns_immediately_if_already_cancelled(self):
        ev = threading.Event()
        ev.set()
        loop = self._make_loop(cancel_event=ev)
        t0 = time.perf_counter()
        result = loop._cancellable_sleep(10.0)
        elapsed = time.perf_counter() - t0
        assert result is True
        assert elapsed < 0.1


# ─── P0 — Stuck detector uses RAW content, not truncated display ─────────────


class TestStuckDetectorRawContent:
    """The stuck-loop detector must hash the raw tool output, not the
    UI-truncated string.  Two different long outputs that happen to share
    a prefix up to max_tool_result_chars would otherwise collide and
    trigger a false stuck-break.
    """

    def test_source_passes_raw_result_content_to_detector(self):
        import inspect
        source = _loop_combined_source()
        # Locate the stuck-detector.record call.
        idx = source.find("stuck_detector.record(")
        assert idx != -1, "stuck_detector.record(...) call is missing"
        snippet = source[idx : idx + 400]
        assert "result.content" in snippet, (
            "stuck_detector.record must consume the RAW result.content, not "
            "the UI-truncated display string."
        )
        # Negative check: must not feed the truncated `content_str` into the
        # detector (the truncation now happens only for the messages[] append).
        # We look specifically for the old pattern `content_str,\n` inside the
        # snippet — a bare mention of `content_str` is not a regression.
        assert "result.is_error,\n" not in snippet or "content_str,\n                    " not in snippet, (
            "stuck_detector.record appears to still receive the truncated "
            "content_str — regression."
        )


# ─── P0 — Parallel-batch stuck-detector dedup ────────────────────────────────


class TestStuckDetectorBatchDedup:
    """Legitimate parallel fanout of identical (name, args) tools in a single
    batch must not be counted as 'consecutive identical calls'.

    Bug: if a step fires `[read_file(a), read_file(a), read_file(a)]` in
    parallel, the detector's threshold=3 would be hit within that one step,
    producing a bogus dedup_break.  Fix: dedup (name, args) keys within a
    batch before recording.
    """

    def test_detector_not_triggered_by_three_identical_parallel_reads(self):
        """Direct white-box test: simulate the per-batch recording logic with
        dedup and assert no stuck feedback is produced.
        """
        detector = le._StuckLoopDetector(threshold=3)
        # Simulate 3 identical parallel reads in a single batch.
        name = "read_file"
        args = {"path": "/tmp/a"}
        content = "file contents here"

        # Build the same dedup set the loop uses.
        seen: set[str] = set()
        feedbacks = []
        for _ in range(3):
            key = le._StuckLoopDetector._make_key(name, args)
            if key in seen:
                continue
            seen.add(key)
            fb = detector.record(name, args, content)
            feedbacks.append(fb)

        # Only one record() call should have been made, and it cannot trigger
        # the detector on a single observation.
        assert feedbacks == [None], (
            f"Expected a single non-trigger record, got: {feedbacks}"
        )

    def test_detector_still_triggers_across_three_consecutive_steps(self):
        """Positive control: genuinely stuck pattern (same call across 3
        separate steps) must still trigger.
        """
        detector = le._StuckLoopDetector(threshold=3)
        name, args, content = "run_shell", {"command": "sleep 1"}, "done"

        results = [detector.record(name, args, content) for _ in range(3)]
        # First two record() calls: no trigger.  Third: STUCK LOOP feedback.
        assert results[0] is None
        assert results[1] is None
        assert results[2] is not None
        assert "STUCK LOOP" in results[2]

    def test_source_has_batch_dedup_set(self):
        """Source-level check: the loop must allocate a per-batch seen-set."""
        source = _loop_combined_source()
        # Anchor on the dedup comment and look for a seen-set just below.
        assert "seen_batch_keys" in source, (
            "AgentLoop.run must dedup (name, args) keys within a single tool "
            "batch before feeding the stuck detector."
        )


# ─── P0 — Cancellation check inside stream loop ──────────────────────────────


class TestStreamLoopCancellationCheck:
    """Streaming must poll cancellation inside the event loop so Ctrl-C
    doesn't wait for the provider to close the stream."""

    def test_stream_loop_has_cancel_check(self):
        import inspect
        # C4 Phase 3 (2026-04-25): streaming machinery moved to
        # ``runtime.loop.llm_runner.LLMRunner.call_streaming``.  The AgentLoop
        # ``_call_streaming`` delegate is a one-liner — inspect the runner.
        from jyagent.runtime.loop.llm_runner import LLMRunner
        source = inspect.getsource(LLMRunner.call_streaming)
        # The check must appear before the `etype = event.get(...)` dispatch
        # so that an in-flight stream can be short-circuited on cancel.
        assert "for event in stream:" in source
        assert "self._is_cancelled()" in source, (
            "LLMRunner.call_streaming must check self._is_cancelled() inside "
            "the event loop to short-circuit on Ctrl-C."
        )
        # Sanity: the cancel check lands before the event dispatch.
        iter_idx = source.find("for event in stream:")
        cancel_idx = source.find("self._is_cancelled()", iter_idx)
        dispatch_idx = source.find("event.get(", iter_idx)
        assert iter_idx < cancel_idx < dispatch_idx, (
            "cancel check must sit between `for event in stream:` and the "
            "event.get(...) dispatch."
        )



# ─── P0 — Verification gate boundary ─────────────────────────────────────────


class TestVerificationGateBoundary:
    """The verification gate must not inject `[VERIFICATION]` on the final
    allowed step — there's no iteration left for the model to reply, and the
    dangling user message would leak into the persisted session.

    Source-level assertions only: a full runtime integration would require
    booting real adapters.  The guard logic is a simple `step + 1 < max_steps`
    check, which is easy to verify in the source.
    """

    def test_gate_has_boundary_guard(self):
        import inspect
        source = _loop_combined_source()
        # The gate must include the boundary guard.
        assert "step + 1 < cfg.max_steps" in source, (
            "Verification gate is missing the `step + 1 < cfg.max_steps` "
            "boundary guard — it would inject [VERIFICATION] at max_steps-1 "
            "and leave a dangling unanswered user message."
        )

    def test_max_steps_handler_pops_dangling_verification(self):
        """Defense-in-depth: if somehow a verification prompt ends up at the
        tail of messages when max_steps is hit, the loop must clear it.

        After the runtime-finalize-funnel refactor (2026-04), cleanup is
        guaranteed by routing every exit through ``_finalize_run`` which
        unconditionally calls ``_strip_dangling_verification``.  See
        ``TestVerificationCleanupParity.test_no_bare_LoopResult_returns_in_run_impl``
        for the static funnel invariant covering all exit paths at once.
        """
        import inspect
        source = inspect.getsource(le.AgentLoop._run_impl)
        anchor = source.find("# ── max_steps exit ")
        assert anchor != -1, (
            "Could not find the max_steps exit anchor — refactor may have "
            "renamed the section"
        )
        tail_block = source[anchor : anchor + 2500]
        assert "_finalize_run(" in tail_block, (
            "max_steps exit must funnel through _finalize_run() (which calls "
            "_strip_dangling_verification unconditionally)"
        )
        assert 'status="max_steps"' in tail_block, (
            "max_steps exit must report status='max_steps'"
        )


# ─── P0 — Retry jitter ───────────────────────────────────────────────────────


class TestRetryJitter:
    """Retry backoff must include randomised jitter so parallel sub-agents
    don't all retry in lockstep after a 529 overload.

    The engine uses "equal jitter": half the delay is deterministic, half is
    uniform random in [0, base/2].  We verify empirically that across many
    samples the delay varies.
    """

    def _sleeps_for_attempts(self, attempts: int = 5, samples: int = 30) -> list[float]:
        """Monkey-patch _cancellable_sleep to capture the delay values the
        retry loop would sleep for across many invocations."""
        from unittest.mock import patch

        class _DummyTransient(Exception):
            pass

        # Pretend every exception is transient so retries are exercised.
        with patch.object(le, "_is_transient_error", return_value=True):
            delays: list[float] = []

            class _Loop(le.AgentLoop):
                # Bypass __init__ — we only need the retry code path.
                def __init__(self):  # type: ignore[no-redef]
                    self._config = le.LoopConfig(retry_attempts=attempts, retry_base_delay=1.0)
                    self._callbacks = le.LoopCallbacks()
                    self._cancel_event = None
                    self._model_spec = None

                def _cancellable_sleep(self, seconds: float) -> bool:
                    delays.append(seconds)
                    return False  # never cancel

                def _call_streaming(self, *a, **kw):
                    raise _DummyTransient("fail")

                def _call_complete(self, *a, **kw):
                    raise _DummyTransient("fail")

            for _ in range(samples):
                loop = _Loop()
                try:
                    loop._call_llm_with_retry(context={}, options=None, step=0)
                except _DummyTransient:
                    pass
            return delays

    def test_retry_delays_are_jittered(self):
        """Across many retry sequences the delay for a given attempt must
        vary — otherwise there's no jitter."""
        delays = self._sleeps_for_attempts(attempts=3, samples=20)
        assert len(delays) >= 30, f"expected many delays, got {len(delays)}"
        # Every sample produced `retry_attempts` delays (3), so we can slice.
        first_attempt = [delays[i] for i in range(0, len(delays), 3)]
        assert len(set(first_attempt)) > 1, (
            "All first-attempt delays identical — jitter missing."
        )

    def test_retry_delays_bounded_by_equal_jitter_formula(self):
        """Equal-jitter bounds: delay ∈ [base/2, base].  For attempt a with
        base_delay=1.0, that's [2^a/2, 2^a]."""
        delays = self._sleeps_for_attempts(attempts=3, samples=20)
        # Chunk into per-run sequences of 3.
        for i in range(0, len(delays), 3):
            chunk = delays[i : i + 3]
            if len(chunk) < 3:
                continue
            for attempt, d in enumerate(chunk):
                base = 2 ** attempt
                lo = base / 2
                hi = base
                assert lo - 1e-6 <= d <= hi + 1e-6, (
                    f"attempt {attempt}: delay {d} outside equal-jitter "
                    f"window [{lo}, {hi}]"
                )


# ─── P0 — Streaming retry deduplication hooks ────────────────────────────────


class _FakeRuntimeStream:
    """Mimics LLMStream: yields a deterministic list of events."""

    def __init__(self, events):
        self._events = events
        self._final = None
        for ev in events:
            if ev.get("type") in ("done", "error"):
                self._final = ev.get("message")

    def __iter__(self):
        yield from self._events

    def get_final_message(self):
        return self._final or {"role": "assistant", "content": [], "stop_reason": "stop"}

    def close(self):
        pass


class _FakeRuntimeOwner:
    """Minimal LLMOwner replacement for streaming tests."""

    def __init__(self, event_seqs):
        """event_seqs: list of event lists; each call to .stream() consumes one."""
        self._seqs = list(event_seqs)
        self._calls = 0

        class _spec:
            provider = "anthropic"
            model = "claude-opus-4-6"

            @staticmethod
            def label():
                return "anthropic:claude-opus-4-6"

        self.model_spec = _spec()

    def stream(self, context, options=None, model_spec=None):
        events = self._seqs[self._calls]
        self._calls += 1
        return _FakeRuntimeStream(events)

    def complete(self, *a, **kw):
        raise AssertionError("streaming tests must not call complete()")


def _make_loop_for_streaming(owner, *, buffered: bool = False, callbacks=None):
    """Build an AgentLoop bypassing normal __init__ for focused unit tests."""
    loop = le.AgentLoop.__new__(le.AgentLoop)
    loop._runtime_owner = owner
    loop._config = le.LoopConfig(
        streaming=True,
        buffered_streaming=buffered,
        retry_attempts=2,
        retry_base_delay=0.001,  # tiny — we never actually sleep in these tests
    )
    loop._callbacks = callbacks or le.LoopCallbacks()
    loop._tool_source = None
    loop._model_spec = None
    loop._cancel_event = None
    loop._executor = le._tool_dispatch_executor
    return loop


class TestBufferedStreaming:
    """buffered_streaming=True must defer on_text_delta until `done`."""

    def test_live_mode_fires_deltas_during_stream(self):
        # Live mode: each delta fires immediately.
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
        loop = _make_loop_for_streaming(owner, buffered=False, callbacks=cbs)

        text, tools, stop, msg = loop._call_streaming(context={}, options=None)
        assert text == "Hi there"
        assert stop == "stop"
        # Live: two delta fires with partials.
        assert emitted == ["Hi ", "there"]

    def test_buffered_mode_defers_until_done(self):
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
        call_order = []  # ordered record of (kind, payload)
        cbs = le.LoopCallbacks(
            on_text_delta=lambda t: call_order.append(("delta", t)),
        )
        loop = _make_loop_for_streaming(owner, buffered=True, callbacks=cbs)

        text, tools, stop, msg = loop._call_streaming(context={}, options=None)
        assert text == "Hi there"
        # Buffered: exactly one delta fire, with the full text.
        assert call_order == [("delta", "Hi there")], (
            f"Buffered mode should emit a single flush; got {call_order}"
        )

    def test_uses_final_message_text_when_stream_has_no_text_deltas(self):
        done_msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "stop",
            "usage": {},
        }
        owner = _FakeRuntimeOwner([[{"type": "done", "message": done_msg}]])
        emitted = []
        cbs = le.LoopCallbacks(on_text_delta=lambda t: emitted.append(t))
        loop = _make_loop_for_streaming(owner, buffered=False, callbacks=cbs)

        text, tools, stop, msg = loop._call_streaming(context={}, options=None)
        assert text == "Hello!"
        assert emitted == ["Hello!"]
        assert stop == "stop"
        assert msg is done_msg


class TestStreamRetryCallbacks:
    """on_stream_retry must fire with partial text on transient-error retry."""

    def test_partial_text_attached_to_error(self):
        """Streaming exceptions gain a `partial_stream_text` attribute holding
        whatever text the attempt had emitted before failing."""
        # Fake stream yields some text, then a `done` event with stop_reason=error.
        err_msg = {
            "role": "assistant",
            "content": [],
            "stop_reason": "error",
            "error_message": "simulated network blip",
            "usage": {},
        }
        events = [
            {"type": "text_delta", "text": "partial "},
            {"type": "text_delta", "text": "answer..."},
            {"type": "done", "message": err_msg},
        ]
        owner = _FakeRuntimeOwner([events])
        loop = _make_loop_for_streaming(owner, buffered=False)

        with pytest.raises(RuntimeError) as excinfo:
            loop._call_streaming(context={}, options=None)
        assert hasattr(excinfo.value, "partial_stream_text")
        assert excinfo.value.partial_stream_text == "partial answer..."

    def test_retry_fires_on_stream_retry_with_partial(self):
        """_call_llm_with_retry must fire on_stream_retry("transient_error",
        partial_text) before the retry sleep."""
        # First stream: partial text then transient error.
        # Second stream: clean success.
        err_msg = {
            "role": "assistant", "content": [], "stop_reason": "error",
            "error_message": "503 Service Unavailable", "usage": {},
        }
        done_msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "full answer"}],
            "stop_reason": "stop", "usage": {},
        }
        first = [
            {"type": "text_delta", "text": "partial"},
            {"type": "done", "message": err_msg},
        ]
        second = [
            {"type": "text_delta", "text": "full answer"},
            {"type": "done", "message": done_msg},
        ]
        owner = _FakeRuntimeOwner([first, second])

        retry_signals = []
        cbs = le.LoopCallbacks(
            on_stream_retry=lambda reason, partial: retry_signals.append((reason, partial)),
            on_text_delta=lambda t: None,
        )
        loop = _make_loop_for_streaming(owner, buffered=False, callbacks=cbs)
        # Treat the simulated 503 as transient.
        from unittest.mock import patch
        with patch.object(le, "_is_transient_error", return_value=True):
            text, _, stop, _ = loop._call_llm_with_retry(context={}, options=None, step=0)
        assert stop == "stop"
        assert text == "full answer"
        # Exactly one on_stream_retry fired with the transient-error reason
        # and the partial text from the failed attempt.
        assert retry_signals == [("transient_error", "partial")], retry_signals


class TestTruncationRecoveryEmitsStreamRetry:
    """Truncation recovery must also fire on_stream_retry so UIs use the same
    visual treatment for duplication that follows."""

    def test_source_fires_on_stream_retry_on_truncation(self):
        import inspect
        source = _loop_combined_source()
        # Anchor on the truncation block.
        idx = source.find("_is_truncated(stop_reason, tool_call_blocks)")
        assert idx != -1, "truncation-recovery block not found"
        snippet = source[idx : idx + 1500]
        assert '_fire("on_truncation")' in snippet, (
            "existing on_truncation callback missing"
        )
        assert '_fire("on_stream_retry", "truncation"' in snippet, (
            "truncation-recovery must also fire on_stream_retry so UIs can "
            "dedupe the replayed text"
        )


class TestLoopCallbacksHasNewHook:
    """The new on_stream_retry callback must be part of LoopCallbacks."""

    def test_dataclass_has_field(self):
        cbs = le.LoopCallbacks()
        # Default None is fine — presence of the attribute is what we need.
        assert hasattr(cbs, "on_stream_retry")
        assert cbs.on_stream_retry is None

    def test_dataclass_field_is_assignable(self):
        called = []
        cbs = le.LoopCallbacks(
            on_stream_retry=lambda reason, partial: called.append((reason, partial))
        )
        assert cbs.on_stream_retry is not None
        cbs.on_stream_retry("test", "hello")
        assert called == [("test", "hello")]



# ─── P0 — Compaction preserves thinking adjacent to tool_use ─────────────────


class TestCompactionPreservesThinkingAdjacency:
    """Anthropic extended-thinking requires that `thinking` blocks remain
    paired with their following `tool_use` block (the pair is signed).  The
    compaction pass must NOT strip thinking blocks from messages that also
    contain tool_use — doing so invalidates the signature and the provider
    rejects the next turn.
    """

    def _make_conv(self, n_padding: int = 20):
        """Build a conversation that exceeds the compaction threshold so
        compaction actually runs.  Includes one assistant message with a
        `thinking + tool_use` pair that MUST be preserved, and one assistant
        message with a standalone thinking block that may be stripped.
        """
        # One assistant message: thinking + tool_use (must preserve pairing).
        signed_assistant = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me think...", "signature": "SIG-123"},
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "/x"}},
            ],
        }
        # Another assistant message: standalone thinking, no tool_use.
        standalone_assistant = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Just musing...", "signature": "SIG-456"},
                {"type": "text", "text": "Here's my answer."},
            ],
        }
        # Pad with enough long user/tool-result messages to force compaction.
        padding: list[dict] = []
        for i in range(n_padding):
            padding.append({"role": "user", "content": "x" * 5_000})
            padding.append({
                "role": "tool_result",
                "tool_call_id": f"pad{i}",
                "tool_name": "read_file",
                "content": "y" * 5_000,
            })
        # Order: paired-assistant early, standalone-assistant early, padding, keep-intact tail.
        msgs = [signed_assistant, standalone_assistant] + padding + [
            {"role": "user", "content": "final prompt"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ]
        return msgs, signed_assistant, standalone_assistant

    def test_thinking_preserved_when_paired_with_tool_use(self):
        msgs, _, _ = self._make_conv(n_padding=30)
        # Aggressive token budget — force compaction to run.
        compacted = le._compact_messages(msgs, max_tokens=1000, compact_chars=200, batch=le.ToolBatch.empty())
        # Must be a different list (compaction ran).
        assert compacted is not msgs, "compaction did not run — test setup issue"

        # Locate the signed assistant message by its tool_use id.
        signed_after = None
        for m in compacted:
            if m.get("role") == "assistant" and isinstance(m.get("content"), list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("id") == "t1":
                        signed_after = m
                        break
        assert signed_after is not None, "signed assistant message lost"
        # The thinking block MUST survive.
        block_types = [b.get("type") for b in signed_after["content"] if isinstance(b, dict)]
        assert "thinking" in block_types, (
            "`thinking` block was stripped from an assistant message that "
            "also contains `tool_use` — this invalidates Anthropic "
            "extended-thinking signatures."
        )
        assert "tool_use" in block_types
        # Preserve order: thinking before tool_use (signature binding).
        assert block_types.index("thinking") < block_types.index("tool_use")

    def test_standalone_thinking_still_stripped_for_token_economy(self):
        msgs, _, _ = self._make_conv(n_padding=30)
        compacted = le._compact_messages(msgs, max_tokens=1000, compact_chars=200, batch=le.ToolBatch.empty())
        # Find the standalone assistant (text "Here's my answer.").
        stand = None
        for m in compacted:
            if m.get("role") == "assistant" and isinstance(m.get("content"), list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("type") == "text" \
                            and "my answer" in b.get("text", ""):
                        stand = m
                        break
        assert stand is not None, "standalone assistant message lost"
        block_types = [b.get("type") for b in stand["content"] if isinstance(b, dict)]
        assert "thinking" not in block_types, (
            "standalone `thinking` blocks (no tool_use) SHOULD still be "
            "stripped for token economy — only signature-pair thinking is "
            "protected."
        )


# ─── P0 — Tool timeout uses daemon thread (no pool leak) ─────────────────────


class TestDaemonThreadTimeout:
    """`_execute_tool_with_timeout` must NOT leak a pool slot when a tool
    body runs past its deadline.  Daemon threads accomplish this by holding
    no pool slot at all.
    """

    def test_timeout_returns_clean_error(self):
        """A tool body that exceeds the timeout returns a structured error."""
        def _slow(label: str = "x"):
            time.sleep(5.0)  # way over the 0.3s test timeout
            return "done"

        batch = _make_batch(functions={"slow_tool": _slow}, parallel_safe=False)
        t0 = time.perf_counter()
        result = le._execute_tool_with_timeout(
            "slow_tool", {"label": "x"}, batch, 1,
        )
        elapsed = time.perf_counter() - t0
        # Timeout should be honored within a small margin (inner +10s slack
        # only applies to run_shell; other tools honor the literal timeout).
        assert result.is_error
        assert "timed out" in result.content.lower()
        assert elapsed < 2.0, f"timeout wrapper waited too long: {elapsed:.2f}s"

    def test_many_consecutive_timeouts_do_not_leak_dispatch_pool(self):
        """Repeated timeouts must NOT exhaust a fixed-size pool.

        The regression signature of the OLD design (`_tool_body_executor.submit`
        + `future.cancel`): each timed-out tool pins a worker thread until the
        thread actually returns.  Once the pool's 16 workers are saturated,
        any NEW tool submission blocks until a worker frees — so a fast tool
        submitted after a flood of long-running timeouts would inherit the
        blocked-pool latency.

        The daemon-thread design holds no pool slot so a fast tool runs
        immediately regardless of how many long-running bodies are orphaned.
        """
        def _slow(label: str = "x"):
            # Sleep for longer than the total test budget so orphaned bodies
            # are definitely still running when we issue the fast tool.
            time.sleep(60.0)
            return "done"

        def _fast(label: str = "x"):
            return "ok"

        slow_batch = _make_batch(functions={"slow": _slow}, parallel_safe=False)
        fast_batch = _make_batch(functions={"fast": _fast}, parallel_safe=False)

        # Fire 20 timeouts.  Under the old design this would pin 16+ pool
        # workers indefinitely.  Budget: 20 × 1s ≈ 20s + overhead.
        for _ in range(20):
            r = le._execute_tool_with_timeout(
                "slow", {"label": "x"}, slow_batch, 1,
            )
            assert r.is_error and "timed out" in r.content.lower()

        # After the flood, a fast tool must still complete quickly.  Under
        # the old design, the pool would be fully saturated by the 16+
        # orphaned _slow workers, each still sleeping ~60s; a new submit
        # would block up to ~60 seconds waiting for a worker to free.
        # Under the new design: daemon thread, instant start.
        t0 = time.perf_counter()
        result = le._execute_tool_with_timeout(
            "fast", {"label": "x"}, fast_batch, 5,
        )
        elapsed_fast = time.perf_counter() - t0
        assert not result.is_error
        assert result.content == "ok"
        assert elapsed_fast < 1.0, (
            f"fast tool took {elapsed_fast:.2f}s after 20 timeouts — "
            "pool starvation suspected (regression of the leak fix)."
        )

    def test_daemon_thread_is_actually_spawned(self):
        """Source-level check that the implementation uses daemon threads
        rather than a pool submit.

        We match on positive patterns (daemon thread machinery) and on
        explicit absence of the old pool-submit code idiom, not on bare
        mentions of `future.cancel()` (which appear as rationale in the
        docstring and must not fail the test).
        """
        import ast
        import inspect

        source = inspect.getsource(le._execute_tool_with_timeout)
        tree = ast.parse(source)
        fn_node = tree.body[0]
        assert isinstance(fn_node, ast.FunctionDef)

        # Strip the docstring so we only look at executable statements.
        body = fn_node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        code_nodes = "\n".join(ast.unparse(n) for n in body)

        # Positive signals: daemon thread + Event-based wait.
        assert "threading.Thread(" in code_nodes, (
            "_execute_tool_with_timeout must spawn a dedicated thread per call"
        )
        assert "daemon=True" in code_nodes, (
            "tool-body thread must be a daemon so it never blocks exit"
        )
        assert ".wait(" in code_nodes, (
            "must use an Event-based wait to honor the timeout"
        )

        # Negative signals: none of the old pool-submit/future-cancel idiom.
        assert "pool.submit(" not in code_nodes, (
            "Tool body no longer goes through a pool — daemon-thread per call"
        )
        assert "_tool_body_executor.submit(" not in code_nodes
        assert "future.cancel(" not in code_nodes, (
            "future.cancel() is a no-op on thread futures and must not be "
            "used as the timeout strategy"
        )


# ─── Compaction preserves thinking paired with normalized `tool_call` ────────


class TestCompactionThinkingPairedWithToolCall:
    """The engine sees normalized assistant messages where the tool-invocation
    block type is `tool_call` (see runtime.types.ToolCallBlock).  Compaction's
    thinking-block pruner must therefore protect adjacency with BOTH the
    provider-native `tool_use` name AND the normalized `tool_call` name —
    otherwise signed-thinking/tool_use pairs get stripped in production even
    though the `tool_use`-only test passes on synthetic input.
    """

    def _make_conv(self, tool_block_type: str, n_padding: int = 30):
        signed_assistant = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Reasoning...", "signature": "SIG-AB"},
                {
                    "type": tool_block_type,
                    "id": "tcall",
                    "name": "read_file",
                    # `tool_call` uses `arguments`, `tool_use` uses `input`
                    ("arguments" if tool_block_type == "tool_call" else "input"): {"path": "/x"},
                },
            ],
        }
        padding = []
        for i in range(n_padding):
            padding.append({"role": "user", "content": "x" * 5_000})
            padding.append({
                "role": "tool_result",
                "tool_call_id": f"p{i}",
                "tool_name": "read_file",
                "content": "y" * 5_000,
            })
        return [signed_assistant] + padding + [
            {"role": "user", "content": "final prompt"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ], signed_assistant

    def test_thinking_preserved_when_paired_with_tool_call(self):
        msgs, _ = self._make_conv(tool_block_type="tool_call")
        compacted = le._compact_messages(msgs, max_tokens=1000, compact_chars=200, batch=le.ToolBatch.empty())
        assert compacted is not msgs, "compaction did not run — test setup issue"
        # Locate the signed assistant message.
        signed_after = None
        for m in compacted:
            if m.get("role") == "assistant" and isinstance(m.get("content"), list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("id") == "tcall":
                        signed_after = m
                        break
        assert signed_after is not None, "signed assistant message lost"
        block_types = [b.get("type") for b in signed_after["content"] if isinstance(b, dict)]
        assert "thinking" in block_types, (
            "`thinking` block was stripped from a normalized assistant message "
            "that contains a `tool_call` block — this breaks Anthropic extended "
            "thinking signatures when the message is re-serialized for the next "
            "turn."
        )
        # Order preserved: thinking before tool_call.
        assert block_types.index("thinking") < block_types.index("tool_call")


# ─── Verification cleanup parity across exit paths ───────────────────────────


class TestVerificationCleanupParity:
    """The dangling-[VERIFICATION] cleanup must run on every terminal path that
    fires AFTER the gate could have injected the prompt — not just max_steps.

    Otherwise a KeyboardInterrupt or uncaught exception between the gate's
    inject-and-continue step and the model's follow-up reply leaves the
    unanswered user message at the tail of `messages`, poisoning the next
    persisted turn.
    """

    def test_helper_strips_only_trailing_verification(self):
        """Direct test of the helper: idempotent, guards non-dict tail."""
        # Positive: trailing VERIFICATION user message is popped.
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": "[VERIFICATION] please self-check"},
        ]
        le._strip_dangling_verification(msgs)
        assert len(msgs) == 2
        assert msgs[-1]["role"] == "assistant"

        # Negative: non-VERIFICATION tail is left alone.
        msgs2 = [{"role": "user", "content": "normal"}]
        le._strip_dangling_verification(msgs2)
        assert len(msgs2) == 1

        # Empty list is a no-op.
        empty: list = []
        le._strip_dangling_verification(empty)
        assert empty == []

        # Non-dict tail is a no-op (guard against malformed input).
        mixed = [{"role": "user", "content": "hi"}, "not-a-dict"]
        le._strip_dangling_verification(mixed)
        assert mixed == [{"role": "user", "content": "hi"}, "not-a-dict"]

    def test_source_calls_helper_in_max_steps_path(self):
        """After the runtime-finalize-funnel refactor, the max_steps exit
        funnels through ``_finalize_run`` which always strips."""
        import inspect
        source = inspect.getsource(le.AgentLoop._run_impl)
        anchor = source.find("# ── max_steps exit ")
        assert anchor != -1
        tail_block = source[anchor : anchor + 3000]
        assert "_finalize_run(" in tail_block, (
            "max_steps path must exit through _finalize_run()"
        )

    def test_source_calls_helper_in_keyboardinterrupt_path(self):
        """The outer KeyboardInterrupt handler (the one that finalises the
        trace with status='interrupted') must funnel through _finalize_run.

        There is an inner ``except KeyboardInterrupt: raise`` nested inside
        the max_steps fallback try/except — that one is intentionally a
        bare re-raise, so we skip past it and assert on the outer handler.
        """
        import inspect
        source = inspect.getsource(le.AgentLoop._run_impl)
        # Find every ``except KeyboardInterrupt:`` and keep the one whose
        # body contains the interrupted-status return.
        import re as _re
        outer_handler = None
        for m in _re.finditer(r"except KeyboardInterrupt:", source):
            window = source[m.start() : m.start() + 800]
            if 'status="interrupted"' in window:
                outer_handler = window
                break
        assert outer_handler is not None, (
            "Could not locate the outer KeyboardInterrupt handler in _run_impl"
        )
        assert "_finalize_run(" in outer_handler, (
            "KeyboardInterrupt handler must exit through _finalize_run() so "
            "Ctrl-C mid-turn doesn't leak the unanswered [VERIFICATION] prompt "
            "into the next persisted session"
        )

    def test_source_calls_helper_in_exception_path(self):
        """The outer generic Exception handler (the one that finalises the
        trace with status='error') must funnel through _finalize_run."""
        import inspect
        import re as _re
        source = inspect.getsource(le.AgentLoop._run_impl)
        outer_handler = None
        for m in _re.finditer(r"except Exception as e:", source):
            window = source[m.start() : m.start() + 1500]
            if 'status="error"' in window:
                outer_handler = window
                break
        assert outer_handler is not None, (
            "Could not locate the outer Exception handler in _run_impl"
        )
        assert "_finalize_run(" in outer_handler, (
            "generic Exception handler must also funnel through _finalize_run()"
        )

    def test_no_bare_LoopResult_returns_in_run_impl(self):
        """**Funnel invariant.**  Every exit from ``_run_impl`` must go
        through ``_finalize_run`` so that dangling [VERIFICATION] cleanup
        and trace finalisation cannot be bypassed by a new exit path.

        This is the static guard that prevents regression of the bug class
        the funnel was built to eliminate (3 exit paths historically forgot
        to call ``_strip_dangling_verification``).  If you intentionally
        need a new exit path, route it through ``_finalize_run`` — do not
        weaken this test.
        """
        import inspect
        import re as _re
        source = _loop_combined_source()
        # Look for `return LoopResult(` — the pre-funnel pattern.  The
        # only acceptable return shape is `return _finalize_run(...)`.
        bare_returns = _re.findall(r"return\s+LoopResult\s*\(", source)
        assert bare_returns == [], (
            f"Found {len(bare_returns)} bare `return LoopResult(...)` in "
            f"_run_impl — every exit must funnel through _finalize_run() so "
            f"_strip_dangling_verification cannot be bypassed.  See "
            f"data/memory/topics/runtime-review-2026-04-25.md for context."
        )

    def test_finalize_run_always_strips_dangling_verification(self):
        """Behavioural guarantee of the funnel: calling ``_finalize_run`` on
        a messages list ending in a [VERIFICATION] prompt always pops it,
        regardless of whether trace is None or which status is reported."""
        for status in ("max_steps", "interrupted", "error", "cost_limit",
                       "dedup_break", "completed"):
            msgs = [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "[VERIFICATION] please self-check"},
            ]
            le._finalize_run(
                status=status,
                text="", final_text="",
                messages=msgs, steps=1,
                total_input_tokens=0, total_output_tokens=0,
                tool_calls_count=0,
                trace=None,
            )
            assert len(msgs) == 2, (
                f"_finalize_run(status={status!r}) must strip dangling "
                f"[VERIFICATION] but messages still has {len(msgs)} entries"
            )
            assert msgs[-1]["role"] == "assistant"


# ─── max_tool_workers honours LoopConfig ─────────────────────────────────────


class TestMaxToolWorkersApplied:
    """`LoopConfig.max_tool_workers` must actually cap concurrent tool-body
    execution.  Historically the value was threaded through `_execute_tools`
    but never applied — the shared dispatch pool (8 workers) always dictated
    parallelism.  A per-call BoundedSemaphore now honours the config.
    """

    def test_semaphore_caps_concurrent_bodies(self):
        """With max_workers=2, only 2 of 5 parallel-safe tools run at once."""
        import concurrent.futures as cf

        active = 0
        peak = 0
        lock = threading.Lock()
        event = threading.Event()

        def _tracked(label: str = "x"):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            # Hold until enough observations are in — but short overall so
            # even if the cap is broken the test finishes.
            event.wait(0.3)
            with lock:
                active -= 1
            return f"{label}-done"

        n = 5
        blocks = [
            ToolCallRequest(id=f"id{i}", name="t", input={"label": f"t{i}"})
            for i in range(n)
        ]
        batch = _make_batch(functions={"t": _tracked}, parallel_safe=True)

        t0 = time.perf_counter()
        caller = cf.ThreadPoolExecutor(max_workers=1)
        fut = caller.submit(
            _execute_tools,
            blocks=blocks,
            batch=batch,
            concurrent_mode=True,
            max_workers=2,
            timeout=5,
        )
        # Give the batch a moment to saturate the cap, then release.
        time.sleep(0.1)
        event.set()
        results = fut.result(timeout=5.0)
        caller.shutdown(wait=True)
        elapsed = time.perf_counter() - t0

        assert len(results) == n
        assert all(not r.is_error for _, r in results)
        assert peak <= 2, (
            f"max_tool_workers=2 was ignored — saw peak={peak} concurrent bodies"
        )
        # Sanity: with cap=2 and 5 tools of ~0.3s hold-then-release, the
        # floor is two hold waves (~0.3s + ~0.3s). Generous upper bound so
        # CI jitter doesn't flake.
        assert elapsed < 3.0, f"unexpected slowdown: {elapsed:.2f}s"


# ─── _CostTracker reports lower bound when pricing is unknown ────────────────


class TestCostTrackerLowerBound:
    """Unknown pricing must report a lower-bound ``cost`` with
    ``has_unpriced_usage`` set — the old ``known_cost → None`` sentinel
    silently disabled the budget check on any unpriced call.
    """

    def test_cost_excludes_unpriced_and_flags_it(self):
        tracker = le._CostTracker()
        # One priced call at a known-to-exist model; one call at a bogus
        # provider that lookup will miss.
        tracker.record(
            {"input_tokens": 1000, "output_tokens": 500},
            "anthropic", "claude-opus-4-6",
        )
        priced_only = tracker.cost
        assert priced_only > 0, "priced call should have produced non-zero cost"
        assert not tracker.has_unpriced_usage
        assert tracker.unpriced_calls == 0

        tracker.record(
            {"input_tokens": 1000, "output_tokens": 500},
            "bogus-provider", "does-not-exist",
        )
        # Unpriced call must NOT alter the running total, only the flag.
        assert tracker.cost == priced_only, (
            "unpriced call tokens must not enter the running total — they'd "
            "fabricate a number that isn't backed by pricing data"
        )
        assert tracker.has_unpriced_usage
        assert tracker.unpriced_calls == 1

    def test_budget_gate_does_not_silently_disable_on_unpriced(self):
        """Source-level assertion: the budget check uses tracker.cost, not a
        None-returning sentinel — so the gate still fires even when some
        calls were unpriced."""
        import inspect
        source = inspect.getsource(le.AgentLoop._run_impl)
        assert "cost_tracker.cost" in source, (
            "budget check must consume cost_tracker.cost (partial total)"
        )
        assert "cost_tracker.known_cost" not in source, (
            "regression: known_cost returned None on unpriced usage, which "
            "silently disabled the budget"
        )
        assert "has_unpriced_usage" in source, (
            "loop must surface has_unpriced_usage via on_warning so the "
            "lower-bound nature of the budget is visible"
        )


# ─── Per-step ToolBatch isolates dispatch from concurrent registry mutation ──


class TestToolBatchSnapshotIsolation:
    """``ToolRegistry.freeze()`` must produce an immutable per-step snapshot
    such that a concurrent ``register()``/``unregister()``/schema mutation
    cannot affect any helper that consumed the snapshot.

    Codex review 2026-04-25 (Part 1 #4, #11, #12) flagged three concrete
    races the engine had under the old "live registry reads everywhere"
    design:

      * #4  — ``get_schema(name)`` was a live unlocked lookup, so validation
              could pair the per-step ``functions`` snapshot with a *new*
              schema or no schema mid-step.
      * #11 — ``is_parallel_safe(name)`` was a live unlocked read called
              three times during partition, so the same batch could see a
              tool as serial in one place and parallel in another.
      * #12 — ``snapshot()`` returned shared schema dict references, so a
              caller that retained a registered schema dict could mutate
              the snapshot through aliasing.

    These tests exercise the snapshot contract directly.
    """

    def test_freeze_is_atomic_under_concurrent_registration(self):
        """Race a thread that hammers register()/unregister() against
        repeated freeze() calls.  Each frozen batch must be internally
        consistent: every name in ``functions`` must also have a schema
        and metadata entry; ``parallel_safe`` membership must agree with
        ``schema_map`` membership for any tool registered with that flag.
        """
        reg = ToolRegistry()

        def _churn(stop: threading.Event, idx_holder: list):
            i = 0
            while not stop.is_set():
                name = f"churn_{i % 8}"
                if i % 2 == 0:
                    reg.register(
                        name, lambda **_: "ok",
                        {"name": name, "input_schema": {"type": "object"}},
                        parallel_safe=(i % 4 == 0),
                        timeout_hint=10,
                    )
                else:
                    reg.unregister(name)
                i += 1
                idx_holder[0] = i

        stop = threading.Event()
        idx_holder = [0]
        churner = threading.Thread(target=_churn, args=(stop, idx_holder), daemon=True)
        churner.start()

        try:
            # Take 200 freezes against the churn.  Every batch must satisfy
            # the consistency invariants below, regardless of churn timing.
            for _ in range(200):
                batch = reg.freeze()
                for name in batch.functions:
                    assert name in batch.schema_map, (
                        f"tool {name!r} in batch.functions but missing from "
                        f"batch.schema_map — non-atomic freeze()"
                    )
                # parallel_safe must be a subset of registered tools (a tool
                # can't be parallel_safe if it isn't in the batch).
                assert batch.parallel_safe <= set(batch.functions), (
                    f"parallel_safe contains names absent from functions: "
                    f"{batch.parallel_safe - set(batch.functions)}"
                )
        finally:
            stop.set()
            churner.join(timeout=2.0)

        assert idx_holder[0] > 100, (
            "churner did not run enough iterations for the test to be "
            f"meaningful (only {idx_holder[0]} cycles)"
        )

    def test_batch_isolates_schema_from_post_freeze_mutation(self):
        """Codex Part 1 #12: the OLD ``snapshot()`` returned the original
        schema dict by reference, so a caller mutating the dict it had
        passed to ``register()`` could change what the snapshot exposed.
        ``freeze()`` deep-copies — post-freeze mutation must not leak in.
        """
        reg = ToolRegistry()
        live_schema = {
            "name": "shared",
            "description": "ORIGINAL",
            "input_schema": {"type": "object", "properties": {}},
        }
        reg.register("shared", lambda **_: "ok", live_schema, parallel_safe=True)

        batch = reg.freeze()
        # Mutate the dict we registered with.  A correctly-isolating
        # freeze() must not propagate this into the frozen schema_map.
        live_schema["description"] = "MUTATED"
        live_schema["input_schema"]["properties"]["sneaky"] = {"type": "string"}

        snap = batch.get_schema("shared")
        assert snap is not None
        assert snap["description"] == "ORIGINAL", (
            "freeze() did not deep-copy schema['description']; post-register "
            "mutation leaked into the snapshot"
        )
        assert "sneaky" not in snap["input_schema"]["properties"], (
            "freeze() did not deep-copy nested schema dicts; post-register "
            "mutation leaked into input_schema"
        )

    def test_dispatch_partition_consistent_when_parallel_safe_flips_mid_step(self):
        """Codex Part 1 #11: ``_execute_tools`` historically called
        ``registry.is_parallel_safe(name)`` three times during partition
        (the outer `any(...)` check, the contiguous-group head check, and
        the contiguous-group extend check).  If a concurrent register()
        flipped the flag between those reads, the same batch could be
        partitioned inconsistently.

        With ``ToolBatch`` the flag lives in a frozen ``frozenset``: no
        amount of concurrent registry churn can change what a frozen
        batch sees.  This test verifies the invariant by mutating the
        registry while ``_execute_tools`` runs against a previously
        frozen batch.
        """
        reg = ToolRegistry()

        def _slow(label: str = "x"):
            time.sleep(0.05)
            return f"{label}-done"

        # Register `t` as parallel_safe.
        reg.register(
            "t", _slow,
            {"name": "t", "input_schema": {"type": "object"}},
            parallel_safe=True,
        )
        batch = reg.freeze()

        # Sanity: the frozen batch sees `t` as parallel-safe.
        assert batch.is_parallel_safe("t")

        # Concurrently flip `t`'s parallel_safe flag (re-register) while
        # _execute_tools runs.  The frozen batch must NOT observe the
        # flip.  Without the snapshot, the partition logic could see
        # "parallel_safe" for `any(...)` but "not parallel_safe" by the
        # time the inner while-loop reads — a mismatched partition.
        flipper_done = threading.Event()

        def _flip_repeatedly():
            for i in range(50):
                if flipper_done.is_set():
                    return
                reg.register(
                    "t", _slow,
                    {"name": "t", "input_schema": {"type": "object"}},
                    parallel_safe=(i % 2 == 0),
                )

        flipper = threading.Thread(target=_flip_repeatedly, daemon=True)
        flipper.start()

        try:
            blocks = [
                ToolCallRequest(id=f"id{i}", name="t", input={"label": f"t{i}"})
                for i in range(8)
            ]
            results = _execute_tools(
                blocks=blocks,
                batch=batch,  # frozen — flipper cannot affect this
                concurrent_mode=True,
                max_workers=4,
                timeout=5,
            )
            # All tools must complete successfully.  If partition was
            # inconsistent we'd see dispatch errors or hangs.
            assert len(results) == len(blocks)
            for block, r in results:
                assert not r.is_error, (
                    f"tool {block.id} failed under concurrent registry churn — "
                    f"frozen batch should isolate dispatch from registry: {r.content!r}"
                )
        finally:
            flipper_done.set()
            flipper.join(timeout=2.0)

        # Post-condition: the frozen batch's view is unchanged regardless
        # of what the registry now thinks.
        assert batch.is_parallel_safe("t"), (
            "frozen batch's parallel_safe membership changed during dispatch — "
            "frozenset is supposed to be immutable"
        )
