"""Regression tests: ContextVar state propagates across every
thread/executor.submit spawn site in the agent loop.

Codex review item #7 from the 2026-05 loop-runtime co-review.

Background
----------
``threading.Thread`` and ``concurrent.futures.ThreadPoolExecutor.submit``
do NOT auto-propagate ``ContextVar`` state — workers start in the
default empty context.  See Python 3.14.3 ``contextvars`` docs and
this project's MEMORY.md ("ContextVar is NOT auto-propagated by
ThreadPoolExecutor.submit()...").

Today the loop deliberately avoids ContextVars for tool state (per
the daemon-thread gotcha in MEMORY.md), but ANY future tool, provider
SDK, or tracing integration that uses ContextVars will silently lose
state at the four spawn sites flagged by the codex review:

  1. ``tool_invocation.execute_tool_with_timeout``: daemon thread per
     tool body
  2. ``tool_dispatch.execute_tools``: ``pool.submit`` per parallel-safe
     tool in a batch
  3. ``llm_runner.LLMRunner.call_complete``: daemon worker for the
     SDK ``complete()`` call (only when cancel_event is wired up)
  4. ``llm_runner.LLMRunner.call_streaming``: daemon cancel watcher
     (only when cancel_event is wired up)

Each spawn site now snapshots the parent's context with
``contextvars.copy_context()`` and routes the worker through
``ctx.run``.  These tests pin that contract.
"""
from __future__ import annotations

import contextvars
import threading
import time

import pytest

from jyagent.runtime.loop import llm_runner as le_lr
from jyagent.runtime.loop import tool_invocation
from jyagent.runtime.loop import tool_dispatch
from jyagent.runtime.loop.llm_types import ToolCallRequest
from jyagent.runtime.tools.registry import ToolRegistry


# Module-scope ContextVar exercised by every test.  Default is the
# sentinel "<unset>" so a worker that doesn't see propagation reports
# the sentinel rather than a stale value from a previous test.
_TEST_CV: contextvars.ContextVar[str] = contextvars.ContextVar(
    "jyagent_test_cv", default="<unset>",
)


# ─────────────────────────────────────────────────────────────────────────
# Site 1: execute_tool_with_timeout daemon thread
# ─────────────────────────────────────────────────────────────────────────


class TestToolBodyDaemonInheritsCV:
    def test_tool_body_sees_parent_cv(self):
        """The tool body daemon thread must see the CV value the
        dispatch thread had set immediately before spawning it."""
        observed: list[str] = []

        def tool_fn() -> str:
            observed.append(_TEST_CV.get())
            return "ok"

        reg = ToolRegistry()
        reg.register(
            "probe_tool", tool_fn,
            {"name": "probe_tool", "input_schema": {"type": "object"}},
        )
        batch = reg.freeze()

        token = _TEST_CV.set("from-parent")
        try:
            result = tool_invocation.execute_tool_with_timeout(
                "probe_tool", {}, batch, default_timeout=5,
            )
        finally:
            _TEST_CV.reset(token)

        assert not result.is_error, result.content
        assert observed == ["from-parent"], (
            f"daemon thread did not inherit parent CV: observed={observed}"
        )

    def test_tool_body_cv_mutation_does_not_leak_back(self):
        """A ``CV.set`` inside the tool body stays local to the
        daemon's context copy — the parent thread's view is unchanged."""
        def mutating_tool() -> str:
            _TEST_CV.set("mutated-by-tool")
            return "ok"

        reg = ToolRegistry()
        reg.register(
            "mutating_tool", mutating_tool,
            {"name": "mutating_tool", "input_schema": {"type": "object"}},
        )
        batch = reg.freeze()

        token = _TEST_CV.set("parent-original")
        try:
            tool_invocation.execute_tool_with_timeout(
                "mutating_tool", {}, batch, default_timeout=5,
            )
            # Parent's view must still be the original — the tool
            # mutated its own copy of the context, not ours.
            assert _TEST_CV.get() == "parent-original"
        finally:
            _TEST_CV.reset(token)


# ─────────────────────────────────────────────────────────────────────────
# Site 2: execute_tools parallel pool.submit (per-batch fan-out)
# ─────────────────────────────────────────────────────────────────────────


class TestParallelDispatchInheritsCV:
    def test_each_parallel_worker_sees_parent_cv(self):
        """A parallel-safe batch spawns N workers via pool.submit; each
        must inherit the dispatcher's CV state.  Without per-submit
        context copying, the workers would start in the default empty
        context and report ``"<unset>"``."""
        observed_lock = threading.Lock()
        observed: list[str] = []

        def probe_tool(label: str = "x") -> str:
            with observed_lock:
                observed.append(_TEST_CV.get())
            # Brief sleep so the runs actually overlap — proves we
            # aren't accidentally re-using the same Context object
            # (which cannot be entered concurrently).
            time.sleep(0.05)
            return f"{label}-done"

        reg = ToolRegistry()
        reg.register(
            "probe_tool", probe_tool,
            {"name": "probe_tool", "input_schema": {
                "type": "object",
                "properties": {"label": {"type": "string"}},
            }},
            parallel_safe=True,
        )
        batch = reg.freeze()

        n = 5
        blocks = [
            ToolCallRequest(id=f"id{i}", name="probe_tool", input={"label": f"t{i}"})
            for i in range(n)
        ]
        token = _TEST_CV.set("from-dispatcher")
        try:
            results = tool_dispatch.execute_tools(
                blocks=blocks,
                batch=batch,
                concurrent_mode=True,
                max_workers=4,
                timeout=5,
            )
        finally:
            _TEST_CV.reset(token)

        assert len(results) == n
        for _, r in results:
            assert not r.is_error, r.content
        # Every parallel worker must have observed the dispatcher's CV.
        assert observed == ["from-dispatcher"] * n, (
            f"some workers did not inherit dispatcher CV: observed={observed}"
        )

    def test_parallel_workers_do_not_share_context(self):
        """Each pool.submit gets a FRESH context copy — a single
        ``Context`` cannot be entered concurrently.  If the
        implementation accidentally shared one ``Context`` across
        workers, ``ctx.run`` would raise ``RuntimeError`` on the second
        entrant.  This test creates enough overlap to surface that bug."""
        barrier = threading.Barrier(parties=4)
        errors: list[BaseException] = []

        def overlap_tool(label: str = "x") -> str:
            try:
                # All 4 workers MUST be inside their context.run
                # simultaneously — barrier blocks until they are.
                barrier.wait(timeout=3)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
                raise
            return f"{label}-done"

        reg = ToolRegistry()
        reg.register(
            "overlap_tool", overlap_tool,
            {"name": "overlap_tool", "input_schema": {
                "type": "object",
                "properties": {"label": {"type": "string"}},
            }},
            parallel_safe=True,
        )
        batch = reg.freeze()

        blocks = [
            ToolCallRequest(id=f"id{i}", name="overlap_tool", input={"label": f"t{i}"})
            for i in range(4)
        ]
        results = tool_dispatch.execute_tools(
            blocks=blocks,
            batch=batch,
            concurrent_mode=True,
            max_workers=4,
            timeout=10,
        )
        assert errors == [], errors
        for _, r in results:
            assert not r.is_error, r.content


# ─────────────────────────────────────────────────────────────────────────
# Site 3: LLMRunner.call_complete daemon worker (cancel-event path)
# ─────────────────────────────────────────────────────────────────────────


class _StubOwner:
    """Minimal LLMClient stand-in — captures the CV inside complete()."""
    def __init__(self):
        self.observed_cv: str | None = None

        class _spec:
            provider = "anthropic"
            model = "test-model"
        self.model_spec = _spec()

    def complete(self, context, options=None, model_spec=None):
        # Read the CV from the daemon worker's context — must reflect
        # what the dispatcher had set, not the default.
        self.observed_cv = _TEST_CV.get()
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "stop",
            "usage": {},
        }

    def stream(self, *a, **kw):
        raise AssertionError("call_complete tests must not stream")


class TestCallCompleteDaemonInheritsCV:
    def test_complete_worker_sees_parent_cv(self):
        owner = _StubOwner()
        cancel_event = threading.Event()  # never fires; presence triggers daemon path
        runner = le_lr.LLMRunner.__new__(le_lr.LLMRunner)
        runner.runtime_owner = owner
        runner.config = le_lr.LoopConfig(streaming=False)
        runner._callbacks = le_lr.LoopCallbacks()
        runner._cancel_event = cancel_event
        runner.model_spec = None

        token = _TEST_CV.set("complete-cv")
        try:
            runner.call_complete(context={}, options=None)
        finally:
            _TEST_CV.reset(token)

        assert owner.observed_cv == "complete-cv", (
            f"call_complete daemon did not inherit parent CV: "
            f"observed={owner.observed_cv!r}"
        )


# ─────────────────────────────────────────────────────────────────────────
# Site 4: structural — every relevant module imports contextvars
# (catches a future contributor adding a new spawn site without
#  threading the propagation through)
# ─────────────────────────────────────────────────────────────────────────


class TestSpawnSitesUseCopyContext:
    """Every module that spawns a thread or pool task in the loop must
    import ``contextvars`` and call ``copy_context().run`` somewhere.
    Catches drift if a new spawn site lands without the propagation."""

    def _module_has_copy_context_call(self, module) -> bool:
        import inspect
        src = inspect.getsource(module)
        return "copy_context()" in src and ("ctx.run" in src or ".run," in src)

    def test_tool_invocation_uses_copy_context(self):
        assert self._module_has_copy_context_call(tool_invocation)

    def test_tool_dispatch_uses_copy_context(self):
        assert self._module_has_copy_context_call(tool_dispatch)

    def test_llm_runner_uses_copy_context(self):
        assert self._module_has_copy_context_call(le_lr)
