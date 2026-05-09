"""Regression tests for the tool_executor split (codex review item #6).

In 2026-05 the 621-line ``runtime/loop/tool_executor.py`` was split into
three focused modules:

* ``runtime/loop/tool_pool.py``        — shared executor + lifecycle
* ``runtime/loop/tool_invocation.py``  — single-call invocation + timeout
* ``runtime/loop/tool_dispatch.py``    — fan-out across sub-batches

``tool_executor.py`` remains as a thin facade that re-exports every
public name and uses PEP-562 ``__getattr__`` to passthrough live pool
state (``tool_dispatch_executor`` / ``tool_dispatch_lock`` /
``tool_dispatch_cap``).

These tests pin the contracts that matter in practice:

1. Each leaf module hosts the right symbols and is independently
   importable.
2. The facade re-exports every name callers depend on, with object
   identity preserved (so ``isinstance``/``is`` checks against either
   module work).
3. The PEP-562 read-passthrough returns the LIVE pool, not a snapshot.
4. Pool growth via the facade is observed through the leaf module
   (and vice-versa) — there is exactly one canonical state.
5. The ``tool_executor`` facade does NOT carry behavior — none of the
   actual function bodies live in it any more.
"""
from __future__ import annotations

import inspect

from jyagent.runtime.loop import tool_executor as te
from jyagent.runtime.loop import tool_pool
from jyagent.runtime.loop import tool_invocation
from jyagent.runtime.loop import tool_dispatch


# ─────────────────────────────────────────────────────────────────────────
# Structural: the right symbols live in the right modules
# ─────────────────────────────────────────────────────────────────────────


class TestSymbolsLandedInTheRightLeafModule:
    def test_pool_state_lives_in_tool_pool(self):
        """Mutable pool state is in tool_pool, NOT in the facade."""
        assert hasattr(tool_pool, "tool_dispatch_executor")
        assert hasattr(tool_pool, "tool_dispatch_lock")
        assert hasattr(tool_pool, "tool_dispatch_cap")
        assert callable(tool_pool.get_tool_dispatch_executor)

    def test_invocation_helpers_live_in_tool_invocation(self):
        assert callable(tool_invocation.execute_tool)
        assert callable(tool_invocation.execute_tool_with_timeout)
        assert callable(tool_invocation._accepts_cancel_event)

    def test_fanout_lives_in_tool_dispatch(self):
        assert callable(tool_dispatch.execute_tools)

    def test_facade_holds_no_behavior_only_aliases(self):
        """``tool_executor.py`` has zero function definitions of its own —
        every public name is sourced from a leaf module.  This pins the
        split: a future contributor cannot quietly re-inline a function
        body into the facade.
        """
        # Source code of the facade module is small (under 100 lines) and
        # contains no ``def`` keyword introducing a runtime function.
        # Module-level ``def __getattr__`` and ``def __dir__`` are PEP-562
        # hooks, not behavior — explicitly allowed.
        src = inspect.getsource(te)
        # Count ``def `` occurrences; only PEP-562 hooks are allowed.
        defs = [line.strip() for line in src.splitlines() if line.lstrip().startswith("def ")]
        allowed = {"def __getattr__(name: str):", "def __dir__():"}
        unexpected = [d for d in defs if d not in allowed]
        assert not unexpected, (
            f"tool_executor.py grew a real function definition: {unexpected}.  "
            f"Behavior belongs in tool_pool / tool_invocation / tool_dispatch."
        )


# ─────────────────────────────────────────────────────────────────────────
# Identity: the facade re-exports the SAME function objects (not copies)
# ─────────────────────────────────────────────────────────────────────────


class TestFacadeReExportsSameObjects:
    def test_execute_tool_identity(self):
        assert te.execute_tool is tool_invocation.execute_tool

    def test_execute_tool_with_timeout_identity(self):
        assert te.execute_tool_with_timeout is tool_invocation.execute_tool_with_timeout

    def test_execute_tools_identity(self):
        assert te.execute_tools is tool_dispatch.execute_tools

    def test_get_executor_identity(self):
        assert te.get_tool_dispatch_executor is tool_pool.get_tool_dispatch_executor

    def test_accepts_cancel_event_identity(self):
        assert te._accepts_cancel_event is tool_invocation._accepts_cancel_event


# ─────────────────────────────────────────────────────────────────────────
# PEP-562 passthrough: facade reads always reflect live tool_pool state
# ─────────────────────────────────────────────────────────────────────────


class TestFacadePassthroughReturnsLiveState:
    def test_lock_passthrough_is_canonical(self):
        """The lock object exposed via the facade IS the canonical lock —
        callers that acquire it are coordinating with the real growth path."""
        assert te.tool_dispatch_lock is tool_pool.tool_dispatch_lock

    def test_executor_passthrough_observes_growth(self):
        """Growing the pool through ``te.get_tool_dispatch_executor`` must
        be observable on the next facade read.  This is the bug the
        passthrough exists to prevent — a snapshotted reference would
        point at the dead, shut-down pool after growth."""
        # Snapshot to restore at the end so this test is independent.
        saved_exec = tool_pool.tool_dispatch_executor
        saved_cap = tool_pool.tool_dispatch_cap
        try:
            # Force a growth event regardless of where the pool currently is.
            te.get_tool_dispatch_executor(8)
            baseline_cap = te.tool_dispatch_cap
            target = baseline_cap + 8
            grown = te.get_tool_dispatch_executor(target)
            assert te.tool_dispatch_executor is grown
            assert te.tool_dispatch_executor is tool_pool.tool_dispatch_executor
            assert te.tool_dispatch_cap >= target
            assert tool_pool.tool_dispatch_cap == te.tool_dispatch_cap
        finally:
            tool_pool.tool_dispatch_executor = saved_exec
            tool_pool.tool_dispatch_cap = saved_cap

    def test_unknown_attribute_raises(self):
        """Names outside the passthrough whitelist still raise
        AttributeError — the facade is not a permissive proxy."""
        import pytest
        with pytest.raises(AttributeError):
            te.this_attribute_does_not_exist  # noqa: B018

    def test_dir_includes_passthrough_names(self):
        """``dir(tool_executor)`` must list the passthrough names so
        IDE autocomplete + ``hasattr`` checks behave intuitively."""
        listing = dir(te)
        for name in ("tool_dispatch_executor", "tool_dispatch_lock", "tool_dispatch_cap"):
            assert name in listing, listing
