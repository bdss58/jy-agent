"""Regression tests for codex review 2026-04-29 fixes.

Covers:
    B-2 — ``ToolBatch.from_source`` deep-copies schemas + wraps maps in
          MappingProxyType.
    H-1 — ``check_background`` carries ``timeout_hint=360`` and
          ``mutating=True`` (kill branch SIGTERM/SIGKILLs the process group).
          (See also tests/test_background.py for the registry-flag check.)
    M-2 — ``dispatch_agent`` is no longer parallel-safe (serialised
          coarse-grained sub-agent dispatches).
    M-5 — ``ToolBatch.with_overlay`` accepts an explicit ``mutating`` set.
    L-3 — ``LoopCheckpoint.from_json`` filters unknown fields so a newer-
          version checkpoint loads cleanly in older code.

The B-1 (verification gate consults batch.is_mutating) regression lives
in ``tests/test_tracing_and_verification.py`` next to the rest of the
verification-gate tests.
"""

from __future__ import annotations

import json

import pytest


# ─── B-2: ToolBatch.from_source deep-copy + readonly ────────────────────────


class TestB2ToolBatchFromSource:
    """``ToolBatch.from_source`` mirrors ``ToolRegistry.freeze()`` safety
    properties for the non-registry tool_source path: deep-copied schemas,
    MappingProxyType views, inherited metadata classification.
    """

    def _make_base(self):
        from jyagent.runtime.tools.registry import ToolRegistry
        reg = ToolRegistry()
        reg.register(
            "real_tool",
            lambda: "ok",
            {"name": "real_tool", "input_schema": {"type": "object"}},
            mutating=True,
            timeout_hint=42,
        )
        return reg.freeze()

    def test_from_source_deep_copies_schemas(self):
        """Mutating the input schema list AFTER from_source() must not
        affect the resulting batch's stored schema."""
        from jyagent.runtime.tools.registry import ToolBatch
        src_schema = {
            "name": "src_tool",
            "input_schema": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
            },
        }
        src_schemas = [src_schema]
        src_functions = {"src_tool": lambda: "x"}

        batch = ToolBatch.from_source(src_schemas, src_functions, base=self._make_base())

        # Mutate the input list/element after the call.
        src_schema["name"] = "mutated_name"
        src_schema["input_schema"]["properties"]["a"]["type"] = "integer"
        src_schemas.append({"name": "leaked", "input_schema": {"type": "object"}})

        stored = batch.get_schema("src_tool")
        assert stored is not None, "src_tool not found in batch"
        assert stored["name"] == "src_tool"
        assert stored["input_schema"]["properties"]["a"]["type"] == "string"
        # And the appended-after-the-fact schema must NOT be visible.
        assert batch.get_schema("leaked") is None

    def test_from_source_schemas_are_not_same_object(self):
        """batch.schemas[0] must NOT be the same object as the input element."""
        from jyagent.runtime.tools.registry import ToolBatch
        src_schema = {"name": "src_tool", "input_schema": {"type": "object"}}
        src_schemas = [src_schema]

        batch = ToolBatch.from_source(
            src_schemas, {"src_tool": lambda: "x"}, base=self._make_base(),
        )
        assert len(batch.schemas) == 1
        assert batch.schemas[0] is not src_schema, (
            "from_source did not deep-copy the schema (identity matched)"
        )
        # Sanity: equality preserved (deep-copy, not transformation).
        assert batch.schemas[0] == src_schema

    def test_from_source_functions_is_readonly(self):
        """batch.functions must raise TypeError on mutation (MappingProxyType)."""
        from jyagent.runtime.tools.registry import ToolBatch
        batch = ToolBatch.from_source(
            [{"name": "src_tool", "input_schema": {"type": "object"}}],
            {"src_tool": lambda: "x"},
            base=self._make_base(),
        )
        with pytest.raises(TypeError):
            batch.functions["another"] = lambda: None  # type: ignore[index]
        with pytest.raises(TypeError):
            batch.schema_map["another"] = {}  # type: ignore[index]

    def test_from_source_inherits_metadata_from_base(self):
        """Metadata classification (parallel_safe, mutating, timeout
        hints, large_input_keys, compaction_priority) is inherited from
        the base batch verbatim."""
        from jyagent.runtime.tools.registry import ToolBatch
        base = self._make_base()
        batch = ToolBatch.from_source(
            [{"name": "real_tool", "input_schema": {"type": "object"}}],
            {"real_tool": lambda: "ok"},
            base=base,
        )
        assert batch.is_mutating("real_tool") is True
        assert batch.get_timeout_hint("real_tool") == 42
        assert batch.version == base.version


# ─── M-2: dispatch_agent is serial ───────────────────────────────────────────


class TestM2DispatchAgentSerial:
    """``dispatch_agent`` is now serial — sub-agents are coarse-grained and
    serialising at the top level avoids cross-pool reentrancy under high
    parallel-tool fan-outs.  Background path is unaffected (it returns
    immediately after the grace period for ``background=True``).
    """

    def test_dispatch_agent_is_serial(self):
        from jyagent.runtime.tools.registry import get_registry
        batch = get_registry().freeze()
        assert batch.is_parallel_safe("dispatch_agent") is False, (
            "dispatch_agent must be serial (M-2 review 2026-04-29) — "
            "found parallel_safe=True"
        )
        # Sanity: it remains classified as mutating (the original A1 flag).
        assert batch.is_mutating("dispatch_agent") is True


# ─── M-5: with_overlay accepts mutating= ────────────────────────────────────


class TestM5OverlayMutating:
    """``with_overlay(mutating=...)`` lets callers flag a side-effecting
    overlay tool without going through the registry path.  The resulting
    classification unions into any pre-existing mutating set.
    """

    def test_overlay_mutating_flag_propagates(self):
        from jyagent.runtime.tools.registry import ToolBatch
        base = ToolBatch.empty()
        new = base.with_overlay(
            functions={"foo": lambda: "ok"},
            schemas=[{"name": "foo", "input_schema": {"type": "object"}}],
            mutating={"foo"},
        )
        assert new.is_mutating("foo") is True
        # Without the flag (default), overlaid tools are non-mutating.
        plain = base.with_overlay(
            functions={"bar": lambda: "ok"},
            schemas=[{"name": "bar", "input_schema": {"type": "object"}}],
        )
        assert plain.is_mutating("bar") is False

    def test_overlay_unions_into_existing_mutating(self):
        """Adding a new mutator must not drop pre-existing mutating
        classifications inherited from the base batch."""
        from jyagent.runtime.tools.registry import ToolRegistry
        reg = ToolRegistry()
        reg.register(
            "old_mutator",
            lambda: "ok",
            {"name": "old_mutator", "input_schema": {"type": "object"}},
            mutating=True,
        )
        base = reg.freeze()
        assert base.is_mutating("old_mutator") is True

        new = base.with_overlay(
            functions={"new_mutator": lambda: "ok"},
            schemas=[{"name": "new_mutator", "input_schema": {"type": "object"}}],
            mutating={"new_mutator"},
        )
        assert new.is_mutating("new_mutator") is True
        assert new.is_mutating("old_mutator") is True, (
            "overlay clobbered base's mutating set"
        )


# ─── L-3: LoopCheckpoint.from_json filters unknown fields ───────────────────


class TestL3CheckpointForwardCompat:
    """A checkpoint authored by a NEWER agent version (with extra fields)
    must load cleanly in an OLDER agent that doesn't know those fields.
    Missing-field tolerance comes for free via dataclass defaults.
    """

    def test_extra_fields_are_filtered(self):
        from jyagent.runtime.loop.checkpoint import LoopCheckpoint, iso_utc_now
        payload = {
            "run_id": "future-run",
            "step": 7,
            "saved_at": iso_utc_now(),
            "messages": [],
            "total_input_tokens": 100,
            "total_output_tokens": 50,
            "tool_calls_count": 3,
            "todos": [],
            "provider": "anthropic",
            "model": "claude-9.5-opus",
            "status": "in_progress",
            "error": None,
            # Hypothetical newer-version fields the older code doesn't know.
            "phase": "exploration",
            "memory_pointer": "mem-abc123",
            "experimental_flags": {"streaming_v2": True},
        }
        cp = LoopCheckpoint.from_json(json.dumps(payload))
        # Known fields populated as expected.
        assert cp.run_id == "future-run"
        assert cp.step == 7
        assert cp.tool_calls_count == 3
        # Unknown fields silently filtered (no AttributeError, no crash).
        assert not hasattr(cp, "phase")
        assert not hasattr(cp, "memory_pointer")

    def test_missing_optional_fields_use_defaults(self):
        """Loading a minimal payload (only required fields) succeeds via
        dataclass defaults."""
        from jyagent.runtime.loop.checkpoint import LoopCheckpoint, iso_utc_now
        payload = {
            "run_id": "minimal-run",
            "step": 0,
            "saved_at": iso_utc_now(),
        }
        cp = LoopCheckpoint.from_json(json.dumps(payload))
        assert cp.run_id == "minimal-run"
        assert cp.messages == []
        assert cp.todos == []
        assert cp.tool_calls_count == 0
        assert cp.provider is None

