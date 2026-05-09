# tests/test_tracing_and_verification.py — Tests for tracing and verification.
#
# JSONL trace file logger
# Pre-completion verification gate

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Tracing
# ═══════════════════════════════════════════════════════════════════════════

from jyagent.runtime.loop.tracing import RunTrace, SpanEvent, get_tracer, TRACE_ENABLED


class TestSpanEvent:
    """SpanEvent data structure."""

    def test_to_dict_drops_internal_fields(self):
        """_start_ns is excluded from serialization."""
        s = SpanEvent(step=1, event_type="tool_call", tool_name="run_shell")
        d = s.to_dict()
        assert "_start_ns" not in d
        assert d["step"] == 1
        assert d["event_type"] == "tool_call"

    def test_to_dict_drops_none_values(self):
        """None optional fields are omitted from dict."""
        s = SpanEvent(step=0, event_type="llm_call")
        d = s.to_dict()
        assert "tool_name" not in d
        assert "tokens_in" not in d
        assert "error" not in d

    def test_timing(self):
        """_begin / _end computes duration_ms."""
        s = SpanEvent(step=0, event_type="llm_call")
        s._begin()
        # Small sleep to ensure nonzero
        import time; time.sleep(0.005)
        s._end()
        assert s.duration_ms > 0


class TestRunTrace:
    """RunTrace lifecycle and flush."""

    def test_start_sets_fields(self):
        t = RunTrace()
        t.start("anthropic", "claude-sonnet-4")
        assert t.provider == "anthropic"
        assert t.model == "claude-sonnet-4"
        assert t.start_time  # non-empty ISO string

    def test_add_span_accumulates(self):
        t = RunTrace()
        t.start("openai", "gpt-5")
        t.add_span(step=0, event_type="llm_call", tokens_in=100, tokens_out=50)
        t.add_span(step=0, event_type="tool_call", tool_name="read_file")
        assert len(t.spans) == 2
        assert t.spans[0].event_type == "llm_call"
        assert t.spans[1].tool_name == "read_file"

    def test_span_context_manager(self):
        """span() context manager auto-times and appends."""
        t = RunTrace()
        t.start("test", "model")
        with t.span(step=1, event_type="tool_call", tool_name="edit_file") as s:
            s.success = True
        assert len(t.spans) == 1
        assert t.spans[0].tool_name == "edit_file"
        assert t.spans[0].duration_ms >= 0

    def test_span_context_manager_captures_error(self):
        """span() records exceptions."""
        t = RunTrace()
        t.start("test", "model")
        with pytest.raises(ValueError):
            with t.span(step=0, event_type="tool_call", tool_name="bad") as s:
                raise ValueError("boom")
        assert len(t.spans) == 1
        assert t.spans[0].success is False
        assert "boom" in t.spans[0].error

    def test_tool_args_summary_truncation(self):
        """Tool args > 200 chars are truncated."""
        t = RunTrace()
        t.start("test", "model")
        big_args = {"data": "x" * 500}
        with t.span(step=0, event_type="tool_call", tool_args=big_args):
            pass
        assert len(t.spans[0].tool_args_summary) <= 200

    def test_flush_writes_jsonl(self):
        """flush() writes valid JSON to data/traces/."""
        t = RunTrace()
        t.start("anthropic", "claude-sonnet-4")
        t.add_span(step=0, event_type="llm_call", tokens_in=1000, tokens_out=200)
        t.finish(status="completed", total_steps=1, total_cost_usd=0.005)

        with tempfile.TemporaryDirectory() as tmpdir:
            traces_dir = Path(tmpdir) / "traces"
            # Patch the TRACES_DIR
            with mock.patch("jyagent.runtime.loop.tracing.TRACES_DIR", traces_dir):
                t.flush()

            files = list(traces_dir.glob("*.jsonl"))
            assert len(files) == 1

            with open(files[0]) as f:
                data = json.loads(f.read())

            assert data["trace_id"] == t.trace_id
            assert data["provider"] == "anthropic"
            assert data["status"] == "completed"
            assert len(data["spans"]) == 1
            assert data["spans"][0]["tokens_in"] == 1000

    def test_finish_sets_end_fields(self):
        t = RunTrace()
        t.start("test", "m")
        t.finish(status="error", total_steps=3, total_cost_usd=0.1)
        assert t.status == "error"
        assert t.total_steps == 3
        assert t.total_cost_usd == 0.1
        assert t.end_time  # non-empty


class TestGetTracer:
    """get_tracer() factory respects AGENT_TRACE_ENABLED."""

    def test_disabled_returns_none(self):
        """When TRACE_ENABLED is False, get_tracer() returns None."""
        with mock.patch("jyagent.runtime.loop.tracing.TRACE_ENABLED", False):
            assert get_tracer() is None

    def test_enabled_returns_run_trace(self):
        """When TRACE_ENABLED is True, get_tracer() returns a RunTrace."""
        with mock.patch("jyagent.runtime.loop.tracing.TRACE_ENABLED", True):
            t = get_tracer()
            assert isinstance(t, RunTrace)


# ═══════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════

from jyagent.runtime.loop.verification import (
    should_verify,
    build_verification_prompt,
    _VERIFICATION_MARKER,
    _has_mutation,
    _already_injected,
)
from jyagent.runtime.tools.registry import ToolBatch, ToolRegistry


# A frozen ToolBatch that flags edit_file/write_file/run_shell as mutating
# (matching the historical VERIFY_TOOL_NAMES set the gate used to consult
# directly).  Built once per test via the helper below — cheap, immutable,
# and identical to what the production registry would produce after
# ``jyagent.tools`` registers its core tools with mutating=True.
def _mutating_batch() -> ToolBatch:
    reg = ToolRegistry()
    for name in ("edit_file", "write_file", "run_shell"):
        reg.register(
            name,
            lambda **_kw: "ok",
            {"name": name, "input_schema": {"type": "object"}},
            mutating=True,
        )
    # Also register a non-mutating reader so tests that assert "no
    # mutation" against read_file have a registered tool (the gate's
    # unknown-name default is False anyway, but being explicit catches
    # accidental classification flips).
    reg.register(
        "read_file",
        lambda **_kw: "data",
        {"name": "read_file", "input_schema": {"type": "object"}},
    )
    return reg.freeze()


class TestShouldVerify:
    """should_verify() gate logic."""

    def test_disabled_returns_false(self):
        """When VERIFICATION_ENABLED is False, always returns False."""
        with mock.patch("jyagent.runtime.loop.verification.VERIFICATION_ENABLED", False):
            msgs = [{"role": "tool_result", "tool_name": "edit_file", "content": "ok"}]
            assert should_verify(msgs, tool_calls_count=1, batch=_mutating_batch()) is False

    def test_no_tool_calls_returns_false(self):
        """Zero tool calls → no verification."""
        with mock.patch("jyagent.runtime.loop.verification.VERIFICATION_ENABLED", True):
            assert should_verify([], tool_calls_count=0, batch=_mutating_batch()) is False

    def test_no_mutation_returns_false(self):
        """Tool calls but no file mutations → no verification."""
        with mock.patch("jyagent.runtime.loop.verification.VERIFICATION_ENABLED", True):
            msgs = [{"role": "tool_result", "tool_name": "read_file", "content": "data"}]
            assert should_verify(msgs, tool_calls_count=1, batch=_mutating_batch()) is False

    def test_mutation_triggers_verification(self):
        """edit_file tool result + enabled → should verify."""
        with mock.patch("jyagent.runtime.loop.verification.VERIFICATION_ENABLED", True):
            msgs = [{"role": "tool_result", "tool_name": "edit_file", "content": "ok"}]
            assert should_verify(msgs, tool_calls_count=1, batch=_mutating_batch()) is True

    def test_already_injected_returns_false(self):
        """Re-arming contract (latent-bug fix, 2026-05): the gate is no
        longer blocked by the mere PRESENCE of a prior verification
        marker.  The caller (``step.run_step``) is responsible for
        passing ``since_index = max(turn_start_idx, last_verification_idx
        + 1)``.  When that floor advances past every mutation in the
        scan window, the gate naturally returns False — no marker scan
        needed.

        This test pins the new contract: with the floor placed AFTER
        the lone tool_result, the gate must close even though
        ``VERIFICATION_ENABLED=1`` and ``tool_calls_count=1``.
        """
        with mock.patch("jyagent.runtime.loop.verification.VERIFICATION_ENABLED", True):
            msgs = [
                {"role": "tool_result", "tool_name": "edit_file", "content": "ok"},
                {"role": "user", "content": f"{_VERIFICATION_MARKER} Before you finish..."},
            ]
            # Caller's contract: bump the floor past the prior mutation
            # at index 0, mirroring what step.run_step does after an
            # injection.  The gate must close.
            assert should_verify(
                msgs, tool_calls_count=1,
                since_index=2,  # past every existing message
                batch=_mutating_batch(),
            ) is False

            # And the legacy "scan everything" call (since_index=0) NOW
            # returns True against the same fixture — pre-existing
            # mutation exceeds the floor.  This is intentional: the
            # marker-presence gate was redundant with the index floor and
            # actively wrong (it blocked re-firing after new mutation
            # rounds).  See ``should_verify``'s docstring for rationale.
            assert should_verify(
                msgs, tool_calls_count=1, batch=_mutating_batch(),
            ) is True

    def test_run_shell_counts_as_mutation(self):
        """run_shell is registered with mutating=True."""
        with mock.patch("jyagent.runtime.loop.verification.VERIFICATION_ENABLED", True):
            msgs = [{"role": "tool_result", "tool_name": "run_shell", "content": "ok"}]
            assert should_verify(msgs, tool_calls_count=1, batch=_mutating_batch()) is True

    def test_write_file_counts_as_mutation(self):
        """write_file is registered with mutating=True."""
        with mock.patch("jyagent.runtime.loop.verification.VERIFICATION_ENABLED", True):
            msgs = [{"role": "tool_result", "tool_name": "write_file", "content": "ok"}]
            assert should_verify(msgs, tool_calls_count=1, batch=_mutating_batch()) is True

    def test_dynamic_mcp_mutating_tool_triggers_verification(self):
        """A dynamic MCP-flagged-mutating tool — one not in the historical
        hard-coded VERIFY_TOOL_NAMES set — must arm the verification gate as
        soon as it's used.

        This test injects the mutating flag via ``with_overlay(mutating=...)``
        — the same code path an MCP integration uses to add a freshly
        registered tool to the per-step ToolBatch — and asserts the gate
        fires for a tool result with that name.
        """
        with mock.patch("jyagent.runtime.loop.verification.VERIFICATION_ENABLED", True):
            batch = ToolBatch.empty().with_overlay(mutating={"mcp_dyn_writer"})
            msgs = [{"role": "tool_result", "tool_name": "mcp_dyn_writer", "content": "ok"}]
            assert should_verify(msgs, tool_calls_count=1, batch=batch) is True

    def test_only_nonmutating_tools_skips_verification(self):
        """A turn that only used read-only tools (e.g. read_file) must NOT arm
        the verification gate, even when VERIFICATION_ENABLED=1 and at least
        one tool was called.
        """
        with mock.patch("jyagent.runtime.loop.verification.VERIFICATION_ENABLED", True):
            batch = _mutating_batch()
            msgs = [{"role": "tool_result", "tool_name": "read_file", "content": "data"}]
            assert should_verify(msgs, tool_calls_count=1, batch=batch) is False

    def test_since_index_excludes_prior_turn_mutations(self):
        """Replayed historical mutations must NOT re-arm verification on a
        non-mutating turn.

        ``_has_mutation`` previously scanned the entire ``messages`` list, so
        a prior turn's ``edit_file`` (still present in the persisted history)
        would trigger verification on a follow-up turn that only did read-only
        work.  The engine passes ``since_index = len(messages)`` at turn start
        so the scan is bounded to this-turn appends.
        """
        with mock.patch("jyagent.runtime.loop.verification.VERIFICATION_ENABLED", True):
            # Prior turn left a write_file in history; current turn only
            # called read_file.  Without since_index the gate would
            # incorrectly fire.
            history = [
                {"role": "tool_result", "tool_name": "write_file", "content": "ok"},
                {"role": "assistant", "content": "done"},
            ]
            turn_start = len(history)
            this_turn = [
                {"role": "tool_result", "tool_name": "read_file", "content": "data"},
            ]
            msgs = history + this_turn
            batch = _mutating_batch()
            assert should_verify(
                msgs, tool_calls_count=1, since_index=turn_start, batch=batch,
            ) is False, "since_index must exclude prior-turn mutations"

            # Sanity: without since_index the legacy behaviour DOES fire
            # (proves the test is exercising the right boundary).
            assert should_verify(msgs, tool_calls_count=1, batch=batch) is True

    def test_since_index_includes_current_turn_mutations(self):
        """When this-turn has a mutation past the boundary, verify fires."""
        with mock.patch("jyagent.runtime.loop.verification.VERIFICATION_ENABLED", True):
            history = [{"role": "user", "content": "do thing"}]
            turn_start = len(history)
            this_turn = [
                {"role": "tool_result", "tool_name": "edit_file", "content": "ok"},
            ]
            msgs = history + this_turn
            assert should_verify(
                msgs, tool_calls_count=1, since_index=turn_start, batch=_mutating_batch(),
            ) is True


class TestBuildVerificationPrompt:
    """build_verification_prompt() output."""

    def test_starts_with_marker(self):
        prompt = build_verification_prompt([])
        assert prompt.startswith(_VERIFICATION_MARKER)

    def test_contains_key_checks(self):
        prompt = build_verification_prompt([])
        assert "Syntax" in prompt or "syntax" in prompt.lower()
        assert "test" in prompt.lower()
        assert "fix them now" in prompt.lower()

    def test_reasonable_length(self):
        """Prompt should be concise (~200 words)."""
        prompt = build_verification_prompt([])
        word_count = len(prompt.split())
        assert word_count < 300  # generous upper bound


class TestHasMutation:
    """_has_mutation internal helper."""

    def test_empty_messages(self):
        assert _has_mutation([], batch=_mutating_batch()) is False

    def test_tool_result_with_mutation_tool(self):
        """Detects tool_name at message level."""
        msgs = [{"role": "tool_result", "tool_name": "edit_file", "content": "ok"}]
        assert _has_mutation(msgs, batch=_mutating_batch()) is True

    def test_tool_result_without_mutation(self):
        msgs = [{"role": "tool_result", "tool_name": "read_file", "content": "data"}]
        assert _has_mutation(msgs, batch=_mutating_batch()) is False

    def test_anthropic_format_content_blocks(self):
        """Detects tool_result inside content blocks (Anthropic format)."""
        msgs = [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_name": "write_file", "content": "ok"},
            ],
        }]
        assert _has_mutation(msgs, batch=_mutating_batch()) is True

    def test_openai_format_role_tool(self):
        """Detects role=tool with name field (OpenAI format)."""
        msgs = [{"role": "tool", "name": "edit_file", "content": "ok"}]
        assert _has_mutation(msgs, batch=_mutating_batch()) is True


class TestAlreadyInjected:
    """_already_injected internal helper."""

    def test_no_user_messages(self):
        assert _already_injected([]) is False

    def test_marker_present(self):
        msgs = [{"role": "user", "content": f"{_VERIFICATION_MARKER} checking..."}]
        assert _already_injected(msgs) is True

    def test_marker_absent(self):
        msgs = [{"role": "user", "content": "please fix the bug"}]
        assert _already_injected(msgs) is False

    def test_marker_in_content_blocks(self):
        """Detects marker in list-style content blocks."""
        msgs = [{
            "role": "user",
            "content": [{"type": "text", "text": f"{_VERIFICATION_MARKER} checking..."}],
        }]
        assert _already_injected(msgs) is True

    def test_only_checks_most_recent_user(self):
        """Only inspects the last user message."""
        msgs = [
            {"role": "user", "content": f"{_VERIFICATION_MARKER} old check"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "new request"},
        ]
        assert _already_injected(msgs) is False


# ═══════════════════════════════════════════════════════════════════════════
# Integration: VERIFY_TOOL_NAMES has been removed.  Mutating-classification
# now lives on the registry's ``mutating=True`` flag and is consulted via
# ``ToolBatch.is_mutating(...)``; see
# TestShouldVerify.test_dynamic_mcp_mutating_tool_triggers_verification for
# the regression test that pins the new behaviour.
# ═══════════════════════════════════════════════════════════════════════════
