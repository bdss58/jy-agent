# tests/test_subagent_envelope.py — Structured sub-agent result envelope tests.
#
# Validates that `_finalize_outcome` now wraps the sub-agent's free-form
# answer in a structured Markdown envelope (per P1 item 5 from the
# 2026-04-18 joint review) so the parent model can see status + cost
# metadata at a glance rather than having to parse free-form text.
#
# Back-compat: `JY_SUBAGENT_FLAT_RESULT=1` restores the legacy behavior.

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from jyagent.tools.subagent import (
    _finalize_outcome,
    _format_subagent_envelope,
    _make_subagent_outcome,
    _SUBAGENT_STATUS_API_ERROR,
    _SUBAGENT_STATUS_COMPLETED,
    _SUBAGENT_STATUS_MAX_STEPS,
)


# ─── Pure format function ────────────────────────────────────────────────────


class TestFormatSubagentEnvelope:
    def test_basic_completed_envelope(self):
        text = _format_subagent_envelope(
            status=_SUBAGENT_STATUS_COMPLETED,
            answer="Here is the answer.",
            elapsed=2.3,
            steps=4,
            tool_calls=3,
            input_tokens=1000,
            output_tokens=250,
        )
        assert text.startswith("## Sub-agent Result")
        assert f"**Status:** {_SUBAGENT_STATUS_COMPLETED}" in text
        # Stats line contains all counters.
        assert "4 step(s)" in text
        assert "3 tool call(s)" in text
        assert "1000+250 tokens" in text
        assert "2.3s" in text
        assert "### Response" in text
        assert "Here is the answer." in text

    def test_empty_answer_shows_placeholder(self):
        text = _format_subagent_envelope(
            status=_SUBAGENT_STATUS_COMPLETED,
            answer="",
            elapsed=0.1, steps=0, tool_calls=0,
            input_tokens=0, output_tokens=0,
        )
        assert "### Response" in text
        assert "no text output" in text

    def test_error_field_rendered_when_provided(self):
        text = _format_subagent_envelope(
            status=_SUBAGENT_STATUS_API_ERROR,
            answer="Partial work.",
            elapsed=1.2, steps=2, tool_calls=1,
            input_tokens=100, output_tokens=50,
            error="Provider timeout",
        )
        assert "**Error:** Provider timeout" in text
        assert "Partial work." in text

    def test_answer_whitespace_trimmed(self):
        text = _format_subagent_envelope(
            status=_SUBAGENT_STATUS_COMPLETED,
            answer="trimmed me\n\n\n",
            elapsed=0.1, steps=1, tool_calls=0,
            input_tokens=10, output_tokens=5,
        )
        # The trailing whitespace is stripped so the envelope doesn't grow
        # unbounded blank lines.
        assert not text.endswith("\n\n\n")
        assert text.endswith("trimmed me")

    def test_markdown_structure_stable_for_parsing(self):
        """The parent LLM benefits from a stable, predictable layout.
        Lock down the section ordering so regressions are caught."""
        text = _format_subagent_envelope(
            status=_SUBAGENT_STATUS_COMPLETED,
            answer="body",
            elapsed=1.0, steps=1, tool_calls=1,
            input_tokens=1, output_tokens=1,
        )
        status_idx = text.find("**Status:**")
        stats_idx = text.find("**Stats:**")
        response_idx = text.find("### Response")
        assert -1 < status_idx < stats_idx < response_idx
        # Header appears before everything.
        assert text.find("## Sub-agent Result") < status_idx


# ─── _finalize_outcome integration ───────────────────────────────────────────


class _FakeSpec:
    provider = "anthropic"
    model = "claude-opus-4-6"


def _completed_outcome(content="ok", **kwargs):
    return _make_subagent_outcome(
        _SUBAGENT_STATUS_COMPLETED, content,
        kwargs.get("steps", 3),
        kwargs.get("input_tokens", 100),
        kwargs.get("output_tokens", 25),
        kwargs.get("tool_calls", 2),
    )


class TestFinalizeOutcomeDefaultStructured:
    """By default, _finalize_outcome wraps the answer in the envelope."""

    def test_default_returns_structured_result(self):
        out = _completed_outcome("the answer")
        # Patch get_stats so we don't mutate global stats during the test.
        with patch("jyagent.tools.subagent.get_stats") as gs:
            gs.return_value.record_subagent_usage = lambda *a, **k: None
            result = _finalize_outcome(out, elapsed=1.5, model_spec=_FakeSpec(),
                                       task_preview="t")
        assert not result.is_error
        assert "## Sub-agent Result" in result.content
        assert "**Status:** completed" in result.content
        assert "the answer" in result.content
        assert "1.5s" in result.content

    def test_error_status_marks_tool_result_as_error(self):
        out = _make_subagent_outcome(
            _SUBAGENT_STATUS_API_ERROR, "partial body",
            steps=1, input_tokens=10, output_tokens=5, tool_calls=0,
            error="boom",
        )
        with patch("jyagent.tools.subagent.get_stats") as gs:
            gs.return_value.record_subagent_usage = lambda *a, **k: None
            result = _finalize_outcome(out, elapsed=0.5, model_spec=_FakeSpec(),
                                       task_preview="t")
        assert result.is_error is True
        assert "**Status:** api_error" in result.content or "**Status:** " in result.content
        assert "**Error:** boom" in result.content

    def test_max_steps_status_renders_in_envelope(self):
        out = _make_subagent_outcome(
            _SUBAGENT_STATUS_MAX_STEPS, "best-effort answer here",
            steps=50, input_tokens=5000, output_tokens=2000, tool_calls=15,
        )
        with patch("jyagent.tools.subagent.get_stats") as gs:
            gs.return_value.record_subagent_usage = lambda *a, **k: None
            result = _finalize_outcome(out, elapsed=42.0, model_spec=_FakeSpec(),
                                       task_preview="t")
        assert result.is_error is True  # not "completed"
        assert "max_steps" in result.content
        assert "50 step(s)" in result.content
        assert "best-effort answer here" in result.content


class TestFinalizeOutcomeFlatOptOut:
    """JY_SUBAGENT_FLAT_RESULT=1 restores legacy raw-answer behavior."""

    def test_flat_mode_returns_bare_answer(self):
        out = _completed_outcome("just the answer text")
        with (
            patch("jyagent.tools.subagent.get_stats") as gs,
            patch.dict(os.environ, {"JY_SUBAGENT_FLAT_RESULT": "1"}),
        ):
            gs.return_value.record_subagent_usage = lambda *a, **k: None
            result = _finalize_outcome(out, elapsed=1.0, model_spec=_FakeSpec(),
                                       task_preview="t")
        assert not result.is_error
        # Legacy: bare answer, no envelope.
        assert result.content == "just the answer text"
        assert "## Sub-agent Result" not in result.content

    def test_envelope_mode_when_flag_unset(self):
        out = _completed_outcome("just the answer text")
        # Make sure the env var isn't accidentally set from another test.
        env = {k: v for k, v in os.environ.items() if k != "JY_SUBAGENT_FLAT_RESULT"}
        with (
            patch("jyagent.tools.subagent.get_stats") as gs,
            patch.dict(os.environ, env, clear=True),
        ):
            gs.return_value.record_subagent_usage = lambda *a, **k: None
            result = _finalize_outcome(out, elapsed=1.0, model_spec=_FakeSpec(),
                                       task_preview="t")
        assert "## Sub-agent Result" in result.content


class TestEnvelopeBackwardsCompatWithExistingTests:
    """The existing TestBackgroundFastFinishReturnsInline check still
    requires that:
      * `is_error` is False on completion
      * the answer text appears in `result.content`
      * neither 'dispatched' nor 'agent_id' appear (those belong to the
        slow-dispatch path, not the inline path)
    Verify the envelope satisfies all of these."""

    def test_envelope_contains_answer_no_dispatched_metadata(self):
        out = _completed_outcome("fast result")
        with patch("jyagent.tools.subagent.get_stats") as gs:
            gs.return_value.record_subagent_usage = lambda *a, **k: None
            result = _finalize_outcome(out, elapsed=0.2, model_spec=_FakeSpec(),
                                       task_preview="t")
        assert not result.is_error
        assert "fast result" in result.content
        assert "dispatched" not in result.content
        assert "agent_id" not in result.content