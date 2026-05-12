"""Unit tests for ``ui.loop_result_presenter.present_loop_result``.

The presenter was extracted from ``agent.run()`` during the LIGHT-CLEANUP
follow-up; before this test it was only exercised through the full run
loop.  These tests pin the per-status mapping behavior directly so a
future refactor can't silently change which fields propagate to
ConversationMemory.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from jyagent.ui.loop_result_presenter import (
    PresentedResult,
    present_loop_result,
)


@dataclass
class _FakeLoopResult:
    status: str
    text: str = ""
    final_text: str = ""
    messages: list = None
    error: str | None = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []


@dataclass
class _FakeConfig:
    max_steps: int = 50


class _FakeStreamingUI:
    def __init__(self):
        self.flushed = 0

    def flush_trailing_newline(self):
        self.flushed += 1


def _present(status: str, **kw) -> PresentedResult:
    result = _FakeLoopResult(status=status, **kw)
    return present_loop_result(result, _FakeConfig(), _FakeStreamingUI())


def test_completed_passes_through_text_and_final_text():
    p = _present("completed", text="hello", final_text="world", messages=[{"r": "x"}])
    assert p.response == "hello"
    assert p.final_text == "world"
    assert p.planner_messages == [{"r": "x"}]


def test_completed_flushes_trailing_newline():
    result = _FakeLoopResult(status="completed", text="t", final_text="f")
    ui = _FakeStreamingUI()
    present_loop_result(result, _FakeConfig(), ui)
    assert ui.flushed == 1


def test_max_steps_substitutes_placeholder_when_text_empty():
    p = _present("max_steps", text="", final_text="")
    assert "maximum reasoning steps" in p.response
    # final_text passes through (loop may have produced partial narrative)
    assert p.final_text == ""


def test_max_steps_keeps_text_when_present():
    p = _present("max_steps", text="partial answer", final_text="partial answer")
    assert p.response == "partial answer"
    assert p.final_text == "partial answer"


def test_interrupted_clears_final_text():
    p = _present("interrupted", text="partial", final_text="should be dropped")
    assert p.response == "partial"
    assert p.final_text == ""


def test_error_with_text_appends_error_marker():
    p = _present("error", text="some output", error="boom")
    assert p.response == "some output\n\n[Error: boom]"
    assert p.final_text == ""


def test_error_without_text_uses_error_message():
    p = _present("error", text="", error="boom")
    assert p.response == "Error during planning: boom"
    assert p.final_text == ""


def test_cost_limit_appends_warning_to_text():
    p = _present("cost_limit", text="answer", final_text="answer", error="budget hit")
    assert p.response == "answer\n\n⚠️ budget hit"
    assert p.final_text == "answer"


def test_cost_limit_uses_warning_when_text_empty():
    p = _present("cost_limit", text="", final_text="", error="budget hit")
    assert p.response == "\n\n⚠️ budget hit"


def test_dedup_break_appends_loop_warning():
    p = _present("dedup_break", text="ans", final_text="ans")
    assert "Loop detected" in p.response
    assert p.response.startswith("ans")
    assert p.final_text == "ans"


def test_unknown_status_falls_back_to_text_or_unknown():
    p = _present("totally_new_status", text="")
    assert p.response == "Unknown error"
    assert p.final_text == ""

    p2 = _present("totally_new_status", text="t")
    assert p2.response == "t"


def test_planner_messages_pass_through_unchanged_for_every_status():
    msgs = [{"role": "assistant", "content": "x"}]
    for status in (
        "completed", "max_steps", "interrupted", "error",
        "cost_limit", "dedup_break", "made_up_status",
    ):
        p = _present(status, messages=list(msgs), error="e")
        assert p.planner_messages == msgs, f"messages changed for status={status!r}"
