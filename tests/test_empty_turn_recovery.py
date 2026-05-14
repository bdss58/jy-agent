# test_empty_turn_recovery.py — Bug B fix.
#
# Bug B: the Anthropic API occasionally returns a structurally-valid
# assistant turn that carries NO visible output — e.g. only an empty
# extended-thinking block:
#
#     {"role": "assistant",
#      "content": [{"type": "thinking", "thinking": "", "signature": ""}],
#      "stop_reason": "stop",
#      "usage": {"output_tokens": 63, ...}}
#
# Real-world incident (session ``fe07d3bc-e5dc-49c3-aa7e-5190c2b98b72``,
# seq=1014, 2026-05-14 19:51 → 20:01): mid-task, right after a failing
# pytest result, the model emitted this exact shape.  The no-tool branch
# of ``run_step`` accepted it as a clean terminal completion and the
# user stared at nothing for 10 minutes before typing "continue".
#
# Fix: detect the empty turn and inject a corrective ``[EMPTY_TURN]``
# user message, capped at ``max_empty_turn_corrections`` so a model
# that genuinely has nothing to say isn't pestered forever.

from __future__ import annotations

import pytest

# Re-use the FakeLoop helpers from the canonical step-runner suite.
from test_step_runner import (
    FakeLoop,
    _build_state,
    _llm_text_only,
)

from jyagent.runtime.loop.finalize import (
    EMPTY_TURN_MARKER,
    build_empty_turn_correction,
    looks_like_empty_turn,
    strip_dangling_verification,
)
from jyagent.runtime.loop.step import StepContinue, StepTerminate, run_step


# ─── Detector precision ─────────────────────────────────────────────────────


class TestEmptyTurnDetector:
    """Pin the detector's true / false positive surface.

    Failure mode: zero visible text AND zero tool calls.  Thinking
    blocks (even non-empty ones) don't count — the user can't see them.
    """

    def test_empty_thinking_only_fires(self):
        # The exact shape from session fe07d3bc seq=1014.
        msg = {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "", "signature": ""}],
            "stop_reason": "stop",
        }
        assert looks_like_empty_turn("", [], msg) is True

    def test_non_empty_thinking_still_fires_when_no_text(self):
        # Thinking is not visible output — a turn with only a populated
        # thinking block is still a "silent stop" from the user's view.
        msg = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I should think harder.", "signature": "sig"}
            ],
        }
        assert looks_like_empty_turn("", [], msg) is True

    def test_no_content_at_all_fires(self):
        msg = {"role": "assistant", "content": []}
        assert looks_like_empty_turn("", [], msg) is True

    def test_none_message_fires(self):
        # Defensive: provider returned no structured message at all.
        assert looks_like_empty_turn("", [], None) is True

    def test_whitespace_only_text_fires(self):
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "   \n  "}],
        }
        assert looks_like_empty_turn("   \n  ", [], msg) is True

    def test_visible_text_does_not_fire(self):
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Here's the result."}],
        }
        assert looks_like_empty_turn("Here's the result.", [], msg) is False

    def test_tool_call_present_does_not_fire(self):
        # Even with no text, a real tool_use block means the loop will
        # continue via the tool-dispatch path — not an empty turn.
        msg = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "x", "name": "run_shell", "input": {}}
            ],
        }
        assert looks_like_empty_turn("", [{"id": "x"}], msg) is False

    def test_thinking_plus_text_does_not_fire(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "...", "signature": "s"},
                {"type": "text", "text": "Done."},
            ],
        }
        assert looks_like_empty_turn("Done.", [], msg) is False


# ─── Correction builder ─────────────────────────────────────────────────────


class TestEmptyTurnCorrection:
    def test_correction_starts_with_marker(self):
        msg = build_empty_turn_correction()
        assert msg.startswith(EMPTY_TURN_MARKER)
        # Must mention what's expected next so the model has a clear ask.
        assert "tool" in msg.lower() or "summarize" in msg.lower()


# ─── Loop semantics — Bug B end-to-end via run_step ─────────────────────────


def _llm_empty_thinking() -> tuple:
    """LLM response replicating the exact session-fe07d3bc seq=1014 shape."""
    return (
        "",  # step_text — extract_text() over thinking-only content
        [],  # no tool_call_blocks
        "stop",
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "", "signature": ""}],
            "usage": {"input_tokens": 1, "output_tokens": 63},
        },
    )


class TestEmptyTurnGate:
    def test_first_empty_turn_continues_and_injects_correction(self):
        loop = FakeLoop(llm_response=_llm_empty_thinking())
        state = _build_state(loop)

        outcome = run_step(loop, state)

        # Loop must NOT terminate — that was the original bug.
        assert isinstance(outcome, StepContinue)
        # Two messages appended: the (empty) assistant attempt + correction.
        assert len(state.messages) == 2
        assert state.messages[0]["role"] == "assistant"
        assert state.messages[1]["role"] == "user"
        assert state.messages[1]["content"].startswith(EMPTY_TURN_MARKER)
        assert state.empty_turn_corrections == 1
        # Bug A counter must NOT have moved (different failure mode).
        assert state.prose_tool_call_corrections == 0

    def test_clean_text_after_correction_terminates_completed(self):
        # Step 1: empty turn → continue.  Step 2: clean text → terminate.
        loop = FakeLoop(llm_response=_llm_empty_thinking())
        state = _build_state(loop)

        first = run_step(loop, state)
        assert isinstance(first, StepContinue)

        loop.llm_response = _llm_text_only("here is the summary")
        state.step = 1
        second = run_step(loop, state)

        assert isinstance(second, StepTerminate)
        assert second.result.status == "completed"
        assert "here is the summary" in second.result.text

    def test_correction_capped_after_max_retries(self):
        # Force the cap low and keep returning empty turns.
        loop = FakeLoop(llm_response=_llm_empty_thinking())
        state = _build_state(loop)
        state.max_empty_turn_corrections = 1

        first = run_step(loop, state)
        assert isinstance(first, StepContinue)
        assert state.empty_turn_corrections == 1

        # Second empty turn exceeds cap → must terminate, not loop.
        state.step = 1
        second = run_step(loop, state)
        assert isinstance(second, StepTerminate)
        assert second.result.status == "completed"


# ─── Dangling-cleanup contract ──────────────────────────────────────────────


class TestStripDanglingEmptyTurn:
    """Belt-and-braces: if the run terminates before the correction is
    answered (max_steps, KeyboardInterrupt, etc.), ``finalize_run`` must
    strip the unanswered ``[EMPTY_TURN]`` user message so it does not
    leak into the persisted session and poison the next turn.
    """

    def test_strips_trailing_unanswered_empty_turn(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": []},
            {"role": "user", "content": build_empty_turn_correction()},
        ]
        strip_dangling_verification(messages)
        assert len(messages) == 2
        assert messages[-1]["role"] == "assistant"

    def test_idempotent_when_already_answered(self):
        # Empty-turn correction WAS answered → tail is assistant, nothing
        # to strip.
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": build_empty_turn_correction()},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ]
        before = list(messages)
        strip_dangling_verification(messages)
        assert messages == before


# ─── Codex-review follow-up: narrowed-detector + boundary + finalize_run ────


class TestEmptyTurnDetectorNarrowing:
    """Codex review (2026-05-14) — narrow the detector so unknown / future
    visible content blocks (e.g. ``image``, ``citation``, ``ui_card``) do
    NOT register as empty.  Without this guard, a future provider feature
    that emits real user-visible output via a non-text block type would
    be silently retried, erasing the response.
    """

    def test_visible_image_block_does_not_fire(self):
        # An ``image`` block is user-visible — the gate must NOT fire even
        # though step_text is empty and there are no tool calls.
        msg = {
            "role": "assistant",
            "content": [
                {"type": "image", "source": {"type": "base64", "data": "..."}}
            ],
        }
        assert looks_like_empty_turn("", [], msg) is False

    def test_unknown_block_type_does_not_fire(self):
        # Future-proofing: any block type outside the invisible allowlist
        # must be treated as visible by default.
        msg = {
            "role": "assistant",
            "content": [{"type": "ui_card", "payload": {"x": 1}}],
        }
        assert looks_like_empty_turn("", [], msg) is False

    def test_thinking_plus_visible_image_does_not_fire(self):
        # Mixed: invisible thinking + visible image → visible wins.
        msg = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "", "signature": ""},
                {"type": "image", "source": {"type": "base64", "data": "..."}},
            ],
        }
        assert looks_like_empty_turn("", [], msg) is False

    def test_redacted_thinking_only_still_fires(self):
        # ``redacted_thinking`` is also user-invisible (server-side
        # safety redaction) — should be treated the same as ``thinking``.
        msg = {
            "role": "assistant",
            "content": [{"type": "redacted_thinking", "data": "encrypted-blob"}],
        }
        assert looks_like_empty_turn("", [], msg) is True

    def test_non_dict_block_does_not_fire(self):
        # Defensive: a malformed non-dict block is treated as visible
        # (safe default — don't retry on garbage input).
        msg = {"role": "assistant", "content": ["not-a-dict-block"]}
        assert looks_like_empty_turn("", [], msg) is False


class TestEmptyTurnGateBoundary:
    """Codex review (2026-05-14) — pin the ``step + 1 < cfg.max_steps``
    boundary.  On the LAST allowed step the gate must NOT inject a
    corrective prompt: the follow-up reply has no iteration left to run,
    and a dangling ``[EMPTY_TURN]`` would otherwise leak into the
    persisted session.
    """

    def test_does_not_fire_on_last_allowed_step(self):
        from jyagent.runtime.loop.config import LoopConfig

        loop = FakeLoop(llm_response=_llm_empty_thinking())
        # Force a tight cap: max_steps=2 means the last allowed step
        # index is 1, i.e. step + 1 == max_steps blocks the gate.
        loop._config = LoopConfig(
            max_steps=2, streaming=False, compact_messages=False,
            todos_enabled=False, fallback_on_max_steps=False,
        )
        state = _build_state(loop)
        state.step = 1  # last allowed step

        outcome = run_step(loop, state)

        # Gate suppressed → falls through to terminal completion path.
        assert isinstance(outcome, StepTerminate)
        # No correction injected, counter untouched.
        assert state.empty_turn_corrections == 0
        # No dangling correction in the persisted tail.
        if state.messages:
            tail = state.messages[-1]
            content = tail.get("content", "") if isinstance(tail, dict) else ""
            assert not (isinstance(content, str) and content.startswith(EMPTY_TURN_MARKER))


class TestEmptyTurnFinalizeRunIntegration:
    """Codex review (2026-05-14) — the unit-level
    ``strip_dangling_verification`` test exercises the helper directly;
    this drives the REAL terminal cleanup path via ``finalize_run`` so
    we catch any future regression that decouples them (e.g. someone
    adds an exit path that constructs ``LoopResult`` directly).
    """

    def test_finalize_run_strips_dangling_empty_turn_for_all_statuses(self):
        from jyagent.runtime.loop import finalize as _finalize

        # Same status matrix the [VERIFICATION] regression test uses —
        # every terminal status must funnel through the same cleanup.
        for status in ("max_steps", "interrupted", "error", "cost_limit",
                       "dedup_break", "completed"):
            msgs = [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": []},
                {"role": "user", "content": build_empty_turn_correction()},
            ]
            _finalize.finalize_run(
                status=status,
                text="", final_text="",
                messages=msgs, steps=1,
                total_input_tokens=0, total_output_tokens=0,
                tool_calls_count=0,
                trace=None,
            )
            assert len(msgs) == 2, (
                f"finalize_run(status={status!r}) must strip dangling "
                f"[EMPTY_TURN] but messages still has {len(msgs)} entries"
            )
            assert msgs[-1]["role"] == "assistant"
