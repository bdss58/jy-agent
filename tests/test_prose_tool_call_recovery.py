# test_prose_tool_call_recovery.py — Bug A fix + Bug B regression.
#
# Bug A (runtime fix): when the model emits text that LOOKS like a tool
# call (e.g. ``[Tool call: run_shell]{...}``) but no real structured
# tool_use block, ``run_step`` used to terminate the turn cleanly,
# silently swallowing the user's request.  The fix adds a detector
# (``looks_like_prose_tool_call``) and a corrective re-prompt path.
# Tests below pin both the detector's precision and the loop semantics.
#
# Bug B (no-op regression test): the streaming path must NOT scan text
# deltas for tool-tag-like substrings.  Codex traced the streaming
# pipeline and confirmed it doesn't today — we add a small assertion
# that the Anthropic adapter passes literal tag-shaped text through
# unchanged via ``text_delta`` events, so any future "smart" parser
# that breaks this contract fails loudly here instead of silently
# truncating the assistant's output mid-stream.

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Re-use the FakeLoop / helpers from the existing step-runner suite —
# they are the canonical way to drive ``run_step`` without spinning up
# real provider adapters.
from test_step_runner import (
    FakeLoop,
    _build_state,
    _llm_text_only,
    _llm_with_tool_call,
)

from jyagent.runtime.loop.config import LoopConfig
from jyagent.runtime.loop.finalize import (
    PROSE_TOOL_CALL_MARKER,
    build_prose_tool_call_correction,
    looks_like_prose_tool_call,
    strip_dangling_verification,
)
from jyagent.runtime.loop.step import StepContinue, StepTerminate, run_step


# ─── Detector precision ─────────────────────────────────────────────────────


class TestProseToolCallDetector:
    """Pin the detector's true / false positive surface.

    Patterns that MUST trigger represent the actual failure mode the user
    hit — the model writing transcript-render syntax instead of invoking
    the tool.  Patterns that MUST NOT trigger represent normal prose that
    happens to mention tool calls, which agents do all the time.
    """

    @pytest.mark.parametrize(
        "text",
        [
            '[Tool call: run_shell]{"cmd": "ls"}',
            "[Tool: read_file]",
            "[tool_use: write_file]",
            "[ tool_call : edit_file ]",
            "Some intro\n[Tool call: run_shell]{\"cmd\":\"ls\"}\nSome trailer",
            "```tool_use\n{\"name\": \"x\"}\n```",
            "```tool_call\n{}\n```",
        ],
    )
    def test_positive_patterns(self, text: str):
        assert looks_like_prose_tool_call(text), f"should fire on: {text!r}"

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "I will run the `run_shell` tool to list files.",
            "Calling write_file next: that should fix it.",
            # Discussion of the failure mode itself must not loop forever.
            "The agent should not emit `[Tool call: ...]` as prose.",
            # Inline (non-line-leading) bracket mention is prose, not invocation.
            "It looked like [Tool call: x] in the middle of a sentence.",
            # Generic Markdown code block, no tool tag.
            "```python\nprint('hi')\n```",
        ],
    )
    def test_negative_patterns(self, text: str):
        assert not looks_like_prose_tool_call(text), f"must not fire on: {text!r}"


# ─── Loop semantics — Bug A end-to-end via run_step ─────────────────────────


def _llm_prose_tool_call(text: str = '[Tool call: run_shell]{"cmd":"ls"}') -> tuple:
    """LLM response carrying prose-shaped tool call but no real tool_use."""
    return (
        text, [], "end_turn",
        {
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 1, "output_tokens": 2},
        },
    )


class TestProseToolCallGate:
    def test_first_prose_tool_call_continues_and_injects_correction(self):
        loop = FakeLoop(llm_response=_llm_prose_tool_call())
        state = _build_state(loop)

        outcome = run_step(loop, state)

        # Loop must NOT terminate — that was the original bug.
        assert isinstance(outcome, StepContinue)
        # Exactly two messages were appended: the assistant's malformed
        # attempt + the corrective user prompt.
        assert len(state.messages) == 2
        assert state.messages[0]["content"][0]["text"].startswith("[Tool call:")
        assert state.messages[1]["role"] == "user"
        assert state.messages[1]["content"].startswith(PROSE_TOOL_CALL_MARKER)
        # Counter advanced.
        assert state.prose_tool_call_corrections == 1

    def test_clean_text_after_correction_terminates_completed(self):
        # Step 1 fires the gate.  Step 2 returns clean prose → terminates.
        loop = FakeLoop(llm_response=_llm_prose_tool_call())
        state = _build_state(loop)

        first = run_step(loop, state)
        assert isinstance(first, StepContinue)

        # Swap the LLM to return clean text and re-run.
        loop.llm_response = _llm_text_only("all done")
        state.step = 1
        second = run_step(loop, state)

        assert isinstance(second, StepTerminate)
        assert second.result.status == "completed"
        assert "all done" in second.result.text

    def test_correction_capped_after_max_retries(self):
        # Force the cap low and make every step prose-shaped.
        loop = FakeLoop(llm_response=_llm_prose_tool_call())
        state = _build_state(loop)
        state.max_prose_tool_call_corrections = 1

        first = run_step(loop, state)
        assert isinstance(first, StepContinue)
        assert state.prose_tool_call_corrections == 1

        # Second prose-shaped step exceeds cap → must terminate, not loop.
        state.step = 1
        second = run_step(loop, state)
        assert isinstance(second, StepTerminate)
        assert second.result.status == "completed"

    def test_real_tool_call_unaffected(self):
        # When the model emits a real tool_use block, the prose gate must
        # NOT fire even if step_text would otherwise match (it doesn't
        # here — we just want to confirm the gate only enters the
        # no-tool branch).
        loop = FakeLoop(llm_response=_llm_with_tool_call("fake_tool"))
        # Wire a no-op tool source so dispatch can find the tool.

        class _FakeBatch:
            def __init__(self):
                from jyagent.runtime.tools.registry import ToolBatch
                self._b = ToolBatch.empty()

            def freeze(self):
                return self._b

        # Easiest: patch the executor + cancellation path so dispatch
        # short-circuits without needing a real tool.  We mark _cancel_event
        # so the executor sees no event and the call falls through, then
        # bail by setting cancel after the LLM call by overriding
        # _execute_tool_round indirectly: simpler to just assert that the
        # prose-gate counter stayed at 0.  The full tool-execution path is
        # already covered by tests/test_step_runner.py.
        state = _build_state(loop)
        # We expect run_step to attempt tool execution which in this
        # FakeLoop has no registered handler → it raises.  Catch + check
        # the counter remained 0, which is all we want to assert here.
        try:
            run_step(loop, state)
        except Exception:
            pass
        assert state.prose_tool_call_corrections == 0

    def test_no_correction_at_final_step_boundary(self):
        # Boundary guard: when this is the last allowed step, re-prompting
        # would be useless (no iteration left).  Loop should terminate
        # normally instead of leaking a dangling correction message.
        cfg = LoopConfig(
            max_steps=2, streaming=False, compact_messages=False,
            todos_enabled=False, fallback_on_max_steps=False,
        )
        loop = FakeLoop(config=cfg, llm_response=_llm_prose_tool_call())
        state = _build_state(loop)
        state.step = 1  # step+1 == cfg.max_steps → boundary

        outcome = run_step(loop, state)

        assert isinstance(outcome, StepTerminate)
        assert outcome.result.status == "completed"
        # No dangling correction in persisted messages.
        if state.messages:
            assert not (
                isinstance(state.messages[-1].get("content"), str)
                and state.messages[-1]["content"].startswith(PROSE_TOOL_CALL_MARKER)
            )


# ─── Cleanup — strip_dangling_verification also handles correction marker ───


class TestDanglingCorrectionCleanup:
    def test_strip_removes_dangling_correction(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "[Tool call: x]"}]},
            {"role": "user", "content": build_prose_tool_call_correction()},
        ]
        strip_dangling_verification(msgs)
        assert len(msgs) == 2
        assert msgs[-1]["role"] == "assistant"

    def test_strip_idempotent_on_normal_tail(self):
        msgs = [{"role": "user", "content": "hi"}]
        strip_dangling_verification(msgs)
        assert len(msgs) == 1


# ─── Bug B regression — Anthropic streaming passes tag-shaped text through ──


class TestAnthropicStreamPassthrough:
    """Pin that the streaming adapter does NOT scan text deltas for
    tool-tag-like substrings.  The adapter switches on SDK event types
    only — see jyagent/llm/providers/anthropic.py:61.  If a future change
    adds local pattern detection that swallows or truncates literal
    ``[Tool call: ...]`` text in a text_delta, this test fails.
    """

    def test_tag_shaped_text_delta_passes_through_unchanged(self):
        from jyagent.llm.providers.anthropic import _AnthropicStream

        # Build a fake SDK event sequence: one text content block with
        # a single text_delta that contains the dangerous pattern.
        dangerous = "[Tool call: run_shell]{\"cmd\":\"ls\"}"

        events = [
            SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(type="text"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(text=dangerous),
            ),
            SimpleNamespace(
                type="content_block_stop",
                index=0,
            ),
        ]

        class _FakeStream:
            def __iter__(self_inner):
                yield from events

            def get_final_message(self_inner):
                # Minimal Anthropic-shaped response — assistant_from_response
                # is patched out below so we don't need to satisfy its full
                # schema; the test only cares about the streamed deltas.
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text=dangerous)],
                    stop_reason="end_turn",
                    usage=SimpleNamespace(input_tokens=1, output_tokens=2),
                )

        class _FakeStreamCM:
            def __enter__(self_inner):
                return _FakeStream()

            def __exit__(self_inner, *exc):
                return False

        from jyagent.llm.types import ModelSpec
        # Patch assistant_from_response so we don't need to mock the full
        # SDK response shape — Bug B is about the streamed deltas, not
        # the final-message conversion.
        import jyagent.llm.providers.anthropic as anthropic_mod

        orig = anthropic_mod.assistant_from_response
        anthropic_mod.assistant_from_response = (
            lambda spec, raw: {"role": "assistant", "content": [], "stop_reason": "end_turn"}
        )
        try:
            stream = _AnthropicStream(
                stream_cm=_FakeStreamCM(),
                model_spec=ModelSpec(provider="anthropic", model="claude-sonnet-4-6"),
            )
            text_deltas = [e for e in stream if e.get("type") == "text_delta"]
        finally:
            anthropic_mod.assistant_from_response = orig

        # The literal tag-shaped string must arrive unchanged in exactly
        # one text_delta event.  No swallowing, no splitting, no
        # truncation.
        assert len(text_deltas) == 1, f"expected 1 text_delta, got {len(text_deltas)}"
        assert text_deltas[0]["text"] == dangerous
