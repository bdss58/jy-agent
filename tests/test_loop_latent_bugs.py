"""Regression tests for three latent bugs in the agent loop runtime.

Each bug was on the review board after the C4 refactor sweep and is
fixed in the companion source change.  The tests are grouped so a
contributor can pin one fix at a time without pulling the whole file.

─── Bug #1 — Verification gate re-arming after new mutations ────────────────

The verification gate fires when the model is about to emit a final
answer and the current turn included mutating tool calls.  Previously
it was one-shot per run (``state.verification_injected: bool``): after
the first fire, a second round of mutations would never re-arm.  Now
the gate re-fires whenever NEW mutations land at indices strictly
greater than the most recently injected verification marker.

─── Bug #2 — Tool-call ID collision in denial / result maps ─────────────────

``step_tools._execute_tool_round`` used to key the denial set and
result map by ``block.id`` (provider-assigned string).  A malformed
provider emitting two tool_call blocks with the same ``id`` would
silently collapse them — one call disappeared from stuck-loop
detection, from ``/history``, and from positional subagent pairing.
Fix: key by Python object identity (``id(block)``) so two distinct
ToolCallRequest objects with duplicate ``.id`` strings cannot collide.

─── Bug #3 — Shallow copy of tool input dict ─────────────────────────────────

``_execute_tool`` used to pass ``**tool_input`` straight through (or
``dict(tool_input)`` on the cancel-injection branch only).  Tool
bodies that mutate a nested list/dict parameter corrupted the
persisted ``ToolCallRequest.input`` — the recorded call drifted from
the call the model actually issued.  Fix: ``copy.deepcopy(tool_input)``
unconditionally before the call, with ``_cancel_event`` passed as a
separate explicit kwarg.
"""

from __future__ import annotations

import copy
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from unittest import mock

import pytest

from jyagent.runtime.loop.callbacks import LoopCallbacks
from jyagent.runtime.loop.config import LoopConfig
from jyagent.runtime.loop.llm_types import ModelSpec, ToolCallRequest
from jyagent.runtime.loop.step import RunState, StepContinue, run_step
from jyagent.runtime.loop import tool_executor as te
from jyagent.runtime.tools.registry import ToolBatch, ToolRegistry
from jyagent.runtime.tools.result import ToolResult


# ─── Shared test doubles ─────────────────────────────────────────────────────


@dataclass
class _FakeOwner:
    model_spec: ModelSpec = field(
        default_factory=lambda: ModelSpec(provider="anthropic", model="claude-sonnet-4-6")
    )


class FakeLoop:
    """Minimal AgentLoop look-alike supporting _fire / _fire_with_return."""

    def __init__(self, *, callbacks: LoopCallbacks, llm_responses: list[tuple]):
        self._config = LoopConfig(
            max_steps=10, streaming=False, compact_messages=False,
            todos_enabled=False, fallback_on_max_steps=False,
            # Disable reflection/checkpoint to keep the test hermetic.
        )
        self._callbacks = callbacks
        self._runtime_owner = _FakeOwner()
        self._model_spec: ModelSpec | None = None
        self._cancel = False
        self._tool_source = None
        self._executor = None
        self._cancel_event = None
        self._partial_side_effects: list = []
        self._todos: list = []
        self._run_id: str = ""
        self._session_id: str = ""
        self._llm_responses = list(llm_responses)
        self._llm_idx = 0
        self.fired: list[tuple] = []

    def _is_cancelled(self) -> bool:
        return self._cancel

    def _fire(self, name: str, *args: Any) -> None:
        self.fired.append((name, args))
        cb = getattr(self._callbacks, name, None)
        if cb is not None:
            try:
                cb(*args)
            except Exception:
                pass

    def _fire_with_return(self, name: str, *args: Any, default: Any = None) -> Any:
        self.fired.append((name, args))
        cb = getattr(self._callbacks, name, None)
        if cb is None:
            return default
        try:
            return cb(*args)
        except Exception:
            return default

    def _call_llm_with_retry(self, context, opts, step):
        resp = self._llm_responses[self._llm_idx]
        self._llm_idx += 1
        return resp

    def _call_complete(self, context, opts):
        return self._call_llm_with_retry(context, opts, 0)

    def _call_streaming(self, context, opts):
        return self._call_llm_with_retry(context, opts, 0)

    def _write_checkpoint(self, **kwargs) -> None:
        pass


def _tool_use_resp(name: str, tool_input: dict, call_id: str = "c1"):
    """Build a provider response that requests a single tool call."""
    block = ToolCallRequest(id=call_id, name=name, input=tool_input)
    msg = {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": call_id, "name": name, "input": tool_input},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    return ("", [block], "tool_use", msg)


def _text_resp(text: str = "done"):
    """Build a terminal text-only response (no tool calls)."""
    msg = {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    return (text, [], "stop", msg)


def _build_state(loop, *, messages=None) -> RunState:
    state = RunState.prepare_for_run(
        loop, system_prompt="sys", messages=messages or [], initial_todos=None,
    )
    state.step = 0
    return state


# ═══════════════════════════════════════════════════════════════════════════
# Bug #1 — Verification gate re-arming
# ═══════════════════════════════════════════════════════════════════════════


class TestVerificationReArming:
    """The verification gate must re-fire on subsequent mutation rounds.

    Scenario: model mutates → gate fires → model does more mutations →
    model tries to return.  The OLD code locked the gate via the
    one-shot ``state.verification_injected`` flag; the second "about to
    return" moment never re-verified.  The NEW code tracks
    ``state.last_verification_idx`` and passes a scan floor to
    ``should_verify`` so each verification only sees mutations newer
    than itself.
    """

    def test_last_verification_idx_starts_none(self):
        """Fresh RunState has no verification history."""
        loop = FakeLoop(callbacks=LoopCallbacks(), llm_responses=[])
        state = _build_state(loop)
        assert state.last_verification_idx is None

    def test_verification_fires_twice_across_two_mutation_rounds(self):
        """Drive a full 2-round mutation/verification dance through run_step.

        Timeline (each line is one run_step call):
          step 0: model → run_shell (mutating)
          step 1: model → text (no tools)  → gate fires, inject verification
          step 2: model → run_shell (mutating, post-verify-response)
          step 3: model → text (no tools)  → gate MUST re-fire
        """
        # Build a mutating tool batch.
        reg = ToolRegistry()
        reg.register(
            "run_shell",
            lambda **kw: "shell-ok",
            {"name": "run_shell",
             "input_schema": {"type": "object",
                              "properties": {"command": {"type": "string"}}}},
            mutating=True,
        )
        batch = reg.freeze()

        responses = [
            _tool_use_resp("run_shell", {"command": "true"}, call_id="a1"),
            _text_resp("first answer"),                    # triggers verify #1
            _tool_use_resp("run_shell", {"command": "id"}, call_id="a2"),
            _text_resp("second answer"),                   # should trigger verify #2
        ]
        loop = FakeLoop(callbacks=LoopCallbacks(), llm_responses=responses)
        # Pin the verification-enabled env flag for this test only.
        with mock.patch("jyagent.runtime.loop.verification.VERIFICATION_ENABLED", True):
            state = _build_state(loop)

            # Point run_step at our hand-built batch so both run_shell
            # dispatches resolve to the test stub.
            with mock.patch(
                "jyagent.runtime.loop.step._prepare_step_batch",
                return_value=batch,
            ):
                # Step 0 — model calls run_shell.  Tool dispatches, state
                # continues.  tool_calls_count moves to 1.
                state.step = 0
                out = run_step(loop, state)
                assert isinstance(out, StepContinue)
                assert state.tool_calls_count == 1

                # Step 1 — model returns text.  No tool calls, mutations
                # seen this turn → gate fires.  Expect:
                #   * verification appended at tail (content starts with
                #     the marker); the LAST message is the marker user
                #     message.
                #   * state.last_verification_idx = len(messages) - 1
                prev_len = len(state.messages)
                state.step = 1
                out = run_step(loop, state)
                assert isinstance(out, StepContinue)
                assert state.last_verification_idx is not None
                assert state.last_verification_idx == len(state.messages) - 1
                tail = state.messages[-1]
                assert tail.get("role") == "user"
                assert tail.get("content", "").startswith("[VERIFICATION]")
                first_verify_idx = state.last_verification_idx

                # Step 2 — model calls run_shell again (post-verification).
                state.step = 2
                out = run_step(loop, state)
                assert isinstance(out, StepContinue)
                assert state.tool_calls_count == 2

                # Step 3 — model returns text.  NEW mutations have landed
                # since the first verification, so the gate MUST re-fire.
                state.step = 3
                out = run_step(loop, state)
                assert isinstance(out, StepContinue), (
                    f"verification did not re-arm: outcome={out!r} "
                    f"messages-tail-role={state.messages[-1].get('role')}"
                )
                # A NEW verification marker must sit at the tail.
                assert state.last_verification_idx is not None
                assert state.last_verification_idx > first_verify_idx, (
                    "last_verification_idx did not advance — gate did not re-fire"
                )
                tail = state.messages[-1]
                assert tail.get("content", "").startswith("[VERIFICATION]")

    def test_verification_does_not_fire_twice_without_new_mutations(self):
        """Belt-and-suspenders: if the gate were called twice in a row
        with no new mutations in between, it must stay closed.  Exercise
        directly via should_verify() since the engine wouldn't do this
        on its own — but a future refactor could, and the contract is
        worth pinning.
        """
        from jyagent.runtime.loop.verification import (
            should_verify,
            build_verification_prompt,
        )

        reg = ToolRegistry()
        reg.register(
            "run_shell",
            lambda **kw: "ok",
            {"name": "run_shell", "input_schema": {"type": "object"}},
            mutating=True,
        )
        batch = reg.freeze()

        messages = [
            # Simulated this-turn history: mutation → verification injected.
            {"role": "assistant", "content": [{"type": "tool_use", "id": "a1", "name": "run_shell", "input": {}}]},
            {"role": "tool_result", "tool_call_id": "a1", "tool_name": "run_shell", "content": "ok"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": build_verification_prompt([])},
        ]
        # Nothing new since the marker (index 3) → gate closed.
        with mock.patch("jyagent.runtime.loop.verification.VERIFICATION_ENABLED", True):
            assert should_verify(
                messages,
                tool_calls_count=1,
                since_index=4,  # last_verification_idx (3) + 1
                batch=batch,
            ) is False


# ═══════════════════════════════════════════════════════════════════════════
# Bug #2 — Tool-call ID collision
# ═══════════════════════════════════════════════════════════════════════════


class TestToolCallIdCollision:
    """Two ToolCallRequest objects with duplicate ``.id`` strings must
    survive Phase 2's denial / result-map bookkeeping.  Python object
    identity (``id(block)``) is the correct key — distinct blocks have
    distinct object ids by construction.
    """

    def test_duplicate_ids_produce_two_tool_result_messages(self):
        """Drive _execute_tool_round with two blocks sharing ``.id``.

        Both must execute, both must get a tool_result appended, and
        ``state.tool_calls_count`` must advance by 2.  The OLD
        ``{b.id: r for ...}`` keying collapsed them into one entry.
        """
        from jyagent.runtime.loop.step_tools import _execute_tool_round

        # Two distinct ToolCallRequest objects carrying the SAME id.
        # A legitimate adapter would never emit this; a malformed
        # proxy envelope can.
        blocks = [
            ToolCallRequest(id="dup", name="reader", input={"x": 1}),
            ToolCallRequest(id="dup", name="reader", input={"x": 2}),
        ]
        # Sanity: they really are distinct Python objects.
        assert blocks[0] is not blocks[1]
        assert id(blocks[0]) != id(blocks[1])

        reg = ToolRegistry()
        reg.register(
            "reader",
            lambda x: f"read-{x}",
            {"name": "reader",
             "input_schema": {"type": "object",
                              "properties": {"x": {"type": "integer"}}}},
        )
        batch = reg.freeze()

        loop = FakeLoop(callbacks=LoopCallbacks(), llm_responses=[])
        state = _build_state(loop)
        state.step = 0

        outcome, tuples = _execute_tool_round(loop, state, batch, blocks)
        assert outcome is None
        # BOTH must have executed, NOT collapsed.
        assert state.tool_calls_count == 2, (
            f"expected 2 tool calls counted, got {state.tool_calls_count} "
            f"— duplicate-id collapse regression"
        )
        # Both tool_result messages must be present in transcript order.
        tool_results = [m for m in state.messages if m.get("role") == "tool_result"]
        assert len(tool_results) == 2, (
            f"expected 2 tool_result messages, got {len(tool_results)}"
        )
        # And — importantly — their content must reflect the DISTINCT inputs.
        contents = [tr.get("content") for tr in tool_results]
        assert "read-1" in contents[0] and "read-2" in contents[1], (
            f"duplicate-id collapse returned wrong results: {contents!r}"
        )

    def test_duplicate_id_with_denial_does_not_leak_into_executed(self):
        """Edge case: a denied block and an executed block share ``.id``.

        The denial must apply only to the denied object-identity; the
        other block must still execute.  Under the OLD keying, the
        single ``{b.id: r}`` mapping picked whichever came last —
        ambiguous and wrong.
        """
        from jyagent.runtime.loop.step_tools import _execute_tool_round

        blocks = [
            ToolCallRequest(id="same", name="reader", input={"x": 1}),
            ToolCallRequest(id="same", name="reader", input={"x": 2}),
        ]
        denied_object = blocks[0]  # only the FIRST is denied

        executed_inputs: list[dict] = []

        def _reader(x):
            executed_inputs.append({"x": x})
            return f"read-{x}"

        reg = ToolRegistry()
        reg.register(
            "reader",
            _reader,
            {"name": "reader",
             "input_schema": {"type": "object",
                              "properties": {"x": {"type": "integer"}}}},
        )
        batch = reg.freeze()

        # Deny only the first object; identity-gate ensures the second
        # (same .id string) still runs.
        def _gate(name, args):
            # The engine re-reads block.input immediately before calling
            # the gate, so match by input value.
            return "deny" if args.get("x") == 1 else None

        callbacks = LoopCallbacks(on_tool_pre_execute=_gate)
        loop = FakeLoop(callbacks=callbacks, llm_responses=[])
        state = _build_state(loop)
        state.step = 0

        outcome, tuples = _execute_tool_round(loop, state, batch, blocks)
        assert outcome is None
        # One executed, one denied.
        assert len(executed_inputs) == 1
        assert executed_inputs[0] == {"x": 2}

        tool_results = [m for m in state.messages if m.get("role") == "tool_result"]
        assert len(tool_results) == 2
        # First result is the denial; second is the real read.
        assert "Denied" in tool_results[0]["content"]
        assert "read-2" in tool_results[1]["content"]


# ═══════════════════════════════════════════════════════════════════════════
# Bug #3 — Shallow input-dict copy
# ═══════════════════════════════════════════════════════════════════════════


class TestToolInputDeepCopy:
    """Tool bodies that mutate nested parameters must NOT corrupt the
    original ``ToolCallRequest.input`` dict.  ``_execute_tool`` now
    deep-copies before the call.
    """

    def test_nested_list_mutation_does_not_leak(self):
        """A tool body that appends to a received list must leave the
        caller's dict pristine — both top-level and the nested list.
        """
        reg = ToolRegistry()

        def _mutator(paths: list):
            # Realistic anti-pattern: tool thinks it owns its inputs
            # and mutates the list it received.
            paths.append("surprise!")
            return f"saw {len(paths)} paths"

        reg.register(
            "mutator",
            _mutator,
            {"name": "mutator",
             "input_schema": {"type": "object",
                              "properties": {"paths": {"type": "array"}}}},
        )
        batch = reg.freeze()

        # This dict is the one held by ToolCallRequest.input and
        # persisted in the transcript.  It MUST survive the call
        # unchanged.
        original_input = {"paths": ["a", "b"]}
        snapshot = copy.deepcopy(original_input)

        result = te.execute_tool("mutator", original_input, batch)
        assert not result.is_error
        # The tool saw 3 paths (deepcopy added "surprise!" to its copy).
        assert "saw 3 paths" in result.content
        # The caller's dict is unchanged — nested list NOT mutated.
        assert original_input == snapshot, (
            f"tool mutated caller's input dict: "
            f"before={snapshot!r} after={original_input!r}"
        )

    def test_nested_dict_mutation_does_not_leak(self):
        """Same guarantee for nested dicts."""
        reg = ToolRegistry()

        def _mutator(config: dict):
            config["injected"] = True
            return "configured"

        reg.register(
            "mutator",
            _mutator,
            {"name": "mutator",
             "input_schema": {"type": "object",
                              "properties": {"config": {"type": "object"}}}},
        )
        batch = reg.freeze()

        original_input = {"config": {"verbose": False}}
        snapshot = copy.deepcopy(original_input)

        result = te.execute_tool("mutator", original_input, batch)
        assert not result.is_error
        assert original_input == snapshot
        # Specifically: the "injected" key must NOT have escaped.
        assert "injected" not in original_input["config"]

    def test_deepcopy_also_applies_on_cancel_event_branch(self):
        """The cancel-event injection branch used to be the ONLY place
        with any copy at all (and it was shallow).  The new code
        deep-copies on every path, cancel-event or not.  Verify the
        cancel path preserves both invariants: (a) the nested dict is
        isolated from the caller, (b) ``_cancel_event`` never leaks
        into the caller's input.
        """
        reg = ToolRegistry()

        def _coop(config: dict, _cancel_event=None):
            config["seen_event"] = _cancel_event is not None
            return "ok"

        reg.register(
            "coop",
            _coop,
            {"name": "coop",
             "input_schema": {"type": "object",
                              "properties": {"config": {"type": "object"}}}},
        )
        batch = reg.freeze()

        original_input = {"config": {"retries": 3}}
        snapshot = copy.deepcopy(original_input)
        ev = threading.Event()

        result = te.execute_tool(
            "coop", original_input, batch, cancel_event=ev,
        )
        assert not result.is_error
        # Caller's dict is clean: no seen_event leak, no _cancel_event leak.
        assert original_input == snapshot, (
            f"cancel-event branch corrupted caller input: "
            f"before={snapshot!r} after={original_input!r}"
        )
        assert "_cancel_event" not in original_input

    def test_deepcopy_handles_empty_and_none_input(self):
        """Edge cases: None and {} must not crash the deepcopy path."""
        reg = ToolRegistry()
        reg.register(
            "noarg",
            lambda: "done",
            {"name": "noarg", "input_schema": {"type": "object"}},
        )
        batch = reg.freeze()

        r1 = te.execute_tool("noarg", {}, batch)
        r2 = te.execute_tool("noarg", None, batch)  # type: ignore[arg-type]
        assert not r1.is_error
        assert not r2.is_error
