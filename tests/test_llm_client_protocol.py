"""Tests for the ``LLMClient`` Protocol contract.

The runtime used to import the concrete ``LLMOwner`` class.  The
``LLMClient`` Protocol keeps the engine dependent only on the contract; any
object satisfying it can drive ``AgentLoop``.

These tests guard the contract:

  1. The real ``LLMOwner`` satisfies ``LLMClient`` structurally.  This
     is the **non-regression** check — if anyone changes ``LLMOwner``'s
     signature for ``complete()``/``stream()``/``model_spec``, this test
     fails BEFORE the engine breaks at runtime.

  2. A pure-fake client that does NOT import ``LLMOwner`` can drive
     ``AgentLoop`` to completion.  This is the **liveness** check — it
     proves the Protocol is the real coupling boundary, not a polite
     fiction that secretly relies on duck-typed access to LLMOwner
     internals.
"""

from __future__ import annotations

import threading

import pytest

from jyagent.runtime import LLMClient, AgentLoop, LoopConfig
from jyagent.llm.types import LLMOptions, ModelSpec


# ─── Non-regression: real LLMOwner satisfies the Protocol ────────────────────


def test_llmowner_satisfies_llmclient_protocol():
    """If anyone narrows ``LLMOwner.complete``/``.stream``/``.model_spec``,
    this test fails BEFORE the engine breaks at runtime.

    Imported lazily so the test file itself doesn't take the
    ``jyagent.llm`` dependency at module load — exercising the same
    isolation the runtime now enjoys.
    """
    from jyagent.llm import LLMOwner

    spec = ModelSpec(provider="anthropic", model="claude-sonnet-4-7")
    owner = LLMOwner(spec)
    assert isinstance(owner, LLMClient), (
        "LLMOwner no longer satisfies the LLMClient Protocol — most likely "
        "a method signature drifted (model_spec / complete / stream).  "
        "Update the Protocol in jyagent/runtime/loop/llm_client.py to match, "
        "or fix LLMOwner to honour the contract."
    )


# ─── Liveness: a pure fake can drive AgentLoop ───────────────────────────────


class _FakeOneShotClient:
    """Minimal LLMClient: returns a single no-tool assistant message and stops.

    Deliberately does NOT import or subclass anything from jyagent.llm.
    If this works, the Protocol is the real boundary.
    """

    def __init__(self, spec: ModelSpec, reply_text: str = "all done"):
        self._spec = spec
        self._reply_text = reply_text
        self.complete_calls: int = 0

    @property
    def model_spec(self) -> ModelSpec:
        return self._spec

    def complete(
        self,
        context: dict,
        *,
        options: LLMOptions,
        model_spec: ModelSpec | None = None,
    ) -> dict:
        self.complete_calls += 1
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": self._reply_text}],
            # Use the engine's normalized stop_reason ('stop'), not the
            # Anthropic wire-format value 'end_turn'.  Real adapters
            # translate end_turn -> stop at the decode boundary; the runtime
            # only ever sees the normalized value (see
            # jyagent.llm.types.StopReason).  Sticking 'end_turn' here
            # made this fake incompatible with the provider-output
            # validator — caught by Codex review commit 2026-05.
            "stop_reason": "stop",
            "usage": {"input_tokens": 5, "output_tokens": 7},
        }

    def stream(self, *args, **kwargs):  # pragma: no cover — non-streaming test
        raise NotImplementedError("this fake is non-streaming only")


def test_fake_llmclient_can_drive_agentloop():
    """A pure fake satisfying ``LLMClient`` drives ``AgentLoop`` to a
    ``completed`` result without any imports from jyagent.llm beyond
    the value types (``LLMOptions``/``ModelSpec``)."""
    spec = ModelSpec(provider="anthropic", model="claude-sonnet-4-7")
    fake = _FakeOneShotClient(spec, reply_text="hello from a fake client")

    # Structural check first — defends against the test silently passing
    # if the Protocol later grows a method the fake doesn't implement.
    assert isinstance(fake, LLMClient), (
        "_FakeOneShotClient no longer satisfies LLMClient — Protocol "
        "grew a method that this fake does not implement.  Either "
        "update the fake or revisit the Protocol's minimal-surface goal."
    )

    cfg = LoopConfig(
        max_steps=3,
        streaming=False,
        compact_messages=False,
        truncate_large_inputs=False,
    )
    loop = AgentLoop(fake, cfg)

    messages = [{"role": "user", "content": "say hi"}]
    result = loop.run(system_prompt="be brief", messages=messages)

    assert result.status == "completed", (
        f"loop did not complete on a one-shot fake reply: status={result.status!r}, "
        f"error={result.error!r}"
    )
    assert "hello from a fake client" in result.text
    assert fake.complete_calls == 1, (
        f"engine should have called complete() exactly once, got {fake.complete_calls}"
    )
    assert result.total_input_tokens == 5
    assert result.total_output_tokens == 7


def test_llmclient_protocol_minimal_surface():
    """The Protocol must stay minimal: model_spec property + complete + stream.

    If a fourth method creeps in, every alternative client (test fakes,
    third-party providers) breaks silently.  The Protocol boundary exists to
    keep the runtime → llm contract small.
    """
    import inspect
    members = {
        name for name, value in inspect.getmembers(LLMClient)
        if not name.startswith("_")
    }
    expected = {"model_spec", "complete", "stream"}
    assert members == expected, (
        f"LLMClient Protocol surface drifted.  "
        f"Expected {expected}, got {members}.  "
        f"If you intentionally added a new required method, every "
        f"non-LLMOwner client (test fakes, alternative providers) now "
        f"silently fails Protocol membership — update this assertion "
        f"intentionally."
    )
