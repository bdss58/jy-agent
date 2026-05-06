"""End-to-end tests: LLMRunner calls the message validator at the provider boundary
when gating is enabled, and skips it otherwise.

Covers both the non-streaming (``call_complete``) and streaming
(``call_streaming`` terminal event) paths.  Uses a fake LLM client so
we can inject malformed output deterministically.
"""

import os
from typing import Iterator
from unittest.mock import MagicMock

import pytest

from jyagent.runtime.loop.config import LoopConfig
from jyagent.runtime.loop.callbacks import LoopCallbacks
from jyagent.runtime.loop.llm_runner import LLMRunner
from jyagent.runtime.loop.llm_types import ModelSpec, LLMOptions
from jyagent.llm.validation import MessageValidationError


# ─── Fakes ───────────────────────────────────────────────────────────────────


class _FakeClient:
    """Minimal LLMClient fake; complete()/stream() return whatever we set."""
    def __init__(self, model_spec: ModelSpec):
        self._model_spec = model_spec
        self._complete_return = None
        self._stream_events: list[dict] = []

    @property
    def model_spec(self) -> ModelSpec:
        return self._model_spec

    def complete(self, context, *, options, model_spec=None):
        return self._complete_return

    def stream(self, context, *, options, model_spec=None):
        # Minimal stream object satisfying LLMStream duck-type
        events = list(self._stream_events)

        class _Stream:
            def __iter__(self) -> Iterator[dict]:
                return iter(events)
            def get_final_message(self):
                for ev in reversed(events):
                    if ev.get("type") in {"done", "error"}:
                        return ev["message"]
                return {"role": "assistant", "content": []}
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, *_): pass

        return _Stream()


def _make_runner(client: _FakeClient, *, validate: bool) -> LLMRunner:
    cfg = LoopConfig(validate_provider_output=validate, streaming=False)
    return LLMRunner(
        runtime_owner=client,
        config=cfg,
        callbacks=LoopCallbacks(),
        model_spec=ModelSpec(provider="anthropic", model="claude-3-7-sonnet-20250219"),
    )


def _good_message(**overrides) -> dict:
    msg = {
        "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "stop",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    msg.update(overrides)
    return msg


# ─── call_complete ────────────────────────────────────────────────────────────


class TestCallCompleteValidationGate:
    def test_validation_disabled_passes_malformed_silently(self, monkeypatch):
        """When the gate is off, malformed output flows through to downstream
        code unchanged (this is the legacy behavior — explicit silence)."""
        # Disable both triggers so config=False genuinely means "off".
        # (The pytest invocation may set JYAGENT_VALIDATE_PROVIDER_OUTPUT=1
        # globally; this test verifies the disabled behavior in isolation.)
        monkeypatch.delenv("JYAGENT_VALIDATE_PROVIDER_OUTPUT", raising=False)
        client = _FakeClient(ModelSpec(provider="anthropic", model="x"))
        client._complete_return = {"role": "assistant"}  # missing content
        runner = _make_runner(client, validate=False)
        # Malformed: missing required 'content'. Should not raise.
        text, tool_calls, stop_reason, msg = runner.call_complete({}, LLMOptions())
        assert msg == {"role": "assistant"}

    def test_validation_enabled_raises_on_missing_content(self):
        client = _FakeClient(ModelSpec(provider="anthropic", model="x"))
        client._complete_return = {"role": "assistant"}  # missing content
        runner = _make_runner(client, validate=True)
        with pytest.raises(MessageValidationError) as exc_info:
            runner.call_complete({}, LLMOptions())
        # Path must name the provider boundary so the error points at adapter
        # drift, not generic engine state.
        assert "provider(complete)" in exc_info.value.path
        assert "content" in exc_info.value.path

    def test_validation_enabled_raises_on_bad_stop_reason(self):
        client = _FakeClient(ModelSpec(provider="anthropic", model="x"))
        client._complete_return = _good_message(stop_reason="max_tokens")  # wrong
        runner = _make_runner(client, validate=True)
        with pytest.raises(MessageValidationError) as exc_info:
            runner.call_complete({}, LLMOptions())
        assert "stop_reason" in exc_info.value.path

    def test_validation_enabled_passes_well_formed(self):
        client = _FakeClient(ModelSpec(provider="anthropic", model="x"))
        client._complete_return = _good_message()
        runner = _make_runner(client, validate=True)
        text, tool_calls, stop_reason, msg = runner.call_complete({}, LLMOptions())
        assert text == "hello"
        assert stop_reason == "stop"
        assert tool_calls == []

    def test_env_var_overrides_off_config(self, monkeypatch):
        """JYAGENT_VALIDATE_PROVIDER_OUTPUT=1 forces validation even when
        LoopConfig.validate_provider_output=False."""
        client = _FakeClient(ModelSpec(provider="anthropic", model="x"))
        client._complete_return = {"role": "assistant"}  # malformed
        runner = _make_runner(client, validate=False)
        monkeypatch.setenv("JYAGENT_VALIDATE_PROVIDER_OUTPUT", "1")
        with pytest.raises(MessageValidationError):
            runner.call_complete({}, LLMOptions())

    def test_env_var_truthy_values(self, monkeypatch):
        client = _FakeClient(ModelSpec(provider="anthropic", model="x"))
        client._complete_return = {"role": "assistant"}
        runner = _make_runner(client, validate=False)
        for val in ["1", "true", "YES", "On"]:
            monkeypatch.setenv("JYAGENT_VALIDATE_PROVIDER_OUTPUT", val)
            with pytest.raises(MessageValidationError):
                runner.call_complete({}, LLMOptions())

    def test_env_var_falsy_values(self, monkeypatch):
        client = _FakeClient(ModelSpec(provider="anthropic", model="x"))
        client._complete_return = {"role": "assistant"}  # malformed
        runner = _make_runner(client, validate=False)
        for val in ["0", "false", "no", "", "something_else"]:
            monkeypatch.setenv("JYAGENT_VALIDATE_PROVIDER_OUTPUT", val)
            # Should NOT raise — falsy env + config off.
            runner.call_complete({}, LLMOptions())


# ─── Streaming path ───────────────────────────────────────────────────────────


class TestCallStreamingValidationGate:
    def test_streaming_done_event_validates(self):
        client = _FakeClient(ModelSpec(provider="anthropic", model="x"))
        client._stream_events = [
            {"type": "text_delta", "text": "hi"},
            {"type": "done", "message": {"role": "assistant"}},  # malformed
        ]
        runner = _make_runner(client, validate=True)
        with pytest.raises(MessageValidationError) as exc_info:
            runner.call_streaming({}, LLMOptions())
        assert "stream:done" in exc_info.value.path

    def test_streaming_error_event_validates(self):
        """An error terminal event also flows through the validator.
        Error events carry an AssistantMessage with error_message + stop_reason='error';
        validator must still accept the canonical shape, but catch malformed ones."""
        client = _FakeClient(ModelSpec(provider="anthropic", model="x"))
        # Malformed error event: message missing required 'content' field.
        client._stream_events = [
            {"type": "error", "message": {"role": "assistant"}},
        ]
        runner = _make_runner(client, validate=True)
        with pytest.raises(MessageValidationError) as exc_info:
            runner.call_streaming({}, LLMOptions())
        assert "stream:error" in exc_info.value.path

    def test_streaming_good_message_passes(self):
        client = _FakeClient(ModelSpec(provider="anthropic", model="x"))
        client._stream_events = [
            {"type": "text_delta", "text": "hello"},
            {"type": "done", "message": _good_message()},
        ]
        runner = _make_runner(client, validate=True)
        text, tool_calls, stop_reason, msg = runner.call_streaming({}, LLMOptions())
        assert stop_reason == "stop"

    def test_streaming_validation_disabled_skips(self, monkeypatch):
        monkeypatch.delenv("JYAGENT_VALIDATE_PROVIDER_OUTPUT", raising=False)
        client = _FakeClient(ModelSpec(provider="anthropic", model="x"))
        client._stream_events = [
            {"type": "done", "message": {"role": "assistant"}},
        ]
        runner = _make_runner(client, validate=False)
        # Should not raise — gate is off.
        text, tool_calls, stop_reason, msg = runner.call_streaming({}, LLMOptions())
        # stop_reason defaults to "stop" when missing.
        assert stop_reason == "stop"


# ─── Gate helper unit test ───────────────────────────────────────────────────


class TestValidationGateHelper:
    def test_config_true_wins(self, monkeypatch):
        monkeypatch.delenv("JYAGENT_VALIDATE_PROVIDER_OUTPUT", raising=False)
        client = _FakeClient(ModelSpec(provider="anthropic", model="x"))
        runner = _make_runner(client, validate=True)
        assert runner._should_validate_provider_output() is True

    def test_neither_returns_false(self, monkeypatch):
        monkeypatch.delenv("JYAGENT_VALIDATE_PROVIDER_OUTPUT", raising=False)
        client = _FakeClient(ModelSpec(provider="anthropic", model="x"))
        runner = _make_runner(client, validate=False)
        assert runner._should_validate_provider_output() is False

    def test_env_only_returns_true(self, monkeypatch):
        monkeypatch.setenv("JYAGENT_VALIDATE_PROVIDER_OUTPUT", "yes")
        client = _FakeClient(ModelSpec(provider="anthropic", model="x"))
        runner = _make_runner(client, validate=False)
        assert runner._should_validate_provider_output() is True
