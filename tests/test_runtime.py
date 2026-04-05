# Tests for provider-neutral runtime transforms and adapters.

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jyagent.runtime import RuntimeOwner, get_adapter
from jyagent.runtime.history import normalize_anthropic_tool_call_id, transform_messages_for_target
from jyagent.runtime.providers.anthropic import (
    AnthropicAdapter,
    _assistant_from_response as anthropic_assistant_from_response,
)
from jyagent.runtime.providers.openai import (
    OpenAIAdapter,
    _assistant_from_response as openai_assistant_from_response,
    _convert_messages as openai_convert_messages,
)
from jyagent.runtime.types import ModelSpec


class _AnthropicBlock:
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class _OpenAIItem:
    def __init__(self, item_type, **kwargs):
        self.type = item_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeManagedStream:
    def __init__(self, events):
        self._events = list(events)
        self.iterated = False

    def __iter__(self):
        self.iterated = True
        yield from self._events


class _FakeOpenAIManagedStream(_FakeManagedStream):
    def __init__(self, events, final_response, *, final_error=None, state=None):
        super().__init__(events)
        self._final_response = final_response
        self._final_error = final_error
        self._state = state if state is not None else SimpleNamespace()

    def get_final_response(self):
        if self._final_error is not None:
            raise self._final_error
        return self._final_response


class _FakeAnthropicManagedStream(_FakeManagedStream):
    def __init__(self, events, final_message):
        super().__init__(events)
        self._final_message = final_message

    def get_final_message(self):
        return self._final_message


class _FakeStreamManager:
    def __init__(self, stream):
        self._stream = stream
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self._stream

    def __exit__(self, exc_type, exc, tb):
        self.exited = True
        return False


class _FakeOpenAIResponsesAPI:
    def __init__(self, manager, *, retrieve_result=None, retrieve_error=None):
        self._manager = manager
        self._retrieve_result = retrieve_result
        self._retrieve_error = retrieve_error
        self.stream_calls = []
        self.create_calls = []
        self.retrieve_calls = []

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        return self._manager

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        raise AssertionError("complete() should use responses.stream()")

    def retrieve(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        if self._retrieve_error is not None:
            raise self._retrieve_error
        if self._retrieve_result is None:
            raise AssertionError("retrieve() should not be called without a configured fake response")
        return self._retrieve_result


class _FakeAnthropicMessagesAPI:
    def __init__(self, manager):
        self._manager = manager
        self.stream_calls = []
        self.create_calls = []

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        return self._manager

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        raise AssertionError("complete() should use messages.stream()")


class TestRuntimeTransforms:
    def test_readable_thinking_becomes_tagged_text_cross_provider(self):
        messages = [
            {
                "role": "assistant",
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "stop",
                "content": [
                    {"type": "thinking", "thinking": "private reasoning"},
                    {"type": "text", "text": "final answer"},
                ],
            }
        ]

        transformed = transform_messages_for_target(messages, ModelSpec("openai", "gpt-5-mini"))

        assert transformed[0]["content"][0]["type"] == "text"
        assert "<thinking>" in transformed[0]["content"][0]["text"]
        assert "private reasoning" in transformed[0]["content"][0]["text"]

    def test_opaque_reasoning_is_dropped_for_foreign_provider(self):
        messages = [
            {
                "role": "assistant",
                "provider": "openai",
                "model": "gpt-5-mini",
                "stop_reason": "stop",
                "content": [
                    {"type": "thinking", "thinking": "", "encrypted_content": "opaque", "redacted": True},
                    {"type": "text", "text": "answer"},
                ],
            }
        ]

        transformed = transform_messages_for_target(messages, ModelSpec("anthropic", "claude-sonnet-4-20250514"))

        assert transformed[0]["content"] == [{"type": "text", "text": "answer"}]

    def test_error_and_aborted_assistant_messages_are_skipped(self):
        messages = [
            {
                "role": "assistant",
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "error",
                "content": [{"type": "text", "text": "partial"}],
            },
            {"role": "user", "content": "next"},
        ]

        transformed = transform_messages_for_target(messages, ModelSpec("openai", "gpt-5-mini"))

        assert transformed == [{"role": "user", "content": "next"}]

    def test_orphaned_tool_calls_get_synthetic_tool_results(self):
        messages = [
            {
                "role": "assistant",
                "provider": "openai",
                "model": "gpt-5-mini",
                "stop_reason": "tool_use",
                "content": [{"type": "tool_call", "id": "call-1", "name": "echo", "arguments": {"text": "x"}}],
            }
        ]

        transformed = transform_messages_for_target(messages, ModelSpec("anthropic", "claude-sonnet-4-20250514"))

        assert transformed[1]["role"] == "tool_result"
        assert transformed[1]["tool_call_id"] == normalize_anthropic_tool_call_id("call-1")
        assert transformed[1]["is_error"] is True
        assert transformed[1]["content"] == "No result provided"

    def test_anthropic_tool_call_ids_are_normalized(self):
        tool_call_id = "tool|id:with*bad/chars"
        normalized = normalize_anthropic_tool_call_id(tool_call_id)

        assert "|" not in normalized
        assert ":" not in normalized
        assert "*" not in normalized
        assert "/" not in normalized
        assert len(normalized) <= 64


class TestRuntimeAdapters:
    def test_openai_tool_result_errors_get_error_prefix(self):
        messages = [
            {
                "role": "tool_result",
                "tool_call_id": "call-1",
                "tool_name": "echo",
                "content": "boom",
                "is_error": True,
            }
        ]

        converted = openai_convert_messages(ModelSpec("openai", "gpt-5-mini"), messages)

        assert converted[0]["type"] == "function_call_output"
        assert converted[0]["output"].startswith("Error: boom")

    def test_anthropic_response_normalizes_to_assistant_message_shape(self):
        response = SimpleNamespace(
            id="msg_1",
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=10, output_tokens=4),
            content=[
                _AnthropicBlock("text", text="hello"),
                _AnthropicBlock("tool_use", id="tool_1", name="echo", input={"value": "x"}),
            ],
        )

        message = anthropic_assistant_from_response(ModelSpec("anthropic", "claude-sonnet-4-20250514"), response)

        assert message["provider"] == "anthropic"
        assert message["api"] == "anthropic-messages"
        assert message["stop_reason"] == "tool_use"
        assert message["content"][0] == {"type": "text", "text": "hello"}
        assert message["content"][1]["type"] == "tool_call"
        assert message["content"][1]["arguments"] == {"value": "x"}

    def test_openai_response_normalizes_to_assistant_message_shape(self):
        response = SimpleNamespace(
            id="resp_1",
            error=None,
            incomplete_details=None,
            usage=SimpleNamespace(
                input_tokens=11,
                output_tokens=5,
                total_tokens=16,
                input_tokens_details=SimpleNamespace(cached_tokens=2),
            ),
            output=[
                _OpenAIItem(
                    "message",
                    id="msg_1",
                    phase="final_answer",
                    content=[_OpenAIItem("output_text", text="hello")],
                ),
                _OpenAIItem("function_call", call_id="call_1", name="echo", arguments=json.dumps({"value": "x"})),
            ],
        )

        message = openai_assistant_from_response(ModelSpec("openai", "gpt-5-mini"), response)

        assert message["provider"] == "openai"
        assert message["api"] == "openai-responses"
        assert message["phase"] == "final_answer"
        assert message["stop_reason"] == "tool_use"
        assert message["content"][0] == {"type": "text", "text": "hello"}
        assert message["content"][1]["type"] == "tool_call"
        assert message["content"][1]["arguments"] == {"value": "x"}

    def test_openai_complete_uses_stream_for_text_response(self, monkeypatch):
        final_response = SimpleNamespace(
            id="resp_text",
            error=None,
            incomplete_details=None,
            usage=SimpleNamespace(input_tokens=11, output_tokens=5, total_tokens=16),
            output=[
                _OpenAIItem(
                    "message",
                    id="msg_text",
                    phase="final_answer",
                    content=[_OpenAIItem("output_text", text="hello from stream")],
                ),
            ],
        )
        managed_stream = _FakeOpenAIManagedStream(
            [SimpleNamespace(type="response.output_text.delta", delta="hello ")],
            final_response,
        )
        manager = _FakeStreamManager(managed_stream)
        responses_api = _FakeOpenAIResponsesAPI(manager)
        adapter = OpenAIAdapter()
        monkeypatch.setattr(
            adapter,
            "_client",
            lambda: SimpleNamespace(responses=responses_api),
        )

        message = adapter.complete(ModelSpec("openai", "gpt-5-mini"), {"messages": []})

        assert message["content"] == [{"type": "text", "text": "hello from stream"}]
        assert message["stop_reason"] == "stop"
        assert managed_stream.iterated is True
        assert manager.entered is True
        assert manager.exited is True
        assert responses_api.create_calls == []
        assert len(responses_api.stream_calls) == 1

    def test_openai_complete_uses_stream_for_tool_call_response(self, monkeypatch):
        final_response = SimpleNamespace(
            id="resp_tool",
            error=None,
            incomplete_details=None,
            usage=SimpleNamespace(input_tokens=7, output_tokens=3, total_tokens=10),
            output=[
                _OpenAIItem(
                    "function_call",
                    call_id="call_1",
                    name="echo",
                    arguments=json.dumps({"value": "x"}),
                ),
            ],
        )
        managed_stream = _FakeOpenAIManagedStream(
            [SimpleNamespace(type="response.function_call_arguments.delta", delta="{")],
            final_response,
        )
        manager = _FakeStreamManager(managed_stream)
        responses_api = _FakeOpenAIResponsesAPI(manager)
        adapter = OpenAIAdapter()
        monkeypatch.setattr(
            adapter,
            "_client",
            lambda: SimpleNamespace(responses=responses_api),
        )

        message = adapter.complete(ModelSpec("openai", "gpt-5-mini"), {"messages": []})

        assert message["stop_reason"] == "tool_use"
        assert message["content"] == [
            {"type": "tool_call", "id": "call_1", "name": "echo", "arguments": {"value": "x"}}
        ]
        assert managed_stream.iterated is True
        assert manager.exited is True
        assert responses_api.create_calls == []

    def test_openai_complete_recovers_missing_completed_event_via_retrieve(self, monkeypatch):
        retrieved_response = SimpleNamespace(
            id="resp_recovered_text",
            error=None,
            incomplete_details=None,
            usage=SimpleNamespace(input_tokens=9, output_tokens=4, total_tokens=13),
            output=[
                _OpenAIItem(
                    "message",
                    id="msg_recovered_text",
                    phase="final_answer",
                    content=[_OpenAIItem("output_text", text="recovered text")],
                ),
            ],
        )
        missing_completed = RuntimeError("Didn't receive a `response.completed` event.")
        managed_stream = _FakeOpenAIManagedStream(
            [
                SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_recovered_text")),
                SimpleNamespace(type="response.output_text.delta", delta="recover"),
            ],
            None,
            final_error=missing_completed,
        )
        manager = _FakeStreamManager(managed_stream)
        responses_api = _FakeOpenAIResponsesAPI(manager, retrieve_result=retrieved_response)
        adapter = OpenAIAdapter()
        monkeypatch.setattr(
            adapter,
            "_client",
            lambda: SimpleNamespace(responses=responses_api),
        )

        message = adapter.complete(ModelSpec("openai", "gpt-5-mini"), {"messages": []})

        assert message["content"] == [{"type": "text", "text": "recovered text"}]
        assert message["runtime_warnings"] == [
            "Recovered OpenAI stream after missing terminal event via responses.retrieve()."
        ]
        assert responses_api.retrieve_calls[0]["response_id"] == "resp_recovered_text"
        assert manager.exited is True

    def test_openai_complete_recovers_tool_call_via_retrieve(self, monkeypatch):
        retrieved_response = SimpleNamespace(
            id="resp_recovered_tool",
            error=None,
            incomplete_details=None,
            usage=SimpleNamespace(input_tokens=7, output_tokens=3, total_tokens=10),
            output=[
                _OpenAIItem(
                    "function_call",
                    call_id="call_recovered",
                    name="echo",
                    arguments=json.dumps({"value": "retrieved"}),
                ),
            ],
        )
        missing_completed = RuntimeError("Didn't receive a `response.completed` event.")
        managed_stream = _FakeOpenAIManagedStream(
            [
                SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_recovered_tool")),
                SimpleNamespace(type="response.function_call_arguments.delta", delta="{"),
            ],
            None,
            final_error=missing_completed,
        )
        manager = _FakeStreamManager(managed_stream)
        responses_api = _FakeOpenAIResponsesAPI(manager, retrieve_result=retrieved_response)
        adapter = OpenAIAdapter()
        monkeypatch.setattr(
            adapter,
            "_client",
            lambda: SimpleNamespace(responses=responses_api),
        )

        message = adapter.complete(ModelSpec("openai", "gpt-5-mini"), {"messages": []})

        assert message["stop_reason"] == "tool_use"
        assert message["content"] == [
            {"type": "tool_call", "id": "call_recovered", "name": "echo", "arguments": {"value": "retrieved"}}
        ]
        assert message["runtime_warnings"] == [
            "Recovered OpenAI stream after missing terminal event via responses.retrieve()."
        ]
        assert responses_api.retrieve_calls[0]["response_id"] == "resp_recovered_tool"

    def test_openai_complete_recovers_from_partial_stream_snapshot_when_retrieve_fails(self, monkeypatch):
        snapshot = SimpleNamespace(
            id="resp_snapshot_tool",
            output=[
                _OpenAIItem(
                    "function_call",
                    call_id="call_snapshot",
                    name="echo",
                    arguments=json.dumps({"value": "snapshot"}),
                ),
            ],
            usage=None,
            error=None,
            incomplete_details=None,
        )
        missing_completed = RuntimeError("Didn't receive a `response.completed` event.")
        managed_stream = _FakeOpenAIManagedStream(
            [
                SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_snapshot_tool")),
                SimpleNamespace(type="response.function_call_arguments.delta", delta="{"),
            ],
            None,
            final_error=missing_completed,
            state=SimpleNamespace(_ResponseStreamState__current_snapshot=snapshot),
        )
        manager = _FakeStreamManager(managed_stream)
        responses_api = _FakeOpenAIResponsesAPI(
            manager,
            retrieve_error=RuntimeError("retrieve failed"),
        )
        adapter = OpenAIAdapter()
        monkeypatch.setattr(
            adapter,
            "_client",
            lambda: SimpleNamespace(responses=responses_api),
        )

        message = adapter.complete(ModelSpec("openai", "gpt-5-mini"), {"messages": []})

        assert message["stop_reason"] == "tool_use"
        assert message["content"] == [
            {"type": "tool_call", "id": "call_snapshot", "name": "echo", "arguments": {"value": "snapshot"}}
        ]
        assert message["usage"] == {}
        assert message["runtime_warnings"] == [
            "Recovered OpenAI stream after missing terminal event from partial stream snapshot."
        ]
        assert responses_api.retrieve_calls[0]["response_id"] == "resp_snapshot_tool"

    def test_anthropic_complete_uses_stream_for_text_response(self, monkeypatch):
        final_response = SimpleNamespace(
            id="msg_text",
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=4),
            content=[_AnthropicBlock("text", text="hello from stream")],
        )
        managed_stream = _FakeAnthropicManagedStream(
            [SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text="hello "))],
            final_response,
        )
        manager = _FakeStreamManager(managed_stream)
        messages_api = _FakeAnthropicMessagesAPI(manager)
        adapter = AnthropicAdapter()
        monkeypatch.setattr(
            adapter,
            "_client",
            lambda: SimpleNamespace(messages=messages_api),
        )

        message = adapter.complete(ModelSpec("anthropic", "claude-sonnet-4"), {"messages": []})

        assert message["content"] == [{"type": "text", "text": "hello from stream"}]
        assert message["stop_reason"] == "stop"
        assert managed_stream.iterated is True
        assert manager.entered is True
        assert manager.exited is True
        assert messages_api.create_calls == []
        assert len(messages_api.stream_calls) == 1

    def test_anthropic_complete_uses_stream_for_tool_call_response(self, monkeypatch):
        final_response = SimpleNamespace(
            id="msg_tool",
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=8, output_tokens=2),
            content=[_AnthropicBlock("tool_use", id="tool_1", name="echo", input={"value": "x"})],
        )
        managed_stream = _FakeAnthropicManagedStream(
            [
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="input_json_delta", partial_json="{"),
                )
            ],
            final_response,
        )
        manager = _FakeStreamManager(managed_stream)
        messages_api = _FakeAnthropicMessagesAPI(manager)
        adapter = AnthropicAdapter()
        monkeypatch.setattr(
            adapter,
            "_client",
            lambda: SimpleNamespace(messages=messages_api),
        )

        message = adapter.complete(ModelSpec("anthropic", "claude-sonnet-4"), {"messages": []})

        assert message["stop_reason"] == "tool_use"
        assert message["content"] == [
            {"type": "tool_call", "id": "tool_1", "name": "echo", "arguments": {"value": "x"}}
        ]
        assert managed_stream.iterated is True
        assert manager.exited is True
        assert messages_api.create_calls == []

    def test_runtime_owner_complete_text_uses_stream_backed_complete(self, monkeypatch):
        class _FakeRuntimeStream:
            def __init__(self, final_message):
                self._final_message = final_message
                self.iterated = False
                self.closed = False

            def __iter__(self):
                self.iterated = True
                yield {"type": "text_delta", "text": "partial"}

            def get_final_message(self):
                return self._final_message

            def close(self):
                self.closed = True

        final_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "silent answer"}],
            "provider": "openai",
            "model": "gpt-5-mini",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "stop_reason": "stop",
        }
        fake_stream = _FakeRuntimeStream(final_message)
        adapter = get_adapter("openai")
        monkeypatch.setattr(
            adapter,
            "stream",
            lambda model_spec, context, options=None: fake_stream,
        )

        owner = RuntimeOwner(ModelSpec("openai", "gpt-5-mini"))
        text = owner.complete_text("hello", system_prompt="system")

        assert text == "silent answer"
        assert fake_stream.iterated is True
        assert fake_stream.closed is True
