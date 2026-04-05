# Tests for provider-neutral runtime transforms and adapters.

import json
import logging
import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jyagent.runtime.core as runtime_core
from jyagent.runtime import RuntimeOptions, RuntimeOwner, get_adapter
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


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextmanager
def _capture_logger(name, level=logging.INFO):
    target = logging.getLogger(name)
    handler = _ListHandler()
    original_level = target.level
    original_propagate = target.propagate
    target.setLevel(level)
    target.propagate = False
    target.addHandler(handler)
    try:
        yield handler.records
    finally:
        target.removeHandler(handler)
        target.setLevel(original_level)
        target.propagate = original_propagate


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
    def __init__(self, events, *, iter_error=None):
        self._events = list(events)
        self._iter_error = iter_error
        self.iterated = False

    def __iter__(self):
        self.iterated = True
        yield from self._events
        if self._iter_error is not None:
            raise self._iter_error


class _FakeOpenAIManagedStream(_FakeManagedStream):
    def __init__(self, events, final_response, *, final_error=None, state=None, iter_error=None):
        super().__init__(events, iter_error=iter_error)
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

    def test_openai_replayed_assistant_items_omit_status(self):
        messages = [
            {
                "role": "assistant",
                "id": "msg_tool_turn",
                "provider": "openai",
                "model": "gpt-5-mini",
                "stop_reason": "tool_use",
                "phase": "commentary",
                "content": [
                    {"type": "text", "text": "I will inspect the directory."},
                    {
                        "type": "thinking",
                        "id": "rs_tool_turn",
                        "thinking": "Need the file count.",
                        "summary": ["Check current directory"],
                        "encrypted_content": "opaque-reasoning",
                    },
                    {"type": "tool_call", "id": "call_dir", "name": "list_directory", "arguments": {"path": "."}},
                ],
            },
            {
                "role": "tool_result",
                "tool_call_id": "call_dir",
                "tool_name": "list_directory",
                "content": "📁 jy-agent/",
                "is_error": False,
            },
        ]

        converted = openai_convert_messages(ModelSpec("openai", "gpt-5-mini"), messages)

        assert [item["type"] for item in converted] == ["message", "reasoning", "function_call", "function_call_output"]
        assert all("status" not in item for item in converted)
        assert converted[0]["id"] == "msg_tool_turn"
        assert converted[0]["phase"] == "commentary"
        assert converted[1]["id"] == "rs_tool_turn"
        assert converted[1]["summary"] == [{"type": "summary_text", "text": "Check current directory"}]
        assert converted[1]["encrypted_content"] == "opaque-reasoning"
        assert converted[2]["call_id"] == "call_dir"
        assert converted[2]["arguments"] == json.dumps({"path": "."}, ensure_ascii=False)

    def test_openai_post_tool_replay_preserves_context_without_status(self):
        messages = [
            {"role": "user", "content": "Count the current directory entries."},
            {
                "role": "assistant",
                "provider": "openai",
                "model": "gpt-5.4",
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "thinking",
                        "id": "rs_followup",
                        "thinking": "",
                        "summary": [],
                        "encrypted_content": "opaque-followup",
                        "redacted": True,
                    },
                    {"type": "tool_call", "id": "call_followup", "name": "list_directory", "arguments": {"path": "."}},
                ],
            },
            {
                "role": "tool_result",
                "tool_call_id": "call_followup",
                "tool_name": "list_directory",
                "content": "📁 jy-agent/ (13 entries)",
                "is_error": False,
            },
        ]

        converted = openai_convert_messages(ModelSpec("openai", "gpt-5.4"), messages)

        assert [item["type"] for item in converted] == ["message", "reasoning", "function_call", "function_call_output"]
        assert all("status" not in item for item in converted)
        assert converted[1]["id"] == "rs_followup"
        assert converted[1]["encrypted_content"] == "opaque-followup"
        assert converted[2]["call_id"] == "call_followup"
        assert converted[2]["name"] == "list_directory"
        assert converted[2]["arguments"] == json.dumps({"path": "."}, ensure_ascii=False)
        assert converted[3] == {
            "type": "function_call_output",
            "call_id": "call_followup",
            "output": "📁 jy-agent/ (13 entries)",
        }

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

    def test_openai_request_kwargs_accept_reasoning_effort(self):
        adapter = OpenAIAdapter()

        kwargs = adapter._request_kwargs(
            ModelSpec("openai", "gpt-5-mini"),
            {"messages": []},
            RuntimeOptions(reasoning={"effort": "high"}),
        )

        assert kwargs["reasoning"] == {"effort": "high"}

    def test_openai_request_kwargs_accept_reasoning_summary(self):
        adapter = OpenAIAdapter()

        kwargs = adapter._request_kwargs(
            ModelSpec("openai", "gpt-5-mini"),
            {"messages": []},
            RuntimeOptions(reasoning={"summary": "concise"}),
        )

        assert kwargs["reasoning"] == {"summary": "concise"}

    def test_openai_request_kwargs_accept_combined_reasoning_config(self):
        adapter = OpenAIAdapter()

        kwargs = adapter._request_kwargs(
            ModelSpec("openai", "gpt-5-mini"),
            {"messages": []},
            RuntimeOptions(reasoning={"effort": "medium", "summary": "detailed"}),
        )

        assert kwargs["reasoning"] == {"effort": "medium", "summary": "detailed"}

    def test_openai_request_kwargs_reject_anthropic_reasoning_shape(self):
        adapter = OpenAIAdapter()

        with pytest.raises(ValueError, match="unsupported keys"):
            adapter._request_kwargs(
                ModelSpec("openai", "gpt-5-mini"),
                {"messages": []},
                RuntimeOptions(reasoning={"type": "adaptive"}),
            )

    def test_openai_request_kwargs_reject_deprecated_generate_summary(self):
        adapter = OpenAIAdapter()

        with pytest.raises(ValueError, match="generate_summary"):
            adapter._request_kwargs(
                ModelSpec("openai", "gpt-5-mini"),
                {"messages": []},
                RuntimeOptions(reasoning={"generate_summary": "concise"}),  # type: ignore[arg-type]
            )

    def test_openai_request_kwargs_reject_unknown_reasoning_keys(self):
        adapter = OpenAIAdapter()

        with pytest.raises(ValueError, match="unsupported keys"):
            adapter._request_kwargs(
                ModelSpec("openai", "gpt-5-mini"),
                {"messages": []},
                RuntimeOptions(reasoning={"effort": "low", "foo": "bar"}),  # type: ignore[arg-type]
            )

    def test_anthropic_request_kwargs_accept_disabled_thinking(self):
        adapter = AnthropicAdapter()

        kwargs = adapter._request_kwargs(
            ModelSpec("anthropic", "claude-sonnet-4"),
            {"messages": []},
            RuntimeOptions(max_output_tokens=2048, reasoning={"type": "disabled"}),
        )

        assert kwargs["thinking"] == {"type": "disabled"}

    def test_anthropic_request_kwargs_accept_adaptive_thinking(self):
        adapter = AnthropicAdapter()

        kwargs = adapter._request_kwargs(
            ModelSpec("anthropic", "claude-sonnet-4"),
            {"messages": []},
            RuntimeOptions(max_output_tokens=2048, reasoning={"type": "adaptive"}),
        )

        assert kwargs["thinking"] == {"type": "adaptive"}

    def test_anthropic_request_kwargs_accept_adaptive_thinking_with_display(self):
        adapter = AnthropicAdapter()

        kwargs = adapter._request_kwargs(
            ModelSpec("anthropic", "claude-sonnet-4"),
            {"messages": []},
            RuntimeOptions(
                max_output_tokens=2048,
                reasoning={"type": "adaptive", "display": "omitted"},
            ),
        )

        assert kwargs["thinking"] == {"type": "adaptive", "display": "omitted"}

    def test_anthropic_request_kwargs_accept_enabled_thinking(self):
        adapter = AnthropicAdapter()

        kwargs = adapter._request_kwargs(
            ModelSpec("anthropic", "claude-sonnet-4"),
            {"messages": []},
            RuntimeOptions(
                max_output_tokens=4096,
                reasoning={"type": "enabled", "budget_tokens": 1024, "display": "summarized"},
            ),
        )

        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 1024, "display": "summarized"}

    def test_anthropic_request_kwargs_reject_openai_reasoning_shape(self):
        adapter = AnthropicAdapter()

        with pytest.raises(ValueError, match="unsupported keys"):
            adapter._request_kwargs(
                ModelSpec("anthropic", "claude-sonnet-4"),
                {"messages": []},
                RuntimeOptions(max_output_tokens=2048, reasoning={"effort": "high"}),
            )

    def test_anthropic_request_kwargs_reject_enabled_thinking_without_budget(self):
        adapter = AnthropicAdapter()

        with pytest.raises(ValueError, match="requires 'budget_tokens'"):
            adapter._request_kwargs(
                ModelSpec("anthropic", "claude-sonnet-4"),
                {"messages": []},
                RuntimeOptions(
                    max_output_tokens=2048,
                    reasoning={"type": "enabled"},  # type: ignore[arg-type]
                ),
            )

    def test_anthropic_request_kwargs_reject_enabled_thinking_below_min_budget(self):
        adapter = AnthropicAdapter()

        with pytest.raises(ValueError, match="must be >= 1024"):
            adapter._request_kwargs(
                ModelSpec("anthropic", "claude-sonnet-4"),
                {"messages": []},
                RuntimeOptions(
                    max_output_tokens=2048,
                    reasoning={"type": "enabled", "budget_tokens": 512},
                ),
            )

    def test_anthropic_request_kwargs_reject_enabled_thinking_budget_at_or_above_max_output_tokens(self):
        adapter = AnthropicAdapter()

        with pytest.raises(ValueError, match="less than max_output_tokens"):
            adapter._request_kwargs(
                ModelSpec("anthropic", "claude-sonnet-4"),
                {"messages": []},
                RuntimeOptions(
                    max_output_tokens=1024,
                    reasoning={"type": "enabled", "budget_tokens": 1024},
                ),
            )

    def test_anthropic_request_kwargs_reject_enabled_thinking_without_max_output_tokens(self):
        adapter = AnthropicAdapter()

        with pytest.raises(ValueError, match="requires RuntimeOptions.max_output_tokens"):
            adapter._request_kwargs(
                ModelSpec("anthropic", "claude-sonnet-4"),
                {"messages": []},
                RuntimeOptions(reasoning={"type": "enabled", "budget_tokens": 1024}),
            )

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

    def test_openai_complete_recovers_malformed_sse_json_via_retrieve(self, monkeypatch):
        retrieved_response = SimpleNamespace(
            id="resp_recovered_json",
            error=None,
            incomplete_details=None,
            usage=SimpleNamespace(input_tokens=8, output_tokens=3, total_tokens=11),
            output=[
                _OpenAIItem(
                    "message",
                    id="msg_recovered_json",
                    phase="final_answer",
                    content=[_OpenAIItem("output_text", text="recovered after json error")],
                ),
            ],
        )
        json_error = json.JSONDecodeError(
            "Expecting ',' delimiter",
            '{"type":"response.created","response":{"id":"resp_recovered_json""}}',
            62,
        )
        managed_stream = _FakeOpenAIManagedStream(
            [SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_recovered_json"))],
            None,
            iter_error=json_error,
        )
        manager = _FakeStreamManager(managed_stream)
        responses_api = _FakeOpenAIResponsesAPI(manager, retrieve_result=retrieved_response)
        adapter = OpenAIAdapter()
        monkeypatch.setattr(
            adapter,
            "_client",
            lambda: SimpleNamespace(responses=responses_api),
        )

        with _capture_logger("jyagent.runtime.providers.openai") as records:
            message = adapter.complete(ModelSpec("openai", "gpt-5-mini"), {"messages": []})

        assert message["content"] == [{"type": "text", "text": "recovered after json error"}]
        assert message["runtime_warnings"] == [
            "Recovered OpenAI stream after malformed SSE JSON via responses.retrieve()."
        ]
        assert responses_api.retrieve_calls[0]["response_id"] == "resp_recovered_json"
        assert [record.event for record in records] == ["llm.request.started", "llm.request.succeeded"]
        assert records[1].payload["runtime_warnings"] == message["runtime_warnings"]

    def test_openai_success_logging_stays_summary_only(self, monkeypatch):
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

        with _capture_logger("jyagent.runtime.providers.openai") as records:
            adapter.complete(
                ModelSpec("openai", "gpt-5-mini"),
                {"messages": [{"role": "user", "content": "secret prompt"}]},
                RuntimeOptions(metadata={"component": "planner", "mode": "stream", "step": 1}),
            )

        assert [record.event for record in records] == ["llm.request.started", "llm.request.succeeded"]
        started_payload = records[0].payload
        success_payload = records[1].payload
        assert started_payload["message_count"] == 1
        assert started_payload["metadata"] == {"component": "planner", "mode": "stream", "step": 1}
        assert "request" not in started_payload
        assert success_payload["output_text_chars"] == len("hello from stream")
        assert success_payload["tool_call_names"] == []
        serialized = json.dumps([record.payload for record in records], ensure_ascii=False)
        assert "secret prompt" not in serialized

    def test_openai_failure_logging_redacts_and_truncates_payloads(self, monkeypatch):
        secret = "sk-test-secret-123456789"
        long_prompt = f"prefix {secret} " + ("x" * 200)

        class _FailingResponsesAPI:
            def stream(self, **kwargs):
                raise RuntimeError(f"backend rejected {secret}")

        adapter = OpenAIAdapter()
        monkeypatch.setenv("OPENAI_API_KEY", secret)
        monkeypatch.setenv("AGENT_LOG_MAX_TEXT_CHARS", "80")
        monkeypatch.setattr(
            adapter,
            "_client",
            lambda: SimpleNamespace(responses=_FailingResponsesAPI()),
        )

        with _capture_logger("jyagent.runtime.providers.openai") as records:
            with pytest.raises(RuntimeError, match="backend rejected"):
                adapter.complete(
                    ModelSpec("openai", "gpt-5-mini"),
                    {"messages": [{"role": "user", "content": long_prompt}]},
                )

        assert [record.event for record in records] == ["llm.request.started", "llm.request.failed"]
        failure_payload = records[1].payload
        serialized = json.dumps(failure_payload, ensure_ascii=False)
        assert secret not in serialized
        assert "[REDACTED]" in serialized
        assert "[truncated " in serialized

    def test_openai_recovery_logs_success_without_failure(self, monkeypatch):
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

        with _capture_logger("jyagent.runtime.providers.openai") as records:
            message = adapter.complete(ModelSpec("openai", "gpt-5-mini"), {"messages": []})

        assert message["runtime_warnings"] == [
            "Recovered OpenAI stream after missing terminal event via responses.retrieve()."
        ]
        assert [record.event for record in records] == ["llm.request.started", "llm.request.succeeded"]
        assert records[1].payload["runtime_warnings"] == message["runtime_warnings"]

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

    def test_openai_complete_recovers_malformed_sse_json_from_partial_stream_snapshot(self, monkeypatch):
        snapshot = SimpleNamespace(
            id="resp_snapshot_json",
            output=[
                _OpenAIItem(
                    "function_call",
                    call_id="call_snapshot_json",
                    name="echo",
                    arguments=json.dumps({"value": "snapshot json"}),
                ),
            ],
            usage=None,
            error=None,
            incomplete_details=None,
        )
        json_error = json.JSONDecodeError(
            "Expecting ',' delimiter",
            '{"type":"response.created","response":{"id":"resp_snapshot_json""}}',
            61,
        )
        managed_stream = _FakeOpenAIManagedStream(
            [SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_snapshot_json"))],
            None,
            iter_error=json_error,
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
            {"type": "tool_call", "id": "call_snapshot_json", "name": "echo", "arguments": {"value": "snapshot json"}}
        ]
        assert message["runtime_warnings"] == [
            "Recovered OpenAI stream after malformed SSE JSON from partial stream snapshot."
        ]
        assert responses_api.retrieve_calls[0]["response_id"] == "resp_snapshot_json"

    def test_openai_unrecoverable_malformed_sse_json_logs_failure_with_sanitized_snippet(self, monkeypatch):
        secret = "sk-test-secret-123456789"
        bad_doc = (
            '{"type":"response.created","secret":"'
            + secret
            + '" "payload":"'
            + ("x" * 160)
            + '"}'
        )
        error_pos = bad_doc.index(secret) + (len(secret) // 2)
        json_error = json.JSONDecodeError("Expecting ',' delimiter", bad_doc, error_pos)
        managed_stream = _FakeOpenAIManagedStream(
            [SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_bad_json"))],
            None,
            iter_error=json_error,
        )
        manager = _FakeStreamManager(managed_stream)
        responses_api = _FakeOpenAIResponsesAPI(
            manager,
            retrieve_error=RuntimeError("retrieve failed"),
        )
        adapter = OpenAIAdapter()
        monkeypatch.setenv("OPENAI_API_KEY", secret)
        monkeypatch.setenv("AGENT_LOG_MAX_TEXT_CHARS", "60")
        monkeypatch.setattr(
            adapter,
            "_client",
            lambda: SimpleNamespace(responses=responses_api),
        )

        with _capture_logger("jyagent.runtime.providers.openai") as records:
            with pytest.raises(json.JSONDecodeError):
                adapter.complete(ModelSpec("openai", "gpt-5-mini"), {"messages": []})

        assert [record.event for record in records] == ["llm.request.started", "llm.request.failed"]
        failure_payload = records[1].payload
        assert failure_payload["stage"] == "stream_iter"
        assert failure_payload["response_id"] == "resp_bad_json"
        assert failure_payload["json_error_position"] == error_pos
        assert failure_payload["json_error_snippet"]
        serialized = json.dumps(failure_payload, ensure_ascii=False)
        assert secret not in serialized
        assert "[REDACTED]" in serialized
        assert "[truncated " in failure_payload["json_error_snippet"]

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

    def test_anthropic_success_logging_stays_summary_only(self, monkeypatch):
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

        with _capture_logger("jyagent.runtime.providers.anthropic") as records:
            adapter.complete(
                ModelSpec("anthropic", "claude-sonnet-4"),
                {"messages": [{"role": "user", "content": "private prompt"}]},
                RuntimeOptions(metadata={"component": "subagent", "mode": "loop_complete", "step": 2}),
            )

        assert [record.event for record in records] == ["llm.request.started", "llm.request.succeeded"]
        assert records[0].payload["message_count"] == 1
        assert records[0].payload["metadata"] == {"component": "subagent", "mode": "loop_complete", "step": 2}
        serialized = json.dumps([record.payload for record in records], ensure_ascii=False)
        assert "private prompt" not in serialized

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
        captured = {}
        monkeypatch.setattr(
            runtime_core,
            "get_reasoning_config_for_provider",
            lambda provider, *, max_output_tokens=None: {"effort": "high"},
        )
        monkeypatch.setattr(
            adapter,
            "stream",
            lambda model_spec, context, options=None: (captured.__setitem__("options", options), fake_stream)[1],
        )

        owner = RuntimeOwner(ModelSpec("openai", "gpt-5-mini"))
        text = owner.complete_text("hello", system_prompt="system")

        assert text == "silent answer"
        assert captured["options"].reasoning == {"effort": "high"}
        assert fake_stream.iterated is True
        assert fake_stream.closed is True
