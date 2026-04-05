# Tests for planner utilities (ToolResult, truncation, error detection)

import json
import os
import sys
import logging
from contextlib import contextmanager
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jyagent.planner as planner
from jyagent.planner import (
    ToolResult, _is_error_result, _result_content,
    _truncate_tool_result, _compact_working_messages, _execute_tool, plan_next_action,
)
from jyagent.registry import get_registry
from jyagent.runtime import RuntimeOwner, get_adapter
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


class _OpenAIItem:
    def __init__(self, item_type, **kwargs):
        self.type = item_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeOpenAIManagedStream:
    def __init__(self, events, final_response, *, final_error=None, state=None, iter_error=None):
        self._events = list(events)
        self._final_response = final_response
        self._final_error = final_error
        self._state = state if state is not None else SimpleNamespace()
        self._iter_error = iter_error

    def __iter__(self):
        yield from self._events
        if self._iter_error is not None:
            raise self._iter_error

    def get_final_response(self):
        if self._final_error is not None:
            raise self._final_error
        return self._final_response


class _FakeStreamManager:
    def __init__(self, stream):
        self._stream = stream

    def __enter__(self):
        return self._stream

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeOpenAIResponsesAPI:
    def __init__(self, managers, *, retrieve_error=None):
        self._managers = list(managers)
        self._retrieve_error = retrieve_error
        self.stream_calls = []
        self.retrieve_calls = []

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        if not self._managers:
            raise AssertionError("No fake stream managers left")
        return self._managers.pop(0)

    def retrieve(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        if self._retrieve_error is not None:
            raise self._retrieve_error
        raise AssertionError("retrieve() should not succeed in this test")


class TestToolResult:
    def test_success(self):
        r = ToolResult("OK")
        assert r.content == "OK"
        assert r.is_error is False
        assert str(r) == "OK"

    def test_error(self):
        r = ToolResult("Something failed", is_error=True)
        assert r.is_error is True
        assert "failed" in str(r)

    def test_repr(self):
        r = ToolResult("test content")
        assert "ToolResult" in repr(r)


class TestErrorDetection:
    def test_tool_result_error(self):
        assert _is_error_result(ToolResult("fail", is_error=True)) is True

    def test_tool_result_success(self):
        assert _is_error_result(ToolResult("ok")) is False

    def test_string_error(self):
        assert _is_error_result("Error: file not found") is True

    def test_string_success(self):
        assert _is_error_result("Success") is False

    def test_error_calling_tool(self):
        assert _is_error_result("Error calling tool X") is True


class TestResultContent:
    def test_tool_result(self):
        r = ToolResult("content here")
        assert _result_content(r) == "content here"

    def test_string(self):
        assert _result_content("plain string") == "plain string"


class TestTruncation:
    def test_short_not_truncated(self):
        result = _truncate_tool_result("short", max_chars=100)
        assert result == "short"

    def test_long_truncated(self):
        long_text = "x" * 10000
        result = _truncate_tool_result(long_text, max_chars=1000)
        assert len(result) <= 1100  # some slack for the truncation message
        assert "truncated" in result.lower()

    def test_error_never_truncated(self):
        long_error = "Error: " + "x" * 10000
        result = _truncate_tool_result(long_error, is_error=True, max_chars=100)
        assert len(result) == len(long_error)  # preserved fully


class TestCompactWorkingMessages:
    def test_under_threshold(self):
        messages = [
            {"role": "user", "content": "short message"},
            {"role": "assistant", "content": "short reply"},
        ]
        compacted = _compact_working_messages(messages, max_tokens=100000)
        assert len(compacted) == 2  # unchanged

    def test_over_threshold_truncates_tool_results(self):
        """Compact should truncate old tool_result blocks, not drop messages."""
        long_result = "x" * 50000
        messages = [
            {"role": "user", "content": [{"type": "tool_result", "content": long_result}]},
            {"role": "assistant", "content": "processing..."},
            {"role": "user", "content": "what happened?"},
            {"role": "assistant", "content": "done"},
        ]
        
        compacted = _compact_working_messages(messages, max_tokens=100)
        # Should still have same number of messages (doesn't drop)
        assert len(compacted) == 4
        # But the old tool result should be truncated
        first_content = compacted[0]["content"]
        tool_result = first_content[0]["content"]
        assert len(tool_result) < len(long_result)


class TestToolInputValidation:
    def test_rejects_non_object_input(self):
        reg = get_registry()

        def _tool(path: str) -> str:
            return path

        schema = {
            "name": "_test_non_object",
            "description": "test",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        }
        reg.register("_test_non_object", _tool, schema)
        try:
            result = _execute_tool("_test_non_object", "not-a-dict", {"_test_non_object": _tool})
            assert result.is_error is True
            assert "expected object input" in result.content
        finally:
            reg.unregister("_test_non_object")

    def test_rejects_invalid_type(self):
        reg = get_registry()

        def _tool(timeout: int) -> str:
            return f"timeout={timeout}"

        schema = {
            "name": "_test_invalid_type",
            "description": "test",
            "input_schema": {
                "type": "object",
                "properties": {
                    "timeout": {"type": "integer"},
                },
                "required": ["timeout"],
            },
        }
        reg.register("_test_invalid_type", _tool, schema)
        try:
            result = _execute_tool("_test_invalid_type", {"timeout": "fast"}, {"_test_invalid_type": _tool})
            assert result.is_error is True
            assert "input.timeout must be integer" in result.content
        finally:
            reg.unregister("_test_invalid_type")

    def test_rejects_enum_violation(self):
        reg = get_registry()

        def _tool(action: str) -> str:
            return action

        schema = {
            "name": "_test_enum",
            "description": "test",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["connect", "status"]},
                },
                "required": ["action"],
            },
        }
        reg.register("_test_enum", _tool, schema)
        try:
            result = _execute_tool("_test_enum", {"action": "launch"}, {"_test_enum": _tool})
            assert result.is_error is True
            assert "must be one of" in result.content
        finally:
            reg.unregister("_test_enum")

    def test_rejects_numeric_constraint_violation(self):
        reg = get_registry()

        def _tool(limit: int) -> str:
            return str(limit)

        schema = {
            "name": "_test_maximum",
            "description": "test",
            "input_schema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "maximum": 10},
                },
                "required": ["limit"],
            },
        }
        reg.register("_test_maximum", _tool, schema)
        try:
            result = _execute_tool("_test_maximum", {"limit": 11}, {"_test_maximum": _tool})
            assert result.is_error is True
            assert "input.limit must be <= 10" in result.content
        finally:
            reg.unregister("_test_maximum")


class TestPlannerFallback:
    def test_stream_response_uses_configured_reasoning(self, monkeypatch):
        class _DummySpinner:
            def start(self):
                pass

            def stop(self):
                pass

        class _DummyStats:
            def record_usage(self, usage, provider="", model=""):
                self.recorded = (usage, provider, model)

        class _FakeRuntimeStream:
            def __init__(self, final_message):
                self._final_message = final_message
                self.closed = False

            def __iter__(self):
                yield {"type": "text_delta", "text": "answer"}

            def get_final_message(self):
                return self._final_message

            def close(self):
                self.closed = True

        captured = {}
        final_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "answer"}],
            "provider": "openai",
            "model": "gpt-5-mini",
            "usage": {"input_tokens": 1, "output_tokens": 2},
            "stop_reason": "stop",
        }
        fake_stream = _FakeRuntimeStream(final_message)

        class _FakeOwner:
            model_spec = ModelSpec("openai", "gpt-5-mini")

            def stream(self, context, options=None):
                captured["options"] = options
                return fake_stream

        monkeypatch.setattr(planner, "_ThinkingSpinner", _DummySpinner)
        monkeypatch.setattr(planner, "_stream_write", lambda _text: None)
        monkeypatch.setattr(planner, "get_stats", lambda: _DummyStats())
        monkeypatch.setattr(
            planner,
            "get_reasoning_config_for_provider",
            lambda provider, *, max_output_tokens=None, model=None: {"effort": "high", "summary": "concise"},
        )

        result = planner._stream_response(_FakeOwner(), {"messages": []}, 2048)

        assert captured["options"].reasoning == {"effort": "high", "summary": "concise"}
        assert result[0] == "answer"
        assert fake_stream.closed is True

    def test_max_steps_fallback_uses_stream_backed_complete(self, monkeypatch):
        tool_name = "_test_planner_fallback_tool"

        class _DummyStats:
            def __init__(self):
                self.recorded_usage = []

            def new_turn(self):
                pass

            def record_usage(self, usage, provider="", model=""):
                self.recorded_usage.append((usage, provider, model))

            def record_tool_call(self):
                pass

        class _FakeRuntimeStream:
            def __init__(self, final_message):
                self._final_message = final_message
                self.iterated = False
                self.closed = False

            def __iter__(self):
                self.iterated = True
                yield {"type": "text_delta", "text": "fallback"}

            def get_final_message(self):
                return self._final_message

            def close(self):
                self.closed = True

        def _tool():
            return "tool output"

        schema = {
            "name": tool_name,
            "description": "planner fallback tool",
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        }
        reg = get_registry()
        reg.register(tool_name, _tool, schema)

        dummy_stats = _DummyStats()
        written = []
        final_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "fallback answer"}],
            "provider": "openai",
            "model": "gpt-5-mini",
            "usage": {"input_tokens": 2, "output_tokens": 3},
            "stop_reason": "stop",
        }
        fake_stream = _FakeRuntimeStream(final_message)
        adapter = get_adapter("openai")
        captured = {}

        def _fake_stream_with_retry(runtime_owner, context, max_output_tokens, step, all_text, working_messages):
            return (
                "",
                [planner._ToolCallRequest(id="call_1", name=tool_name, input={})],
                "tool_use",
                {
                    "role": "assistant",
                    "content": [{"type": "tool_call", "id": "call_1", "name": tool_name, "arguments": {}}],
                    "provider": "openai",
                    "model": "gpt-5-mini",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "stop_reason": "tool_use",
                },
            )

        monkeypatch.setattr(planner, "_stream_with_retry", _fake_stream_with_retry)
        monkeypatch.setattr(planner, "get_stats", lambda: dummy_stats)
        monkeypatch.setattr(planner, "_stream_write", written.append)
        monkeypatch.setattr(
            planner,
            "get_reasoning_config_for_provider",
            lambda provider, *, max_output_tokens=None, model=None: {"effort": "high"},
        )

        def _fake_adapter_stream(model_spec, context, options=None):
            captured["options"] = options
            return fake_stream

        monkeypatch.setattr(
            adapter,
            "stream",
            _fake_adapter_stream,
        )

        try:
            result, final_text, working_messages = plan_next_action(
                RuntimeOwner(ModelSpec("openai", "gpt-5-mini")),
                [],
                "system prompt",
                max_steps=1,
            )
        finally:
            reg.unregister(tool_name)

        assert result == "fallback answer"
        assert final_text == "fallback answer"
        assert working_messages[-1]["content"] == [{"type": "text", "text": "fallback answer"}]
        assert written == ["fallback answer"]
        assert captured["options"].reasoning == {"effort": "high"}
        assert fake_stream.iterated is True
        assert fake_stream.closed is True
        assert dummy_stats.recorded_usage == [
            ({"input_tokens": 2, "output_tokens": 3}, "openai", "gpt-5-mini")
        ]


class TestPlannerWarnings:
    def test_runtime_warnings_are_printed_without_aborting_turn(self, monkeypatch, capsys):
        class _DummyStats:
            def __init__(self):
                self.recorded_usage = []

            def new_turn(self):
                pass

            def record_usage(self, usage, provider="", model=""):
                self.recorded_usage.append((usage, provider, model))

            def record_tool_call(self):
                pass

        class _FakeRuntimeStream:
            def __init__(self, final_message):
                self._final_message = final_message
                self.closed = False

            def __iter__(self):
                yield {"type": "text_delta", "text": "warning-safe answer"}

            def get_final_message(self):
                return self._final_message

            def close(self):
                self.closed = True

        dummy_stats = _DummyStats()
        final_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "warning-safe answer"}],
            "provider": "openai",
            "model": "gpt-5-mini",
            "usage": {"input_tokens": 4, "output_tokens": 5},
            "stop_reason": "stop",
            "runtime_warnings": [
                "Recovered OpenAI stream after missing terminal event via responses.retrieve()."
            ],
        }
        fake_stream = _FakeRuntimeStream(final_message)
        adapter = get_adapter("openai")

        monkeypatch.setattr(planner, "get_stats", lambda: dummy_stats)
        monkeypatch.setattr(
            adapter,
            "stream",
            lambda model_spec, context, options=None: fake_stream,
        )

        result, final_text, working_messages = plan_next_action(
            RuntimeOwner(ModelSpec("openai", "gpt-5-mini")),
            [],
            "system prompt",
            max_steps=1,
        )

        output = capsys.readouterr().out

        assert "Recovered OpenAI stream after missing terminal event via responses.retrieve()." in output
        assert result == "warning-safe answer"
        assert final_text == "warning-safe answer"
        assert working_messages[-1]["runtime_warnings"] == final_message["runtime_warnings"]
        assert fake_stream.closed is True
        assert dummy_stats.recorded_usage == [
            ({"input_tokens": 4, "output_tokens": 5}, "openai", "gpt-5-mini")
        ]


class TestPlannerLogging:
    def test_stream_retry_logs_attempt_context(self, monkeypatch):
        attempts = {"count": 0}

        def _fake_stream_response(runtime_owner, context, max_output_tokens, metadata=None):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("timeout from backend")
            return ("ok", [], "stop", {"role": "assistant", "content": []})

        monkeypatch.setattr(planner, "_stream_response", _fake_stream_response)
        monkeypatch.setattr(planner.time, "sleep", lambda _seconds: None)

        with _capture_logger("jyagent.planner") as records:
            result = planner._stream_with_retry(
                RuntimeOwner(ModelSpec("openai", "gpt-5-mini")),
                {"messages": []},
                2048,
                0,
                "",
                [],
            )

        assert result[0] == "ok"
        assert [record.event for record in records] == ["planner.stream.retry"]
        assert records[0].payload["step"] == 1
        assert records[0].payload["attempt"] == 1
        assert records[0].payload["retry_in_seconds"] == 2

    def test_json_decode_stream_failure_retries_as_transient(self, monkeypatch):
        attempts = {"count": 0}

        def _fake_stream_response(runtime_owner, context, max_output_tokens, metadata=None):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise json.JSONDecodeError(
                    "Expecting ',' delimiter",
                    '{"type":"response.created","response":{"id":"resp_retry""}}',
                    56,
                )
            return ("ok", [], "stop", {"role": "assistant", "content": []})

        monkeypatch.setattr(planner, "_stream_response", _fake_stream_response)
        monkeypatch.setattr(planner.time, "sleep", lambda _seconds: None)

        with _capture_logger("jyagent.planner") as records:
            result = planner._stream_with_retry(
                RuntimeOwner(ModelSpec("openai", "gpt-5-mini")),
                {"messages": []},
                2048,
                0,
                "",
                [],
            )

        assert result[0] == "ok"
        assert [record.event for record in records] == ["planner.stream.retry"]
        assert records[0].payload["transient"] is True
        assert records[0].payload["error_type"] == "JSONDecodeError"

    def test_plan_next_action_retries_after_discarded_snapshot_recovery(self, monkeypatch):
        class _DummySpinner:
            def start(self):
                pass

            def stop(self):
                pass

        class _DummyStats:
            def __init__(self):
                self.recorded_usage = []

            def new_turn(self):
                pass

            def record_usage(self, usage, provider="", model=""):
                self.recorded_usage.append((usage, provider, model))

            def record_tool_call(self):
                pass

        json_error = json.JSONDecodeError(
            "Expecting ',' delimiter",
            '{"type":"response.created","response":{"id":"resp_retry_bad""}}',
            60,
        )
        discarded_snapshot = SimpleNamespace(
            id="resp_retry_bad",
            output=[
                _OpenAIItem(
                    "message",
                    id="msg_retry_bad",
                    content=[_OpenAIItem("output_text", text="")],
                ),
            ],
            usage=None,
            error=None,
            incomplete_details=None,
        )
        bad_stream = _FakeOpenAIManagedStream(
            [SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_retry_bad"))],
            None,
            iter_error=json_error,
            state=SimpleNamespace(_ResponseStreamState__current_snapshot=discarded_snapshot),
        )
        good_response = SimpleNamespace(
            id="resp_retry_good",
            error=None,
            incomplete_details=None,
            usage=SimpleNamespace(input_tokens=3, output_tokens=2, total_tokens=5),
            output=[
                _OpenAIItem(
                    "message",
                    id="msg_retry_good",
                    content=[_OpenAIItem("output_text", text="retried answer")],
                ),
            ],
        )
        good_stream = _FakeOpenAIManagedStream(
            [SimpleNamespace(type="response.output_text.delta", delta="retried answer")],
            good_response,
        )
        responses_api = _FakeOpenAIResponsesAPI(
            [_FakeStreamManager(bad_stream), _FakeStreamManager(good_stream)],
            retrieve_error=RuntimeError("retrieve failed"),
        )
        adapter = get_adapter("openai")
        dummy_stats = _DummyStats()

        monkeypatch.setattr(adapter, "_client", lambda: SimpleNamespace(responses=responses_api))
        monkeypatch.setattr(planner, "_ThinkingSpinner", _DummySpinner)
        monkeypatch.setattr(planner, "get_stats", lambda: dummy_stats)
        monkeypatch.setattr(planner.time, "sleep", lambda _seconds: None)

        with _capture_logger("jyagent.planner") as planner_records, _capture_logger(
            "jyagent.runtime.providers.openai"
        ) as provider_records:
            result, final_text, working_messages = plan_next_action(
                RuntimeOwner(ModelSpec("openai", "gpt-5-mini")),
                [],
                "system prompt",
                max_steps=1,
            )

        assert result == "retried answer"
        assert final_text == "retried answer"
        assert working_messages[-1]["content"] == [{"type": "text", "text": "retried answer"}]
        assert dummy_stats.recorded_usage == [
            ({"input_tokens": 3, "output_tokens": 2, "cache_read_input_tokens": 0, "total_tokens": 5}, "openai", "gpt-5-mini")
        ]
        assert [record.event for record in planner_records] == ["planner.stream.retry"]
        assert [record.event for record in provider_records] == [
            "llm.request.started",
            "llm.request.recovery_discarded",
            "llm.request.failed",
            "llm.request.started",
            "llm.request.succeeded",
        ]
        assert responses_api.retrieve_calls[0]["response_id"] == "resp_retry_bad"

    def test_plan_next_action_surfaces_error_after_discarded_snapshot_recovery_retries_exhausted(self, monkeypatch):
        class _DummySpinner:
            def start(self):
                pass

            def stop(self):
                pass

        class _DummyStats:
            def new_turn(self):
                pass

            def record_usage(self, usage, provider="", model=""):
                pass

            def record_tool_call(self):
                pass

        def _bad_manager(response_id: str):
            json_error = json.JSONDecodeError(
                "Expecting ',' delimiter",
                f'{{"type":"response.created","response":{{"id":"{response_id}""}}}}',
                len(response_id) + 46,
            )
            discarded_snapshot = SimpleNamespace(
                id=response_id,
                output=[
                    _OpenAIItem(
                        "function_call",
                        call_id=f"call_{response_id}",
                        name="echo",
                        arguments='{"value":',
                    ),
                ],
                usage=None,
                error=None,
                incomplete_details=None,
            )
            stream = _FakeOpenAIManagedStream(
                [SimpleNamespace(type="response.created", response=SimpleNamespace(id=response_id))],
                None,
                iter_error=json_error,
                state=SimpleNamespace(_ResponseStreamState__current_snapshot=discarded_snapshot),
            )
            return _FakeStreamManager(stream)

        responses_api = _FakeOpenAIResponsesAPI(
            [_bad_manager("resp_bad_1"), _bad_manager("resp_bad_2"), _bad_manager("resp_bad_3")],
            retrieve_error=RuntimeError("retrieve failed"),
        )
        adapter = get_adapter("openai")

        monkeypatch.setattr(adapter, "_client", lambda: SimpleNamespace(responses=responses_api))
        monkeypatch.setattr(planner, "_ThinkingSpinner", _DummySpinner)
        monkeypatch.setattr(planner, "get_stats", lambda: _DummyStats())
        monkeypatch.setattr(planner.time, "sleep", lambda _seconds: None)

        with _capture_logger("jyagent.planner") as planner_records, _capture_logger(
            "jyagent.runtime.providers.openai"
        ) as provider_records:
            result, final_text, working_messages = plan_next_action(
                RuntimeOwner(ModelSpec("openai", "gpt-5-mini")),
                [],
                "system prompt",
                max_steps=1,
            )

        assert result.startswith("Error during streaming: Expecting ',' delimiter")
        assert final_text == ""
        assert working_messages == []
        assert [record.event for record in planner_records] == [
            "planner.stream.retry",
            "planner.stream.retry",
            "planner.stream.failed",
        ]
        assert [record.event for record in provider_records] == [
            "llm.request.started",
            "llm.request.recovery_discarded",
            "llm.request.failed",
            "llm.request.started",
            "llm.request.recovery_discarded",
            "llm.request.failed",
            "llm.request.started",
            "llm.request.recovery_discarded",
            "llm.request.failed",
        ]
        assert len(responses_api.retrieve_calls) == 3

    def test_fatal_stream_failure_logs_and_preserves_return_shape(self, monkeypatch):
        class _FakeOwner:
            model_spec = ModelSpec("openai", "gpt-5-mini")

        monkeypatch.setattr(
            planner,
            "_stream_response",
            lambda runtime_owner, context, max_output_tokens, metadata=None: (_ for _ in ()).throw(RuntimeError("bad request")),
        )

        with _capture_logger("jyagent.planner") as records:
            result, final_text, working_messages = plan_next_action(_FakeOwner(), [], "system prompt", max_steps=1)

        assert result == "Error during streaming: bad request"
        assert final_text == ""
        assert working_messages == []
        assert [record.event for record in records] == ["planner.stream.failed"]
        assert records[0].payload["step"] == 1
        assert records[0].payload["attempt"] == 1

    def test_fallback_failure_logs_context(self, monkeypatch):
        tool_name = "_test_planner_fallback_failure_tool"

        def _tool():
            return "tool output"

        schema = {
            "name": tool_name,
            "description": "planner fallback tool",
            "input_schema": {"type": "object", "properties": {}},
        }
        reg = get_registry()
        reg.register(tool_name, _tool, schema)

        class _DummyStats:
            def new_turn(self):
                pass

            def record_usage(self, usage, provider="", model=""):
                pass

            def record_tool_call(self):
                pass

        def _fake_stream_with_retry(runtime_owner, context, max_output_tokens, step, all_text, working_messages):
            return (
                "",
                [planner._ToolCallRequest(id="call_1", name=tool_name, input={})],
                "tool_use",
                {
                    "role": "assistant",
                    "content": [{"type": "tool_call", "id": "call_1", "name": tool_name, "arguments": {}}],
                    "provider": "openai",
                    "model": "gpt-5-mini",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "stop_reason": "tool_use",
                },
            )

        monkeypatch.setattr(planner, "_stream_with_retry", _fake_stream_with_retry)
        monkeypatch.setattr(planner, "get_stats", lambda: _DummyStats())
        monkeypatch.setattr(
            get_adapter("openai"),
            "stream",
            lambda model_spec, context, options=None: (_ for _ in ()).throw(RuntimeError("fallback exploded")),
        )

        try:
            with _capture_logger("jyagent.planner") as records:
                result, final_text, working_messages = plan_next_action(
                    RuntimeOwner(ModelSpec("openai", "gpt-5-mini")),
                    [],
                    "system prompt",
                    max_steps=1,
                )
        finally:
            reg.unregister(tool_name)

        assert result == "I've reached my maximum reasoning steps. Please try rephrasing your request."
        assert final_text == ""
        assert working_messages[-1]["tool_name"] == tool_name
        assert [record.event for record in records] == ["planner.fallback.failed"]
        assert records[0].payload["fallback"] is True
