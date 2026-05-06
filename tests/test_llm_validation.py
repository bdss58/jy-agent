"""Tests for jyagent/llm/validation.py.

The validators are pure functions; these tests exercise:
  * Happy paths (shapes that the engine actually consumes)
  * Every error branch (each _check_* helper + every missing-required)
  * Forward-compat tolerance for unknown optional keys / unknown block types
  * Path-pointer correctness on nested failures (e.g. 'messages[3].content[1].arguments')
"""

import pytest

from jyagent.llm.validation import (
    MessageValidationError,
    validate_assistant_message,
    validate_context,
    validate_message,
    validate_stream_event,
    validate_tool_result_message,
    validate_user_message,
)


# ─── Happy paths ─────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_user_message(self):
        msg = validate_user_message({"role": "user", "content": "hi"})
        assert msg["role"] == "user"

    def test_assistant_text_only(self):
        msg = validate_assistant_message({
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
        })
        assert msg["content"][0]["type"] == "text"

    def test_assistant_with_tool_call(self):
        msg = validate_assistant_message({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "using tool"},
                {"type": "tool_call", "id": "tc1", "name": "read", "arguments": {"path": "/x"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5,
                      "cache_read_input_tokens": 100},
        })
        assert msg["stop_reason"] == "tool_use"

    def test_assistant_with_thinking_anthropic_style(self):
        msg = validate_assistant_message({
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "let me think", "signature": "sig"},
                {"type": "text", "text": "answer"},
            ],
        })
        assert len(msg["content"]) == 2

    def test_assistant_with_thinking_openai_summary(self):
        msg = validate_assistant_message({
            "role": "assistant",
            "content": [
                {"type": "thinking", "summary": ["step 1", "step 2"], "id": "rs_abc"},
            ],
        })
        assert msg["content"][0]["summary"] == ["step 1", "step 2"]

    def test_tool_result_with_str_content(self):
        msg = validate_tool_result_message({
            "role": "tool_result", "tool_call_id": "tc1",
            "tool_name": "read", "content": "ok", "is_error": False,
        })
        assert msg["content"] == "ok"

    def test_tool_result_with_list_content(self):
        # Some providers return structured tool_result content.
        msg = validate_tool_result_message({
            "role": "tool_result", "tool_call_id": "tc1",
            "tool_name": "read", "content": [{"type": "text", "text": "ok"}],
            "is_error": False,
        })
        assert isinstance(msg["content"], list)

    def test_validate_message_dispatch(self):
        """validate_message routes by role."""
        assert validate_message({"role": "user", "content": "x"})["role"] == "user"
        assert validate_message({
            "role": "assistant", "content": [],
        })["role"] == "assistant"
        assert validate_message({
            "role": "tool_result", "tool_call_id": "a", "tool_name": "b",
            "content": "c", "is_error": False,
        })["role"] == "tool_result"

    def test_context_full(self):
        ctx = validate_context({
            "system_prompt": "You are helpful.",
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
            ],
            "tools": [{"name": "t", "description": "d", "input_schema": {}}],
        })
        assert len(ctx["messages"]) == 2

    def test_context_no_system_prompt(self):
        """system_prompt is optional."""
        ctx = validate_context({"messages": []})
        assert ctx["messages"] == []


class TestStreamEvents:
    def test_text_delta(self):
        ev = validate_stream_event({"type": "text_delta", "text": "hi", "content_index": 0})
        assert ev["type"] == "text_delta"

    def test_thinking_delta(self):
        ev = validate_stream_event({"type": "thinking_delta", "text": "t", "content_index": 1})
        assert ev["type"] == "thinking_delta"

    def test_tool_call_delta(self):
        ev = validate_stream_event({"type": "tool_call_delta", "delta": "{\"x\": 1}", "content_index": 2})
        assert ev["type"] == "tool_call_delta"

    def test_done_event_carries_assistant_message(self):
        ev = validate_stream_event({
            "type": "done",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        })
        assert ev["type"] == "done"

    def test_error_event_carries_assistant_message(self):
        ev = validate_stream_event({
            "type": "error",
            "message": {"role": "assistant", "content": [], "stop_reason": "error",
                        "error_message": "timeout"},
        })
        assert ev["type"] == "error"

    def test_bare_lifecycle_events(self):
        """start, *_start, *_end events need only 'type'."""
        for t in ("start", "text_start", "text_end",
                  "thinking_start", "thinking_end",
                  "tool_call_start", "tool_call_end"):
            ev = validate_stream_event({"type": t, "content_index": 0})
            assert ev["type"] == t


# ─── Forward compatibility ───────────────────────────────────────────────────


class TestForwardCompat:
    def test_unknown_optional_keys_tolerated(self):
        """A newer provider may add fields the engine doesn't read yet."""
        msg = validate_assistant_message({
            "role": "assistant",
            "content": [{"type": "text", "text": "x"}],
            "provider": "anthropic",
            "model": "claude-3-7-sonnet",
            "response_id": "msg_123",
            "phase": "final_answer",
            "brand_new_field_2030": "something",  # unknown, tolerated
        })
        assert msg["provider"] == "anthropic"

    def test_unknown_content_block_type_tolerated(self):
        """Engine passes through unknown block types — future SDKs may add
        new types (e.g. 'image', 'audio', 'citation') before we special-case them."""
        msg = validate_assistant_message({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "x"},
                {"type": "image", "source": {"url": "..."}},  # unknown, tolerated
            ],
        })
        assert len(msg["content"]) == 2

    def test_unknown_stream_event_type_rejected(self):
        """Stream events are rejected on unknown type — unlike content blocks,
        the engine's stream loop dispatches on type and silent drop would
        lose content mid-stream."""
        with pytest.raises(MessageValidationError) as exc_info:
            validate_stream_event({"type": "novel_event_type"})
        assert "unknown stream event type" in exc_info.value.reason


# ─── Error branches ──────────────────────────────────────────────────────────


class TestErrors:
    def test_not_a_dict(self):
        with pytest.raises(MessageValidationError) as e:
            validate_message("string instead of dict")
        assert e.value.path == "message"
        assert "expected dict" in e.value.reason

    def test_missing_role(self):
        with pytest.raises(MessageValidationError) as e:
            validate_message({"content": "x"})
        assert e.value.path == "message.role"

    def test_unknown_role(self):
        with pytest.raises(MessageValidationError) as e:
            validate_message({"role": "system", "content": "x"})
        assert "unknown role" in e.value.reason

    def test_user_content_wrong_type(self):
        with pytest.raises(MessageValidationError) as e:
            validate_user_message({"role": "user", "content": ["not a string"]})
        assert e.value.path == "message.content"
        assert "expected str" in e.value.reason

    def test_assistant_content_not_list(self):
        with pytest.raises(MessageValidationError) as e:
            validate_assistant_message({"role": "assistant", "content": "x"})
        assert e.value.path == "message.content"
        assert "expected list" in e.value.reason

    def test_content_block_missing_type(self):
        with pytest.raises(MessageValidationError) as e:
            validate_assistant_message({
                "role": "assistant",
                "content": [{"text": "forgot type"}],
            })
        assert "message.content[0].type" in e.value.path

    def test_text_block_missing_text(self):
        with pytest.raises(MessageValidationError) as e:
            validate_assistant_message({
                "role": "assistant",
                "content": [{"type": "text"}],
            })
        assert e.value.path == "message.content[0].text"

    def test_tool_call_missing_id(self):
        with pytest.raises(MessageValidationError) as e:
            validate_assistant_message({
                "role": "assistant",
                "content": [{"type": "tool_call", "name": "t", "arguments": {}}],
            })
        assert e.value.path == "message.content[0].id"

    def test_tool_call_empty_id_rejected(self):
        with pytest.raises(MessageValidationError) as e:
            validate_assistant_message({
                "role": "assistant",
                "content": [{"type": "tool_call", "id": "", "name": "t", "arguments": {}}],
            })
        assert "non-empty" in e.value.reason

    def test_tool_call_arguments_not_dict(self):
        with pytest.raises(MessageValidationError) as e:
            validate_assistant_message({
                "role": "assistant",
                "content": [{"type": "tool_call", "id": "a", "name": "t",
                             "arguments": "{not a dict}"}],
            })
        assert e.value.path == "message.content[0].arguments"

    def test_bad_stop_reason(self):
        with pytest.raises(MessageValidationError) as e:
            validate_assistant_message({
                "role": "assistant", "content": [],
                "stop_reason": "made_up_reason",
            })
        assert "unknown stop_reason" in e.value.reason

    def test_usage_non_int_field(self):
        with pytest.raises(MessageValidationError) as e:
            validate_assistant_message({
                "role": "assistant", "content": [],
                "usage": {"input_tokens": "one hundred"},
            })
        assert e.value.path == "message.usage.input_tokens"

    def test_usage_bool_rejected_for_int_field(self):
        """bool is an int subclass in Python — validator must reject it
        because it's not a legitimate token count."""
        with pytest.raises(MessageValidationError):
            validate_assistant_message({
                "role": "assistant", "content": [],
                "usage": {"input_tokens": True},
            })

    def test_tool_result_missing_is_error(self):
        with pytest.raises(MessageValidationError) as e:
            validate_tool_result_message({
                "role": "tool_result", "tool_call_id": "a",
                "tool_name": "t", "content": "ok",
            })
        assert e.value.path == "message.is_error"

    def test_tool_result_content_wrong_type(self):
        with pytest.raises(MessageValidationError) as e:
            validate_tool_result_message({
                "role": "tool_result", "tool_call_id": "a",
                "tool_name": "t", "content": 42, "is_error": False,
            })
        assert e.value.path == "message.content"

    def test_context_messages_missing(self):
        with pytest.raises(MessageValidationError) as e:
            validate_context({"system_prompt": "x"})
        assert e.value.path == "context.messages"

    def test_nested_path_pointer_in_context(self):
        """Path pointer must point into the exact bad message/block."""
        with pytest.raises(MessageValidationError) as e:
            validate_context({
                "messages": [
                    {"role": "user", "content": "ok"},
                    {"role": "user", "content": "ok"},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": "ok"},
                        {"type": "tool_call", "id": "x", "name": "n"},  # missing arguments
                    ]},
                ],
            })
        assert e.value.path == "context.messages[2].content[1].arguments"

    def test_stream_done_event_bad_inner_message(self):
        with pytest.raises(MessageValidationError) as e:
            validate_stream_event({
                "type": "done",
                "message": {"role": "assistant"},  # missing content
            })
        assert e.value.path == "event.message.content"

    def test_inherits_from_TypeError(self):
        """Runtime catch sites that handle TypeError (SDK adapter layer,
        retry loop) will naturally absorb validation failures too."""
        with pytest.raises(TypeError):
            validate_message({"role": "bad"})


# ─── Integration: normalized adapter output shapes ───────────────────────────


class TestRealAdapterShapes:
    """Spot-check shapes that Anthropic / OpenAI adapters actually produce."""

    def test_anthropic_complete_shape(self):
        """Shape from _anthropic_helpers.build_normalized_message (approx)."""
        msg = validate_assistant_message({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "tool_call", "id": "toolu_01ABC", "name": "read_file",
                 "arguments": {"path": "/etc/hosts"}},
            ],
            "provider": "anthropic",
            "api": "messages",
            "model": "claude-3-7-sonnet-20250219",
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 1234,
                "output_tokens": 56,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 1000,
            },
            "response_id": "msg_01XYZ",
        })
        assert msg["usage"]["cache_read_input_tokens"] == 1000

    def test_openai_responses_shape(self):
        """Shape from OpenAI Responses adapter (approx)."""
        msg = validate_assistant_message({
            "role": "assistant",
            "content": [
                {"type": "thinking", "summary": ["reasoning..."], "id": "rs_1",
                 "status": "completed"},
                {"type": "text", "text": "answer"},
            ],
            "provider": "openai",
            "api": "responses",
            "model": "gpt-5",
            "stop_reason": "stop",
            "usage": {"input_tokens": 500, "output_tokens": 100},
            "response_id": "resp_1",
        })
        assert msg["stop_reason"] == "stop"
