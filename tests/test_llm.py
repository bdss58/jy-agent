"""Comprehensive test suite for jyagent/llm module.

Covers types, messages, streams, core, and provider helper modules.
Uses unittest.mock throughout — no actual anthropic/openai SDK required.
"""

from __future__ import annotations

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ════════════════════════════════════════════════════════════════════════════════
# §1  types.py
# ════════════════════════════════════════════════════════════════════════════════

from jyagent.llm.types import (
    AssistantMessage,
    ThinkingBlock,
    Usage,
    compute_total_tokens,
    ModelSpec,
)


class TestComputeTotalTokens:
    def test_both_present(self):
        usage: Usage = {"input_tokens": 100, "output_tokens": 50}
        assert compute_total_tokens(usage) == 150

    def test_only_input(self):
        usage: Usage = {"input_tokens": 100}
        assert compute_total_tokens(usage) == 100

    def test_only_output(self):
        usage: Usage = {"output_tokens": 42}
        assert compute_total_tokens(usage) == 42

    def test_empty_usage(self):
        usage: Usage = {}
        assert compute_total_tokens(usage) == 0


class TestAssistantMessageRequiredFields:
    """AssistantMessage must have 'role' and 'content'."""

    def test_minimal_valid(self):
        msg: AssistantMessage = {"role": "assistant", "content": []}
        assert msg["role"] == "assistant"
        assert msg["content"] == []

    def test_with_optional_fields(self):
        msg: AssistantMessage = {
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "stop_reason": "stop",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
        assert msg["provider"] == "anthropic"


class TestThinkingBlock:
    """ThinkingBlock requires 'type', other fields optional."""

    def test_required_type(self):
        tb: ThinkingBlock = {"type": "thinking"}
        assert tb["type"] == "thinking"

    def test_with_optional_fields(self):
        tb: ThinkingBlock = {
            "type": "thinking",
            "thinking": "deep thoughts",
            "signature": "sig123",
            "redacted": False,
        }
        assert tb["thinking"] == "deep thoughts"


class TestModelSpec:
    def test_label(self):
        spec = ModelSpec(provider="anthropic", model="claude-sonnet-4-6")
        assert spec.label() == "anthropic:claude-sonnet-4-6"

    def test_frozen(self):
        spec = ModelSpec(provider="openai", model="gpt-4")
        with pytest.raises(AttributeError):
            spec.provider = "x"  # type: ignore[misc]


# ════════════════════════════════════════════════════════════════════════════════
# §2  messages.py
# ════════════════════════════════════════════════════════════════════════════════

from jyagent.llm.messages import (
    assistant_text,
    inject_missing_tool_results,
    normalize_anthropic_tool_call_id,
    thinking_to_text_block,
    transform_messages_for_target,
)


# ── assistant_text ─────────────────────────────────────────────────────────────

class TestAssistantText:
    def test_single_text_block(self):
        msg: AssistantMessage = {
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
        }
        assert assistant_text(msg) == "hello"

    def test_multiple_text_blocks(self):
        msg: AssistantMessage = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hello "},
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "world"},
            ],
        }
        assert assistant_text(msg) == "hello world"

    def test_no_text_blocks(self):
        msg: AssistantMessage = {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "hmm"}],
        }
        assert assistant_text(msg) == ""

    def test_empty_content(self):
        msg: AssistantMessage = {"role": "assistant", "content": []}
        assert assistant_text(msg) == ""


# ── normalize_anthropic_tool_call_id ───────────────────────────────────────────

class TestNormalizeAnthropicToolCallId:
    def test_clean_id_unchanged(self):
        assert normalize_anthropic_tool_call_id("call_abc123") == "call_abc123"

    def test_special_chars_sanitized(self):
        assert normalize_anthropic_tool_call_id("call:abc!@#def") == "call_abc___def"

    def test_truncation_to_64(self):
        long_id = "a" * 100
        result = normalize_anthropic_tool_call_id(long_id)
        assert len(result) == 64

    def test_empty_fallback(self):
        # All chars removed → fallback to "tool_call"
        assert normalize_anthropic_tool_call_id("") == "tool_call"

    def test_all_special_chars_produce_underscores(self):
        assert normalize_anthropic_tool_call_id("!!!") == "___"

    def test_hyphens_and_underscores_kept(self):
        assert normalize_anthropic_tool_call_id("a-b_c") == "a-b_c"


# ── thinking_to_text_block ─────────────────────────────────────────────────────

class TestThinkingToTextBlock:
    def test_normal_thinking(self):
        block: ThinkingBlock = {"type": "thinking", "thinking": "Let me think"}
        result = thinking_to_text_block(block)
        assert result is not None
        assert result["type"] == "text"
        assert "<thinking>" in result["text"]
        assert "Let me think" in result["text"]
        assert "</thinking>" in result["text"]

    def test_empty_thinking_returns_none(self):
        block: ThinkingBlock = {"type": "thinking", "thinking": ""}
        assert thinking_to_text_block(block) is None

    def test_whitespace_only_returns_none(self):
        block: ThinkingBlock = {"type": "thinking", "thinking": "   \n  "}
        assert thinking_to_text_block(block) is None

    def test_missing_thinking_key(self):
        block: ThinkingBlock = {"type": "thinking"}
        assert thinking_to_text_block(block) is None


# ── transform_messages_for_target ──────────────────────────────────────────────

class TestTransformMessagesForTarget:
    def _anthropic_spec(self):
        return ModelSpec(provider="anthropic", model="claude-sonnet-4-6")

    def _openai_spec(self):
        return ModelSpec(provider="openai", model="gpt-5.4")

    def test_user_messages_pass_through(self):
        messages = [{"role": "user", "content": "hello"}]
        result = transform_messages_for_target(messages, self._anthropic_spec())
        assert len(result) == 1
        assert result[0]["content"] == "hello"

    def test_same_model_thinking_preserved(self):
        """Thinking blocks from the same model are kept as-is."""
        target = self._anthropic_spec()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "deep", "signature": "s1"},
                    {"type": "text", "text": "hi"},
                ],
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
            }
        ]
        result = transform_messages_for_target(messages, target)
        assistant = result[0]
        thinking_blocks = [b for b in assistant["content"] if b["type"] == "thinking"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "deep"

    def test_cross_model_thinking_converted_to_text(self):
        """Thinking blocks from different models become <thinking> text blocks."""
        target = self._openai_spec()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "reasoning here"},
                    {"type": "text", "text": "answer"},
                ],
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
            }
        ]
        result = transform_messages_for_target(messages, target)
        assistant = result[0]
        # Should have converted thinking to text
        text_blocks = [b for b in assistant["content"] if b["type"] == "text"]
        assert len(text_blocks) == 2
        thinking_text = [b for b in text_blocks if "<thinking>" in b["text"]]
        assert len(thinking_text) == 1

    def test_cross_model_redacted_thinking_dropped(self):
        """Redacted thinking blocks from different models are dropped."""
        target = self._openai_spec()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "", "redacted": True, "signature": "s"},
                    {"type": "text", "text": "answer"},
                ],
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
            }
        ]
        result = transform_messages_for_target(messages, target)
        assistant = result[0]
        thinking_blocks = [
            b for b in assistant["content"]
            if b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) == 0

    def test_tool_call_ids_normalized_for_anthropic(self):
        """Tool-call IDs are sanitized when targeting Anthropic."""
        target = self._anthropic_spec()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "id": "call:special!id", "name": "foo", "arguments": {}},
                ],
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
            },
            {
                "role": "tool_result",
                "tool_call_id": "call:special!id",
                "tool_name": "foo",
                "content": "ok",
                "is_error": False,
            },
        ]
        result = transform_messages_for_target(messages, target)
        # The assistant tool_call id should be sanitized
        assistant = result[0]
        tc_block = [b for b in assistant["content"] if b["type"] == "tool_call"][0]
        assert tc_block["id"] == "call_special_id"
        # Corresponding tool_result should also be updated
        tr = [m for m in result if m.get("role") == "tool_result"][0]
        assert tr["tool_call_id"] == "call_special_id"

    def test_unknown_block_types_pass_through(self):
        """Unknown block types are preserved for forward compatibility."""
        target = self._anthropic_spec()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "unknown_future_type", "data": "xyz"},
                    {"type": "text", "text": "answer"},
                ],
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
            }
        ]
        result = transform_messages_for_target(messages, target)
        assistant = result[0]
        unknown = [b for b in assistant["content"] if b["type"] == "unknown_future_type"]
        assert len(unknown) == 1
        assert unknown[0]["data"] == "xyz"


# ── inject_missing_tool_results ────────────────────────────────────────────────

class TestInjectMissingToolResults:
    def test_dangling_tool_calls_get_synthetic_error(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "id": "tc1", "name": "foo", "arguments": {}},
                    {"type": "tool_call", "id": "tc2", "name": "bar", "arguments": {}},
                ],
            },
            # Only tc1 has a result
            {
                "role": "tool_result",
                "tool_call_id": "tc1",
                "tool_name": "foo",
                "content": "ok",
                "is_error": False,
            },
        ]
        result = inject_missing_tool_results(messages)
        tool_results = [m for m in result if m.get("role") == "tool_result"]
        assert len(tool_results) == 2
        synthetic = [tr for tr in tool_results if tr["tool_call_id"] == "tc2"]
        assert len(synthetic) == 1
        assert synthetic[0]["is_error"] is True

    def test_existing_results_preserved(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "id": "tc1", "name": "foo", "arguments": {}},
                ],
            },
            {
                "role": "tool_result",
                "tool_call_id": "tc1",
                "tool_name": "foo",
                "content": "result data",
                "is_error": False,
            },
        ]
        result = inject_missing_tool_results(messages)
        tool_results = [m for m in result if m.get("role") == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0]["content"] == "result data"
        assert tool_results[0]["is_error"] is False

    def test_error_and_aborted_messages_skipped(self):
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "err"}],
                "stop_reason": "error",
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "aborted"}],
                "stop_reason": "aborted",
            },
            {"role": "user", "content": "retry"},
        ]
        result = inject_missing_tool_results(messages)
        # Error/aborted assistants dropped, user kept
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_no_tool_calls_unchanged(self):
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "just text"}],
            },
        ]
        result = inject_missing_tool_results(messages)
        assert len(result) == 1

    def test_multiple_assistants_flush_pending(self):
        """When a new assistant appears, pending tool_calls from prior should be flushed."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "id": "tc1", "name": "a", "arguments": {}},
                ],
            },
            # No result for tc1 — then another assistant
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "next"}],
            },
        ]
        result = inject_missing_tool_results(messages)
        tool_results = [m for m in result if m.get("role") == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_call_id"] == "tc1"
        assert tool_results[0]["is_error"] is True


# ════════════════════════════════════════════════════════════════════════════════
# §3  streams.py
# ════════════════════════════════════════════════════════════════════════════════

from jyagent.llm.streams import BaseStream, ErrorStream, make_error_assistant_message


class TestMakeErrorAssistantMessage:
    def test_basic_error(self):
        spec = ModelSpec(provider="test", model="test-model")
        err = ValueError("something broke")
        msg = make_error_assistant_message(spec, err)
        assert msg["role"] == "assistant"
        assert msg["stop_reason"] == "error"
        assert "ValueError" in msg["error_message"]
        assert "something broke" in msg["error_message"]
        assert msg["provider"] == "test"
        assert msg["model"] == "test-model"

    def test_with_api_field(self):
        spec = ModelSpec(provider="test", model="m")
        msg = make_error_assistant_message(spec, RuntimeError("x"), api="my-api")
        assert msg["api"] == "my-api"

    def test_with_partial_content(self):
        spec = ModelSpec(provider="test", model="m")
        partial = [{"type": "text", "text": "partial answer"}]
        msg = make_error_assistant_message(spec, RuntimeError("x"), partial_content=partial)
        assert msg["content"] == partial


class TestErrorStream:
    def test_yields_start_then_error(self):
        spec = ModelSpec(provider="test", model="m")
        stream = ErrorStream(spec, ValueError("boom"))
        events = list(stream)
        assert len(events) == 2
        assert events[0]["type"] == "start"
        assert events[1]["type"] == "error"
        assert "ValueError" in events[1]["message"]["error_message"]

    def test_get_final_message(self):
        spec = ModelSpec(provider="test", model="m")
        stream = ErrorStream(spec, ValueError("boom"))
        msg = stream.get_final_message()
        assert msg["stop_reason"] == "error"

    def test_context_manager(self):
        spec = ModelSpec(provider="test", model="m")
        with ErrorStream(spec, ValueError("x")) as s:
            events = list(s)
        assert len(events) == 2


class TestBaseStream:
    def _make_stream(self):
        mock_cm = MagicMock()
        spec = ModelSpec(provider="test", model="m")
        return BaseStream(mock_cm, spec)

    def test_consumed_flag_starts_false(self):
        stream = self._make_stream()
        assert stream._consumed is False

    def test_close_idempotent(self):
        stream = self._make_stream()
        stream.close()
        stream.close()  # Should not raise
        assert stream._closed is True

    def test_close_with_active_stream(self):
        mock_cm = MagicMock()
        spec = ModelSpec(provider="test", model="m")
        stream = BaseStream(mock_cm, spec)
        stream._stream = "not None"  # simulate active stream
        stream.close()
        mock_cm.__exit__.assert_called_once()
        assert stream._closed is True

    def test_context_manager(self):
        stream = self._make_stream()
        with stream as s:
            assert s is stream
        assert stream._closed is True


# ════════════════════════════════════════════════════════════════════════════════
# §4  providers/_openai_helpers.py
# ════════════════════════════════════════════════════════════════════════════════

from jyagent.llm.providers._openai_helpers import (
    validate_openai_reasoning,
    map_stop_reason as openai_map_stop_reason,
    usage_from_response as openai_usage_from_response,
    assistant_from_response as openai_assistant_from_response,
    supports_openai_reasoning_effort,
)


class TestValidateOpenAIReasoning:
    def test_valid_efforts_accepted(self):
        for effort in ("none", "low", "medium", "high", "xhigh"):
            result = validate_openai_reasoning({"effort": effort})
            assert result["effort"] == effort

    def test_invalid_effort_rejected(self):
        with pytest.raises(ValueError, match="effort"):
            validate_openai_reasoning({"effort": "ultra"})

    def test_non_dict_rejected(self):
        with pytest.raises(ValueError, match="dict"):
            validate_openai_reasoning("not a dict")

    def test_model_check_unsupported(self):
        with pytest.raises(ValueError, match="not supported"):
            validate_openai_reasoning({"effort": "high"}, model="gpt-4o")

    def test_model_check_supported(self):
        result = validate_openai_reasoning({"effort": "high"}, model="gpt-5.4")
        assert result["effort"] == "high"

    def test_no_effort_accepted(self):
        result = validate_openai_reasoning({})
        assert result == {}


class TestOpenAIMapStopReason:
    def test_completed_no_tools(self):
        assert openai_map_stop_reason("completed", has_tool_calls=False) == "stop"

    def test_completed_with_tools(self):
        assert openai_map_stop_reason("completed", has_tool_calls=True) == "tool_use"

    def test_incomplete(self):
        assert openai_map_stop_reason("incomplete") == "length"

    def test_failed(self):
        assert openai_map_stop_reason("failed") == "error"

    def test_none(self):
        assert openai_map_stop_reason(None) == "stop"

    def test_unknown(self):
        assert openai_map_stop_reason("something_else") == "error"


class TestOpenAIUsageFromResponse:
    def test_normal_usage(self):
        usage_obj = SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            input_tokens_details=SimpleNamespace(cached_tokens=20),
        )
        result = openai_usage_from_response(usage_obj)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["cache_read_input_tokens"] == 20
        assert result["total_tokens"] == 150

    def test_none_usage(self):
        result = openai_usage_from_response(None)
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["total_tokens"] == 0

    def test_no_cached_tokens(self):
        usage_obj = SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            input_tokens_details=None,
        )
        result = openai_usage_from_response(usage_obj)
        assert result["cache_read_input_tokens"] == 0


class TestOpenAIAssistantFromResponse:
    def _spec(self):
        return ModelSpec(provider="openai", model="gpt-5.4")

    def test_text_response(self):
        response = SimpleNamespace(
            id="resp_123",
            status="completed",
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="Hello!")],
                ),
            ],
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                input_tokens_details=None,
            ),
        )
        msg = openai_assistant_from_response(self._spec(), response)
        assert msg["role"] == "assistant"
        assert msg["stop_reason"] == "stop"
        text_blocks = [b for b in msg["content"] if b["type"] == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Hello!"

    def test_tool_call_with_valid_json(self):
        response = SimpleNamespace(
            id="resp_456",
            status="completed",
            output=[
                SimpleNamespace(
                    type="function_call",
                    call_id="call_1",
                    name="search",
                    arguments='{"query": "test"}',
                ),
            ],
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                input_tokens_details=None,
            ),
        )
        msg = openai_assistant_from_response(self._spec(), response)
        assert msg["stop_reason"] == "tool_use"
        tc = [b for b in msg["content"] if b["type"] == "tool_call"][0]
        assert tc["name"] == "search"
        assert tc["arguments"] == {"query": "test"}

    def test_tool_call_with_bad_json(self):
        response = SimpleNamespace(
            id="resp_789",
            status="completed",
            output=[
                SimpleNamespace(
                    type="function_call",
                    call_id="call_2",
                    name="broken",
                    arguments="not valid json{{{",
                ),
            ],
            usage=SimpleNamespace(
                input_tokens=1,
                output_tokens=1,
                input_tokens_details=None,
            ),
        )
        msg = openai_assistant_from_response(self._spec(), response)
        tc = [b for b in msg["content"] if b["type"] == "tool_call"][0]
        assert "_parse_error" in tc["arguments"]

    def test_reasoning_blocks(self):
        response = SimpleNamespace(
            id="resp_r1",
            status="completed",
            output=[
                SimpleNamespace(
                    type="reasoning",
                    summary=[SimpleNamespace(text="Step 1"), SimpleNamespace(text="Step 2")],
                ),
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="Final answer")],
                ),
            ],
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=20,
                input_tokens_details=None,
            ),
        )
        msg = openai_assistant_from_response(self._spec(), response)
        thinking = [b for b in msg["content"] if b["type"] == "thinking"]
        assert len(thinking) == 1
        assert "Step 1" in thinking[0]["thinking"]
        assert "Step 2" in thinking[0]["thinking"]


# ════════════════════════════════════════════════════════════════════════════════
# §5  providers/_anthropic_helpers.py
# ════════════════════════════════════════════════════════════════════════════════

from jyagent.llm.providers._anthropic_helpers import (
    map_stop_reason as anthropic_map_stop_reason,
    usage_from_response as anthropic_usage_from_response,
    assistant_from_response as anthropic_assistant_from_response,
    convert_messages as anthropic_convert_messages,
    convert_tools as anthropic_convert_tools,
)


class TestAnthropicMapStopReason:
    def test_max_tokens(self):
        assert anthropic_map_stop_reason("max_tokens") == "length"

    def test_tool_use(self):
        assert anthropic_map_stop_reason("tool_use") == "tool_use"

    def test_end_turn(self):
        assert anthropic_map_stop_reason("end_turn") == "stop"

    def test_stop_sequence(self):
        assert anthropic_map_stop_reason("stop_sequence") == "stop"

    def test_pause_turn(self):
        assert anthropic_map_stop_reason("pause_turn") == "stop"

    def test_none(self):
        assert anthropic_map_stop_reason(None) == "stop"

    def test_refusal(self):
        assert anthropic_map_stop_reason("refusal") == "error"

    def test_unknown(self):
        assert anthropic_map_stop_reason("something_else") == "error"


class TestAnthropicUsageFromResponse:
    def test_full_usage(self):
        usage_obj = SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=20,
        )
        result = anthropic_usage_from_response(usage_obj)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["cache_creation_input_tokens"] == 10
        assert result["cache_read_input_tokens"] == 20
        assert result["total_tokens"] == 150

    def test_none_fields(self):
        usage_obj = SimpleNamespace(
            input_tokens=None,
            output_tokens=None,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
        )
        result = anthropic_usage_from_response(usage_obj)
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["total_tokens"] == 0


class TestAnthropicAssistantFromResponse:
    def _spec(self):
        return ModelSpec(provider="anthropic", model="claude-sonnet-4-6")

    def test_text_block(self):
        response = SimpleNamespace(
            id="msg_123",
            content=[SimpleNamespace(type="text", text="Hello")],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=5,
                output_tokens=3,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )
        msg = anthropic_assistant_from_response(self._spec(), response)
        assert msg["role"] == "assistant"
        assert msg["stop_reason"] == "stop"
        assert msg["content"][0] == {"type": "text", "text": "Hello"}

    def test_thinking_block(self):
        response = SimpleNamespace(
            id="msg_t",
            content=[
                SimpleNamespace(type="thinking", thinking="deep thought", signature="sig_abc"),
                SimpleNamespace(type="text", text="Answer"),
            ],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=20,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )
        msg = anthropic_assistant_from_response(self._spec(), response)
        tb = [b for b in msg["content"] if b["type"] == "thinking"]
        assert len(tb) == 1
        assert tb[0]["thinking"] == "deep thought"
        assert tb[0]["signature"] == "sig_abc"
        assert tb[0].get("redacted") is not True

    def test_redacted_thinking_block(self):
        response = SimpleNamespace(
            id="msg_r",
            content=[
                SimpleNamespace(type="redacted_thinking", data="encrypted_data"),
                SimpleNamespace(type="text", text="Answer"),
            ],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=5,
                output_tokens=5,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )
        msg = anthropic_assistant_from_response(self._spec(), response)
        tb = [b for b in msg["content"] if b["type"] == "thinking"]
        assert len(tb) == 1
        assert tb[0]["redacted"] is True
        assert tb[0]["signature"] == "encrypted_data"

    def test_tool_use_block(self):
        response = SimpleNamespace(
            id="msg_tu",
            content=[
                SimpleNamespace(type="tool_use", id="tu_1", name="search", input={"q": "test"}),
            ],
            stop_reason="tool_use",
            usage=SimpleNamespace(
                input_tokens=5,
                output_tokens=5,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )
        msg = anthropic_assistant_from_response(self._spec(), response)
        tc = [b for b in msg["content"] if b["type"] == "tool_call"]
        assert len(tc) == 1
        assert tc[0]["id"] == "tu_1"
        assert tc[0]["name"] == "search"
        assert tc[0]["arguments"] == {"q": "test"}
        assert msg["stop_reason"] == "tool_use"


class TestAnthropicConvertMessages:
    def _spec(self):
        return ModelSpec(provider="anthropic", model="claude-sonnet-4-6")

    def test_user_message(self):
        messages = [{"role": "user", "content": "hi"}]
        result = anthropic_convert_messages(self._spec(), messages)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "hi"}

    def test_assistant_with_thinking(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig1"},
                    {"type": "text", "text": "answer"},
                ],
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
            }
        ]
        result = anthropic_convert_messages(self._spec(), messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        blocks = result[0]["content"]
        # Should have thinking and text blocks
        assert any(b.get("type") == "thinking" for b in blocks)

    def test_tool_results_grouped(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "id": "tc1", "name": "foo", "arguments": {}},
                    {"type": "tool_call", "id": "tc2", "name": "bar", "arguments": {}},
                ],
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
            },
            {
                "role": "tool_result",
                "tool_call_id": "tc1",
                "tool_name": "foo",
                "content": "result1",
                "is_error": False,
            },
            {
                "role": "tool_result",
                "tool_call_id": "tc2",
                "tool_name": "bar",
                "content": "result2",
                "is_error": False,
            },
        ]
        result = anthropic_convert_messages(self._spec(), messages)
        # Should have 2 messages: assistant, then user (grouped tool results)
        assert len(result) == 2
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "user"
        # The user message content should be a list of tool_results
        assert isinstance(result[1]["content"], list)
        assert len(result[1]["content"]) == 2


class TestAnthropicConvertTools:
    def test_json_schema_wrapping(self):
        tools = [
            {
                "name": "search",
                "description": "Search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ]
        result = anthropic_convert_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert result[0]["description"] == "Search the web"
        assert result[0]["input_schema"]["type"] == "object"

    def test_empty_tools(self):
        assert anthropic_convert_tools([]) == []
        assert anthropic_convert_tools(None) == []

    def test_missing_optional_fields(self):
        tools = [{"name": "basic"}]
        result = anthropic_convert_tools(tools)
        assert result[0]["description"] == ""
        assert result[0]["input_schema"] == {"type": "object", "properties": {}}


# ════════════════════════════════════════════════════════════════════════════════
# §6  providers/_anthropic_reasoning.py
# ════════════════════════════════════════════════════════════════════════════════

from jyagent.llm.providers._anthropic_reasoning import (
    validate_anthropic_reasoning,
    build_anthropic_request_reasoning,
)


class TestValidateAnthropicReasoning:
    def test_disabled_config(self):
        result = validate_anthropic_reasoning({"type": "disabled"}, model="claude-sonnet-4-6")
        assert result["type"] == "disabled"

    def test_adaptive_config(self):
        result = validate_anthropic_reasoning(
            {"type": "adaptive"},
            model="claude-sonnet-4-6",
        )
        assert result["type"] == "adaptive"
        # Should default effort to medium
        assert result.get("effort") == "medium"

    def test_budget_tokens_migration_error(self):
        with pytest.raises(ValueError, match="budget_tokens"):
            validate_anthropic_reasoning(
                {"budget_tokens": 5000},
                model="claude-sonnet-4-6",
            )

    def test_enabled_type_migration_error(self):
        """Old 'enabled' type should trigger migration error."""
        with pytest.raises(ValueError, match="budget"):
            validate_anthropic_reasoning(
                {"type": "enabled"},
                model="claude-sonnet-4-6",
            )

    def test_effort_validation_valid(self):
        for effort in ("low", "medium", "high"):
            result = validate_anthropic_reasoning(
                {"effort": effort},
                model="claude-sonnet-4-6",
            )
            assert result["effort"] == effort

    def test_effort_max_only_opus(self):
        """effort 'max' only supported by claude-opus-4-6."""
        with pytest.raises(ValueError, match="max"):
            validate_anthropic_reasoning(
                {"effort": "max"},
                model="claude-sonnet-4-6",
            )

    def test_effort_max_opus_accepted(self):
        result = validate_anthropic_reasoning(
            {"effort": "max"},
            model="claude-opus-4-6",
        )
        assert result["effort"] == "max"

    def test_invalid_effort_rejected(self):
        with pytest.raises(ValueError, match="effort"):
            validate_anthropic_reasoning(
                {"effort": "ultra"},
                model="claude-sonnet-4-6",
            )

    def test_non_dict_rejected(self):
        with pytest.raises(ValueError, match="dict"):
            validate_anthropic_reasoning("not a dict")

    def test_unsupported_keys_rejected(self):
        with pytest.raises(ValueError, match="unsupported keys"):
            validate_anthropic_reasoning(
                {"type": "adaptive", "unknown_key": True},
                model="claude-sonnet-4-6",
            )

    def test_disabled_with_extra_fields_error(self):
        with pytest.raises(ValueError, match="disabled"):
            validate_anthropic_reasoning(
                {"type": "disabled", "effort": "high"},
                model="claude-sonnet-4-6",
            )

    def test_adaptive_unsupported_model(self):
        with pytest.raises(ValueError, match="adaptive"):
            validate_anthropic_reasoning(
                {"type": "adaptive"},
                model="claude-3-opus-20240229",
            )

    def test_effort_unsupported_model(self):
        with pytest.raises(ValueError, match="not supported"):
            validate_anthropic_reasoning(
                {"effort": "high"},
                model="claude-3-sonnet",
            )

    def test_effort_only_implies_adaptive_on_46(self):
        """If only effort is given on a 4-6 model, type is auto-resolved to adaptive."""
        result = validate_anthropic_reasoning(
            {"effort": "high"},
            model="claude-sonnet-4-6",
        )
        assert result.get("type") == "adaptive"
        assert result["effort"] == "high"

    def test_display_requires_adaptive(self):
        with pytest.raises(ValueError, match="display"):
            validate_anthropic_reasoning(
                {"display": "summarized", "effort": "high"},
                model="claude-opus-4-5",
            )

    def test_adaptive_with_display(self):
        result = validate_anthropic_reasoning(
            {"type": "adaptive", "display": "summarized"},
            model="claude-sonnet-4-6",
        )
        assert result["display"] == "summarized"

    def test_adaptive_claude_opus_4_7_accepted(self):
        result = validate_anthropic_reasoning(
            {"type": "adaptive"},
            model="claude-opus-4-7",
        )
        assert result["type"] == "adaptive"
        assert result["effort"] == "medium"

    def test_effort_max_claude_opus_4_7_accepted(self):
        result = validate_anthropic_reasoning(
            {"effort": "max"},
            model="claude-opus-4-7",
        )
        assert result["effort"] == "max"
        assert result.get("type") == "adaptive"

    def test_adaptive_future_sonnet_4_7_accepted(self):
        result = validate_anthropic_reasoning(
            {"type": "adaptive"},
            model="claude-sonnet-4-7",
        )
        assert result["type"] == "adaptive"

    def test_adaptive_future_minor_4_20_accepted(self):
        result = validate_anthropic_reasoning(
            {"type": "adaptive"},
            model="claude-opus-4-20",
        )
        assert result["type"] == "adaptive"

    def test_model_with_date_suffix_accepted(self):
        result = validate_anthropic_reasoning(
            {"type": "adaptive"},
            model="claude-opus-4-7-20260101",
        )
        assert result["type"] == "adaptive"

    def test_opus_4_5_still_rejects_adaptive(self):
        with pytest.raises(ValueError, match="adaptive"):
            validate_anthropic_reasoning(
                {"type": "adaptive"},
                model="claude-opus-4-5",
            )


class TestBuildAnthropicRequestReasoning:
    def test_disabled(self):
        thinking, output_config = build_anthropic_request_reasoning(
            {"type": "disabled"},
            model="claude-sonnet-4-6",
        )
        assert thinking == {"type": "disabled"}
        assert output_config is None

    def test_adaptive_with_effort(self):
        thinking, output_config = build_anthropic_request_reasoning(
            {"type": "adaptive", "effort": "high"},
            model="claude-sonnet-4-6",
        )
        assert thinking == {"type": "adaptive"}
        assert output_config == {"effort": "high"}

    def test_adaptive_with_display(self):
        thinking, output_config = build_anthropic_request_reasoning(
            {"type": "adaptive", "display": "summarized"},
            model="claude-sonnet-4-6",
        )
        assert thinking == {"type": "adaptive", "display": "summarized"}

    def test_effort_only(self):
        """Effort-only config on opus-4-5 produces no thinking, just output_config."""
        thinking, output_config = build_anthropic_request_reasoning(
            {"effort": "high"},
            model="claude-opus-4-5",
        )
        assert thinking is None
        assert output_config == {"effort": "high"}


# ════════════════════════════════════════════════════════════════════════════════
# §7  core.py — adapter registry & LLMOwner
# ════════════════════════════════════════════════════════════════════════════════

from jyagent.llm.core import register_adapter, get_adapter, list_adapters, _ADAPTERS


class TestAdapterRegistry:
    """Tests for register_adapter / get_adapter / list_adapters."""

    def setup_method(self):
        """Save & restore the adapter registry around each test."""
        self._saved = dict(_ADAPTERS)

    def teardown_method(self):
        _ADAPTERS.clear()
        _ADAPTERS.update(self._saved)

    def test_register_and_get(self):
        mock_adapter = MagicMock()
        mock_adapter.provider = "test_provider"
        mock_adapter.api_name = "test-api"
        register_adapter(mock_adapter)
        assert get_adapter("test_provider") is mock_adapter

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_adapter("nonexistent_provider_xyz")

    def test_list_adapters(self):
        mock1 = MagicMock()
        mock1.provider = "aaa"
        mock2 = MagicMock()
        mock2.provider = "zzz"
        register_adapter(mock1)
        register_adapter(mock2)
        adapters = list_adapters()
        assert "aaa" in adapters
        assert "zzz" in adapters
        # list should be sorted
        assert adapters == sorted(adapters)


class TestRuntimeOwnerCompleteText:
    """Test LLMOwner.complete_text convenience method (mock adapter)."""

    def test_complete_text_returns_concatenated_text(self):
        from jyagent.llm.core import LLMOwner

        mock_adapter = MagicMock()
        mock_adapter.provider = "test_rt"
        mock_adapter.api_name = "test-api"
        mock_adapter.complete.return_value = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "World"},
            ],
            "provider": "test_rt",
            "model": "test-model",
            "stop_reason": "stop",
            "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        }

        saved = dict(_ADAPTERS)
        try:
            register_adapter(mock_adapter)
            # Patch config functions to bypass env-dependent validation
            with patch("jyagent.llm.core.build_model_spec") as mock_bms, \
                 patch("jyagent.llm.core.get_reasoning_config_for_provider") as mock_grc:
                mock_bms.return_value = ModelSpec(provider="test_rt", model="test-model")
                mock_grc.return_value = None
                owner = LLMOwner(ModelSpec(provider="test_rt", model="test-model"))
                result = owner.complete_text("Say hello")
            assert result == "Hello World"
            mock_adapter.complete.assert_called_once()
        finally:
            _ADAPTERS.clear()
            _ADAPTERS.update(saved)

    def test_complete_text_with_only_thinking_returns_empty(self):
        from jyagent.llm.core import LLMOwner

        mock_adapter = MagicMock()
        mock_adapter.provider = "test_rt2"
        mock_adapter.api_name = "test-api"
        mock_adapter.complete.return_value = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "internal reasoning"},
            ],
            "provider": "test_rt2",
            "model": "test-model2",
            "stop_reason": "stop",
            "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        }

        saved = dict(_ADAPTERS)
        try:
            register_adapter(mock_adapter)
            with patch("jyagent.llm.core.build_model_spec") as mock_bms, \
                 patch("jyagent.llm.core.get_reasoning_config_for_provider") as mock_grc:
                mock_bms.return_value = ModelSpec(provider="test_rt2", model="test-model2")
                mock_grc.return_value = None
                owner = LLMOwner(ModelSpec(provider="test_rt2", model="test-model2"))
                result = owner.complete_text("Think about it")
            assert result == ""
        finally:
            _ADAPTERS.clear()
            _ADAPTERS.update(saved)


# ════════════════════════════════════════════════════════════════════════════════
# §8  Edge cases & integration-style tests
# ════════════════════════════════════════════════════════════════════════════════

class TestTransformAndInjectIntegration:
    """End-to-end tests combining transform + inject."""

    def test_cross_model_replay_with_dangling_tool_call(self):
        """Cross-model replay: thinking converted, dangling tool_call gets synthetic result."""
        target = ModelSpec(provider="openai", model="gpt-5.4")
        messages = [
            {"role": "user", "content": "do something"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "planning..."},
                    {"type": "text", "text": "I'll use a tool"},
                    {"type": "tool_call", "id": "tc1", "name": "search", "arguments": {"q": "test"}},
                ],
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
            },
            # No tool_result for tc1
        ]
        result = transform_messages_for_target(messages, target)
        # Should have: user, assistant (thinking->text), synthetic tool_result
        user_msgs = [m for m in result if m.get("role") == "user"]
        assistant_msgs = [m for m in result if m.get("role") == "assistant"]
        tool_results = [m for m in result if m.get("role") == "tool_result"]
        assert len(user_msgs) == 1
        assert len(assistant_msgs) == 1
        assert len(tool_results) == 1
        assert tool_results[0]["is_error"] is True
        # Thinking should have been converted to text (cross-model)
        thinking_blocks = [
            b for b in assistant_msgs[0]["content"] if b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) == 0  # No raw thinking
        text_blocks = [
            b for b in assistant_msgs[0]["content"] if b.get("type") == "text"
        ]
        thinking_as_text = [b for b in text_blocks if "<thinking>" in b["text"]]
        assert len(thinking_as_text) == 1

    def test_encrypted_thinking_without_text_dropped_cross_model(self):
        """Encrypted thinking (no text) should be dropped for cross-model replay."""
        target = ModelSpec(provider="openai", model="gpt-5.4")
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "",
                        "encrypted_content": "encrypted_blob",
                    },
                    {"type": "text", "text": "response"},
                ],
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
            },
        ]
        result = transform_messages_for_target(messages, target)
        assistant = result[0]
        # Encrypted thinking with empty text -> dropped
        thinking_blocks = [b for b in assistant["content"] if b.get("type") == "thinking"]
        assert len(thinking_blocks) == 0
        # Only the text block remains
        assert len(assistant["content"]) == 1
        assert assistant["content"][0]["type"] == "text"


class TestSupportsOpenAIReasoningEffort:
    def test_gpt_5_4_supported(self):
        assert supports_openai_reasoning_effort("gpt-5.4") is True

    def test_gpt_5_4_variant_supported(self):
        assert supports_openai_reasoning_effort("gpt-5.4-turbo") is True

    def test_gpt_5_5_supported(self):
        assert supports_openai_reasoning_effort("gpt-5.5") is True

    def test_gpt_5_5_variant_supported(self):
        assert supports_openai_reasoning_effort("gpt-5.5-mini") is True

    def test_gpt_5_higher_minor_supported(self):
        assert supports_openai_reasoning_effort("gpt-5.10") is True

    def test_gpt_5_3_not_supported(self):
        assert supports_openai_reasoning_effort("gpt-5.3") is False

    def test_gpt_4o_not_supported(self):
        assert supports_openai_reasoning_effort("gpt-4o") is False

    def test_case_insensitive(self):
        assert supports_openai_reasoning_effort("GPT-5.4") is True

    def test_whitespace_stripped(self):
        assert supports_openai_reasoning_effort("  gpt-5.4  ") is True

    def test_malformed_no_minor_not_supported(self):
        assert supports_openai_reasoning_effort("gpt-5.") is False
        assert supports_openai_reasoning_effort("gpt-5.4x") is False
