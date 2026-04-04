# Tests for provider-neutral runtime transforms and adapters.

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jyagent.runtime.history import normalize_anthropic_tool_call_id, transform_messages_for_target
from jyagent.runtime.providers.anthropic import _assistant_from_response as anthropic_assistant_from_response
from jyagent.runtime.providers.openai import (
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
