from __future__ import annotations

import unittest
from unittest.mock import patch

import jyagent.config as config
from jyagent.runtime.providers._openai_helpers import (
    build_request_kwargs,
    convert_messages,
    convert_tools,
    convert_tool_choice,
    assistant_from_response,
    usage_from_response,
    map_stop_reason,
)
from jyagent.runtime.types import ModelSpec, RuntimeOptions


def _context(system_prompt: str = "system") -> dict:
    return {
        "system_prompt": system_prompt,
        "messages": [{"role": "user", "content": "hello"}],
    }


class TestOpenAIRequestKwargs(unittest.TestCase):
    """Tests for build_request_kwargs() targeting the Responses API."""

    def test_gpt_5_4_includes_reasoning_in_reasoning_param(self):
        kwargs = build_request_kwargs(
            ModelSpec(provider="openai", model="gpt-5.4"),
            _context(),
            RuntimeOptions(max_output_tokens=256, reasoning={"effort": "xhigh"}),
        )

        # Reasoning goes into the `reasoning` param, not `reasoning_effort`
        self.assertEqual(kwargs["reasoning"], {"effort": "xhigh"})
        self.assertNotIn("reasoning_effort", kwargs)
        # Responses API uses max_output_tokens (not max_tokens or max_completion_tokens)
        self.assertEqual(kwargs["max_output_tokens"], 256)
        self.assertNotIn("max_tokens", kwargs)
        self.assertNotIn("max_completion_tokens", kwargs)
        # System prompt goes into instructions, not input
        self.assertEqual(kwargs["instructions"], "system")
        # input items should contain only the user message
        self.assertEqual(len(kwargs["input"]), 1)
        self.assertEqual(kwargs["input"][0]["role"], "user")
        # No stream key — streaming is handled by .stream() method
        self.assertNotIn("stream", kwargs)
        # store: False
        self.assertFalse(kwargs["store"])

    def test_o3_uses_instructions_and_max_output_tokens(self):
        """In the Responses API, o-series models use instructions normally."""
        kwargs = build_request_kwargs(
            ModelSpec(provider="openai", model="o3"),
            _context(),
            RuntimeOptions(max_output_tokens=256, reasoning={"effort": "high"}),
        )

        # o-series doesn't support reasoning_effort (only gpt-5.4 does)
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertNotIn("reasoning", kwargs)
        # Responses API: max_output_tokens for all models
        self.assertEqual(kwargs["max_output_tokens"], 256)
        self.assertNotIn("max_tokens", kwargs)
        self.assertNotIn("max_completion_tokens", kwargs)
        # System prompt goes into instructions — no user-message hack needed
        self.assertEqual(kwargs["instructions"], "system")
        # input should only have the user message (no system-in-user hack)
        self.assertEqual(len(kwargs["input"]), 1)
        self.assertEqual(kwargs["input"][0]["role"], "user")
        # No stream key
        self.assertNotIn("stream", kwargs)

    def test_basic_structure(self):
        kwargs = build_request_kwargs(
            ModelSpec(provider="openai", model="gpt-4o"),
            _context("Be helpful"),
            RuntimeOptions(max_output_tokens=100),
        )

        self.assertEqual(kwargs["model"], "gpt-4o")
        self.assertEqual(kwargs["instructions"], "Be helpful")
        self.assertEqual(kwargs["max_output_tokens"], 100)
        self.assertIn("input", kwargs)
        self.assertNotIn("messages", kwargs)
        self.assertFalse(kwargs["store"])

    def test_no_system_prompt_omits_instructions(self):
        kwargs = build_request_kwargs(
            ModelSpec(provider="openai", model="gpt-4o"),
            {"system_prompt": "", "messages": [{"role": "user", "content": "hi"}]},
            RuntimeOptions(),
        )

        self.assertNotIn("instructions", kwargs)


class TestConvertMessages(unittest.TestCase):
    """Tests for convert_messages() producing Responses API input items."""

    def test_user_message(self):
        items = convert_messages(
            ModelSpec(provider="openai", model="gpt-4o"),
            [{"role": "user", "content": "hello"}],
        )
        self.assertEqual(items, [{"role": "user", "content": "hello"}])

    def test_assistant_text_message(self):
        items = convert_messages(
            ModelSpec(provider="openai", model="gpt-4o"),
            [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}],
        )
        self.assertEqual(items, [{"role": "assistant", "content": "hi"}])

    def test_assistant_with_tool_calls(self):
        items = convert_messages(
            ModelSpec(provider="openai", model="gpt-4o"),
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me search"},
                        {"type": "tool_call", "id": "call_1", "name": "search", "arguments": {"q": "test"}},
                    ],
                },
                # Include a matching tool_result to avoid synthetic injection
                {
                    "role": "tool_result",
                    "tool_call_id": "call_1",
                    "tool_name": "search",
                    "content": "result",
                    "is_error": False,
                },
            ],
        )
        # 3 items: assistant text, function_call, function_call_output
        self.assertEqual(len(items), 3)
        # First: assistant text message
        self.assertEqual(items[0], {"role": "assistant", "content": "Let me search"})
        # Second: function_call item
        self.assertEqual(items[1]["type"], "function_call")
        self.assertEqual(items[1]["call_id"], "call_1")
        self.assertEqual(items[1]["name"], "search")
        self.assertEqual(items[1]["arguments"], '{"q": "test"}')
        # Third: function_call_output
        self.assertEqual(items[2]["type"], "function_call_output")
        self.assertEqual(items[2]["call_id"], "call_1")
        self.assertEqual(items[2]["output"], "result")

    def test_tool_result_message(self):
        items = convert_messages(
            ModelSpec(provider="openai", model="gpt-4o"),
            [{
                "role": "tool_result",
                "tool_call_id": "call_1",
                "tool_name": "search",
                "content": "result data",
                "is_error": False,
            }],
        )
        self.assertEqual(items, [{
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "result data",
        }])

    def test_tool_result_error(self):
        items = convert_messages(
            ModelSpec(provider="openai", model="gpt-4o"),
            [{
                "role": "tool_result",
                "tool_call_id": "call_1",
                "tool_name": "search",
                "content": "not found",
                "is_error": True,
            }],
        )
        self.assertEqual(items[0]["output"], "[ERROR] not found")


class TestConvertTools(unittest.TestCase):
    """Tests for convert_tools() producing Responses API flat tool format."""

    def test_basic_tool(self):
        tools = convert_tools([{
            "name": "search",
            "description": "Search the web",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        }])
        self.assertEqual(len(tools), 1)
        tool = tools[0]
        # Flat format (not nested under "function")
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["name"], "search")
        self.assertEqual(tool["description"], "Search the web")
        self.assertTrue(tool["strict"])
        self.assertNotIn("function", tool)  # Not nested!
        # additionalProperties: false should be injected
        self.assertFalse(tool["parameters"]["additionalProperties"])


class TestConvertToolChoice(unittest.TestCase):
    """Tests for convert_tool_choice() producing Responses API format."""

    def test_auto(self):
        self.assertEqual(convert_tool_choice({"type": "auto"}), "auto")

    def test_any(self):
        self.assertEqual(convert_tool_choice({"type": "any"}), "required")

    def test_none(self):
        self.assertEqual(convert_tool_choice({"type": "none"}), "none")

    def test_specific_tool(self):
        result = convert_tool_choice({"type": "tool", "name": "search"})
        # Responses API: flat {"type": "function", "name": "search"}
        self.assertEqual(result, {"type": "function", "name": "search"})

    def test_null(self):
        self.assertIsNone(convert_tool_choice(None))


class TestMapStopReason(unittest.TestCase):
    """Tests for map_stop_reason() with Responses API status values."""

    def test_completed_no_tools(self):
        self.assertEqual(map_stop_reason("completed", has_tool_calls=False), "stop")

    def test_completed_with_tools(self):
        self.assertEqual(map_stop_reason("completed", has_tool_calls=True), "tool_use")

    def test_incomplete(self):
        self.assertEqual(map_stop_reason("incomplete"), "length")

    def test_failed(self):
        self.assertEqual(map_stop_reason("failed"), "error")

    def test_none(self):
        self.assertEqual(map_stop_reason(None), "stop")


class TestUsageFromResponse(unittest.TestCase):
    """Tests for usage_from_response() with Responses API ResponseUsage."""

    def test_none_usage(self):
        result = usage_from_response(None)
        self.assertEqual(result["input_tokens"], 0)
        self.assertEqual(result["output_tokens"], 0)
        self.assertEqual(result["total_tokens"], 0)

    def test_with_cached_tokens(self):
        class InputDetails:
            cached_tokens = 50

        class MockUsage:
            input_tokens = 100
            output_tokens = 200
            input_tokens_details = InputDetails()

        result = usage_from_response(MockUsage())
        self.assertEqual(result["input_tokens"], 100)
        self.assertEqual(result["output_tokens"], 200)
        self.assertEqual(result["total_tokens"], 300)
        self.assertEqual(result["cache_read_input_tokens"], 50)


class TestAssistantFromResponse(unittest.TestCase):
    """Tests for assistant_from_response() with Responses API Response objects."""

    def test_text_response(self):
        class OutputText:
            type = "output_text"
            text = "Hello!"

        class MessageItem:
            type = "message"
            content = [OutputText()]

        class MockUsage:
            input_tokens = 10
            output_tokens = 5
            input_tokens_details = None

        class MockResponse:
            id = "resp_123"
            output = [MessageItem()]
            usage = MockUsage()
            status = "completed"

        spec = ModelSpec(provider="openai", model="gpt-4o")
        msg = assistant_from_response(spec, MockResponse())

        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["api"], "openai-responses")
        self.assertEqual(msg["stop_reason"], "stop")
        self.assertEqual(msg["response_id"], "resp_123")
        self.assertEqual(len(msg["content"]), 1)
        self.assertEqual(msg["content"][0]["type"], "text")
        self.assertEqual(msg["content"][0]["text"], "Hello!")

    def test_function_call_response(self):
        class FunctionCallItem:
            type = "function_call"
            call_id = "call_abc"
            name = "search"
            arguments = '{"q": "test"}'

        class MockUsage:
            input_tokens = 10
            output_tokens = 15
            input_tokens_details = None

        class MockResponse:
            id = "resp_456"
            output = [FunctionCallItem()]
            usage = MockUsage()
            status = "completed"

        spec = ModelSpec(provider="openai", model="gpt-4o")
        msg = assistant_from_response(spec, MockResponse())

        self.assertEqual(msg["stop_reason"], "tool_use")
        self.assertEqual(len(msg["content"]), 1)
        tc = msg["content"][0]
        self.assertEqual(tc["type"], "tool_call")
        self.assertEqual(tc["id"], "call_abc")
        self.assertEqual(tc["name"], "search")
        self.assertEqual(tc["arguments"], {"q": "test"})


class TestOpenAIReasoningConfig(unittest.TestCase):
    @patch.dict("os.environ", {"OPENAI_REASONING_EFFORT": "xhigh"}, clear=False)
    def test_gpt_5_4_accepts_xhigh(self):
        with (
            patch.object(config, "AGENT_PROVIDER", "openai"),
            patch.object(config, "AGENT_MODEL", "gpt-5.4"),
            patch.object(config, "SUPPORTED_RUNTIME_PROVIDERS", {"anthropic", "openai"}),
        ):
            self.assertEqual(
                config.get_reasoning_config_for_provider("openai", model="gpt-5.4"),
                {"effort": "xhigh"},
            )

    @patch.dict("os.environ", {"OPENAI_REASONING_EFFORT": "none"}, clear=False)
    def test_gpt_5_4_accepts_none(self):
        with (
            patch.object(config, "AGENT_PROVIDER", "openai"),
            patch.object(config, "AGENT_MODEL", "gpt-5.4-mini"),
            patch.object(config, "SUPPORTED_RUNTIME_PROVIDERS", {"anthropic", "openai"}),
        ):
            self.assertEqual(
                config.get_reasoning_config_for_provider("openai", model="gpt-5.4-mini"),
                {"effort": "none"},
            )

    @patch.dict("os.environ", {"OPENAI_REASONING_EFFORT": "ultra"}, clear=False)
    def test_gpt_5_4_rejects_unsupported_effort(self):
        with (
            patch.object(config, "AGENT_PROVIDER", "openai"),
            patch.object(config, "AGENT_MODEL", "gpt-5.4"),
            patch.object(config, "SUPPORTED_RUNTIME_PROVIDERS", {"anthropic", "openai"}),
        ):
            with self.assertRaisesRegex(ValueError, "OPENAI_REASONING_EFFORT must be one of"):
                config.get_reasoning_config_for_provider("openai", model="gpt-5.4")

    @patch.dict("os.environ", {"OPENAI_REASONING_EFFORT": "medium"}, clear=False)
    def test_non_gpt_5_4_model_rejects_reasoning_effort(self):
        with (
            patch.object(config, "AGENT_PROVIDER", "openai"),
            patch.object(config, "AGENT_MODEL", "gpt-4o"),
            patch.object(config, "SUPPORTED_RUNTIME_PROVIDERS", {"anthropic", "openai"}),
        ):
            with self.assertRaisesRegex(ValueError, "only supported for OpenAI model 'gpt-5.4'"):
                config.get_reasoning_config_for_provider("openai", model="gpt-4o")
