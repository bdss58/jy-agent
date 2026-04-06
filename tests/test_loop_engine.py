"""Comprehensive unit tests for jyagent.loop_engine."""

from __future__ import annotations

import json
import time
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, call

from jyagent.loop_engine import (
    AgentLoop,
    LoopCallbacks,
    LoopConfig,
    LoopResult,
    ToolCallRequest,
    _compact_messages,
    _extract_text,
    _extract_tool_calls,
    _is_transient_error,
    _is_truncated,
    _truncate_result,
)
from jyagent.toolresult import ToolResult


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_assistant_message(
    text: str = "",
    tool_calls: list[dict] | None = None,
    stop_reason: str = "stop",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> dict:
    """Build a minimal AssistantMessage dict for testing."""
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for tc in (tool_calls or []):
        content.append({
            "type": "tool_call",
            "id": tc.get("id", "tc_1"),
            "name": tc["name"],
            "arguments": tc.get("arguments", {}),
        })
    return {
        "role": "assistant",
        "content": content,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _mock_runtime_owner(complete_side_effect=None, stream_side_effect=None):
    """Create a MagicMock RuntimeOwner with a mock model_spec."""
    owner = MagicMock()
    owner.model_spec = MagicMock()
    owner.model_spec.provider = "anthropic"
    owner.model_spec.model = "claude-sonnet-4-6"
    if complete_side_effect is not None:
        owner.complete.side_effect = complete_side_effect
    if stream_side_effect is not None:
        owner.stream.side_effect = stream_side_effect
    return owner


def _mock_registry(
    schemas=None,
    functions=None,
    parallel_safe_tools=None,
    timeout_hints=None,
):
    """Create a MagicMock registry."""
    reg = MagicMock()
    schemas = schemas or []
    functions = functions or {}
    reg.snapshot.return_value = (1, schemas, functions)
    reg.version = 1

    parallel_safe_set = set(parallel_safe_tools or [])
    reg.is_parallel_safe.side_effect = lambda name: name in parallel_safe_set

    hints = timeout_hints or {}
    reg.get_timeout_hint.side_effect = lambda name: hints.get(name)
    reg.get_schema.return_value = None

    return reg


# ─── Test cases ──────────────────────────────────────────────────────────────

class TestExtractText(unittest.TestCase):
    def test_basic(self):
        msg = _make_assistant_message(text="Hello world")
        self.assertEqual(_extract_text(msg), "Hello world")

    def test_empty(self):
        msg = _make_assistant_message()
        self.assertEqual(_extract_text(msg), "")


class TestExtractToolCalls(unittest.TestCase):
    def test_single(self):
        msg = _make_assistant_message(tool_calls=[{"name": "read_file", "id": "tc1", "arguments": {"path": "a.py"}}])
        calls = _extract_tool_calls(msg)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "read_file")
        self.assertEqual(calls[0].id, "tc1")
        self.assertEqual(calls[0].input, {"path": "a.py"})


class TestIsTruncated(unittest.TestCase):
    def test_truncated(self):
        calls = [ToolCallRequest(id="1", name="t", input={})]
        self.assertTrue(_is_truncated("length", calls))

    def test_not_truncated(self):
        calls = [ToolCallRequest(id="1", name="t", input={})]
        self.assertFalse(_is_truncated("stop", calls))
        self.assertFalse(_is_truncated("length", []))


class TestTruncateResult(unittest.TestCase):
    def test_short_result(self):
        self.assertEqual(_truncate_result("short", 100), "short")

    def test_long_result(self):
        content = "x" * 200
        result = _truncate_result(content, 100)
        self.assertIn("truncated", result)
        self.assertLess(len(result), 200)

    def test_error_not_truncated(self):
        content = "x" * 200
        result = _truncate_result(content, 100, is_error=True)
        self.assertEqual(result, content)


class TestIsTransientError(unittest.TestCase):
    def test_json_decode_error(self):
        err = json.JSONDecodeError("msg", "doc", 0)
        self.assertTrue(_is_transient_error(err))

    def test_timeout(self):
        self.assertTrue(_is_transient_error(Exception("connection timeout")))

    def test_overloaded(self):
        self.assertTrue(_is_transient_error(Exception("API overloaded")))

    def test_529(self):
        self.assertTrue(_is_transient_error(Exception("529 server error")))

    def test_non_transient(self):
        self.assertFalse(_is_transient_error(ValueError("invalid input")))


class TestCompactMessages(unittest.TestCase):
    def test_no_compaction_needed(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = _compact_messages(msgs, 100_000, 2000)
        self.assertIs(result, msgs)  # same list object — no compaction

    def test_compaction_triggers(self):
        # Create messages with a big tool result
        big_content = "x" * 50_000
        msgs = [
            {"role": "tool_result", "content": big_content, "tool_call_id": "t1", "tool_name": "read_file", "is_error": False},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": "next"},
        ]
        result = _compact_messages(msgs, 100, 200)  # very low max_tokens
        self.assertIsNot(result, msgs)
        # First message should be compacted
        self.assertIn("compacted", result[0]["content"])
        self.assertLess(len(result[0]["content"]), len(big_content))


# ─── AgentLoop integration tests ────────────────────────────────────────────

@patch("jyagent.loop_engine.get_registry")
@patch("jyagent.loop_engine.get_reasoning_config_for_provider", return_value=None)
class TestSimpleCompletion(unittest.TestCase):
    """Test 1: Simple completion with no tools — returns text."""

    def test_simple_text_response(self, mock_reasoning, mock_get_reg):
        mock_get_reg.return_value = _mock_registry()

        msg = _make_assistant_message(text="Hello!", input_tokens=5, output_tokens=15)
        owner = _mock_runtime_owner(complete_side_effect=[msg])

        loop = AgentLoop(owner, LoopConfig(streaming=False))
        result = loop.run("system", [{"role": "user", "content": "hi"}])

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.text, "Hello!")
        self.assertEqual(result.final_text, "Hello!")
        self.assertEqual(result.steps, 1)
        self.assertEqual(result.total_input_tokens, 5)
        self.assertEqual(result.total_output_tokens, 15)
        self.assertEqual(result.tool_calls_count, 0)


@patch("jyagent.loop_engine.get_registry")
@patch("jyagent.loop_engine.get_reasoning_config_for_provider", return_value=None)
class TestSingleToolCallLoop(unittest.TestCase):
    """Test 2: Single tool call — calls tool, returns final text."""

    def test_tool_call_then_text(self, mock_reasoning, mock_get_reg):
        def my_tool(path: str) -> ToolResult:
            return ToolResult(f"content of {path}")

        reg = _mock_registry(functions={"read_file": my_tool})
        mock_get_reg.return_value = reg

        # First call: assistant asks to use a tool
        tool_msg = _make_assistant_message(
            tool_calls=[{"name": "read_file", "id": "tc1", "arguments": {"path": "a.py"}}],
            stop_reason="tool_use",
        )
        # Second call: assistant returns final text
        final_msg = _make_assistant_message(text="Done reading!")

        owner = _mock_runtime_owner(complete_side_effect=[tool_msg, final_msg])

        loop = AgentLoop(owner, LoopConfig(streaming=False))
        messages = [{"role": "user", "content": "read a.py"}]
        result = loop.run("system", messages)

        self.assertEqual(result.status, "completed")
        self.assertIn("Done reading!", result.text)
        self.assertEqual(result.tool_calls_count, 1)
        self.assertEqual(result.steps, 2)


@patch("jyagent.loop_engine.get_registry")
@patch("jyagent.loop_engine.get_reasoning_config_for_provider", return_value=None)
class TestMultiStepToolLoop(unittest.TestCase):
    """Test 3: Multi-step tool loop — multiple tool rounds."""

    def test_three_step_loop(self, mock_reasoning, mock_get_reg):
        def my_tool(**kwargs) -> ToolResult:
            return ToolResult("ok")

        reg = _mock_registry(functions={"do_thing": my_tool})
        mock_get_reg.return_value = reg

        tc = {"name": "do_thing", "id": "tc1", "arguments": {}}
        tool_msg1 = _make_assistant_message(tool_calls=[tc], stop_reason="tool_use")
        tool_msg2 = _make_assistant_message(tool_calls=[tc], stop_reason="tool_use")
        final_msg = _make_assistant_message(text="All done")

        owner = _mock_runtime_owner(complete_side_effect=[tool_msg1, tool_msg2, final_msg])

        loop = AgentLoop(owner, LoopConfig(streaming=False))
        result = loop.run("system", [{"role": "user", "content": "go"}])

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.steps, 3)
        self.assertEqual(result.tool_calls_count, 2)


@patch("jyagent.loop_engine.get_registry")
@patch("jyagent.loop_engine.get_reasoning_config_for_provider", return_value=None)
class TestTruncationRecovery(unittest.TestCase):
    """Test 4: Truncation recovery — scales tokens on length + tool_calls."""

    def test_scales_up_on_truncation(self, mock_reasoning, mock_get_reg):
        reg = _mock_registry()
        mock_get_reg.return_value = reg

        # First call: truncated (stop_reason=length with tool calls)
        truncated_msg = _make_assistant_message(
            text="partial",
            tool_calls=[{"name": "read_file", "id": "tc1", "arguments": {}}],
            stop_reason="length",
        )
        # Second call: successful completion
        final_msg = _make_assistant_message(text="Full response")

        owner = _mock_runtime_owner(complete_side_effect=[truncated_msg, final_msg])

        config = LoopConfig(
            streaming=False,
            initial_max_tokens=1000,
            token_scale_factor=2,
            max_tokens_cap=128_000,
        )
        loop = AgentLoop(owner, config)
        result = loop.run("system", [{"role": "user", "content": "go"}])

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.text, "Full response")

        # Verify that the second call used a higher max_tokens
        calls = owner.complete.call_args_list
        self.assertEqual(len(calls), 2)
        first_opts = calls[0][1].get("options") or calls[0][0][1] if len(calls[0][0]) > 1 else calls[0][1]["options"]
        second_opts = calls[1][1].get("options") or calls[1][0][1] if len(calls[1][0]) > 1 else calls[1][1]["options"]
        self.assertEqual(first_opts.max_output_tokens, 1000)
        self.assertEqual(second_opts.max_output_tokens, 2000)


@patch("jyagent.loop_engine.get_registry")
@patch("jyagent.loop_engine.get_reasoning_config_for_provider", return_value=None)
class TestContextCompaction(unittest.TestCase):
    """Test 5: Context compaction triggers when messages are large."""

    def test_compaction_callback_fires(self, mock_reasoning, mock_get_reg):
        reg = _mock_registry()
        mock_get_reg.return_value = reg

        final_msg = _make_assistant_message(text="ok")
        owner = _mock_runtime_owner(complete_side_effect=[final_msg])

        on_compaction = MagicMock()
        callbacks = LoopCallbacks(on_compaction=on_compaction)

        # Create messages with a very large tool result to trigger compaction
        big_content = "x" * 500_000
        messages = [
            {"role": "tool_result", "content": big_content, "tool_call_id": "t1", "tool_name": "f", "is_error": False},
            {"role": "assistant", "content": [{"type": "text", "text": "noted"}]},
            {"role": "user", "content": "next"},
        ]

        config = LoopConfig(
            streaming=False,
            max_working_tokens=100,  # very low to force compaction
            compact_tool_result_chars=200,
        )
        loop = AgentLoop(owner, config, callbacks=callbacks)
        result = loop.run("system", messages)

        self.assertEqual(result.status, "completed")
        on_compaction.assert_called()


@patch("jyagent.loop_engine.get_registry")
@patch("jyagent.loop_engine.get_reasoning_config_for_provider", return_value=None)
class TestRetryOnTransientError(unittest.TestCase):
    """Test 6: Retry on transient error — retries and succeeds."""

    def test_retry_succeeds(self, mock_reasoning, mock_get_reg):
        reg = _mock_registry()
        mock_get_reg.return_value = reg

        final_msg = _make_assistant_message(text="Success!")

        owner = _mock_runtime_owner(complete_side_effect=[
            Exception("connection timeout"),
            final_msg,
        ])

        on_retry = MagicMock()
        callbacks = LoopCallbacks(on_retry=on_retry)

        config = LoopConfig(
            streaming=False,
            retry_attempts=3,
            retry_base_delay=0.01,  # fast for testing
        )
        loop = AgentLoop(owner, config, callbacks=callbacks)
        result = loop.run("system", [{"role": "user", "content": "hi"}])

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.text, "Success!")
        on_retry.assert_called_once()
        self.assertEqual(on_retry.call_args[0][0], 1)  # attempt number


@patch("jyagent.loop_engine.get_registry")
@patch("jyagent.loop_engine.get_reasoning_config_for_provider", return_value=None)
class TestRetryExhaustion(unittest.TestCase):
    """Test 7: Retry exhaustion — all retries fail, returns error."""

    def test_all_retries_fail(self, mock_reasoning, mock_get_reg):
        reg = _mock_registry()
        mock_get_reg.return_value = reg

        owner = _mock_runtime_owner(complete_side_effect=[
            Exception("connection timeout"),
            Exception("connection timeout again"),
            Exception("connection timeout third"),
            Exception("connection timeout fourth"),
        ])

        config = LoopConfig(
            streaming=False,
            retry_attempts=3,
            retry_base_delay=0.01,
        )
        loop = AgentLoop(owner, config)
        result = loop.run("system", [{"role": "user", "content": "hi"}])

        self.assertEqual(result.status, "error")
        self.assertIn("timeout", result.error)


@patch("jyagent.loop_engine.get_registry")
@patch("jyagent.loop_engine.get_reasoning_config_for_provider", return_value=None)
class TestConcurrentToolExecution(unittest.TestCase):
    """Test 8: Concurrent tool execution — parallel-safe tools run concurrently."""

    def test_parallel_tools(self, mock_reasoning, mock_get_reg):
        call_times = []

        def slow_tool(path: str = "") -> ToolResult:
            call_times.append(time.time())
            time.sleep(0.05)
            return ToolResult(f"read {path}")

        reg = _mock_registry(
            functions={"read_file": slow_tool},
            parallel_safe_tools=["read_file"],
        )
        mock_get_reg.return_value = reg

        # Assistant requests 3 parallel read_file calls
        tool_msg = _make_assistant_message(
            tool_calls=[
                {"name": "read_file", "id": f"tc{i}", "arguments": {"path": f"{i}.py"}}
                for i in range(3)
            ],
            stop_reason="tool_use",
        )
        final_msg = _make_assistant_message(text="Done")
        owner = _mock_runtime_owner(complete_side_effect=[tool_msg, final_msg])

        config = LoopConfig(streaming=False, concurrent_tools=True)
        loop = AgentLoop(owner, config)
        result = loop.run("system", [{"role": "user", "content": "read 3 files"}])

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_calls_count, 3)
        # With parallelism, all should start roughly together
        if len(call_times) == 3:
            max_gap = max(call_times) - min(call_times)
            self.assertLess(max_gap, 0.5, "Parallel tools should start near-simultaneously")


@patch("jyagent.loop_engine.get_registry")
@patch("jyagent.loop_engine.get_reasoning_config_for_provider", return_value=None)
class TestMaxStepsReached(unittest.TestCase):
    """Test 9: Max steps reached — returns max_steps status."""

    def test_hits_max_steps(self, mock_reasoning, mock_get_reg):
        def my_tool(**kwargs) -> ToolResult:
            return ToolResult("ok")

        reg = _mock_registry(functions={"do_thing": my_tool})
        mock_get_reg.return_value = reg

        tc = {"name": "do_thing", "id": "tc1", "arguments": {}}
        tool_msg = _make_assistant_message(tool_calls=[tc], stop_reason="tool_use")

        # Always return tool calls — never completes
        owner = _mock_runtime_owner(complete_side_effect=[tool_msg] * 10)

        config = LoopConfig(streaming=False, max_steps=3)
        loop = AgentLoop(owner, config)
        result = loop.run("system", [{"role": "user", "content": "loop forever"}])

        self.assertEqual(result.status, "max_steps")
        self.assertEqual(result.steps, 3)


@patch("jyagent.loop_engine.get_registry")
@patch("jyagent.loop_engine.get_reasoning_config_for_provider", return_value=None)
class TestCallbacksFire(unittest.TestCase):
    """Test 10: Callbacks fire correctly — verify on_text_delta, on_tool_start, on_tool_end."""

    def test_callbacks(self, mock_reasoning, mock_get_reg):
        def my_tool(**kwargs) -> ToolResult:
            return ToolResult("result_data")

        reg = _mock_registry(functions={"my_tool": my_tool})
        mock_get_reg.return_value = reg

        tc = {"name": "my_tool", "id": "tc1", "arguments": {"x": 1}}
        tool_msg = _make_assistant_message(tool_calls=[tc], stop_reason="tool_use")
        final_msg = _make_assistant_message(text="Final answer")

        owner = _mock_runtime_owner(complete_side_effect=[tool_msg, final_msg])

        on_text_delta = MagicMock()
        on_tool_start = MagicMock()
        on_tool_end = MagicMock()
        on_usage = MagicMock()
        on_step_progress = MagicMock()

        callbacks = LoopCallbacks(
            on_text_delta=on_text_delta,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
            on_usage=on_usage,
            on_step_progress=on_step_progress,
        )

        config = LoopConfig(streaming=False)
        loop = AgentLoop(owner, config, callbacks=callbacks)
        result = loop.run("system", [{"role": "user", "content": "go"}])

        self.assertEqual(result.status, "completed")

        # on_text_delta called with final text
        on_text_delta.assert_called_with("Final answer")

        # on_tool_start + on_tool_end called once each
        on_tool_start.assert_called_once_with("my_tool", {"x": 1})
        on_tool_end.assert_called_once()
        end_args = on_tool_end.call_args[0]
        self.assertEqual(end_args[0], "my_tool")
        self.assertIn("result_data", end_args[1])
        self.assertFalse(end_args[2])  # is_error=False

        # on_usage called twice (once per LLM call)
        self.assertEqual(on_usage.call_count, 2)

        # on_step_progress called for each step
        self.assertGreaterEqual(on_step_progress.call_count, 2)


@patch("jyagent.loop_engine.get_registry")
@patch("jyagent.loop_engine.get_reasoning_config_for_provider", return_value=None)
class TestStreamingMode(unittest.TestCase):
    """Test 11: Streaming mode — processes stream events and fires callbacks."""

    def test_streaming_text(self, mock_reasoning, mock_get_reg):
        reg = _mock_registry()
        mock_get_reg.return_value = reg

        final_msg = _make_assistant_message(text="Hello world")

        # Build a mock stream
        events = [
            {"type": "start"},
            {"type": "thinking_start", "content_index": 0},
            {"type": "thinking_delta", "text": "hmm", "content_index": 0},
            {"type": "thinking_end", "content_index": 0},
            {"type": "text_start", "content_index": 1},
            {"type": "text_delta", "text": "Hello ", "content_index": 1},
            {"type": "text_delta", "text": "world", "content_index": 1},
            {"type": "text_end", "content_index": 1},
            {"type": "done", "message": final_msg},
        ]

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter(events))
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.get_final_message.return_value = final_msg

        owner = _mock_runtime_owner()
        owner.stream.return_value = mock_stream

        on_text_delta = MagicMock()
        on_thinking_start = MagicMock()
        on_thinking_stop = MagicMock()
        callbacks = LoopCallbacks(
            on_text_delta=on_text_delta,
            on_thinking_start=on_thinking_start,
            on_thinking_stop=on_thinking_stop,
        )

        config = LoopConfig(streaming=True)
        loop = AgentLoop(owner, config, callbacks=callbacks)
        result = loop.run("system", [{"role": "user", "content": "hi"}])

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.text, "Hello world")

        # Check text delta callbacks
        text_delta_calls = [c[0][0] for c in on_text_delta.call_args_list]
        self.assertEqual(text_delta_calls, ["Hello ", "world"])

        # Thinking callbacks
        on_thinking_start.assert_called()
        on_thinking_stop.assert_called()


@patch("jyagent.loop_engine.get_registry")
@patch("jyagent.loop_engine.get_reasoning_config_for_provider", return_value=None)
class TestStaticToolSource(unittest.TestCase):
    """Test 12: Static tool source — tool_source returns fixed tools."""

    def test_tool_source_used(self, mock_reasoning, mock_get_reg):
        reg = _mock_registry()
        mock_get_reg.return_value = reg

        def my_custom_tool(query: str = "") -> ToolResult:
            return ToolResult(f"searched: {query}")

        tool_schemas = [{"name": "search", "input_schema": {"type": "object", "properties": {}}}]
        tool_functions = {"search": my_custom_tool}

        def tool_source():
            return (tool_schemas, tool_functions)

        tc = {"name": "search", "id": "tc1", "arguments": {"query": "test"}}
        tool_msg = _make_assistant_message(tool_calls=[tc], stop_reason="tool_use")
        final_msg = _make_assistant_message(text="Found it")

        owner = _mock_runtime_owner(complete_side_effect=[tool_msg, final_msg])

        config = LoopConfig(streaming=False)
        loop = AgentLoop(owner, config, tool_source=tool_source)
        result = loop.run("system", [{"role": "user", "content": "search for test"}])

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_calls_count, 1)

        # Verify context was called with tool_schemas
        first_call = owner.complete.call_args_list[0]
        context_arg = first_call[0][0]
        self.assertEqual(context_arg["tools"], tool_schemas)


@patch("jyagent.loop_engine.get_registry")
@patch("jyagent.loop_engine.get_reasoning_config_for_provider", return_value=None)
class TestKeyboardInterruptHandling(unittest.TestCase):
    """Test 13: KeyboardInterrupt handling — returns interrupted status."""

    def test_interrupt_during_llm_call(self, mock_reasoning, mock_get_reg):
        reg = _mock_registry()
        mock_get_reg.return_value = reg

        owner = _mock_runtime_owner(complete_side_effect=KeyboardInterrupt)

        config = LoopConfig(streaming=False)
        loop = AgentLoop(owner, config)
        result = loop.run("system", [{"role": "user", "content": "hi"}])

        self.assertEqual(result.status, "interrupted")
        self.assertIn("Interrupted", result.text)


if __name__ == "__main__":
    unittest.main()
