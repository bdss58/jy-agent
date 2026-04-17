from __future__ import annotations

import inspect
import json
import threading
import time
from unittest.mock import patch, MagicMock

import pytest

from jyagent.tools.subagent import (
    TOOL_SCHEMA,
    CHECK_AGENT_SCHEMA,
    dispatch_agent,
    check_agent,
    _bg_registry,
    _make_subagent_outcome,
    _SUBAGENT_STATUS_COMPLETED,
)


def test_dispatch_agent_schema_does_not_expose_named_presets():
    props = TOOL_SCHEMA["input_schema"]["properties"]

    assert "agent" not in props
    assert "task" in props
    assert "tool_whitelist" in props


def test_dispatch_agent_signature_does_not_accept_agent():
    params = inspect.signature(dispatch_agent).parameters

    assert "agent" not in params


# ─── Helpers for background sub-agent tests ────────────────────────────────

def _completed_outcome(content="done"):
    """Build a successful outcome dict (as returned by _run_subagent)."""
    return _make_subagent_outcome(
        _SUBAGENT_STATUS_COMPLETED, content, steps=1,
        input_tokens=100, output_tokens=50, tool_calls=0,
    )


@pytest.fixture(autouse=True)
def _clean_bg_registry():
    """Ensure the background registry is empty before and after each test."""
    _bg_registry.cancel_all()
    yield
    _bg_registry.cancel_all()


@pytest.fixture(autouse=True)
def _patch_subagent_deps():
    """Patch external dependencies that dispatch_agent touches so tests
    don't need a real Anthropic client or tool registry."""
    mock_registry = MagicMock()
    mock_registry.snapshot.return_value = (0, [], {})

    with (
        patch("jyagent.tools.subagent.get_registry", return_value=mock_registry),
        patch("jyagent.tools.subagent._get_memory_context", return_value=""),
        patch("jyagent.tools.subagent.get_subagent_model_spec") as mock_spec,
        patch("jyagent.tools.subagent._get_runtime_owner") as mock_owner,
        patch("jyagent.tools.subagent.get_stats") as mock_stats,
    ):
        # Provide a minimal model_spec
        spec = MagicMock()
        spec.model = "test-model"
        spec.provider = "anthropic"
        mock_spec.return_value = spec
        mock_owner.return_value = MagicMock(model_spec=spec)
        mock_stats.return_value = MagicMock()
        yield


# ─── Background sub-agent tests ───────────────────────────────────────────


class TestBackgroundFastFinishReturnsInline:
    """dispatch_agent(background=True) where the sub-agent finishes within
    the grace period should return the result inline (not a dispatched JSON)."""

    def test_fast_finish(self):
        outcome = _completed_outcome("fast result")

        with (
            patch("jyagent.tools.subagent._run_subagent", return_value=outcome),
            patch("jyagent.tools.subagent._BG_GRACE_PERIOD", 5),
        ):
            result = dispatch_agent(task="quick task", background=True)

        # Should return the answer inline, not a dispatched envelope
        assert not result.is_error
        assert "fast result" in result.content
        # Must NOT contain dispatch metadata
        assert "dispatched" not in result.content
        assert "agent_id" not in result.content


class TestBackgroundSlowReturnsAgentId:
    """dispatch_agent(background=True) where the sub-agent takes longer
    than the grace period should return JSON with status=dispatched."""

    def test_slow_dispatch(self):
        gate = threading.Event()

        def slow_subagent(*args, **kwargs):
            gate.wait(timeout=10)
            return _completed_outcome("late result")

        with (
            patch("jyagent.tools.subagent._run_subagent", side_effect=slow_subagent),
            patch("jyagent.tools.subagent._BG_GRACE_PERIOD", 0.1),
        ):
            result = dispatch_agent(task="slow task", background=True)
            gate.set()  # unblock the worker so it can clean up

        payload = json.loads(result.content)
        assert payload["status"] == "dispatched"
        assert "agent_id" in payload
        assert isinstance(payload["agent_id"], int)


class TestCheckAgentReturnsResultWhenDone:
    """After a background agent completes, check_agent(agent_id) should
    return the final result text."""

    def test_result_when_done(self):
        gate = threading.Event()

        def blocking_subagent(*args, **kwargs):
            gate.wait(timeout=10)
            return _completed_outcome("the answer is 42")

        with (
            patch("jyagent.tools.subagent._run_subagent", side_effect=blocking_subagent),
            patch("jyagent.tools.subagent._BG_GRACE_PERIOD", 0.1),
        ):
            result = dispatch_agent(task="bg task", background=True)

        payload = json.loads(result.content)
        agent_id = payload["agent_id"]

        # Now let the agent finish
        gate.set()
        # Give the thread a moment to complete
        time.sleep(0.3)

        check_result = check_agent(agent_id=agent_id, action="status")
        assert "the answer is 42" in check_result.content
        assert not check_result.is_error


class TestCheckAgentRunningStatus:
    """check_agent on a still-running agent should return status=running
    with progress info."""

    def test_running_status(self):
        gate = threading.Event()

        def blocking_subagent(*args, **kwargs):
            gate.wait(timeout=10)
            return _completed_outcome("done")

        with (
            patch("jyagent.tools.subagent._run_subagent", side_effect=blocking_subagent),
            patch("jyagent.tools.subagent._BG_GRACE_PERIOD", 0.1),
        ):
            result = dispatch_agent(task="running task", background=True)

        payload = json.loads(result.content)
        agent_id = payload["agent_id"]

        # Agent is still running, check status
        check_result = check_agent(agent_id=agent_id, action="status")
        status_payload = json.loads(check_result.content)

        assert status_payload["status"] == "running"
        assert status_payload["agent_id"] == agent_id
        assert "elapsed_seconds" in status_payload
        assert "step" in status_payload
        assert "max_steps" in status_payload

        # Cleanup
        gate.set()


class TestCheckAgentKill:
    """check_agent(agent_id, action='kill') should cancel the agent."""

    def test_kill(self):
        gate = threading.Event()

        def blocking_subagent(task, context, model_spec, max_steps,
                              tool_schemas, tool_functions,
                              agent_id=None, custom_system_prompt=None,
                              cancel_event=None, progress_ids=None):
            # Block until cancelled or gate is set
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    break
                if gate.is_set():
                    break
                time.sleep(0.05)
            return _completed_outcome("killed result")

        with (
            patch("jyagent.tools.subagent._run_subagent", side_effect=blocking_subagent),
            patch("jyagent.tools.subagent._BG_GRACE_PERIOD", 0.1),
        ):
            result = dispatch_agent(task="killable task", background=True)

        payload = json.loads(result.content)
        agent_id = payload["agent_id"]

        kill_result = check_agent(agent_id=agent_id, action="kill")
        assert f"Agent {agent_id} cancelled" in kill_result.content
        assert not kill_result.is_error

        # Agent should be removed from registry
        assert _bg_registry.get(agent_id) is None

        # Cleanup
        gate.set()


class TestCheckAgentList:
    """check_agent(action='list') should return a list of active agents."""

    def test_list(self):
        gate = threading.Event()

        def blocking_subagent(*args, **kwargs):
            gate.wait(timeout=10)
            return _completed_outcome("done")

        with (
            patch("jyagent.tools.subagent._run_subagent", side_effect=blocking_subagent),
            patch("jyagent.tools.subagent._BG_GRACE_PERIOD", 0.1),
        ):
            r1 = dispatch_agent(task="list task one", background=True)
            r2 = dispatch_agent(task="list task two", background=True)

        id1 = json.loads(r1.content)["agent_id"]
        id2 = json.loads(r2.content)["agent_id"]

        list_result = check_agent(action="list")
        agents = json.loads(list_result.content)

        agent_ids = {a["agent_id"] for a in agents}
        assert id1 in agent_ids
        assert id2 in agent_ids
        assert all("status" in a for a in agents)
        assert all("task" in a for a in agents)

        # Cleanup
        gate.set()


class TestCheckAgentInvalidId:
    """check_agent with a non-existent ID should return an error."""

    def test_invalid_id(self):
        result = check_agent(agent_id=99999, action="status")
        assert result.is_error
        assert "No background agent" in result.content
        assert "99999" in result.content

    def test_missing_id_for_status(self):
        result = check_agent(agent_id=-1, action="status")
        assert result.is_error
        assert "agent_id is required" in result.content


class TestForegroundSoftHandoff:
    """Foreground dispatch_agent that exceeds its timeout should return
    a timeout_handoff with agent_id (not an error)."""

    def test_timeout_handoff(self):
        gate = threading.Event()

        def blocking_subagent(*args, **kwargs):
            gate.wait(timeout=10)
            return _completed_outcome("eventually done")

        with (
            patch("jyagent.tools.subagent._run_subagent", side_effect=blocking_subagent),
            patch("jyagent.tools.subagent._FG_DEFAULT_TIMEOUT", 0.2),
        ):
            result = dispatch_agent(task="long fg task", background=False)

        payload = json.loads(result.content)
        assert payload["status"] == "timeout_handoff"
        assert "agent_id" in payload
        assert isinstance(payload["agent_id"], int)

        # The agent should now be in the background registry
        bg_id = payload["agent_id"]
        agent = _bg_registry.get(bg_id)
        assert agent is not None

        # Cleanup
        gate.set()


class TestCancelEventInterruptsLoop:
    """Setting cancel_event should cause the AgentLoop to stop with
    'interrupted' status."""

    def test_cancel_interrupts(self):
        from jyagent.loop_engine import AgentLoop, LoopConfig

        cancel_event = threading.Event()

        # Create a mock runtime_owner whose complete() returns a tool call,
        # keeping the loop alive long enough for the cancel to be detected.
        mock_owner = MagicMock()
        mock_spec = MagicMock()
        mock_spec.model = "test-model"
        mock_spec.provider = "anthropic"
        mock_owner.model_spec = mock_spec

        call_count = 0

        def fake_complete(context, options=None, model_spec=None):
            nonlocal call_count
            call_count += 1
            # First call returns a tool call to keep the loop going;
            # set cancel so the next iteration's top-of-loop check fires.
            cancel_event.set()
            return {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": f"call_{call_count}",
                        "name": "dummy_tool",
                        "arguments": {},
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

        mock_owner.complete.side_effect = fake_complete

        # Provide a dummy tool so the loop can "execute" it
        def dummy_tool(**kwargs):
            from jyagent.toolresult import ToolResult as TR
            return TR("ok")

        tool_schemas = [{"name": "dummy_tool", "input_schema": {"type": "object", "properties": {}}}]
        tool_functions = {"dummy_tool": dummy_tool}

        # Patch get_reasoning_config_for_provider to avoid model validation errors
        with patch("jyagent.loop_engine.get_reasoning_config_for_provider", return_value=None):
            loop = AgentLoop(
                runtime_owner=mock_owner,
                config=LoopConfig(max_steps=10, streaming=False),
                tool_source=lambda: (tool_schemas, tool_functions),
                cancel_event=cancel_event,
            )
            result = loop.run("system", [{"role": "user", "content": "hello"}])

        assert result.status == "interrupted"
        assert result.steps >= 1
