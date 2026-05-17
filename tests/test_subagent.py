from __future__ import annotations

import inspect
import json
import threading
import time
from unittest.mock import patch, MagicMock

import pytest

from jyagent.tools.schemas import (
    SUBAGENT_SCHEMA as TOOL_SCHEMA,
    CHECK_AGENT_SCHEMA,
)
from jyagent.tools.subagent import (
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
        patch("jyagent.tools.subagent._get_runtime_owner") as mock_owner,
        patch("jyagent.tools.subagent.get_stats") as mock_stats,
    ):
        # Provide a minimal model_spec via the runtime owner
        spec = MagicMock()
        spec.model = "test-model"
        spec.provider = "anthropic"
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

class TestCheckAgentWait:
    """check_agent(agent_id, action='wait') should block until the agent
    finishes (or until wait_timeout_seconds elapses), then return the same
    response shape as action='status'."""

    def test_wait_returns_result_when_agent_finishes_during_window(self):
        """If the agent completes during the wait window, we should get
        the final result without paying for an extra polling turn."""
        gate = threading.Event()

        def slow_then_done(*args, **kwargs):
            # Finish ~0.2s after dispatch — well within the wait window
            gate.wait(timeout=10)
            return _completed_outcome("waited answer")

        with (
            patch("jyagent.tools.subagent._run_subagent", side_effect=slow_then_done),
            patch("jyagent.tools.subagent._BG_GRACE_PERIOD", 0.05),
        ):
            result = dispatch_agent(task="wait task", background=True)
            payload = json.loads(result.content)
            agent_id = payload["agent_id"]

            # Release the worker so it finishes shortly into the wait
            def _release():
                time.sleep(0.2)
                gate.set()
            threading.Thread(target=_release, daemon=True).start()

            t0 = time.monotonic()
            check_result = check_agent(
                agent_id=agent_id, action="wait", wait_timeout_seconds=5,
            )
            elapsed = time.monotonic() - t0

        assert not check_result.is_error
        assert "waited answer" in check_result.content
        # Should have unblocked roughly when the worker finished, NOT after
        # the full 5s timeout.
        assert elapsed < 3.0, f"wait took too long: {elapsed:.2f}s"

    def test_wait_returns_running_status_after_timeout(self):
        """If the agent does NOT finish within wait_timeout_seconds, wait
        should return the running-status payload (same shape as action='status'
        on a still-running agent)."""
        gate = threading.Event()

        def blocking(*args, **kwargs):
            gate.wait(timeout=10)
            return _completed_outcome("never seen")

        with (
            patch("jyagent.tools.subagent._run_subagent", side_effect=blocking),
            patch("jyagent.tools.subagent._BG_GRACE_PERIOD", 0.05),
        ):
            result = dispatch_agent(task="slow task", background=True)
            payload = json.loads(result.content)
            agent_id = payload["agent_id"]

            t0 = time.monotonic()
            check_result = check_agent(
                agent_id=agent_id, action="wait", wait_timeout_seconds=1,
            )
            elapsed = time.monotonic() - t0
            gate.set()  # cleanup

        assert not check_result.is_error
        status_payload = json.loads(check_result.content)
        assert status_payload["status"] == "running"
        assert status_payload["agent_id"] == agent_id
        # Must have actually blocked for ~1s (not returned instantly)
        assert elapsed >= 0.9, f"wait did not actually block: {elapsed:.2f}s"
        assert elapsed < 3.0, f"wait blocked too long: {elapsed:.2f}s"

    def test_wait_returns_immediately_when_already_done(self):
        """If the agent is already done before wait is called, wait should
        return the result immediately without blocking."""
        gate = threading.Event()

        def slow_then_done(*args, **kwargs):
            # Wait briefly so we miss the grace period and go to background,
            # then finish before the test calls wait().
            gate.wait(timeout=10)
            return _completed_outcome("already done")

        with (
            patch("jyagent.tools.subagent._run_subagent", side_effect=slow_then_done),
            patch("jyagent.tools.subagent._BG_GRACE_PERIOD", 0.05),
        ):
            result = dispatch_agent(task="instant task", background=True)
            payload = json.loads(result.content)
            agent_id = payload["agent_id"]

            # Let the worker finish well before we call wait
            gate.set()
            # Poll the future directly — robust against any scheduling jitter
            agent = _bg_registry.get(agent_id)
            assert agent is not None
            agent.future.result(timeout=5)

            t0 = time.monotonic()
            check_result = check_agent(
                agent_id=agent_id, action="wait",
                wait_timeout_seconds=10,
            )
            elapsed = time.monotonic() - t0

        assert not check_result.is_error
        assert "already done" in check_result.content
        # Must not have blocked for anywhere near 10s
        assert elapsed < 1.0, f"wait blocked unexpectedly: {elapsed:.2f}s"



class TestCheckAgentKill:
    """check_agent(agent_id, action='kill') should cancel the agent."""

    def test_kill(self):
        gate = threading.Event()

        def blocking_subagent(task, context, model_spec, max_steps,
                              tool_schemas, tool_functions,
                              agent_id=None, custom_system_prompt=None,
                              cancel_event=None, progress_ids=None,
                              memory_mode="none", cancel_state=None):
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


class TestCheckAgentIdempotencyAndPersistence:
    """G2 + G9 fix (2026-05): check_agent reads must be idempotent and
    recoverable from disk after a wait-timeout-lost result.

    Three guarantees being verified:
      (a) Calling check_agent on a completed agent multiple times
          returns the SAME outcome each time (no consume-on-read).
      (b) The completed agent's outcome is persisted to disk under
          data/sessions/subagents/<pid>-<id>.json.
      (c) After hard-removing the agent from the in-memory registry
          (simulating eviction or a lost wait), check_agent still
          returns the outcome via the disk fallback.
    """

    def _run_one_and_get_id(self, content="persisted answer"):
        gate = threading.Event()

        def slow(*args, **kwargs):
            gate.wait(timeout=10)
            return _completed_outcome(content)

        with (
            patch("jyagent.tools.subagent._run_subagent", side_effect=slow),
            patch("jyagent.tools.subagent._BG_GRACE_PERIOD", 0.05),
        ):
            result = dispatch_agent(task="persist task", background=True)
            payload = json.loads(result.content)
            agent_id = payload["agent_id"]
            gate.set()
            time.sleep(0.3)  # let the worker thread finish

        return agent_id

    def test_repeat_reads_return_same_outcome(self):
        agent_id = self._run_one_and_get_id("the answer is 42")

        r1 = check_agent(agent_id=agent_id, action="status")
        r2 = check_agent(agent_id=agent_id, action="status")
        r3 = check_agent(agent_id=agent_id, action="status")

        assert "the answer is 42" in r1.content
        assert r1.content == r2.content == r3.content
        assert not (r1.is_error or r2.is_error or r3.is_error)

    def test_outcome_persisted_to_disk(self):
        from jyagent.tools.subagent import (
            _subagent_outcome_path,
            _load_subagent_outcome_from_disk,
        )

        agent_id = self._run_one_and_get_id("persisted answer")
        # Trigger persistence by reading once.
        check_agent(agent_id=agent_id, action="status")

        import os
        path = _subagent_outcome_path(agent_id)
        assert os.path.exists(path), f"expected persisted record at {path}"

        record = _load_subagent_outcome_from_disk(agent_id)
        assert record is not None
        assert record["agent_id"] == agent_id
        assert record["outcome"]["content"] == "persisted answer"
        assert record["outcome"]["status"] == _SUBAGENT_STATUS_COMPLETED

    def test_disk_fallback_after_registry_eviction(self):
        """Simulates the wait-timeout-lost scenario: the in-memory record
        is gone, but check_agent still recovers the outcome from disk."""
        agent_id = self._run_one_and_get_id("recovered answer")
        # Read once to ensure persistence has happened, then nuke the
        # in-memory record to simulate eviction / a prior client-timeout
        # path that did remove it.
        check_agent(agent_id=agent_id, action="status")
        _bg_registry._agents.pop(agent_id, None)
        try:
            _bg_registry._completed_order.remove(agent_id)
        except ValueError:
            pass

        # The disk fallback should kick in and rebuild the outcome.
        recovered = check_agent(agent_id=agent_id, action="status")
        assert "recovered answer" in recovered.content
        assert not recovered.is_error


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
        from jyagent.runtime.loop.engine import AgentLoop, LoopConfig

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
            from jyagent.runtime.tools.result import ToolResult as TR
            return TR("ok")

        tool_schemas = [{"name": "dummy_tool", "input_schema": {"type": "object", "properties": {}}}]
        tool_functions = {"dummy_tool": dummy_tool}

        # Patch get_reasoning_config_for_provider to avoid model validation errors.
        # ``_build_runtime_options`` moved from
        # engine.py to runtime/loop/llm_runner.py, so the patch target
        # moved with it (``patch`` must target the module that *looks up*
        # the symbol — not the symbol's defining module).
        with patch("jyagent.runtime.loop.llm_runner.get_reasoning_config_for_provider", return_value=None):
            loop = AgentLoop(
                runtime_owner=mock_owner,
                config=LoopConfig(max_steps=10, streaming=False),
                tool_source=lambda: (tool_schemas, tool_functions),
                cancel_event=cancel_event,
            )
            result = loop.run("system", [{"role": "user", "content": "hello"}])

        assert result.status == "interrupted"
        assert result.steps >= 1
