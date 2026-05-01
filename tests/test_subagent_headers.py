from __future__ import annotations

from jyagent.llm.core import LLMOwner, _ADAPTERS, register_adapter
from jyagent.llm.types import ModelSpec
from jyagent.tools import subagent


class _CapturingAdapter:
    provider = "anthropic"
    api_name = "test-capturing-adapter"

    def __init__(self) -> None:
        self.options = []

    def complete(self, model_spec, context, options=None):
        self.options.append(options)
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "stop",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    def stream(self, model_spec, context, options=None):  # pragma: no cover - not used here
        raise AssertionError("sub-agent regression test should use non-streaming complete()")


def test_shared_runtime_owner_threads_session_metadata_into_subagent(monkeypatch):
    saved_adapters = dict(_ADAPTERS)
    saved_runtime_owner = subagent._runtime_owner
    adapter = _CapturingAdapter()

    try:
        register_adapter(adapter)
        owner = LLMOwner(ModelSpec(provider="anthropic", model="test-model"))
        owner.set_session_id("session-123")
        subagent.set_runtime_owner(owner)

        monkeypatch.setattr(subagent, "_get_memory_context", lambda *a, **kw: "")
        monkeypatch.setattr(
            "jyagent.runtime.loop.llm_runner.get_reasoning_config_for_provider",
            lambda *args, **kwargs: None,
        )

        outcome = subagent._run_subagent(
            task="answer directly",
            context="",
            model_spec=owner.model_spec,
            max_steps=1,
            tool_schemas=[],
            tool_functions={},
        )

        assert outcome["status"] == subagent._SUBAGENT_STATUS_COMPLETED
        assert adapter.options
        assert adapter.options[-1].metadata["session_id"] == "session-123"
    finally:
        _ADAPTERS.clear()
        _ADAPTERS.update(saved_adapters)
        subagent._runtime_owner = saved_runtime_owner
