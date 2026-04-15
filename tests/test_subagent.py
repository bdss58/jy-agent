from __future__ import annotations

import inspect

from jyagent.tools.subagent import TOOL_SCHEMA, dispatch_agent


def test_dispatch_agent_schema_does_not_expose_named_presets():
    props = TOOL_SCHEMA["input_schema"]["properties"]

    assert "agent" not in props
    assert "task" in props
    assert "tool_whitelist" in props


def test_dispatch_agent_signature_does_not_accept_agent():
    params = inspect.signature(dispatch_agent).parameters

    assert "agent" not in params
