# Tests for startup/bootstrap behavior

import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestMainBootstrap:
    def test_create_runtime_owner_uses_env_set_after_main_import(self, monkeypatch):
        monkeypatch.delenv("AGENT_PROVIDER", raising=False)
        monkeypatch.delenv("AGENT_MODEL", raising=False)
        sys.modules.pop("jyagent.config", None)
        sys.modules.pop("jyagent.__main__", None)

        main_module = importlib.import_module("jyagent.__main__")

        assert "jyagent.config" not in sys.modules

        monkeypatch.setenv("AGENT_PROVIDER", "openai")
        monkeypatch.setenv("AGENT_MODEL", "gpt-5-mini")

        runtime_owner = main_module.create_runtime_owner()

        assert runtime_owner.model_spec.provider == "openai"
        assert runtime_owner.model_spec.model == "gpt-5-mini"
