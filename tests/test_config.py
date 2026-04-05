# Tests for central configuration

import os
import sys
import importlib

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jyagent.config as config_module

from jyagent.config import (
    SKIP_DIRS, BINARY_EXTS,
    DEFAULT_MAX_TOKENS, MAX_TOKENS_CAP, DEFAULT_MAX_STEPS,
    MAX_TOOL_RESULT_CHARS, DEFAULT_TOOL_TIMEOUT,
    MEMORY_DIR, TOPICS_DIR, MEMORY_MD_FILE,
    CHARS_PER_TOKEN,
)


def _reload_config():
    return importlib.reload(config_module)


class TestConfig:
    def test_skip_dirs_is_set(self):
        assert isinstance(SKIP_DIRS, set)
        assert '.git' in SKIP_DIRS
        assert 'node_modules' in SKIP_DIRS

    def test_binary_exts_is_set(self):
        assert isinstance(BINARY_EXTS, set)
        assert '.png' in BINARY_EXTS
        assert '.exe' in BINARY_EXTS

    def test_numeric_constants(self):
        assert isinstance(DEFAULT_MAX_TOKENS, int)
        assert DEFAULT_MAX_TOKENS > 0
        assert MAX_TOKENS_CAP > DEFAULT_MAX_TOKENS
        assert DEFAULT_MAX_STEPS > 0
        assert MAX_TOOL_RESULT_CHARS > 0
        assert DEFAULT_TOOL_TIMEOUT > 0

    def test_paths(self):
        assert "memory" in MEMORY_DIR
        assert "topics" in TOPICS_DIR
        assert MEMORY_MD_FILE.endswith("MEMORY.md")

    def test_chars_per_token(self):
        assert CHARS_PER_TOKEN == 4

    def test_provider_neutral_envs(self, monkeypatch):
        monkeypatch.setenv("AGENT_PROVIDER", "openai")
        monkeypatch.setenv("AGENT_MODEL", "gpt-5-mini")
        monkeypatch.setenv("AGENT_MAX_TOKENS", "2048")
        cfg = _reload_config()

        assert cfg.AGENT_PROVIDER == "openai"
        assert cfg.AGENT_MODEL == "gpt-5-mini"
        assert cfg.DEFAULT_MAX_TOKENS == 2048
        assert cfg.get_active_model_spec().provider == "openai"
        assert cfg.get_active_model_spec().model == "gpt-5-mini"

    def test_agent_token_limits_ignore_legacy_anthropic_envs(self, monkeypatch):
        monkeypatch.delenv("AGENT_PROVIDER", raising=False)
        monkeypatch.delenv("AGENT_MODEL", raising=False)
        monkeypatch.delenv("AGENT_MAX_TOKENS", raising=False)
        monkeypatch.delenv("AGENT_MAX_TOKENS_CAP", raising=False)
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test")
        cfg = _reload_config()

        assert cfg.AGENT_PROVIDER == "anthropic"
        assert cfg.AGENT_MODEL == "claude-test"
        assert cfg.DEFAULT_MAX_TOKENS == 16384
        assert cfg.MAX_TOKENS_CAP == 128000

    def test_subagent_and_router_specs_default_to_active_runtime(self, monkeypatch):
        monkeypatch.setenv("AGENT_PROVIDER", "openai")
        monkeypatch.setenv("AGENT_MODEL", "gpt-5-mini")
        monkeypatch.delenv("SKILL_ROUTER_PROVIDER", raising=False)
        monkeypatch.delenv("SKILL_ROUTER_MODEL", raising=False)
        monkeypatch.delenv("SUBAGENT_FAST_PROVIDER", raising=False)
        monkeypatch.delenv("SUBAGENT_FAST_MODEL", raising=False)
        cfg = _reload_config()

        active = cfg.get_active_model_spec()
        router = cfg.get_skill_router_model_spec(active)
        subagent_fast = cfg.get_subagent_model_spec("fast", active)

        assert router.provider == "openai"
        assert router.model == "gpt-5-mini"
        assert subagent_fast.provider == "openai"
        assert subagent_fast.model == "gpt-5-mini"

    def test_openai_reasoning_envs_parse_to_structured_config(self, monkeypatch):
        monkeypatch.setenv("OPENAI_REASONING_EFFORT", "high")
        monkeypatch.setenv("OPENAI_REASONING_SUMMARY", "concise")
        cfg = _reload_config()

        assert cfg.get_reasoning_config_for_provider("openai") == {
            "effort": "high",
            "summary": "concise",
        }

    def test_anthropic_reasoning_envs_parse_to_structured_config(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_THINKING_TYPE", "enabled")
        monkeypatch.setenv("ANTHROPIC_THINKING_BUDGET_TOKENS", "2048")
        monkeypatch.setenv("ANTHROPIC_THINKING_DISPLAY", "omitted")
        cfg = _reload_config()

        assert cfg.get_reasoning_config_for_provider("anthropic", max_output_tokens=4096) == {
            "type": "enabled",
            "budget_tokens": 2048,
            "display": "omitted",
        }

    def test_reasoning_config_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)
        monkeypatch.delenv("OPENAI_REASONING_SUMMARY", raising=False)
        monkeypatch.delenv("ANTHROPIC_THINKING_TYPE", raising=False)
        monkeypatch.delenv("ANTHROPIC_THINKING_BUDGET_TOKENS", raising=False)
        monkeypatch.delenv("ANTHROPIC_THINKING_DISPLAY", raising=False)
        cfg = _reload_config()

        assert cfg.get_reasoning_config_for_provider("openai") is None
        assert cfg.get_reasoning_config_for_provider("anthropic", max_output_tokens=4096) is None

    def test_anthropic_reasoning_budget_env_must_be_integer(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_THINKING_TYPE", "enabled")
        monkeypatch.setenv("ANTHROPIC_THINKING_BUDGET_TOKENS", "not-an-int")
        cfg = _reload_config()

        with pytest.raises(ValueError, match="must be an integer"):
            cfg.get_reasoning_config_for_provider("anthropic", max_output_tokens=4096)
