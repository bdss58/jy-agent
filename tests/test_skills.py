# Tests for the Agent Skills engine.

import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jyagent.agent as agent
import jyagent.config as config
import jyagent.skills as skills


class _DummyCLI:
    def __init__(self):
        self.messages = []

    def print_system(self, message):
        self.messages.append(message)


class _DummyConversation:
    def __init__(self):
        self.cleared = False

    def clear(self):
        self.cleared = True


class _DummyStats:
    def __init__(self):
        self.reset_called = False
        self.active_model = None

    def reset(self):
        self.reset_called = True

    def set_active_model(self, provider, model):
        self.active_model = (provider, model)


class _DummySkillManager:
    def __init__(self):
        self.deactivate_all_called = False

    def deactivate_all(self):
        self.deactivate_all_called = True


class TestSkillIntegration:
    def test_new_command_clears_active_skills(self, monkeypatch):
        cli = _DummyCLI()
        conversation = _DummyConversation()
        stats = _DummyStats()
        skill_mgr = _DummySkillManager()
        runtime_owner = SimpleNamespace(model_spec=SimpleNamespace(provider="openai", model="gpt-5"))

        monkeypatch.setattr(agent, "get_stats", lambda: stats)
        monkeypatch.setattr(agent, "get_skill_manager", lambda: skill_mgr)
        agent._cached_memory_context = "stale"

        agent._cmd_new(cli=cli, runtime_owner=runtime_owner, conversation=conversation)

        assert conversation.cleared is True
        assert skill_mgr.deactivate_all_called is True
        assert stats.reset_called is True
        assert stats.active_model == ("openai", "gpt-5")
        assert agent._cached_memory_context is None
        assert cli.messages == ["Conversation cleared. Starting fresh."]


class TestSkillResources:
    def test_read_resource_blocks_prefix_path_traversal(self, tmp_path):
        skills_root = tmp_path / "skills"
        skill_dir = skills_root / "foo"
        refs_dir = skill_dir / "references"
        sibling_dir = skills_root / "foo-bar"
        refs_dir.mkdir(parents=True)
        sibling_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: foo\n"
            "description: >-\n"
            "  Test skill\n"
            "---\n\n"
            "# Test skill\n",
            encoding="utf-8",
        )
        (refs_dir / "ok.txt").write_text("inside", encoding="utf-8")
        (sibling_dir / "secret.txt").write_text("outside", encoding="utf-8")

        mgr = skills.SkillManager(str(skills_root))
        assert mgr.discover() == ["foo"]

        assert mgr.read_resource("foo", "references/ok.txt") == "inside"
        assert mgr.read_resource("foo", "../foo-bar/secret.txt") is None

    def test_skill_limits_are_sourced_from_config(self):
        assert skills.MAX_INSTRUCTIONS_CHARS == config.MAX_INSTRUCTIONS_CHARS
        assert skills.MAX_RESOURCE_CHARS == config.MAX_RESOURCE_CHARS


class TestSkillDocs:
    def test_claude_and_codex_docs_include_run_shell_600_policy(self):
        repo_root = Path(__file__).resolve().parents[1]

        claude_text = (repo_root / "skills" / "claude-code" / "SKILL.md").read_text(encoding="utf-8").lower()
        assert "run_shell" in claude_text
        assert "timeout=600" in claude_text
        assert "claude -p" in claude_text

        codex_text = (repo_root / "skills" / "codex-cli" / "SKILL.md").read_text(encoding="utf-8").lower()
        assert "run_shell" in codex_text
        assert "timeout=600" in codex_text
        assert "codex exec" in codex_text
        assert "codex review" in codex_text
