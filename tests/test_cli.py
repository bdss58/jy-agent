# Tests for CLI rendering and fatal error handling.

import io
import os
import sys

from rich.console import Console

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jyagent.agent as agent
import jyagent.cli as cli_module


def _make_cli():
    cli = cli_module.CLI.__new__(cli_module.CLI)
    cli.multiline_mode = [False]
    return cli


def _make_console():
    stream = io.StringIO()
    console = Console(
        file=stream,
        force_terminal=False,
        color_system=None,
        highlight=False,
        theme=cli_module.CUSTOM_THEME,
        width=140,
    )
    return stream, console


class TestCLIOutput:
    def test_print_error_preserves_bracketed_text(self, monkeypatch):
        stream, recording_console = _make_console()
        monkeypatch.setattr(cli_module, "console", recording_console)

        cli = _make_cli()
        cli.print_error("Error: [/abs/path/file.py: 10 lines total, showing L1-L10]")

        output = stream.getvalue()
        assert "✖ Error: [/abs/path/file.py: 10 lines total, showing L1-L10]" in output

    def test_print_turn_summary_is_safe_for_bracketed_strings(self, monkeypatch):
        class _DummyStats:
            def turn_summary(self):
                return "[openai:gpt-5-mini] input=[12] output=[7]"

        stream, recording_console = _make_console()
        monkeypatch.setattr(cli_module, "console", recording_console)
        monkeypatch.setattr(cli_module, "get_stats", lambda: _DummyStats())

        cli = _make_cli()
        cli.print_turn_summary()

        output = stream.getvalue()
        assert "[openai:gpt-5-mini] input=[12] output=[7]" in output

    def test_print_history_formats_structured_messages_without_repr_noise(self, monkeypatch):
        stream, recording_console = _make_console()
        monkeypatch.setattr(cli_module, "console", recording_console)

        cli = _make_cli()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "[/abs/path/file.py: 3 lines total, showing L1-L3]\nL1: alpha",
                    },
                    {
                        "type": "tool_call",
                        "id": "call_1",
                        "name": "read_file",
                        "arguments": {
                            "path": "skills/codex-cli/SKILL.md",
                            "line_numbers": True,
                        },
                    },
                    {"type": "thinking", "thinking": "reasoning about [literal] output"},
                    {"type": "thinking", "encrypted_content": "opaque", "redacted": True},
                ],
            },
            {
                "role": "tool_result",
                "tool_name": "read_file",
                "content": "[/abs/path/file.py: 3 lines total, showing L1-L3]\nL1: alpha\nL2: beta",
                "is_error": False,
            },
        ]

        cli.print_history(messages)

        output = stream.getvalue()
        assert "assistant: [/abs/path/file.py: 3 lines total, showing L1-L3]" in output
        assert "↵ L1: alpha (2 lines," in output
        assert 'assistant: tool_call: read_file {"line_numbers": true, "path": "skills/codex-cli/SKILL.md"}' in output
        assert "assistant: thinking: reasoning about [literal] output" in output
        assert "assistant: thinking: [redacted]" in output
        assert "tool_result: (read_file, ok): [/abs/path/file.py: 3 lines total, showing L1-L3]" in output
        assert "↵ L1: alpha ↵ L2: beta (3 lines," in output
        assert "{'type': 'text'" not in output
        assert "[{'type':" not in output


def test_run_top_level_error_with_bracketed_text_is_rendered_safely(monkeypatch):
    stream, recording_console = _make_console()

    class _BoomCLI:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("[/abs/path/file.py: 10 lines total, showing L1-L10]")

    monkeypatch.setattr(agent, "CLI", _BoomCLI)
    monkeypatch.setattr(agent, "console", recording_console)

    agent.run(None)

    output = stream.getvalue()
    assert "Unexpected error in agent: [/abs/path/file.py: 10 lines total, showing L1-L10]" in output
