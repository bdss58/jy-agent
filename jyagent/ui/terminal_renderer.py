"""TerminalRenderer — Rich-on-stdout rendering surface used by ``CLI``.

Owns ALL terminal output: banner, status lines, history, help, errors.
``CLI`` (in :mod:`jyagent.ui.cli`) inherits from this class and adds only
the input-side concerns (prompt_toolkit session, multi-line toggle).
``agent.py`` interacts with a ``CLI`` instance — calls into this class's
methods work transparently via inheritance.

Public methods used by ``agent.py``:
    print_banner / print_system / print_error / print_separator
    print_history / print_help / print_turn_summary / goodbye

Implementation notes:
    * The shared ``console`` Console is a module-level singleton — it owns
      the theme and prompt_toolkit re-uses it indirectly via terminal width
      detection.  Keep it module-level (do not put it on ``self``).
    * All public ``print_*`` methods build :class:`rich.text.Text` objects
      explicitly rather than passing format strings to ``console.print``.
      This is the durable rule: dynamic content must NOT be parsed by Rich
      markup (otherwise a stray ``[`` in user output crashes rendering).
"""
from __future__ import annotations

import json
from typing import Optional

from rich.panel import Panel
from rich.text import Text

from ..runtime.stats import get_stats
from .output import CUSTOM_THEME, console


# ─── Renderer ────────────────────────────────────────────────────────────────


class TerminalRenderer:
    """Rich-on-stdout renderer.  Base class of :class:`jyagent.ui.cli.CLI`."""

    # ─── Banner / lifecycle ──────────────────────────────────────────────

    def print_banner(self, model_label: str = ""):
        """Print the startup banner."""
        if not model_label:
            from ..config import AGENT_PROVIDER, AGENT_MODEL
            model_label = f"{AGENT_PROVIDER}:{AGENT_MODEL}"

        banner_text = Text()
        banner_text.append("jy-agent", style="bold magenta")
        banner_text.append(" — AI Agent in your terminal\n", style="bold")
        banner_text.append(f"Model: {model_label}\n", style="dim")
        banner_text.append("\nCommands: ", style="bold")
        banner_text.append("/help /quit /new /tools /multi /markdown /skills /model\n")
        banner_text.append("\nInput:    ", style="bold")
        banner_text.append("Enter=submit │ Esc→Enter=newline │ /multi=toggle\n")
        banner_text.append("Exit:     ", style="bold")
        banner_text.append("Ctrl-C at prompt to quit")

        console.print(Panel(
            banner_text,
            title="[bold magenta]🤖[/bold magenta]",
            border_style="magenta",
            padding=(0, 2),
        ))

    def goodbye(self):
        """Print goodbye message."""
        console.print("\n[system]👋 Goodbye![/system]")

    # ─── Status line primitives ──────────────────────────────────────────

    def _print_prefixed(self, prefix: str, prefix_style: str, body: str = "", body_style: Optional[str] = None):
        """Print a styled prefix plus literal body text without Rich markup parsing."""
        text = Text()
        body_str = "" if body is None else str(body)
        if body_str and "\n" in body_str and prefix:
            indent = " " * (len(prefix) + 1)
            body_str = body_str.replace("\n", "\n" + indent)

        if prefix:
            text.append(prefix, style=prefix_style)
        if body_str:
            if prefix:
                text.append(" ")
            text.append(body_str, style=body_style)
        console.print(text)

    def print_system(self, msg: str):
        """Print a system message."""
        self._print_prefixed("⚙", "system", msg, body_style="system")

    def print_error(self, msg: str):
        """Print an error message."""
        self._print_prefixed("✖", "error", msg, body_style="error")

    def print_separator(self):
        """Print a subtle separator."""
        console.print("─" * 56, style="dim")

    def print_turn_summary(self):
        """Print a compact turn summary (tokens + cost) after the response."""
        stats = get_stats()
        summary = stats.turn_summary()
        self._print_prefixed("", "dim", f"  {summary}", body_style="dim")

    # ─── History rendering ───────────────────────────────────────────────

    def _preview_history_text(self, text: str, max_chars: int = 300) -> str:
        """Collapse long or multi-line history entries into a compact single-line preview."""
        raw = "" if text is None else str(text)
        line_count = raw.count("\n") + 1 if raw else 1
        if line_count == 1 and len(raw) <= max_chars:
            return raw

        preview = raw.replace("\n", " ↵ ")
        if len(preview) > max_chars:
            preview = preview[: max_chars - 3].rstrip() + "..."
        return f"{preview} ({line_count} lines, {len(raw)} chars)"

    def _format_tool_args_preview(self, arguments: object, max_chars: int = 120) -> str:
        """Serialize tool arguments into a compact single-line preview."""
        try:
            preview = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        except TypeError:
            preview = str(arguments)
        if len(preview) > max_chars:
            preview = preview[: max_chars - 3].rstrip() + "..."
        return preview

    def _format_assistant_history_lines(self, content: list) -> list[str]:
        """Format normalized assistant content blocks for `/history`."""
        lines = []
        for block in content:
            if not isinstance(block, dict):
                lines.append(self._preview_history_text(block))
                continue

            block_type = block.get("type", "")
            if block_type == "text":
                lines.append(self._preview_history_text(block.get("text", "")))
                continue

            if block_type == "tool_call":
                name = block.get("name", "")
                args_preview = self._format_tool_args_preview(block.get("arguments", {}))
                summary = f"tool_call: {name}"
                if args_preview:
                    summary += f" {args_preview}"
                lines.append(self._preview_history_text(summary))
                continue

            if block_type == "thinking":
                thinking = block.get("thinking") or ""
                preview = self._preview_history_text(thinking) if thinking else "[redacted]"
                lines.append(f"thinking: {preview}")
                continue

            lines.append(f"{block_type or 'block'}: {self._preview_history_text(block)}")

        return lines or ["[no content]"]

    def _format_history_lines(self, message: dict) -> list[str]:
        """Format a conversation message into one or more display lines."""
        role = message.get("role")
        content = message.get("content", "")

        if role == "assistant" and isinstance(content, list):
            return self._format_assistant_history_lines(content)

        if role == "tool_result":
            tool_name = message.get("tool_name", "")
            status = "error" if message.get("is_error") else "ok"
            preview = self._preview_history_text(content)
            return [f"({tool_name}, {status}): {preview}"]

        return [str(content)]

    def _history_style_for_role(self, role: str) -> str:
        """Map history roles to Rich styles."""
        if role == "user":
            return "user"
        if role == "assistant":
            return "agent"
        if role == "tool_result":
            return "tool"
        return "system"

    def print_history(self, messages: list):
        """Print conversation history."""
        self.print_separator()
        for msg in messages:
            role = str(msg.get("role", "system"))
            role_style = self._history_style_for_role(role)
            for line in self._format_history_lines(msg):
                self._print_prefixed(f"{role}:", role_style, line)
        self.print_separator()

    # ─── Help ────────────────────────────────────────────────────────────

    def print_help(self):
        """Print help information.

        Built dynamically from :mod:`jyagent.ui.commands` so the dispatcher
        and this list cannot drift apart.
        """
        from .commands import commands_by_group

        help_text = Text()

        groups = commands_by_group()
        for group_name, cmds in groups.items():
            help_text.append(f"{group_name}:\n", style="bold")
            for cmd in cmds:
                help_text.append(f"  {cmd.name:<18}", style="bold cyan")
                help_text.append(f"— {cmd.summary}\n")
            help_text.append("\n")

        help_text.append("Memory:\n", style="bold")
        help_text.append("  Just ask in natural language — e.g. \"记住我喜欢用 Docker\",\n")
        help_text.append("  \"忘掉关于 wan2 的记忆\", \"看看你记了什么\"\n")
        help_text.append("  (uses the manage_memory tool automatically)\n\n")

        help_text.append("Keyboard Shortcuts:\n", style="bold")
        help_text.append("  • Ctrl-C at prompt:            Exit the agent\n")
        help_text.append("  • Ctrl-C during response:      Interrupt current operation, return to prompt\n")
        help_text.append("  • Single-line mode (default):  Enter submits, Meta+Enter adds newline\n")
        help_text.append("  • Multi-line mode (/multi):    Enter adds newline, Meta+Enter submits\n")
        help_text.append("  • Paste multi-line content:    Works automatically (bracketed paste)\n")
        help_text.append("  • Open in $EDITOR:             Ctrl+X then Ctrl+E\n")
        help_text.append("  • Search history:              Ctrl+R\n")
        help_text.append("  • Meta+Enter = Esc then Enter (press Esc, release, press Enter)\n")

        console.print(Panel(
            help_text,
            title="[bold magenta]Help — Available Commands[/bold magenta]",
            border_style="magenta",
            padding=(1, 2),
        ))


__all__ = ["TerminalRenderer", "CUSTOM_THEME", "console"]
