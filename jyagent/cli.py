# CLI module — Modern terminal interface using prompt_toolkit + rich
# Provides: multi-line input, bracketed paste, syntax highlighting, rich output,
# status bar with token/cost tracking
#
# v6: Added status bar with session stats (model, tokens, cost),
#     turn summary after each response, updated banner

import os
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.filters import Condition
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.formatted_text import HTML

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from .session_stats import get_stats


# ─── Rich console with custom theme ──────────────────────────────────────────

CUSTOM_THEME = Theme({
    "agent": "bold green",
    "user": "bold cyan",
    "system": "yellow",
    "error": "bold red",
    "dim": "dim",
    "banner": "bold magenta",
    "tool": "yellow",
    "info": "dim cyan",
})

console = Console(theme=CUSTOM_THEME, highlight=False)

# ─── Prompt toolkit styling ──────────────────────────────────────────────────

PT_STYLE = PTStyle.from_dict({
    "prompt": "bold ansicyan",
    "continuation": "ansigray",
    "bottom-toolbar": "bg:ansibrightblack ansiwhite",
})


# ─── CLI Interface class ─────────────────────────────────────────────────────

class CLI:
    """Modern terminal interface with multi-line support and rich output."""

    def __init__(self, history_file: str = ".agent_history"):
        self.multiline_mode = [False]  # mutable container for toggle

        # Resolve history file path relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        history_path = os.path.join(project_root, history_file)

        self.session = PromptSession(
            history=FileHistory(history_path),
            auto_suggest=AutoSuggestFromHistory(),
            style=PT_STYLE,
            multiline=Condition(lambda: self.multiline_mode[0]),
            enable_open_in_editor=True,  # Ctrl+X Ctrl+E to open in $EDITOR
            mouse_support=False,
        )

    # ─── Input ────────────────────────────────────────────────────────────

    def get_input(self) -> Optional[str]:
        """Get user input with prompt_toolkit. Returns None on EOF/Ctrl+C."""
        try:
            def get_toolbar():
                try:
                    from html import escape as html_escape
                    stats = get_stats()
                    stats_str = html_escape(stats.summary_line())

                    if self.multiline_mode[0]:
                        mode_str = '<b>multi-line</b> │ <b>Enter</b>=newline <b>Meta+Enter</b>=submit'
                    else:
                        mode_str = '<b>single-line</b> │ <b>Enter</b>=submit <b>Meta+Enter</b>=newline'

                    return HTML(f'{stats_str} │ {mode_str}')
                except Exception:
                    return ""

            if self.multiline_mode[0]:
                prompt_text = [("class:prompt", "You ▶ "), ("class:continuation", "[multi] ")]
            else:
                prompt_text = [("class:prompt", "You ▶ ")]

            text = self.session.prompt(
                prompt_text,
                bottom_toolbar=get_toolbar,
            )
            return text
        except (EOFError, KeyboardInterrupt):
            return None

    def toggle_multiline(self):
        """Toggle multi-line mode."""
        self.multiline_mode[0] = not self.multiline_mode[0]
        if self.multiline_mode[0]:
            self.print_system("Multi-line mode ON — Enter=newline, Meta+Enter (Esc then Enter)=submit")
        else:
            self.print_system("Multi-line mode OFF — Enter=submit, Meta+Enter (Esc then Enter)=newline")

    # ─── Output ───────────────────────────────────────────────────────────

    def print_banner(self, model_label: str = ""):
        """Print the startup banner."""
        if not model_label:
            from .config import AGENT_PROVIDER, AGENT_MODEL
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

    def print_system(self, msg: str):
        """Print a system message."""
        console.print(f"[system]⚙ {msg}[/system]")

    def print_error(self, msg: str):
        """Print an error message."""
        console.print(f"[error]✖ {msg}[/error]")

    def print_separator(self):
        """Print a subtle separator."""
        console.print("─" * 56, style="dim")

    def print_turn_summary(self):
        """Print a compact turn summary (tokens + cost) after the response."""
        stats = get_stats()
        summary = stats.turn_summary()
        console.print(f"[dim]  {summary}[/dim]")

    def print_history(self, messages: list):
        """Print conversation history."""
        self.print_separator()
        for msg in messages:
            role = msg['role']
            content = str(msg['content'])
            if role == 'user':
                console.print(f"[user]{role}:[/user] {content}")
            elif role == 'assistant':
                console.print(f"[agent]{role}:[/agent] {content}")
            else:
                console.print(f"[system]{role}:[/system] {content}")
        self.print_separator()

    def print_help(self):
        """Print help information."""
        help_text = Text()
        
        help_text.append("General:\n", style="bold")
        commands_general = [
            ("/quit", "Exit the agent"),
            ("/help", "Show this help message"),
            ("/history", "Show last 10 messages"),
            ("/new", "Save session and start fresh"),
            ("/tools", "List registered tools"),
            ("/model", "Show or switch provider/model"),
            ("/multi", "Toggle multi-line input mode"),
            ("/markdown", "Toggle markdown rendering"),
            ("/stats", "Show session statistics (tokens, cost)"),
        ]
        for cmd, desc in commands_general:
            help_text.append(f"  {cmd:<18}", style="bold cyan")
            help_text.append(f"— {desc}\n")
        
        help_text.append("\nMemory:\n", style="bold")
        help_text.append("  Just ask in natural language — e.g. \"记住我喜欢用 Docker\",\n")
        help_text.append("  \"忘掉关于 wan2 的记忆\", \"看看你记了什么\"\n")
        help_text.append("  (uses the manage_memory tool automatically)\n")
        
        help_text.append("\nSkills:\n", style="bold")
        commands_skills = [
            ("/skills", "List all available skills and status"),
            ("/skill <name>", "Activate a skill"),
            ("/skill -<name>", "Deactivate a skill"),
        ]
        for cmd, desc in commands_skills:
            help_text.append(f"  {cmd:<18}", style="bold cyan")
            help_text.append(f"— {desc}\n")
        
        help_text.append("\n")
        help_text.append("Keyboard Shortcuts:\n", style="bold")
        help_text.append("  • Ctrl-C at prompt:            Exit the agent (saves session)\n")
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

    def goodbye(self):
        """Print goodbye message."""
        console.print("\n[system]👋 Goodbye![/system]")
