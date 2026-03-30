# CLI module — Modern terminal interface using prompt_toolkit + rich
# Provides: multi-line input, bracketed paste, syntax highlighting, rich output
#
# v5: Removed /memory, /remember, /forget from help and banner (consolidated into manage_memory tool)

import os
import sys
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.formatted_text import HTML

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme


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


# ─── Key bindings ─────────────────────────────────────────────────────────────

def _create_key_bindings(multiline_mode: list):
    """Create key bindings.
    
    multiline_mode is a mutable list [bool] so we can toggle it.
    
    Behavior:
    - Single-line mode (default): Enter submits, Meta+Enter / Esc then Enter inserts newline
    - Multi-line mode: Enter inserts newline, Meta+Enter submits
    """
    kb = KeyBindings()

    @kb.add(Keys.Enter)
    def handle_enter(event):
        buf = event.app.current_buffer
        if multiline_mode[0]:
            # In multi-line mode, Enter inserts a newline
            buf.insert_text("\n")
        else:
            # In single-line mode, submit the text
            buf.validate_and_handle()

    @kb.add(Keys.Escape, Keys.Enter)
    def handle_meta_enter(event):
        buf = event.app.current_buffer
        if multiline_mode[0]:
            # In multi-line mode, Meta+Enter submits
            buf.validate_and_handle()
        else:
            # In single-line mode, Meta+Enter inserts newline
            buf.insert_text("\n")

    return kb


# ─── CLI Interface class ─────────────────────────────────────────────────────

class CLI:
    """Modern terminal interface with multi-line support and rich output."""

    def __init__(self, history_file: str = ".agent_history"):
        self.multiline_mode = [False]  # mutable container for toggle
        self.kb = _create_key_bindings(self.multiline_mode)
        
        # Resolve history file path relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        history_path = os.path.join(project_root, history_file)
        
        self.session = PromptSession(
            history=FileHistory(history_path),
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=self.kb,
            style=PT_STYLE,
            multiline=True,  # Allow multi-line display; Enter behavior is controlled by keybindings
            enable_open_in_editor=True,  # Ctrl+X Ctrl+E to open in $EDITOR
            mouse_support=False,
        )

    # ─── Input ────────────────────────────────────────────────────────────

    def get_input(self) -> Optional[str]:
        """Get user input with prompt_toolkit. Returns None on EOF/Ctrl+C."""
        try:
            def get_toolbar():
                if self.multiline_mode[0]:
                    return HTML(
                        '<b>Mode:</b> multi-line │ '
                        '<b>Enter</b>=newline │ '
                        '<b>Meta+Enter</b>=submit │ '
                        '<b>/multi</b>=toggle │ '
                        '<b>Ctrl-C</b>=exit'
                    )
                else:
                    return HTML(
                        '<b>Mode:</b> single-line │ '
                        '<b>Enter</b>=submit │ '
                        '<b>Meta+Enter</b>=newline │ '
                        '<b>/multi</b>=toggle │ '
                        '<b>Ctrl-C</b>=exit'
                    )

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

    def print_banner(self):
        """Print the startup banner."""
        banner_text = Text()
        banner_text.append("Self-Assembled AI Agent", style="bold")
        banner_text.append(" (Self-Evolving)\n", style="bold")
        banner_text.append("Bootstrapped from a single API call\n")
        banner_text.append("✨ Now with rich output & multi-line input!\n\n", style="italic")
        banner_text.append("Commands: ", style="bold")
        banner_text.append("/quit /help /history /clear /tools /multi /markdown\n")
        banner_text.append("          /skills /skill /evolve\n")
        banner_text.append("\nInput:    ", style="bold")
        banner_text.append("Enter=submit │ Meta+Enter=newline │ /multi=toggle mode\n")
        banner_text.append("          Paste multi-line content works automatically!\n")
        banner_text.append("\nExit:     ", style="bold")
        banner_text.append("Ctrl-C at prompt to quit │ Ctrl-C during response to interrupt")

        console.print(Panel(
            banner_text,
            title="[bold magenta]🤖 AI Agent[/bold magenta]",
            border_style="magenta",
            padding=(1, 2),
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
            ("/clear", "Clear conversation history"),
            ("/tools", "List registered tools"),
            ("/evolve", "Manually trigger self-evolution"),
            ("/multi", "Toggle multi-line input mode"),
            ("/markdown", "Toggle markdown rendering for responses"),
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
