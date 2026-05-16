"""CLI module — combines a prompt_toolkit input session with a Rich renderer.

Architecture:
    * :class:`jyagent.ui.terminal_renderer.TerminalRenderer` owns ALL output.
    * This module owns the input layer (prompt_toolkit session, history file,
      multi-line toggle, status-bar/toolbar callbacks) and the public ``CLI``
      class that composes both.

``CLI`` inherits from ``TerminalRenderer`` so the existing call surface
``cli.print_system(...) / cli.print_history(...)`` continues to work
unchanged in ``agent.py``.  The split is in the *file layout* (Renderer
methods live in ``terminal_renderer.py``, input methods live here), which
makes it easy to swap in a Textual or web renderer later — only this file
needs to change to a different concrete renderer base.

The ``console`` symbol is re-exported from :mod:`jyagent.ui.output` so
existing ``from .ui.cli import CLI, console`` imports keep working
(``console`` is the canonical output singleton — see ``ui/output.py``).
"""

import os
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.filters import Condition
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.formatted_text import HTML

from ..runtime.stats import get_stats
from .output import console
from .terminal_renderer import TerminalRenderer


# ─── Prompt toolkit styling ──────────────────────────────────────────────────

PT_STYLE = PTStyle.from_dict({
    "prompt":         "bold ansicyan",
    "continuation":   "ansigray",
    "bottom-toolbar": "bg:ansibrightblack ansiwhite",
})


# ─── CLI = Renderer + Input ──────────────────────────────────────────────────


class CLI(TerminalRenderer):
    """Modern terminal interface with multi-line support and rich output.

    Inherits all rendering from :class:`TerminalRenderer`; adds the
    prompt_toolkit-based input session and the multi-line toggle.
    """

    def __init__(self, history_file: str = ".agent_history"):
        super().__init__()
        self.multiline_mode = [False]  # mutable container for toggle

        # Resolve history file path relative to project root.
        # NOTE: cli.py lives at jyagent/ui/cli.py (depth 2 inside the package),
        # so we need three dirname() calls to escape to the project root.
        # If you ever move this file, update the depth count — see
        # MEMORY.md gotcha "When moving a Python module deeper..."
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
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


__all__ = ["CLI", "console"]
