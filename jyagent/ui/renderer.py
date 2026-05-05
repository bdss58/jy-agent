"""Renderer protocol — the surface ``agent.py`` uses for all display output.

Defines the methods any concrete renderer must implement.  Lets us swap in
a Textual/Web/recording renderer later without editing the agent.  Pair
with :class:`jyagent.ui.terminal_renderer.TerminalRenderer` for the stock
Rich-on-stdout implementation.

This is a `typing.Protocol` (structural), so any class with the right
methods satisfies it — no registration needed.

Scope (deliberate):
    This protocol covers ONLY *rendering*.  Input concerns (``get_input``,
    ``toggle_multiline``, prompt_toolkit session) are intentionally
    excluded — they belong to the ``CLI`` class, which composes a renderer
    with an input source.  ``agent.py`` consumes both via a ``CLI``
    instance (see :class:`jyagent.ui.cli.CLI`); a future Textual / Web
    front-end would provide its own input layer alongside a renderer that
    satisfies *this* protocol.  Keeping the two surfaces separate makes
    the renderer reusable across multiple input layouts.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Renderer(Protocol):
    """Structural type for anything that can render agent output."""

    # Banner / startup
    def print_banner(self, model_label: str = "") -> None: ...

    # Routine status lines
    def print_system(self, msg: str) -> None: ...
    def print_error(self, msg: str) -> None: ...
    def print_separator(self) -> None: ...

    # Higher-level rendering
    def print_history(self, messages: list) -> None: ...
    def print_help(self) -> None: ...
    def print_turn_summary(self) -> None: ...

    # Lifecycle
    def goodbye(self) -> None: ...


__all__ = ["Renderer"]
