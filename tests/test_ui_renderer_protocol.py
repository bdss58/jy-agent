"""Tests for the ui.renderer.Renderer protocol seam.

The point of the protocol is that swapping renderers must be possible
without editing agent.py.  These tests verify:

  1. ``TerminalRenderer`` (the stock Rich-on-stdout renderer) satisfies
     the structural ``Renderer`` protocol.
  2. ``CLI`` (which inherits from TerminalRenderer) also satisfies it.
  3. A trivial alternative renderer (no Rich, just a recording fake)
     also satisfies the protocol — the proof that the seam is real.
"""
from __future__ import annotations

from jyagent.ui.renderer import Renderer
from jyagent.ui.terminal_renderer import TerminalRenderer


def test_terminal_renderer_satisfies_protocol():
    r = TerminalRenderer()
    # runtime_checkable Protocol → isinstance() works.
    assert isinstance(r, Renderer)


def test_cli_satisfies_protocol():
    # Import here to avoid prompt_toolkit cost when not needed.
    from jyagent.ui.cli import CLI
    cli = CLI()
    assert isinstance(cli, Renderer)


def test_alternative_recorder_renderer_satisfies_protocol():
    """Anything with the right shape passes — no inheritance required."""

    class RecordingRenderer:
        def __init__(self):
            self.events: list[tuple] = []

        def print_banner(self, model_label: str = "") -> None:
            self.events.append(("banner", model_label))

        def print_system(self, msg: str) -> None:
            self.events.append(("system", msg))

        def print_error(self, msg: str) -> None:
            self.events.append(("error", msg))

        def print_separator(self) -> None:
            self.events.append(("sep",))

        def print_history(self, messages: list) -> None:
            self.events.append(("history", len(messages)))

        def print_help(self) -> None:
            self.events.append(("help",))

        def print_turn_summary(self) -> None:
            self.events.append(("turn",))

        def goodbye(self) -> None:
            self.events.append(("bye",))

    r = RecordingRenderer()
    assert isinstance(r, Renderer)
    r.print_system("hi")
    r.print_error("oops")
    r.goodbye()
    assert r.events == [("system", "hi"), ("error", "oops"), ("bye",)]
