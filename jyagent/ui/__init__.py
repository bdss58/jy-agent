# UI subpackage — terminal interface (prompt_toolkit + rich) + LoopCallbacks
# implementation that the CLI installs into the agent's run loop.
#
# Public API:
#   CLI                        — input + renderer facade (used by agent.py)
#   console                    — shared Rich Console with the agent theme
#   build_streaming_callbacks  — factory returning a StreamingUI bundle
#   render_final_text          — markdown-rendered final-answer panel
#   TerminalRenderer           — Rich-on-stdout renderer (base class of CLI)

from .cli import CLI
from .output import console
from .terminal import build_streaming_callbacks, render_final_text, StreamingUI
from .terminal_renderer import TerminalRenderer

__all__ = [
    "CLI",
    "console",
    "build_streaming_callbacks",
    "render_final_text",
    "StreamingUI",
    "TerminalRenderer",
]
