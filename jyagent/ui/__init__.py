# UI subpackage — terminal interface (prompt_toolkit + rich) + LoopCallbacks
# implementation that the CLI installs into the agent's run loop.
#
# Public API: import from ``jyagent.ui`` directly.

from .cli import CLI, console
from .terminal import build_streaming_callbacks

__all__ = [
    "CLI",
    "console",
    "build_streaming_callbacks",
]
