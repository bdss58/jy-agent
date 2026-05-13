# jyagent.tools.macos — macOS UI automation primitives.
#
# Reusable building blocks for "click into a non-AppKit / Electron / custom
# canvas app on Mac" workflows (WeChat, certain Tencent / NetEase apps, some
# games). Hard-won during the 2026-05-13 WeChat-Mac image-send session — see
# data/memory/topics/wechat-mac-automation.md for the historical context that
# motivated each module.
#
# Design choices:
#
# - NOT registered as agent tools. These are library functions, called from
#   skill scripts via `python -m jyagent.tools.macos.<module> ...` or imported
#   from short ad-hoc scripts. Keeps the agent's system prompt small and
#   platform-isolated.
#
# - Each module guards its macOS-only dependencies (Quartz, AppKit, the
#   `osascript` / `screencapture` shellouts) at call-time, not import-time.
#   You can import this package on Linux for the pure-Python pieces
#   (canvas_rows is the main one) without pyobjc installed.
#
# - Every module has a `__main__` CLI shim with --help, so skill bodies can
#   invoke them with `run_shell` rather than pasting Python boilerplate.

# Sub-modules are NOT eagerly imported here — that would shadow
# `python -m jyagent.tools.macos.<sub>` execution with a noisy
# RuntimeWarning about `<sub>` already being in sys.modules. Callers
# should import the specific module they need, e.g.::
#
#     from jyagent.tools.macos.canvas_rows import detect_bands
#     from jyagent.tools.macos.mouse import click

__all__: list[str] = []
