# jyagent.macos — macOS UI automation primitives.
#
# Reusable building blocks for "click into a non-AppKit / Electron / custom
# canvas app on Mac" workflows (WeChat, certain Tencent / NetEase apps, some
# games).
#
# Design choices:
#
# - NOT registered as agent tools. These are library functions, called from
#   skill scripts via `python -m jyagent.macos.<module> ...` or imported from
#   short ad-hoc scripts. Keeps the agent's system prompt small and platform-
#   isolated.
#
#   See ``jyagent/tools/__init__.py`` for what an *actual* agent tool looks
#   like (it has to be wired into ``_TOOL_FN_MAP``). Nothing in this package
#   appears there.
#
# - Each module guards its macOS-only dependencies (Quartz, AppKit, the
#   ``osascript`` / ``screencapture`` shellouts) at call-time, not import-
#   time. ``canvas_rows`` (pure PIL) is importable on Linux without pyobjc.
#
# - Every module has a ``__main__`` CLI shim with --help, so skill bodies can
#   invoke them with ``run_shell`` rather than pasting Python boilerplate.
#
# - App-specific row classification *profiles* (e.g. WECHAT_SEARCH_PROFILE)
#   do NOT live here — they belong with the skill that uses them. See
#   ``skills/wechat-mac-send/scripts/profiles.py``.
#
# History: this package lived at ``jyagent.tools.macos`` until 2026-05;
# renamed to drop the misleading ``tools.`` prefix because these helpers are
# not registered agent tools. The old import path still works via a
# deprecation shim under ``jyagent/tools/macos/``.

# Sub-modules are NOT eagerly imported here — that would shadow
# ``python -m jyagent.macos.<sub>`` execution with a noisy
# RuntimeWarning about ``<sub>`` already being in sys.modules. Callers
# should import the specific module they need, e.g.::
#
#     from jyagent.macos.canvas_rows import detect_bands
#     from jyagent.macos.mouse import click

__all__: list[str] = []
