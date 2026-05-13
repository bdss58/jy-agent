"""Send keystrokes to the frontmost macOS app via AppleScript.

This is the lowest-friction path on macOS: AppleScript's
``tell application "System Events" to keystroke / key code`` is the API
that the OS itself uses for menu-key emulation, and works against
non-AppKit apps (Electron, custom canvases, games) that AX cannot read.

Python::

    from jyagent.tools.macos.keys import keystroke, key_code, activate
    activate("WeChat")
    keystroke("f", cmd=True)        # ⌘F
    keystroke("v", cmd=True)        # ⌘V
    key_code(36)                    # Return
    key_code(53)                    # Escape

CLI::

    python -m jyagent.tools.macos.keys activate WeChat
    python -m jyagent.tools.macos.keys keystroke f --cmd
    python -m jyagent.tools.macos.keys keycode 36
    python -m jyagent.tools.macos.keys type "hello world"

Common key codes::

    36   Return / Enter
    48   Tab
    51   Delete (backspace)
    53   Escape
    76   keypad Enter
    123  Left arrow
    124  Right arrow
    125  Down arrow
    126  Up arrow
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Sequence


class KeysError(RuntimeError):
    pass


def _osascript(script: str, *, timeout: float = 5.0) -> str:
    if shutil.which("osascript") is None:  # pragma: no cover
        raise KeysError("osascript not found — this module is macOS-only.")
    try:
        out = subprocess.run(
            ["osascript", "-e", script],
            check=True, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        raise KeysError(
            f"osascript failed (exit {exc.returncode}): {exc.stderr.strip()}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise KeysError(f"osascript timed out after {timeout}s") from exc
    return out.stdout.strip()


def activate(app_name: str) -> None:
    """Bring an app to the foreground (creating its launch if needed)."""
    _osascript(f'tell application "{app_name}" to activate')


def _modifier_clause(cmd: bool, shift: bool, opt: bool, ctrl: bool) -> str:
    mods = []
    if cmd:   mods.append("command down")
    if shift: mods.append("shift down")
    if opt:   mods.append("option down")
    if ctrl:  mods.append("control down")
    if not mods:
        return ""
    return f" using {{{', '.join(mods)}}}"


def keystroke(
    char: str,
    *,
    cmd: bool = False, shift: bool = False,
    opt: bool = False, ctrl: bool = False,
) -> None:
    """Send a single character (with optional modifiers) to the frontmost app.

    For non-printable keys, use :func:`key_code` instead.
    """
    if len(char) != 1:
        raise KeysError(f"keystroke wants one char, got {char!r}")
    # Escape backslash and double-quote for AppleScript string literal.
    esc = char.replace("\\", "\\\\").replace('"', '\\"')
    mods = _modifier_clause(cmd, shift, opt, ctrl)
    _osascript(
        f'tell application "System Events" to keystroke "{esc}"{mods}'
    )


def key_code(
    code: int,
    *,
    cmd: bool = False, shift: bool = False,
    opt: bool = False, ctrl: bool = False,
) -> None:
    """Post a raw macOS key code (e.g. 36 = Return, 53 = Escape)."""
    mods = _modifier_clause(cmd, shift, opt, ctrl)
    _osascript(
        f'tell application "System Events" to key code {int(code)}{mods}'
    )


def type_text(text: str) -> None:
    """Type a string by keystroke. Slow but works in any text field.

    Prefer :func:`jyagent.tools.macos.clipboard.set_text_clipboard` + ⌘V
    for anything more than a few characters — System Events keystroke is
    slow and can drop characters under load.
    """
    esc = text.replace("\\", "\\\\").replace('"', '\\"')
    _osascript(f'tell application "System Events" to keystroke "{esc}"')


# Convenience aliases for the most common keys.
def press_return() -> None: key_code(36)
def press_escape() -> None: key_code(53)
def press_tab()    -> None: key_code(48)
def press_down()   -> None: key_code(125)
def press_up()     -> None: key_code(126)


# ─── CLI shim ────────────────────────────────────────────────────────────────


def _cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m jyagent.tools.macos.keys",
        description="Send keystrokes / key codes to the frontmost macOS app.",
    )
    sub = p.add_subparsers(dest="subcmd", required=True)

    sp_act = sub.add_parser("activate")
    sp_act.add_argument("app")

    sp_ks = sub.add_parser("keystroke")
    sp_ks.add_argument("char")
    for flag in ("cmd", "shift", "opt", "ctrl"):
        sp_ks.add_argument(f"--{flag}", action="store_true")

    sp_kc = sub.add_parser("keycode")
    sp_kc.add_argument("code", type=int)
    for flag in ("cmd", "shift", "opt", "ctrl"):
        sp_kc.add_argument(f"--{flag}", action="store_true")

    sp_t = sub.add_parser("type")
    sp_t.add_argument("text")

    args = p.parse_args(argv)
    try:
        if args.subcmd == "activate":
            activate(args.app)
        elif args.subcmd == "keystroke":
            keystroke(args.char, cmd=args.cmd, shift=args.shift,
                      opt=args.opt, ctrl=args.ctrl)
        elif args.subcmd == "keycode":
            key_code(args.code, cmd=args.cmd, shift=args.shift,
                     opt=args.opt, ctrl=args.ctrl)
        elif args.subcmd == "type":
            type_text(args.text)
    except KeysError as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
