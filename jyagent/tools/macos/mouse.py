"""Synthesize mouse clicks on macOS via Quartz CGEvents.

Usage (Python)::

    from jyagent.tools.macos.mouse import click
    click(406, 365)

Usage (CLI, for skill bodies that prefer run_shell)::

    .venv/bin/python -m jyagent.tools.macos.mouse click 406 365
    .venv/bin/python -m jyagent.tools.macos.mouse move 406 365

Requires ``pyobjc-framework-Quartz``. Install with::

    .venv/bin/python -m ensurepip --upgrade
    .venv/bin/python -m pip install pyobjc-framework-Quartz

Coordinates are **logical screen** coordinates (the same units AppKit uses
for window frames), NOT image pixels. On Retina displays, divide an image-y
from a ``screencapture`` PNG by 2 (and add the panel origin) — see
:func:`jyagent.tools.macos.canvas_rows.image_y_to_screen_y`.

TCC permissions required (one-time):

- System Settings → Privacy & Security → **Accessibility** → toggle ON
  for the terminal that will execute this Python. Without it, events post
  silently and clicks do nothing.
"""

from __future__ import annotations

import time
from typing import Sequence


def _require_quartz():
    try:
        import Quartz  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "mouse module requires pyobjc-framework-Quartz. Install with: "
            ".venv/bin/python -m ensurepip --upgrade && "
            ".venv/bin/python -m pip install pyobjc-framework-Quartz"
        ) from exc
    return Quartz


def move(x: float, y: float, *, settle: float = 0.15) -> None:
    """Move the cursor to (x, y) and pause briefly for the OS to settle."""
    Q = _require_quartz()
    ev = Q.CGEventCreateMouseEvent(
        None, Q.kCGEventMouseMoved, (x, y), Q.kCGMouseButtonLeft
    )
    Q.CGEventPost(Q.kCGHIDEventTap, ev)
    if settle > 0:
        time.sleep(settle)


def click(
    x: float, y: float,
    *,
    button: str = "left",
    pre_move_settle: float = 0.15,
    down_up_gap: float = 0.05,
) -> None:
    """Move to (x, y) and synthesize a single click.

    Parameters
    ----------
    x, y
        Logical screen coordinates (origin top-left on macOS CGEvents API).
    button
        "left" or "right".
    pre_move_settle
        Seconds to pause after the move before posting the button-down.
        Some apps drop click events that arrive in the same frame as the
        cursor move — 150 ms is the smallest value that has worked
        reliably on every app tested so far.
    down_up_gap
        Seconds between button-down and button-up.
    """
    Q = _require_quartz()
    down_type, up_type, btn = {
        "left":  (Q.kCGEventLeftMouseDown,  Q.kCGEventLeftMouseUp,  Q.kCGMouseButtonLeft),
        "right": (Q.kCGEventRightMouseDown, Q.kCGEventRightMouseUp, Q.kCGMouseButtonRight),
    }[button]

    move_ev = Q.CGEventCreateMouseEvent(
        None, Q.kCGEventMouseMoved, (x, y), Q.kCGMouseButtonLeft
    )
    Q.CGEventPost(Q.kCGHIDEventTap, move_ev)
    if pre_move_settle > 0:
        time.sleep(pre_move_settle)

    down_ev = Q.CGEventCreateMouseEvent(None, down_type, (x, y), btn)
    Q.CGEventPost(Q.kCGHIDEventTap, down_ev)
    if down_up_gap > 0:
        time.sleep(down_up_gap)

    up_ev = Q.CGEventCreateMouseEvent(None, up_type, (x, y), btn)
    Q.CGEventPost(Q.kCGHIDEventTap, up_ev)


def double_click(x: float, y: float, *, gap: float = 0.08, **kw) -> None:
    """Two clicks in quick succession. ``gap`` is seconds between the two."""
    click(x, y, **kw)
    time.sleep(gap)
    click(x, y, **kw)


# ─── CLI shim ────────────────────────────────────────────────────────────────


def _cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m jyagent.tools.macos.mouse",
        description="Synthesize mouse events on macOS via Quartz.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("move", "click", "double-click"):
        sp = sub.add_parser(name)
        sp.add_argument("x", type=float)
        sp.add_argument("y", type=float)
        if name != "move":
            sp.add_argument("--button", choices=["left", "right"], default="left")

    args = p.parse_args(argv)
    if args.cmd == "move":
        move(args.x, args.y)
    elif args.cmd == "click":
        click(args.x, args.y, button=args.button)
    elif args.cmd == "double-click":
        double_click(args.x, args.y, button=args.button)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
