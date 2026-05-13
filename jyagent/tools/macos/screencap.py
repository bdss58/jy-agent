"""Thin wrappers around the macOS ``screencapture`` shell command.

Why a module instead of inlining ``screencapture`` calls in skill bodies?

1. Argument quoting differs between zsh / bash / `run_shell`-quoting, and
   getting the ``-R<x>,<y>,<w>,<h>`` region flag right is fiddly.
2. The agent often wants to capture and *then* immediately classify with
   :mod:`canvas_rows`. Bundling produces a single shellout-free Python flow.
3. The module also exposes the Retina scale factor used to convert image
   pixels back to logical screen coords.

CLI::

    .venv/bin/python -m jyagent.tools.macos.screencap full /tmp/full.png
    .venv/bin/python -m jyagent.tools.macos.screencap region 222 92 368 518 /tmp/panel.png
    .venv/bin/python -m jyagent.tools.macos.screencap window /tmp/win.png   # interactive: click target
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence


class ScreencapError(RuntimeError):
    pass


def _require_screencapture() -> str:
    path = shutil.which("screencapture")
    if path is None:  # pragma: no cover
        raise ScreencapError("screencapture not found — this module is macOS-only.")
    return path


def capture_full(out_path: str | Path) -> Path:
    """Capture all attached displays into one PNG. No sound, no UI."""
    sc = _require_screencapture()
    p = Path(out_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sc, "-x", str(p)], check=True, timeout=10)
    return p


def capture_region(
    x: int, y: int, w: int, h: int, out_path: str | Path
) -> Path:
    """Capture a logical-screen rectangle. (x, y) is top-left, in points."""
    sc = _require_screencapture()
    p = Path(out_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    region = f"{x},{y},{w},{h}"
    subprocess.run([sc, "-x", "-R", region, str(p)], check=True, timeout=10)
    return p


def capture_window_interactive(out_path: str | Path) -> Path:
    """Prompt the user to click a window to capture. INTERACTIVE.

    Useful for one-time calibration of new app windows. Not for automation.
    """
    sc = _require_screencapture()
    p = Path(out_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sc, "-w", str(p)], check=True, timeout=120)
    return p


# Retina scale on Apple Silicon / most modern Macs. screencapture writes the
# image at the native resolution of the display, so a 368×518 logical region
# becomes a 736×1036 PNG on a 2x panel.
DEFAULT_RETINA_SCALE = 2


# ─── CLI shim ────────────────────────────────────────────────────────────────


def _cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m jyagent.tools.macos.screencap",
        description="Thin wrappers around macOS screencapture.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_full = sub.add_parser("full")
    sp_full.add_argument("out_path")

    sp_reg = sub.add_parser("region")
    sp_reg.add_argument("x", type=int)
    sp_reg.add_argument("y", type=int)
    sp_reg.add_argument("w", type=int)
    sp_reg.add_argument("h", type=int)
    sp_reg.add_argument("out_path")

    sp_win = sub.add_parser("window")
    sp_win.add_argument("out_path")

    args = p.parse_args(argv)
    try:
        if args.cmd == "full":
            out = capture_full(args.out_path)
        elif args.cmd == "region":
            out = capture_region(args.x, args.y, args.w, args.h, args.out_path)
        elif args.cmd == "window":
            out = capture_window_interactive(args.out_path)
    except ScreencapError as exc:
        print(f"error: {exc}")
        return 2
    print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
