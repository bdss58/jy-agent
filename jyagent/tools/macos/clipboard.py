"""macOS clipboard helpers for pasting images as real attachments (⌘V).

The key trick: apps like WeChat need an image on the clipboard AS AN IMAGE
(NSPasteboardTypePNG / «class PNGf»), NOT as a file path or URL. If you use
``pbcopy`` or drag a POSIX path in, the receiving app pastes the path as
text instead of embedding the picture.

The reliable recipe is an AppleScript one-liner that reads the file with
the PNGf class coercion and sets the clipboard directly.

Python::

    from jyagent.tools.macos.clipboard import (
        set_image_clipboard, set_text_clipboard, clipboard_info,
    )
    set_image_clipboard("/tmp/cat.png")
    info = clipboard_info()
    assert "«class PNGf»" in info

CLI::

    .venv/bin/python -m jyagent.tools.macos.clipboard set-image /tmp/cat.png
    .venv/bin/python -m jyagent.tools.macos.clipboard set-text "文件传输助手"
    .venv/bin/python -m jyagent.tools.macos.clipboard info
    .venv/bin/python -m jyagent.tools.macos.clipboard verify-image  # exit 0 iff PNGf on clipboard

WARNING — text clobbers image
-----------------------------
Pasting text to the clipboard (e.g. the name of the contact you're searching
for) REPLACES any image that was there. Always re-load the image to the
clipboard AFTER any text-paste step and BEFORE the ⌘V that commits the
attachment. :func:`ensure_image_on_clipboard` below is the idempotent form
that checks first and reloads only if needed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence


class ClipboardError(RuntimeError):
    """Raised when a clipboard operation fails or verification does not match."""


def _osascript(script: str, *, timeout: float = 5.0) -> str:
    """Run an osascript one-liner and return stdout (stripped)."""
    if shutil.which("osascript") is None:  # pragma: no cover
        raise ClipboardError("osascript not found — this module is macOS-only.")
    try:
        out = subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        raise ClipboardError(
            f"osascript failed (exit {exc.returncode}): {exc.stderr.strip()}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ClipboardError(f"osascript timed out after {timeout}s") from exc
    return out.stdout.strip()


def set_image_clipboard(path: str | Path) -> None:
    """Load a PNG file onto the clipboard as an image (PNGf class)."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ClipboardError(f"image file not found: {p}")
    # NOTE: POSIX-path alias coercion is the one that reliably round-trips
    # into PNGf on macOS 12+. ``read POSIX file "..." as «class PNGf»`` in a
    # single read works too, but tripping up on the alias form is less common.
    script = (
        f'set the clipboard to '
        f'(read (POSIX file "{p}") as «class PNGf»)'
    )
    _osascript(script)


def set_text_clipboard(text: str) -> None:
    """Put plain text on the clipboard. NOTE: this clobbers any image."""
    # Use pbcopy rather than osascript so arbitrary characters (quotes, CJK,
    # newlines) round-trip without escaping headaches.
    if shutil.which("pbcopy") is None:  # pragma: no cover
        raise ClipboardError("pbcopy not found — this module is macOS-only.")
    try:
        subprocess.run(
            ["pbcopy"],
            input=text,
            text=True,
            check=True,
            timeout=5.0,
        )
    except subprocess.CalledProcessError as exc:
        raise ClipboardError(f"pbcopy failed: {exc.stderr}") from exc


def clipboard_info() -> str:
    """Return osascript's raw ``clipboard info`` output.

    Example output when an image is loaded::

        «class PNGf», 1432189, «class jp2 », 1432189, JPEG picture, ...

    When only text is loaded::

        string, 27, Unicode text, 54, utf8 text, 27
    """
    return _osascript("clipboard info")


def clipboard_has_image() -> bool:
    """True iff the clipboard currently holds a pasteable image."""
    info = clipboard_info()
    return "«class PNGf»" in info or "JPEG picture" in info or "TIFF picture" in info


def ensure_image_on_clipboard(path: str | Path) -> None:
    """Load image if the clipboard doesn't already contain a pasteable image.

    Idempotent. Safe to call right before ⌘V to guarantee the paste will
    land as an image attachment and not as stale text.
    """
    if clipboard_has_image():
        return
    set_image_clipboard(path)
    if not clipboard_has_image():
        raise ClipboardError(
            f"tried to load {path} as image but clipboard still has no image. "
            f"Current info: {clipboard_info()}"
        )


# ─── CLI shim ────────────────────────────────────────────────────────────────


def _cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m jyagent.tools.macos.clipboard",
        description="macOS clipboard helpers (image-as-PNGf, text, verify).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_img = sub.add_parser("set-image")
    sp_img.add_argument("path")

    sp_txt = sub.add_parser("set-text")
    sp_txt.add_argument("text")

    sub.add_parser("info")

    sp_v = sub.add_parser("verify-image")
    sp_v.add_argument(
        "--reload", metavar="PATH", default=None,
        help="If no image is present, reload from this path and re-verify.",
    )

    args = p.parse_args(argv)
    try:
        if args.cmd == "set-image":
            set_image_clipboard(args.path)
            print(clipboard_info())
        elif args.cmd == "set-text":
            set_text_clipboard(args.text)
        elif args.cmd == "info":
            print(clipboard_info())
        elif args.cmd == "verify-image":
            if args.reload:
                ensure_image_on_clipboard(args.reload)
            if not clipboard_has_image():
                print("NO IMAGE on clipboard")
                print(clipboard_info())
                return 1
            print("image on clipboard OK")
            print(clipboard_info())
    except ClipboardError as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
