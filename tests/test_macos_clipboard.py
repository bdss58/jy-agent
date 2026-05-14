"""Tests for jyagent.macos.clipboard.

These tests are pure-Python: we monkeypatch the ``_osascript`` shellout (and
``subprocess.run`` for ``pbcopy``) so the test suite is portable and fast.
Real macOS-only behavior (NSPasteboard round-trip) is implicitly covered by
the manual ``python -m jyagent.macos.clipboard`` smoke runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jyagent.macos import clipboard as cb


# ─── Helpers ────────────────────────────────────────────────────────────────


class _FakeOsascript:
    """Capture each osascript invocation; return canned values."""

    def __init__(self, return_value: str = ""):
        self.calls: list[str] = []
        self.return_value = return_value

    def __call__(self, script: str, *, timeout: float = 5.0) -> str:
        self.calls.append(script)
        return self.return_value


@pytest.fixture
def fake_osascript(monkeypatch):
    fake = _FakeOsascript()
    monkeypatch.setattr(cb, "_osascript", fake)
    return fake


@pytest.fixture
def tmp_png(tmp_path) -> Path:
    p = tmp_path / "img.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    return p


# ─── set_image_clipboard: path escaping (regression test for codex P2 #1) ───


def test_set_image_clipboard_rejects_missing_file(fake_osascript, tmp_path):
    with pytest.raises(cb.ClipboardError, match="not found"):
        cb.set_image_clipboard(tmp_path / "no-such.png")
    assert fake_osascript.calls == []


def test_set_image_clipboard_writes_correct_script(fake_osascript, tmp_png):
    cb.set_image_clipboard(tmp_png)
    assert len(fake_osascript.calls) == 1
    script = fake_osascript.calls[0]
    assert "set the clipboard to" in script
    assert "«class PNGf»" in script
    # Path is interpolated inside double quotes.
    assert f'"{tmp_png}"' in script


def test_set_image_clipboard_escapes_double_quote_in_path(
    fake_osascript, tmp_path
):
    """Path with a literal `"` must not break the AppleScript string."""
    weird_dir = tmp_path / 'has "quote" in dir'
    weird_dir.mkdir()
    p = weird_dir / "img.png"
    p.write_bytes(b"\x89PNG")

    cb.set_image_clipboard(p)
    script = fake_osascript.calls[0]
    # The literal `"` must be escaped as `\"` so the AppleScript string
    # closes only at the outer quote — never in the middle of the path.
    assert '\\"quote\\"' in script
    # And there must be exactly TWO unescaped quotes (the opening and
    # closing of the string literal). Count by replacing escaped ones.
    unescaped = script.replace('\\"', "")
    assert unescaped.count('"') == 2


def test_set_image_clipboard_escapes_backslash_in_path(
    fake_osascript, tmp_path, monkeypatch
):
    """Backslash is AppleScript's escape char — must be doubled."""
    # Backslash isn't legal on macOS APFS so we can't actually create such a
    # file; bypass the existence check to focus on the escaping logic.
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda self: Path(r"/tmp/weird\dir/img.png"),
    )
    cb.set_image_clipboard("/anything")
    script = fake_osascript.calls[0]
    # `\d` in the original path must appear as `\\d` in the script.
    assert r"\\dir" in script


# ─── ensure_image_on_clipboard: ALWAYS reload (regression test for #2) ──────


def test_ensure_image_on_clipboard_always_reloads(monkeypatch, tmp_png):
    """Even when an image is already present, we must reload the requested one.

    Before the fix, ``ensure_image_on_clipboard`` short-circuited on
    ``clipboard_has_image()`` and could leave a stale attachment in place —
    exactly the wrong-attachment failure mode the WeChat workflow guards
    against. After the fix it must always call set_image_clipboard.
    """
    set_image_calls: list[Path] = []

    def fake_set_image(path):
        set_image_calls.append(Path(path))

    # has_image returns True the first time (simulating a stale image
    # already on the clipboard), then True after we re-load.
    monkeypatch.setattr(cb, "set_image_clipboard", fake_set_image)
    monkeypatch.setattr(cb, "clipboard_has_image", lambda: True)

    cb.ensure_image_on_clipboard(tmp_png)
    assert set_image_calls == [tmp_png], (
        "ensure_image_on_clipboard must always reload — stale-image short-"
        "circuit would silently send the wrong attachment."
    )


def test_ensure_image_on_clipboard_raises_when_reload_does_not_stick(
    monkeypatch, tmp_png
):
    monkeypatch.setattr(cb, "set_image_clipboard", lambda p: None)
    monkeypatch.setattr(cb, "clipboard_has_image", lambda: False)
    monkeypatch.setattr(cb, "clipboard_info", lambda: "string, 4")

    with pytest.raises(cb.ClipboardError, match="still has no"):
        cb.ensure_image_on_clipboard(tmp_png)


# ─── clipboard_has_image: type sniffing ─────────────────────────────────────


@pytest.mark.parametrize(
    "info,expected",
    [
        ("«class PNGf», 1432189, JPEG picture, 1432189", True),
        ("JPEG picture, 894123", True),
        ("TIFF picture, 1024", True),
        ("string, 27, Unicode text, 54, utf8 text, 27", False),
        ("", False),
    ],
)
def test_clipboard_has_image_sniffs_known_image_types(
    monkeypatch, info, expected
):
    monkeypatch.setattr(cb, "clipboard_info", lambda: info)
    assert cb.clipboard_has_image() is expected


# ─── set_text_clipboard goes through pbcopy, not osascript ──────────────────


def test_set_text_clipboard_uses_pbcopy(monkeypatch):
    pbcopy_calls: list[dict] = []

    def fake_run(argv, *, input, text, check, timeout):
        pbcopy_calls.append({"argv": argv, "input": input})
        class _R: returncode = 0
        return _R()

    monkeypatch.setattr(cb.subprocess, "run", fake_run)
    monkeypatch.setattr(cb.shutil, "which", lambda name: "/usr/bin/pbcopy")

    cb.set_text_clipboard("文件传输助手")
    assert len(pbcopy_calls) == 1
    assert pbcopy_calls[0]["argv"] == ["pbcopy"]
    assert pbcopy_calls[0]["input"] == "文件传输助手"
