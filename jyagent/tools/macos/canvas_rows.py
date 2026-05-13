"""Pixel-structural row classification for non-AppKit / custom-canvas Mac apps.

When AppleScript / AX cannot read the UI tree (WeChat for Mac, many Electron
apps, custom render canvases), and a vision API is unreliable, we can still
locate clickable rows deterministically by scanning a screenshot for text
"bands" and classifying each by color signatures + band height.

This module is **pure PIL** — no Quartz, no macOS-only deps. The screenshot
itself is captured separately (see :mod:`jyagent.tools.macos.screencap` for
the macOS-specific capture wrapper).

Vocabulary
----------
band
    A vertical run of image rows where at least one pixel in the text column
    is darker than the page background. Bands are separated by gaps of
    background-only rows.
signature
    A named color test we run on each pixel of a band — e.g. "is this pixel
    bright green like a WeChat keyword highlight?".  A band's classification
    is "which signatures fired anywhere inside it".
classification
    The downstream meaning of a band (suggestion row, section header, real
    contact row, etc.). The mapping from (signatures, band_height) to
    classification is application-specific and lives in the caller's
    :class:`RowProfile`.

Typical use
-----------
.. code-block:: python

    from jyagent.tools.macos.canvas_rows import (
        detect_bands, classify_bands, WECHAT_SEARCH_PROFILE,
    )

    bands = detect_bands("/tmp/search-panel.png")
    rows = classify_bands(bands, WECHAT_SEARCH_PROFILE)
    contact_rows = [r for r in rows if r.label == "contact"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence


# ─── Data types ──────────────────────────────────────────────────────────────


Pixel = tuple[int, int, int]
PixelTest = Callable[[Pixel], bool]


@dataclass(frozen=True)
class Band:
    """A vertical run of consecutive rows containing text-dark pixels."""

    y_start: int        # inclusive, in image pixel coords (top-left origin)
    y_end: int          # inclusive
    signatures: frozenset[str]  # which signature names fired anywhere inside

    @property
    def height(self) -> int:
        return self.y_end - self.y_start + 1

    @property
    def y_center(self) -> int:
        return (self.y_start + self.y_end) // 2


@dataclass(frozen=True)
class ClassifiedRow:
    """A band plus its caller-defined classification label."""

    band: Band
    label: str           # e.g. "contact", "suggestion", "section_header", "unknown"


@dataclass(frozen=True)
class RowRule:
    """One classification rule.

    A rule matches a band when:
      - ``required_signatures`` are ALL present in the band, AND
      - ``forbidden_signatures`` are ALL absent, AND
      - ``height_min <= band.height <= height_max``.

    Rules are tried in declaration order; first match wins.
    """

    label: str
    required_signatures: frozenset[str] = frozenset()
    forbidden_signatures: frozenset[str] = frozenset()
    height_min: int = 0
    height_max: int = 10**6

    def matches(self, band: Band) -> bool:
        if not self.required_signatures.issubset(band.signatures):
            return False
        if self.forbidden_signatures & band.signatures:
            return False
        return self.height_min <= band.height <= self.height_max


@dataclass(frozen=True)
class RowProfile:
    """Application-specific config for band detection + classification.

    All offsets / heights are in **image pixels** (Retina @2x doubles the
    logical coords seen on screen — keep that in mind when converting back
    to click coordinates).
    """

    name: str
    # X range of the text column to scan (skip leading icon column).
    text_x_start: int
    text_x_end: int
    # X step for the scan loop. 3 is a good speed/accuracy tradeoff at @2x.
    x_step: int = 3
    # A row counts as "text" if the darkest pixel in [x_start, x_end] is
    # below this threshold. 180 is right for white-background panels.
    text_darkness_threshold: int = 180
    # Bands separated by a gap of <= this many rows are merged. WeChat
    # text bands have 1-3px interior anti-aliasing gaps, so 3 is safe.
    band_merge_gap: int = 3
    # Named signature predicates (pixel-level tests).
    signatures: dict[str, PixelTest] = field(default_factory=dict)
    # Classification rules, tried in order.
    rules: tuple[RowRule, ...] = ()


# ─── Built-in signature predicates ───────────────────────────────────────────
#
# Each takes a single (r, g, b) tuple and returns True if the pixel matches.
# Keep them cheap — they're called for every pixel in every band scan.


def is_wechat_green(px: Pixel) -> bool:
    """WeChat search keyword highlight: bright green text on white bg."""
    r, g, b = px
    return g > 150 and r < 130 and b < 130


def is_near_black_text(px: Pixel) -> bool:
    """Standard near-black UI text — r ≈ g ≈ b, all dark."""
    r, g, b = px
    return r < 80 and abs(r - g) < 8 and abs(g - b) < 8


def is_mid_grey_text(px: Pixel) -> bool:
    """Grey section-header / disabled text. r ≈ g ≈ b, mid range."""
    r, g, b = px
    return 120 < r < 200 and abs(r - g) < 8 and abs(g - b) < 8


# ─── Core detection ──────────────────────────────────────────────────────────


def _open_image(image_or_path):
    """Accept a path str/Path or an already-open PIL Image. Return PIL Image (RGB)."""
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — install Pillow
        raise RuntimeError(
            "canvas_rows requires Pillow. Install with: "
            ".venv/bin/python -m pip install Pillow"
        ) from exc

    if hasattr(image_or_path, "convert"):
        return image_or_path.convert("RGB")
    return Image.open(image_or_path).convert("RGB")


def detect_bands(image_or_path, profile: "RowProfile") -> list[Band]:
    """Scan an image and return all text-bands matching ``profile``.

    A "text band" is a maximal run of consecutive image-rows that contain
    at least one pixel below ``profile.text_darkness_threshold`` in the
    profile's text column. Bands separated by a gap of
    ``<= profile.band_merge_gap`` rows are merged.

    For each band we also record which named signatures fire **anywhere
    inside it** (any pixel in any row matches the predicate). This avoids
    re-scanning later.
    """
    img = _open_image(image_or_path)
    W, H = img.size
    x_end = min(profile.text_x_end, W)
    x_start = max(profile.text_x_start, 0)
    if x_start >= x_end:
        return []

    px = img.load()
    step = max(1, profile.x_step)
    dark_thresh = profile.text_darkness_threshold
    sigs = list(profile.signatures.items())

    # Per-row: (has_text, set-of-signature-names-firing-this-row)
    row_text: list[bool] = [False] * H
    row_sigs: list[set[str]] = [set() for _ in range(H)]

    for y in range(H):
        min_v = 256
        local_sigs: set[str] = set()
        for x in range(x_start, x_end, step):
            r, g, b = px[x, y]
            m = min(r, g, b)
            if m < min_v:
                min_v = m
            for name, test in sigs:
                if name in local_sigs:
                    continue
                if test((r, g, b)):
                    local_sigs.add(name)
        if min_v < dark_thresh:
            row_text[y] = True
        row_sigs[y] = local_sigs

    # Group consecutive text rows into raw bands.
    raw: list[tuple[int, int]] = []
    i = 0
    while i < H:
        if row_text[i]:
            j = i
            while j + 1 < H and row_text[j + 1]:
                j += 1
            raw.append((i, j))
            i = j + 1
        else:
            i += 1

    # Merge bands separated by small gaps.
    merged: list[tuple[int, int]] = []
    for s, e in raw:
        if merged and s - merged[-1][1] - 1 <= profile.band_merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # Aggregate signatures inside each merged band.
    bands: list[Band] = []
    for s, e in merged:
        agg: set[str] = set()
        for y in range(s, e + 1):
            agg.update(row_sigs[y])
        bands.append(Band(y_start=s, y_end=e, signatures=frozenset(agg)))

    return bands


def classify_bands(
    bands: Iterable[Band],
    profile: "RowProfile",
    default_label: str = "unknown",
) -> list[ClassifiedRow]:
    """Apply ``profile.rules`` in order; first match wins, else ``default_label``."""
    out: list[ClassifiedRow] = []
    for band in bands:
        label = default_label
        for rule in profile.rules:
            if rule.matches(band):
                label = rule.label
                break
        out.append(ClassifiedRow(band=band, label=label))
    return out


# ─── Retina coordinate conversion ────────────────────────────────────────────


def image_y_to_screen_y(
    image_y: int, panel_origin_y: int, scale: int = 2
) -> float:
    """Convert an image-pixel Y inside a screenshot to a logical screen Y.

    macOS `screencapture` captures at native (Retina) resolution, so a
    1190×892 logical panel becomes a 2380×1784 PNG on a 2x display. To
    click back at the right spot we divide image-y by the scale and add
    the panel's logical-screen origin.
    """
    return panel_origin_y + image_y / scale


def image_x_to_screen_x(
    image_x: int, panel_origin_x: int, scale: int = 2
) -> float:
    return panel_origin_x + image_x / scale


# ─── Built-in profile: WeChat for Mac search panel ───────────────────────────
#
# Calibrated 2026-05-13 on macOS 14, WeChat 4.x, default theme, English+CJK
# UI. Search panel size: 368×518 logical (= 736×1036 image px at @2x). The
# real-contact row is 67-72 image-px tall (2 lines: name on top, last-msg
# preview on bottom). Query-suggestion rows are 22 image-px tall (1 line).
# Section headers ("功能", "群聊", "相关搜索") are grey-only and short.

WECHAT_SEARCH_PROFILE = RowProfile(
    name="wechat_search_panel",
    text_x_start=110,   # skip icon / avatar column
    text_x_end=700,     # leave a margin past the right edge of text column
    x_step=3,
    text_darkness_threshold=180,
    band_merge_gap=3,
    signatures={
        "green": is_wechat_green,
        "black": is_near_black_text,
        "grey":  is_mid_grey_text,
    },
    rules=(
        # Real contact: tall row, with green-highlighted name + grey preview.
        # Height range covers single-line (no preview yet) up to 2-line.
        RowRule(
            label="contact",
            required_signatures=frozenset({"green"}),
            height_min=50, height_max=85,
        ),
        # Query suggestion: short row, mixes green (the query) + black (suffix).
        RowRule(
            label="suggestion",
            required_signatures=frozenset({"green", "black"}),
            height_min=15, height_max=30,
        ),
        # Section header: short, grey-only, NO green, NO black.
        RowRule(
            label="section_header",
            required_signatures=frozenset({"grey"}),
            forbidden_signatures=frozenset({"green", "black"}),
            height_min=8, height_max=22,
        ),
    ),
)


# ─── CLI shim ────────────────────────────────────────────────────────────────


def _cli(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json
    import sys

    p = argparse.ArgumentParser(
        prog="python -m jyagent.tools.macos.canvas_rows",
        description="Classify rows in a Mac-app screenshot by pixel signatures.",
    )
    p.add_argument("image", help="Path to PNG screenshot (Retina @2x).")
    p.add_argument(
        "--profile", default="wechat_search",
        choices=["wechat_search"],
        help="Which built-in profile to apply. (Add more by editing this module.)",
    )
    p.add_argument(
        "--only", default=None,
        help="If set, print only rows whose label matches this string.",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of pretty text.",
    )
    args = p.parse_args(argv)

    profile = {"wechat_search": WECHAT_SEARCH_PROFILE}[args.profile]
    bands = detect_bands(args.image, profile)
    rows = classify_bands(bands, profile)
    if args.only:
        rows = [r for r in rows if r.label == args.only]

    if args.json:
        json.dump(
            [
                {
                    "label": r.label,
                    "y_start": r.band.y_start,
                    "y_end": r.band.y_end,
                    "y_center": r.band.y_center,
                    "height": r.band.height,
                    "signatures": sorted(r.band.signatures),
                }
                for r in rows
            ],
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        print(f"profile={profile.name}  image={args.image}  rows={len(rows)}")
        for r in rows:
            sigs = ",".join(sorted(r.band.signatures)) or "-"
            print(
                f"  [{r.label:>15s}]  y={r.band.y_start:>4d}..{r.band.y_end:<4d}"
                f"  h={r.band.height:>3d}  sigs={sigs}"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
