"""Row-classification profiles for skills/wechat-mac-send.

These are application-specific :class:`jyagent.macos.canvas_rows.RowProfile`
instances. They live with the skill — not in the generic ``jyagent.macos``
library — because a Tencent reskin of WeChat invalidates the thresholds
without touching the generic detector. Keeping the profile next to the
skill makes "WeChat changed colors" a one-file fix, and prevents future
canvas-app skills (qq-mac-send, netease-cloud-music-mac) from forking
``canvas_rows.py`` to ship their own.

Calibration log
---------------
2026-05-13 — Initial calibration: macOS 14, WeChat 4.x, default theme,
    English + CJK UI. Search panel size 368×518 logical points
    (= 736×1036 image px at @2x Retina). Real-contact rows are 67-72
    image-px tall (2 lines: name + last-message preview). Query-suggestion
    rows are 22 image-px tall (1 line). Section headers ("功能", "群聊",
    "相关搜索") are grey-only and short.

When a calibration change is needed, log the date / WeChat version / what
shifted here, then update the rule heights/thresholds below. Treat this
file as the operational record, not the generic library.
"""

from __future__ import annotations

from jyagent.macos.canvas_rows import (
    RowProfile,
    RowRule,
    is_mid_grey_text,
    is_near_black_text,
    is_wechat_green,
)


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
#
# Lets skill bodies invoke the WeChat-specific classifier without writing
# Python. Mirrors the generic CLI in ``jyagent.macos.canvas_rows`` but
# preloads ``WECHAT_SEARCH_PROFILE``.
#
# Usage::
#
#   .venv/bin/python skills/wechat-mac-send/scripts/profiles.py \
#       /tmp/wx-search.png --only contact --json


def _cli(argv=None) -> int:
    import argparse
    import json
    import sys

    from jyagent.macos.canvas_rows import classify_bands, detect_bands

    p = argparse.ArgumentParser(
        prog="profiles.py",
        description=(
            "Classify rows in a WeChat-for-Mac search-panel screenshot."
        ),
    )
    p.add_argument("image", help="Path to PNG screenshot (Retina @2x).")
    p.add_argument(
        "--only", default=None,
        help="If set, print only rows whose label matches this string.",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of pretty text.",
    )
    args = p.parse_args(argv)

    profile = WECHAT_SEARCH_PROFILE
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
    import sys
    # Allow running directly: PYTHONPATH must include repo root so
    # ``import jyagent.macos.canvas_rows`` resolves.
    raise SystemExit(_cli(sys.argv[1:]))
