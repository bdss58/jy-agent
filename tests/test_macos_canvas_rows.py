"""Tests for jyagent.tools.macos.canvas_rows.

No macOS dependency — pure PIL image synthesis. We fabricate a tiny
"search panel" image that contains:

- A query-suggestion row (green + black text, height ≈ 22 img-px)
- A section header (grey-only, short)
- A real contact row (green name, height ≈ 70 img-px)
- A second suggestion row (sanity: multiple same-label rows)

and verify ``detect_bands`` + ``classify_bands`` identify them correctly.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PIL")

from PIL import Image, ImageDraw

from jyagent.tools.macos.canvas_rows import (
    Band,
    RowProfile,
    RowRule,
    WECHAT_SEARCH_PROFILE,
    classify_bands,
    detect_bands,
    image_x_to_screen_x,
    image_y_to_screen_y,
    is_mid_grey_text,
    is_near_black_text,
    is_wechat_green,
)


# ─── Predicate sanity ────────────────────────────────────────────────────────


def test_green_predicate_catches_wechat_highlight():
    # Typical WeChat keyword-highlight green (hex 07C160 area).
    assert is_wechat_green((7, 193, 96))
    assert is_wechat_green((40, 190, 80))
    # White text, black text, grey text should all be rejected.
    assert not is_wechat_green((255, 255, 255))
    assert not is_wechat_green((20, 20, 20))
    assert not is_wechat_green((150, 150, 150))


def test_black_predicate_catches_dark_neutral_text():
    assert is_near_black_text((15, 15, 15))
    assert is_near_black_text((40, 42, 45))
    # Green and grey must not match.
    assert not is_near_black_text((7, 193, 96))
    assert not is_near_black_text((160, 160, 160))
    # Too bright for "near black".
    assert not is_near_black_text((100, 100, 100))


def test_grey_predicate_is_mid_range_and_neutral():
    assert is_mid_grey_text((150, 150, 150))
    assert is_mid_grey_text((170, 172, 175))
    # Too dark → near-black, not grey.
    assert not is_mid_grey_text((40, 40, 40))
    # Too bright → background, not grey text.
    assert not is_mid_grey_text((220, 220, 220))
    # Not neutral → not grey even if lightness is right.
    assert not is_mid_grey_text((150, 100, 150))


# ─── Retina conversion ───────────────────────────────────────────────────────


def test_image_to_screen_coords_halve_on_retina():
    # A panel anchored at logical screen (222, 92). Image-y 144 → 92 + 72 = 164.
    assert image_y_to_screen_y(144, 92) == 164
    assert image_x_to_screen_x(300, 222) == 222 + 150
    # Non-Retina / 1x stays unscaled.
    assert image_y_to_screen_y(144, 92, scale=1) == 236


# ─── Band detection fixture ──────────────────────────────────────────────────


def _paint_row(draw: ImageDraw.ImageDraw, *, y: int, height: int, color,
               x0: int = 115, x1: int = 300) -> None:
    """Fill a horizontal stripe of ``height`` rows with ``color``."""
    draw.rectangle([x0, y, x1, y + height - 1], fill=color)


@pytest.fixture
def fake_wechat_panel(tmp_path):
    """Synthesize a 736×520 panel with one of every expected row type."""
    W, H = 736, 520
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)

    GREEN = (7, 193, 96)
    BLACK = (20, 20, 20)
    GREY = (160, 160, 160)

    # 1. Suggestion row at y=40, height=22: green (query) + black (suffix).
    _paint_row(d, y=40, height=22, color=GREEN, x0=115, x1=200)
    _paint_row(d, y=40, height=22, color=BLACK, x0=210, x1=400)

    # 2. Another suggestion row at y=80, height=22.
    _paint_row(d, y=80, height=22, color=GREEN, x0=115, x1=180)
    _paint_row(d, y=80, height=22, color=BLACK, x0=190, x1=380)

    # 3. Section header at y=130, height=14, GREY ONLY (no green, no black).
    _paint_row(d, y=130, height=14, color=GREY, x0=115, x1=180)

    # 4. Real contact row at y=170..239, height=70: a continuous 70-px tall
    #    band with green at the top (name) and grey below (preview). In a
    #    real WeChat screenshot the inter-line gap is anti-aliased into a
    #    continuous text "column", which is what we approximate here.
    _paint_row(d, y=170, height=20, color=GREEN, x0=115, x1=300)
    _paint_row(d, y=190, height=2,  color=GREY, x0=115, x1=300)   # bridge
    _paint_row(d, y=192, height=18, color=GREY, x0=115, x1=350)   # subtitle
    _paint_row(d, y=210, height=2,  color=GREY, x0=115, x1=300)   # bridge
    _paint_row(d, y=212, height=28, color=GREY, x0=115, x1=280)

    # 5. Another section header at y=260, height=12.
    _paint_row(d, y=260, height=12, color=GREY, x0=115, x1=170)

    # 6. Group-chat-like contact row at y=295..363, height=68.
    _paint_row(d, y=295, height=20, color=GREEN, x0=115, x1=280)
    _paint_row(d, y=315, height=2,  color=GREY, x0=115, x1=280)
    _paint_row(d, y=317, height=18, color=GREY, x0=115, x1=330)
    _paint_row(d, y=335, height=2,  color=GREY, x0=115, x1=300)
    _paint_row(d, y=337, height=26, color=GREY, x0=115, x1=260)

    path = tmp_path / "panel.png"
    img.save(path)
    return path


# ─── detect_bands ────────────────────────────────────────────────────────────


def test_detect_bands_finds_every_painted_stripe(fake_wechat_panel):
    bands = detect_bands(fake_wechat_panel, WECHAT_SEARCH_PROFILE)
    # We painted 5 visually-distinct bands (two contact rows each merged
    # from their 3 sub-stripes because the gaps are < band_merge_gap=3).
    assert len(bands) == 6, [
        (b.y_start, b.y_end, b.height, sorted(b.signatures)) for b in bands
    ]


def test_detect_bands_aggregates_signatures(fake_wechat_panel):
    bands = detect_bands(fake_wechat_panel, WECHAT_SEARCH_PROFILE)
    # First band is the y=40 suggestion: must have green AND black.
    first = bands[0]
    assert "green" in first.signatures
    assert "black" in first.signatures


def test_detect_bands_returns_empty_for_blank_image(tmp_path):
    blank = Image.new("RGB", (600, 400), (255, 255, 255))
    path = tmp_path / "blank.png"
    blank.save(path)
    assert detect_bands(path, WECHAT_SEARCH_PROFILE) == []


def test_detect_bands_respects_text_column_bounds(tmp_path):
    """A dark stripe OUTSIDE the profile's text column must be ignored."""
    img = Image.new("RGB", (736, 200), (255, 255, 255))
    d = ImageDraw.Draw(img)
    # Paint outside the profile's text_x_start..text_x_end (110..700) range.
    d.rectangle([10, 50, 80, 80], fill=(0, 0, 0))
    path = tmp_path / "outside.png"
    img.save(path)
    assert detect_bands(path, WECHAT_SEARCH_PROFILE) == []


# ─── classify_bands with WECHAT_SEARCH_PROFILE ──────────────────────────────


def test_classify_labels_all_row_types(fake_wechat_panel):
    bands = detect_bands(fake_wechat_panel, WECHAT_SEARCH_PROFILE)
    rows = classify_bands(bands, WECHAT_SEARCH_PROFILE)
    labels = [r.label for r in rows]

    assert labels.count("suggestion") == 2
    assert labels.count("section_header") == 2
    assert labels.count("contact") == 2
    assert "unknown" not in labels


def test_classify_contact_rows_have_expected_height(fake_wechat_panel):
    bands = detect_bands(fake_wechat_panel, WECHAT_SEARCH_PROFILE)
    rows = classify_bands(bands, WECHAT_SEARCH_PROFILE)
    contacts = [r for r in rows if r.label == "contact"]
    assert len(contacts) == 2
    for c in contacts:
        # We painted these as ~70 img-px. Profile allows 50..85.
        assert 50 <= c.band.height <= 85


# ─── Custom RowProfile smoke ────────────────────────────────────────────────


def test_custom_profile_round_trip(tmp_path):
    """A caller can build its own RowProfile for a different app."""
    # Image: one single 30-px-tall dark stripe.
    img = Image.new("RGB", (400, 200), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([50, 40, 300, 70], fill=(10, 10, 10))
    path = tmp_path / "custom.png"
    img.save(path)

    profile = RowProfile(
        name="demo",
        text_x_start=40, text_x_end=350, x_step=2,
        text_darkness_threshold=120,
        band_merge_gap=2,
        signatures={"dark": is_near_black_text},
        rules=(
            RowRule(
                label="item",
                required_signatures=frozenset({"dark"}),
                height_min=20, height_max=50,
            ),
        ),
    )
    rows = classify_bands(detect_bands(path, profile), profile)
    assert len(rows) == 1
    assert rows[0].label == "item"
    assert 20 <= rows[0].band.height <= 50


# ─── Band dataclass helpers ─────────────────────────────────────────────────


def test_band_height_and_center_are_inclusive():
    b = Band(y_start=10, y_end=19, signatures=frozenset())
    assert b.height == 10
    assert b.y_center == 14
