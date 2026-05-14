"""Integration test: load skills/wechat-mac-send/scripts/profiles.py
(the WeChat-specific RowProfile) by file path and verify it correctly
classifies a synthesized search-panel screenshot.

This lives at the top-level tests/ dir rather than inside the skill folder
because pytest discovery from repo root is the simplest reliable runner.
The skill scripts/ dir is NOT a Python package (hyphenated name + spec
makes that explicit), so we import by filesystem path with importlib.util.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image, ImageDraw

from jyagent.macos.canvas_rows import classify_bands, detect_bands


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "skills" / "wechat-mac-send" / "scripts" / "profiles.py"


def _load_wechat_profile():
    spec = importlib.util.spec_from_file_location(
        "wechat_search_profile_under_test", PROFILE_PATH
    )
    assert spec is not None and spec.loader is not None, (
        f"could not build import spec for {PROFILE_PATH}"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WECHAT_SEARCH_PROFILE


def test_profile_file_exists():
    assert PROFILE_PATH.exists(), (
        f"WeChat profile not found at {PROFILE_PATH}. The skill is supposed "
        f"to ship its row-classification profile in scripts/profiles.py."
    )


def test_profile_has_expected_rules():
    profile = _load_wechat_profile()
    labels = {rule.label for rule in profile.rules}
    assert labels == {"contact", "suggestion", "section_header"}, labels
    assert profile.name == "wechat_search_panel"


@pytest.fixture
def fake_wechat_panel(tmp_path):
    """Synthesize a 736×400 panel that matches the calibrated thresholds."""
    img = Image.new("RGB", (736, 400), (255, 255, 255))
    d = ImageDraw.Draw(img)
    GREEN = (7, 193, 96)
    BLACK = (20, 20, 20)
    GREY = (160, 160, 160)

    # Suggestion row at y=40, height=22: green query + black suffix.
    d.rectangle([115, 40, 200, 61], fill=GREEN)
    d.rectangle([210, 40, 400, 61], fill=BLACK)

    # Section header at y=100, height=14: grey only.
    d.rectangle([115, 100, 180, 113], fill=GREY)

    # Contact row at y=140..209, height=70.
    d.rectangle([115, 140, 300, 159], fill=GREEN)
    d.rectangle([115, 160, 300, 161], fill=GREY)
    d.rectangle([115, 162, 350, 179], fill=GREY)
    d.rectangle([115, 180, 300, 181], fill=GREY)
    d.rectangle([115, 182, 280, 209], fill=GREY)

    out = tmp_path / "panel.png"
    img.save(out)
    return out


def test_profile_classifies_synthetic_panel(fake_wechat_panel):
    profile = _load_wechat_profile()
    rows = classify_bands(detect_bands(fake_wechat_panel, profile), profile)
    labels = [r.label for r in rows]
    assert labels.count("suggestion") == 1
    assert labels.count("section_header") == 1
    assert labels.count("contact") == 1


def test_profile_picks_first_contact_band(fake_wechat_panel):
    """The skill orchestrator clicks the FIRST contact band — verify there
    is exactly one in this scene and its y_center falls in the right range."""
    profile = _load_wechat_profile()
    rows = classify_bands(detect_bands(fake_wechat_panel, profile), profile)
    contact_rows = [r for r in rows if r.label == "contact"]
    assert len(contact_rows) == 1
    band = contact_rows[0].band
    # Painted at y=140..209, so y_center ≈ 174.
    assert 165 <= band.y_center <= 185, band.y_center
