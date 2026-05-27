from __future__ import annotations

from pathlib import Path


def test_debug_overlay_user_manual_exists_and_contains_key_strings():
    manual = Path("docs/DEBUG_OVERLAY_USER_MANUAL.md")

    assert manual.exists()
    text = manual.read_text(encoding="utf-8")
    for key in [
        "side:LEFT",
        "cut:NONE",
        "cv:False",
        "cutin_valid_for_safety",
        "Dg",
        "Ds",
        "Df",
        "Dbump",
        "CAIS",
        "Sentinel",
    ]:
        assert key in text
