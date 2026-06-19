"""Deterministic evidence-requirement lookup for the agent tool loop.

The reasoner passes a free-form issue-family hint, so this module maps broad
keywords onto the closest object-specific row in evidence_requirements.csv. It
never raises into the loop: unknown objects or issue hints return no specific
match plus the three general baseline requirements.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config import EVIDENCE_REQUIREMENTS_CSV
from io_utils import read_csv_dicts

GENERAL_REQUIREMENT_IDS = [
    "REQ_GENERAL_OBJECT_PART",
    "REQ_GENERAL_MULTI_IMAGE",
    "REQ_REVIEW_TRUST",
]

_SPECIFIC_KEYWORDS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "car": [
        (
            "REQ_CAR_BODY_PANEL",
            (
                "dent",
                "scratch",
                "scrape",
                "mark",
                "hail",
                "panel",
                "bumper",
                "door",
                "hood",
                "fender",
                "quarter",
                "body",
            ),
        ),
        (
            "REQ_CAR_GLASS_LIGHT_MIRROR",
            (
                "crack",
                "cracked",
                "shatter",
                "shattered",
                "broken",
                "missing",
                "glass",
                "windshield",
                "headlight",
                "taillight",
                "light",
                "mirror",
                "component",
            ),
        ),
        (
            "REQ_CAR_IDENTITY_OR_SIDE",
            (
                "identity",
                "orientation",
                "side",
                "left",
                "right",
                "front",
                "rear",
                "vehicle",
                "color",
                "blue",
                "black",
            ),
        ),
    ],
    "laptop": [
        (
            "REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD",
            (
                "screen",
                "display",
                "keyboard",
                "key",
                "keycap",
                "trackpad",
                "touchpad",
                "stain",
                "liquid",
                "water",
                "coffee",
            ),
        ),
        (
            "REQ_LAPTOP_BODY_HINGE_PORT",
            (
                "hinge",
                "lid",
                "corner",
                "body",
                "base",
                "port",
                "edge",
                "side",
                "crack",
                "dent",
                "broken",
            ),
        ),
    ],
    "package": [
        (
            "REQ_PACKAGE_EXTERIOR",
            (
                "crushed",
                "crush",
                "torn",
                "tear",
                "opened",
                "open",
                "seal",
                "flap",
                "corner",
                "side",
                "box",
                "exterior",
                "packaging",
            ),
        ),
        (
            "REQ_PACKAGE_LABEL_OR_STAIN",
            (
                "water",
                "wet",
                "stain",
                "oil",
                "oily",
                "label",
                "unreadable",
                "surface",
            ),
        ),
        (
            "REQ_PACKAGE_CONTENTS",
            (
                "contents",
                "content",
                "inner",
                "inside",
                "item",
                "product",
                "missing",
                "broken item",
            ),
        ),
    ],
}

_REQUIREMENTS_BY_ID: dict[str, dict[str, str]] | None = None


def _compact(row: dict[str, Any]) -> dict[str, str]:
    return {
        "requirement_id": str(row.get("requirement_id", "")).strip(),
        "applies_to": str(row.get("applies_to", "")).strip(),
        "minimum_image_evidence": str(row.get("minimum_image_evidence", "")).strip(),
    }


def _requirements_by_id() -> dict[str, dict[str, str]]:
    global _REQUIREMENTS_BY_ID
    if _REQUIREMENTS_BY_ID is None:
        _REQUIREMENTS_BY_ID = {
            str(row.get("requirement_id", "")).strip(): _compact(row)
            for row in read_csv_dicts(EVIDENCE_REQUIREMENTS_CSV)
        }
    return _REQUIREMENTS_BY_ID


def _specific_requirement_id(claim_object: str, issue_family: str) -> str | None:
    obj = (claim_object or "").strip().lower()
    hint = (issue_family or "").strip().lower()
    if not obj or not hint:
        return None

    rows = _requirements_by_id()
    for requirement_id, keywords in _SPECIFIC_KEYWORDS.get(obj, []):
        row = rows.get(requirement_id)
        applies_to = row["applies_to"].lower() if row else ""
        if row and (applies_to in hint or hint in applies_to):
            return requirement_id
        if any(keyword in hint for keyword in keywords):
            return requirement_id
    return None


def get_evidence_requirement(claim_object: str, issue_family: str) -> dict:
    """Return the best specific requirement plus always-applicable baselines."""
    rows = _requirements_by_id()
    matched_id = _specific_requirement_id(claim_object, issue_family)
    matched = rows.get(matched_id) if matched_id else None
    always_applies = [
        rows[requirement_id]
        for requirement_id in GENERAL_REQUIREMENT_IDS
        if requirement_id in rows
    ]
    return {
        "claim_object": claim_object,
        "issue_family": issue_family,
        "matched": matched,
        "always_applies": always_applies,
    }


def _self_test() -> None:
    car = get_evidence_requirement("car", "dent or scratch")
    assert car["matched"]["requirement_id"] == "REQ_CAR_BODY_PANEL", car
    package = get_evidence_requirement("package", "seal torn open")
    assert package["matched"]["requirement_id"] == "REQ_PACKAGE_EXTERIOR", package
    unknown = get_evidence_requirement("toy", "crack")
    assert unknown["matched"] is None, unknown
    assert (
        [r["requirement_id"] for r in car["always_applies"]]
        == GENERAL_REQUIREMENT_IDS
    )


if __name__ == "__main__":
    _self_test()
    print("evidence_lookup self-test passed")
