from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from config import (
    CLAIMS_CSV,
    DATASET_ROOT,
    EVIDENCE_REQUIREMENTS_CSV,
    SAMPLE_CLAIMS_CSV,
    USER_HISTORY_CSV,
)
from io_utils import read_csv_dicts, resolve_images

EXPECTED_CLAIM_ROWS = 44
ALLOWED_ISSUE_TYPES = {
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
}
ALLOWED_OBJECT_PARTS = {
    "front_bumper",
    "rear_bumper",
    "door",
    "hood",
    "windshield",
    "side_mirror",
    "headlight",
    "taillight",
    "fender",
    "quarter_panel",
    "body",
    "screen",
    "keyboard",
    "trackpad",
    "hinge",
    "lid",
    "corner",
    "port",
    "base",
    "box",
    "package_corner",
    "package_side",
    "seal",
    "label",
    "contents",
    "item",
    "unknown",
}
ALLOWED_RISK_FLAGS = {
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
}
ALLOWED_SEVERITIES = {"none", "low", "medium", "high", "unknown"}
ALLOWED_CLAIM_STATUSES = {
    "supported",
    "contradicted",
    "not_enough_information",
}


def _case_number(image_path: str) -> int | None:
    parts = Path(image_path).parts
    for part in parts:
        if part.startswith("case_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                return None
    return None


def _report_vocab(
    rows: list[dict[str, str]],
    column: str,
    allowed: set[str],
    split_values: bool = False,
) -> tuple[set[str], set[str], set[str]]:
    observed: set[str] = set()
    for row in rows:
        raw_value = row[column].strip()
        if split_values:
            values = [part.strip() for part in raw_value.split(";") if part.strip()]
            if not values:
                values = ["none"]
        else:
            values = [raw_value]
        observed.update(values)
    return observed, observed - allowed, allowed - observed


def main() -> int:
    failures: list[str] = []

    claims_rows = read_csv_dicts(CLAIMS_CSV)
    sample_rows = read_csv_dicts(SAMPLE_CLAIMS_CSV)
    history_rows = read_csv_dicts(USER_HISTORY_CSV)
    _ = read_csv_dicts(EVIDENCE_REQUIREMENTS_CSV)

    print("== Data audit ==")
    print(f"claims.csv rows: {len(claims_rows)}")
    if len(claims_rows) != EXPECTED_CLAIM_ROWS:
        failures.append(
            f"Expected {EXPECTED_CLAIM_ROWS} claim rows, found {len(claims_rows)}"
        )

    total_images = 0
    per_row_counts: list[int] = []
    missing_files: list[str] = []
    outside_dataset: list[str] = []
    for row in claims_rows:
        resolved = resolve_images(row["image_paths"])
        per_row_counts.append(len(resolved))
        total_images += len(resolved)
        if not 1 <= len(resolved) <= 3:
            failures.append(
                f"user_id={row['user_id']} has {len(resolved)} images; expected 1-3"
            )
        for _, image_path in resolved:
            try:
                image_path.resolve(strict=False).relative_to(
                    DATASET_ROOT.resolve(strict=False)
                )
            except ValueError:
                outside_dataset.append(str(image_path))
            if not image_path.is_file():
                missing_files.append(str(image_path))
    print(f"total referenced images: {total_images}")
    print(f"per-row image counts: {per_row_counts}")
    if missing_files:
        print("missing image files:")
        for path in missing_files:
            print(f"  - {path}")
        failures.append(f"Missing {len(missing_files)} referenced image files")
    if outside_dataset:
        print("paths outside dataset root:")
        for path in outside_dataset:
            print(f"  - {path}")
        failures.append(f"Found {len(outside_dataset)} image paths outside dataset root")

    history_user_ids = {row["user_id"] for row in history_rows}
    missing_users = sorted(
        {row["user_id"] for row in claims_rows if row["user_id"] not in history_user_ids}
    )
    print(f"missing user history ids: {missing_users if missing_users else 'none'}")
    if missing_users:
        failures.append(
            f"Missing user_history rows for {len(missing_users)} claim user ids"
        )

    duplicate_counts = Counter(row["user_id"] for row in claims_rows)
    duplicates = {user_id: count for user_id, count in duplicate_counts.items() if count > 1}
    print(f"duplicate user_ids across claims: {duplicates if duplicates else 'none'}")

    observed_cases = sorted(
        {
            case_number
            for row in claims_rows
            for raw_path in row["image_paths"].split(";")
            if raw_path.strip()
            for case_number in [_case_number(raw_path.strip())]
            if case_number is not None
        }
    )
    if observed_cases:
        case_gaps = [
            case_number
            for case_number in range(observed_cases[0], observed_cases[-1] + 1)
            if case_number not in observed_cases
        ]
        print(f"observed case folders: {observed_cases}")
        print(f"missing case numbers through case_{observed_cases[-1]:03d}: {case_gaps}")
    else:
        failures.append("No case folders could be derived from claims.csv image_paths")

    vocab_checks = [
        ("issue_type", ALLOWED_ISSUE_TYPES, False),
        ("object_part", ALLOWED_OBJECT_PARTS, False),
        ("risk_flags", ALLOWED_RISK_FLAGS, True),
        ("severity", ALLOWED_SEVERITIES, False),
        ("claim_status", ALLOWED_CLAIM_STATUSES, False),
    ]
    for column, allowed, split_values in vocab_checks:
        observed, unexpected, unused = _report_vocab(
            sample_rows,
            column,
            allowed,
            split_values=split_values,
        )
        print(f"{column} observed: {sorted(observed)}")
        print(f"{column} unexpected: {sorted(unexpected)}")
        print(f"{column} unused allowed values: {sorted(unused)}")
        if unexpected:
            failures.append(
                f"Unexpected {column} values in sample_claims.csv: {sorted(unexpected)}"
            )

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
