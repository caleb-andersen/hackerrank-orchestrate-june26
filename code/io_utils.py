from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from config import DATASET_ROOT, OUTPUT_COLUMNS, OUTPUT_CSV

INPUT_COLUMNS = ["user_id", "image_paths", "user_claim", "claim_object"]


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_claims(path: str | Path) -> list[dict[str, str]]:
    return read_csv_dicts(path)


def resolve_images(image_paths: str) -> list[tuple[str, Path]]:
    resolved: list[tuple[str, Path]] = []
    for raw_path in image_paths.split(";"):
        rel_path = raw_path.strip()
        if not rel_path:
            continue
        image_path = DATASET_ROOT / Path(rel_path)
        resolved.append((Path(rel_path).stem, image_path))
    return resolved


def image_ids(image_paths: str) -> list[str]:
    return [image_id for image_id, _ in resolve_images(image_paths)]


def write_output(
    rows: list[dict[str, str]],
    path: str | Path = OUTPUT_CSV,
) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            extrasaction="raise",
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)


def _self_test() -> None:
    sample_path = DATASET_ROOT / "sample_claims.csv"
    rows = read_csv_dicts(sample_path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        suffix=".csv",
        delete=False,
    ) as tmp:
        temp_path = Path(tmp.name)
    try:
        write_output(rows, temp_path)
        with temp_path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
        if header != OUTPUT_COLUMNS:
            raise AssertionError(
                f"header mismatch: expected {OUTPUT_COLUMNS!r}, got {header!r}"
            )
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    _self_test()
    print("io_utils self-test passed")
