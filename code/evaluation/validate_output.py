"""Whole-file output.csv validator for the final submission gate.

This is intentionally separate from `code/validate.py`: that module validates
one model-produced row inside the pipeline, while this script validates the
written CSV exactly as the evaluator will read it. Row-level invariant flags are
reported, not fixed, so the owner sees internal inconsistencies without hiding
the model's real output quality.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from agent.prompts import (  # noqa: E402
    CLAIM_STATUS_VALUES,
    ISSUE_TYPE_VALUES,
    OBJECT_PART_VALUES,
    RISK_FLAG_VALUES,
    SEVERITY_VALUES,
)
from config import CLAIMS_CSV, OUTPUT_COLUMNS, OUTPUT_CSV  # noqa: E402
from io_utils import image_ids, read_claims, write_output  # noqa: E402
from validate import validate_row  # noqa: E402


INPUT_COLUMNS = ["user_id", "image_paths", "user_claim", "claim_object"]
BOOL_FIELDS = ["evidence_standard_met", "valid_image"]
REQUIRED_TEXT_FIELDS = [
    "evidence_standard_met_reason",
    "claim_status_justification",
]


def _header_problem(header: list[str], expected: list[str]) -> str:
    limit = max(len(header), len(expected))
    for index in range(limit):
        got = header[index] if index < len(header) else "<missing>"
        want = expected[index] if index < len(expected) else "<extra>"
        if got != want:
            return f"col {index + 1}: got {got!r} expected {want!r}"
    return "header mismatch"


def _read_output(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        rows = [dict(zip(header, row)) for row in reader]
    return header, rows


def _split_semicolon(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(";")]


def _validate_vocab_and_format(row: dict[str, str], row_number: int) -> list[str]:
    problems: list[str] = []
    prefix = f"[vocab] row {row_number}:"
    claim_object = row.get("claim_object", "")

    for field in BOOL_FIELDS:
        if row.get(field) not in ("true", "false"):
            problems.append(f"{prefix} {field}={row.get(field)!r} is not true/false")

    if row.get("claim_status") not in CLAIM_STATUS_VALUES:
        problems.append(
            f"{prefix} claim_status={row.get('claim_status')!r} not in vocab"
        )
    if row.get("issue_type") not in ISSUE_TYPE_VALUES:
        problems.append(f"{prefix} issue_type={row.get('issue_type')!r} not in vocab")
    if row.get("severity") not in SEVERITY_VALUES:
        problems.append(f"{prefix} severity={row.get('severity')!r} not in vocab")

    object_parts = OBJECT_PART_VALUES.get(claim_object)
    if object_parts is None:
        problems.append(f"{prefix} claim_object={claim_object!r} has no object_part vocab")
    elif row.get("object_part") not in object_parts:
        problems.append(
            f"{prefix} object_part={row.get('object_part')!r} not valid for "
            f"claim_object={claim_object!r}"
        )

    risk_flags = str(row.get("risk_flags", ""))
    risk_tokens = _split_semicolon(risk_flags)
    if risk_flags == "none":
        pass
    elif not risk_flags or any(token == "" for token in risk_tokens):
        problems.append(f"{prefix} risk_flags has an empty token")
    else:
        if "none" in risk_tokens and len(risk_tokens) > 1:
            problems.append(f"{prefix} risk_flags mixes 'none' with other flags")
        for token in risk_tokens:
            if token not in RISK_FLAG_VALUES:
                problems.append(f"{prefix} risk_flag {token!r} not in vocab")

    allowed_image_ids = set(image_ids(row.get("image_paths", "")))
    supporting = str(row.get("supporting_image_ids", ""))
    supporting_tokens = _split_semicolon(supporting)
    if supporting == "none":
        pass
    elif not supporting or any(token == "" for token in supporting_tokens):
        problems.append(f"{prefix} supporting_image_ids has an empty token")
    else:
        for token in supporting_tokens:
            if token not in allowed_image_ids:
                problems.append(
                    f"{prefix} supporting_image_id {token!r} not in row image ids "
                    f"{sorted(allowed_image_ids)!r}"
                )

    for field in REQUIRED_TEXT_FIELDS:
        if not str(row.get(field, "")).strip():
            problems.append(f"{prefix} {field} is empty")

    return problems


def validate_output(
    output_path: str | Path = OUTPUT_CSV,
    claims_path: str | Path = CLAIMS_CSV,
) -> tuple[bool, list[str]]:
    """Validate the emitted output CSV. Missing output is a no-op success."""
    out_path = Path(output_path)
    if not out_path.is_file():
        print("run code/main.py first to produce output.csv")
        return True, []

    problems: list[str] = []
    ok = True

    header, output_rows = _read_output(out_path)
    if header != OUTPUT_COLUMNS:
        ok = False
        problems.append(f"[header] {_header_problem(header, OUTPUT_COLUMNS)}")

    claims = read_claims(claims_path)
    if len(output_rows) != len(claims):
        ok = False
        problems.append(
            f"[row_count] got {len(output_rows)} output rows expected {len(claims)}"
        )

    for index, (claim_row, output_row) in enumerate(zip(claims, output_rows), start=1):
        for field in INPUT_COLUMNS:
            if output_row.get(field) != claim_row.get(field):
                ok = False
                problems.append(
                    f"[input_fidelity] row {index} field {field}: "
                    f"got {output_row.get(field)!r} expected {claim_row.get(field)!r}"
                )
                break
        else:
            continue
        break

    for index, row in enumerate(output_rows, start=1):
        row_problems = _validate_vocab_and_format(row, index)
        if row_problems:
            ok = False
            problems.extend(row_problems)

    # Invariant/schema flags from the row-level validator are report-only here.
    # Checks above drive hard pass/fail; this summary preserves the model's real
    # inconsistencies for the owner without silently rewriting the output.
    flag_counts: Counter[str] = Counter()
    for row in output_rows:
        _, flags = validate_row(row, row.get("claim_object"))
        for severity, _code, _message in flags:
            if severity in {"schema", "invariant"}:
                flag_counts[severity] += 1
    if flag_counts:
        details = ", ".join(f"{key}={flag_counts[key]}" for key in sorted(flag_counts))
        problems.append(f"[row_invariants:report_only] {details}")

    return ok, problems


def _print_summary(ok: bool, problems: list[str]) -> None:
    hard = [p for p in problems if not p.startswith("[row_invariants:report_only]")]
    report_only = [p for p in problems if p.startswith("[row_invariants:report_only]")]
    print("== output.csv validation ==")
    print("PASS" if ok else "FAIL")
    print(f"hard problems: {len(hard)}")
    print(f"report-only invariant summaries: {len(report_only)}")
    for problem in problems[:20]:
        print(f"- {problem}")
    if len(problems) > 20:
        print(f"- ... {len(problems) - 20} more")


def _make_row(claim: dict[str, str]) -> dict[str, str]:
    return {
        "user_id": claim["user_id"],
        "image_paths": claim["image_paths"],
        "user_claim": claim["user_claim"],
        "claim_object": claim["claim_object"],
        "evidence_standard_met": "true",
        "evidence_standard_met_reason": "The image set is sufficient for review.",
        "risk_flags": "none",
        "issue_type": "dent" if claim["claim_object"] == "car" else "unknown",
        "object_part": "rear_bumper" if claim["claim_object"] == "car" else "unknown",
        "claim_status": "supported",
        "claim_status_justification": "The cited image supports the decision.",
        "supporting_image_ids": image_ids(claim["image_paths"])[0],
        "valid_image": "true",
        "severity": "medium",
    }


def _write_claims(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def _self_test() -> None:
    claims = [
        {
            "user_id": "user_a",
            "image_paths": "images/test/case_001/img_1.jpg",
            "user_claim": "Customer: rear bumper dent.",
            "claim_object": "car",
        },
        {
            "user_id": "user_b",
            "image_paths": "images/test/case_002/img_1.jpg",
            "user_claim": "Customer: rear bumper dent.",
            "claim_object": "car",
        },
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        claims_path = root / "claims.csv"
        good_output = root / "good_output.csv"
        bad_header = root / "bad_header.csv"
        bad_rows = root / "bad_rows.csv"
        bad_vocab = root / "bad_vocab.csv"

        _write_claims(claims, claims_path)
        write_output([_make_row(row) for row in claims], good_output)
        ok, problems = validate_output(good_output, claims_path)
        assert ok, problems

        with bad_header.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
            writer.writerow([OUTPUT_COLUMNS[0], OUTPUT_COLUMNS[2], OUTPUT_COLUMNS[1], *OUTPUT_COLUMNS[3:]])
            writer.writerow([_make_row(claims[0]).get(column, "") for column in OUTPUT_COLUMNS])
        ok, problems = validate_output(bad_header, claims_path)
        assert not ok and any(problem.startswith("[header]") for problem in problems)

        write_output([_make_row(claims[0])], bad_rows)
        ok, problems = validate_output(bad_rows, claims_path)
        assert not ok and any(problem.startswith("[row_count]") for problem in problems)

        bad = _make_row(claims[0])
        bad["severity"] = "extreme"
        write_output([bad, _make_row(claims[1])], bad_vocab)
        ok, problems = validate_output(bad_vocab, claims_path)
        assert not ok and any("severity='extreme'" in problem for problem in problems)

        ok, problems = validate_output(root / "missing.csv", claims_path)
        assert ok and problems == []


def main() -> int:
    ok, problems = validate_output()
    _print_summary(ok, problems)
    return 0 if ok else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
        print("validate_output self-test passed")
        raise SystemExit(0)
    raise SystemExit(main())
