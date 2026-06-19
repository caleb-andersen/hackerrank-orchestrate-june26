from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config import OUTPUT_COLUMNS, SAMPLE_CLAIMS_CSV  # noqa: E402
from io_utils import INPUT_COLUMNS, image_ids, read_claims  # noqa: E402
from metrics import score_all  # noqa: E402


REPORT_PATH = Path(__file__).resolve().parent / "evaluation_report.md"
CLAIM_STATUS_LABELS = ["supported", "contradicted", "not_enough_information"]


def predict(row: dict[str, str]) -> dict[str, str]:
    """Offline P1 placeholder predictor matching the future agent interface."""
    supporting_ids = image_ids(row["image_paths"])
    supporting_image_ids = supporting_ids[0] if supporting_ids else "none"
    return {
        "user_id": row["user_id"],
        "image_paths": row["image_paths"],
        "user_claim": row["user_claim"],
        "claim_object": row["claim_object"],
        "evidence_standard_met": "true",
        "evidence_standard_met_reason": (
            "P1 offline stub assumes the submitted image set is reviewable."
        ),
        "risk_flags": "none",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "supported",
        "claim_status_justification": (
            "P1 offline stub uses a constant allowed-vocab placeholder decision."
        ),
        "supporting_image_ids": supporting_image_ids,
        "valid_image": "true",
        "severity": "unknown",
    }


def _input_only(row: dict[str, str]) -> dict[str, str]:
    return {column: row[column] for column in INPUT_COLUMNS}


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(lines)


def _print_field_accuracy(report: dict[str, Any]) -> None:
    print("== Per-field exact-match accuracy ==")
    print(f"{'field':<24} {'correct':>7} {'n':>4} {'accuracy':>9}")
    for item in report["field_accuracy"]:
        print(
            f"{item['field']:<24} {item['correct']:>7} {item['n']:>4} "
            f"{item['accuracy']:>9.3f}"
        )
    print()


def _print_confusion(report: dict[str, Any]) -> None:
    confusion = report["claim_status_confusion"]
    columns = confusion["columns"]
    matrix = confusion["matrix"]
    print("== claim_status confusion matrix ==")
    print("rows=gold, cols=prediction")
    print(f"{'gold \\ pred':<24}" + "".join(f"{column:>24}" for column in columns))
    for label in CLAIM_STATUS_LABELS:
        print(
            f"{label:<24}"
            + "".join(f"{matrix[label][column]:>24}" for column in columns)
        )
    print()
    print("== claim_status per-class scores ==")
    print(f"{'class':<24} {'support':>7} {'precision':>10} {'recall':>10} {'f1':>10}")
    for label, scores in confusion["per_class"].items():
        print(
            f"{label:<24} {scores['support']:>7} {scores['precision']:>10.3f} "
            f"{scores['recall']:>10.3f} {scores['f1']:>10.3f}"
        )
    print(f"macro_f1: {confusion['macro_f1']:.3f}")
    print(
        "contradicted -> not_enough_information: "
        f"{confusion['contra_as_nei']}"
    )
    print(
        "not_enough_information -> contradicted: "
        f"{confusion['nei_as_contra']}"
    )
    print()


def _print_risk_flags(report: dict[str, Any]) -> None:
    risk = report["risk_flags_prf"]
    print("== risk_flags multilabel P/R/F1 ==")
    print(
        "micro: "
        f"P={risk['micro']['precision']:.3f} "
        f"R={risk['micro']['recall']:.3f} "
        f"F1={risk['micro']['f1']:.3f}"
    )
    print(
        "macro: "
        f"P={risk['macro']['precision']:.3f} "
        f"R={risk['macro']['recall']:.3f} "
        f"F1={risk['macro']['f1']:.3f}"
    )
    print()


def _print_routing(report: dict[str, Any]) -> None:
    routing = report["routing"]
    print("== routing counts ==")
    print(f"early_stop_count: {routing['early_stop_count']}")
    print(f"reinspection_count: {routing['reinspection_count']}")
    print(f"post_reinspection_flip_count: {routing['post_reinspection_flip_count']}")
    print()


def _render_report(report: dict[str, Any]) -> str:
    accuracy_rows = [
        [item["field"], item["correct"], item["n"], item["accuracy"]]
        for item in report["field_accuracy"]
    ]
    confusion = report["claim_status_confusion"]
    matrix_rows = [
        [label, *[confusion["matrix"][label][column] for column in confusion["columns"]]]
        for label in CLAIM_STATUS_LABELS
    ]
    class_rows = [
        [
            label,
            scores["support"],
            scores["precision"],
            scores["recall"],
            scores["f1"],
        ]
        for label, scores in confusion["per_class"].items()
    ]
    risk = report["risk_flags_prf"]
    routing = report["routing"]
    coverage_rows = [
        [
            item["field"],
            item["present"],
            item["n"],
            item["present_rate"],
            item["non_empty"],
            item["non_empty_rate"],
        ]
        for item in report["coverage"]
    ]
    return "\n".join(
        [
            "# P1 Evaluation Report",
            "",
            "This report is generated by the offline P1 stub predictor. It validates the evaluation harness before the real model-backed agent is implemented.",
            "",
            f"Rows scored: gold={report['n_gold']}, predictions={report['n_pred']}",
            "",
            "## Per-field Exact Match",
            "",
            _markdown_table(["field", "correct", "n", "accuracy"], accuracy_rows),
            "",
            "## Claim Status Confusion Matrix",
            "",
            "Rows are gold labels; columns are predicted labels.",
            "",
            _markdown_table(["gold \\ pred", *confusion["columns"]], matrix_rows),
            "",
            _markdown_table(
                ["class", "support", "precision", "recall", "f1"],
                class_rows,
            ),
            "",
            f"Macro-F1: {_fmt(confusion['macro_f1'])}",
            "",
            f"Contradicted predicted as not_enough_information: {confusion['contra_as_nei']}",
            "",
            f"Not_enough_information predicted as contradicted: {confusion['nei_as_contra']}",
            "",
            "## Risk Flags Multilabel Metrics",
            "",
            _markdown_table(
                ["scope", "precision", "recall", "f1"],
                [
                    [
                        "micro",
                        risk["micro"]["precision"],
                        risk["micro"]["recall"],
                        risk["micro"]["f1"],
                    ],
                    [
                        "macro",
                        risk["macro"]["precision"],
                        risk["macro"]["recall"],
                        risk["macro"]["f1"],
                    ],
                ],
            ),
            "",
            "## Coverage",
            "",
            _markdown_table(
                ["field", "present", "n", "present_rate", "non_empty", "non_empty_rate"],
                coverage_rows,
            ),
            "",
            "## Routing Metrics",
            "",
            _markdown_table(
                ["metric", "count"],
                [
                    ["early_stop_count", routing["early_stop_count"]],
                    ["reinspection_count", routing["reinspection_count"]],
                    ["post_reinspection_flip_count", routing["post_reinspection_flip_count"]],
                ],
            ),
            "",
            "## Operational Analysis Placeholders",
            "",
            "- Model calls: placeholder; P1 stub makes 0 model calls.",
            "- Token usage: placeholder; P1 stub uses 0 input/output model tokens.",
            "- Images processed: placeholder; P1 stub only parses image IDs and does not inspect images.",
            "- Approximate cost: placeholder; P1 stub cost is $0.",
            "- Latency/runtime: placeholder; add measured wall-clock runtime once the real agent runs.",
            "- TPM/RPM considerations: placeholder; add batching, throttling, retry, and cache notes in P2/P4.",
            "",
        ]
    )


def main() -> int:
    gold = read_claims(SAMPLE_CLAIMS_CSV)
    predictions = [predict(_input_only(row)) for row in gold]

    for prediction in predictions:
        extra_columns = set(prediction) - set(OUTPUT_COLUMNS) - {"_routing"}
        missing_columns = set(OUTPUT_COLUMNS) - set(prediction)
        if extra_columns or missing_columns:
            raise ValueError(
                f"Invalid prediction schema: extra={sorted(extra_columns)}, "
                f"missing={sorted(missing_columns)}"
            )

    report = score_all(gold, predictions)
    _print_field_accuracy(report)
    _print_confusion(report)
    _print_risk_flags(report)
    _print_routing(report)
    REPORT_PATH.write_text(_render_report(report), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
