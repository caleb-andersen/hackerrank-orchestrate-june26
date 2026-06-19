from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def _load_dotenv() -> None:
    """Minimal .env loader (mirrors code/main.py); real env vars win.

    Must run BEFORE building the Azure client so endpoint/key/model names are
    visible. Stub (offline) mode never touches this.
    """
    env_path = CODE_ROOT.parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


from config import OUTPUT_COLUMNS, SAMPLE_CLAIMS_CSV  # noqa: E402
from io_utils import INPUT_COLUMNS, image_ids, read_claims  # noqa: E402
from metrics import normalize, score_all  # noqa: E402


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


def _print_field_confusion(report: dict[str, Any], field: str) -> None:
    confusion = report["field_confusions"][field]
    labels = confusion["labels"]
    matrix = confusion["matrix"]
    header = "gold \\ pred"
    label_width = max([len(header), *(len(label) for label in labels)] or [12])
    label_width = max(label_width + 2, 14)
    column_width = max([len(label) for label in labels] or [8]) + 2
    column_width = max(column_width, 10)

    print(f"== {field} confusion matrix ==")
    print("rows=gold, cols=prediction")
    print(f"{header:<{label_width}}" + "".join(
        f"{label:>{column_width}}" for label in labels
    ))
    for label in labels:
        print(
            f"{label:<{label_width}}"
            + "".join(f"{matrix[label][column]:>{column_width}}" for column in labels)
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


def _print_mismatches(
    gold: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    fields: tuple[str, ...] = ("claim_status", "valid_image", "evidence_standard_met"),
) -> None:
    """Per-row gold-vs-pred diff (file order) for the fields we tune in P3.

    This is the working view for fixing contradicted/NEI confusion: it lists
    only rows that miss, with the predicted justification so the failure mode is
    legible at a glance.
    """
    print("== per-row mismatches (gold -> pred) ==")
    any_miss = False
    for index, (gold_row, pred_row) in enumerate(zip(gold, pred)):
        misses = [
            f"{field}: {normalize(gold_row.get(field))} -> {normalize(pred_row.get(field))}"
            for field in fields
            if normalize(gold_row.get(field)) != normalize(pred_row.get(field))
        ]
        if not misses:
            continue
        any_miss = True
        user = gold_row.get("user_id", "?")
        case = image_ids(gold_row.get("image_paths", ""))
        case_hint = case[0].rsplit("_", 1)[0] if case else "?"
        print(f"row {index:>2} [{user} {case_hint}]: " + " | ".join(misses))
        print(f"        gold_just: {gold_row.get('claim_status_justification', '')[:140]}")
        print(f"        pred_just: {pred_row.get('claim_status_justification', '')[:140]}")
    if not any_miss:
        print("(none)")
    print()


def _render_report(report: dict[str, Any], intro: str, title: str) -> str:
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
    field_confusion_sections: list[str] = []
    for field in ("issue_type", "severity"):
        field_confusion = report["field_confusions"][field]
        labels = field_confusion["labels"]
        field_matrix_rows = [
            [
                label,
                *[field_confusion["matrix"][label][column] for column in labels],
            ]
            for label in labels
        ]
        per_value_rows = [
            [
                label,
                scores["support"],
                scores["precision"],
                scores["recall"],
                scores["f1"],
            ]
            for label, scores in field_confusion["per_value"].items()
        ]
        field_confusion_sections.extend(
            [
                f"### {field}",
                "",
                "Rows are gold labels; columns are predicted labels.",
                "",
                _markdown_table(["gold \\ pred", *labels], field_matrix_rows),
                "",
                _markdown_table(
                    ["value", "support", "precision", "recall", "f1"],
                    per_value_rows,
                ),
                "",
            ]
        )
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
            f"# {title}",
            "",
            intro,
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
            "## Categorical Field Confusions",
            "",
            *field_confusion_sections,
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
            "- Model calls: instrumented per-model token/cost accounting lands in P4 (code/instrument.py).",
            "- Token usage: placeholder; wire instrument.record_call into the loop in P4.",
            "- Images processed: see routing counts above (inspect_calls per row via _routing).",
            "- Approximate cost: placeholder; derive from instrument.snapshot() PRICING in P4.",
            "- Latency/runtime: see the wall-clock printed by the harness run.",
            "- TPM/RPM considerations: placeholder; add batching, throttling, retry, and cache notes in P4.",
            "",
        ]
    )


def _build_predictor(use_cache: bool):
    """Real agent predictor over the Azure client (default eval path, P3+)."""
    _load_dotenv()
    from config import azure_client  # noqa: E402 - lazy: stub mode needs no SDK
    from agent_predictor import make_predictor  # noqa: E402

    return make_predictor(azure_client(), use_cache=use_cache)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the agent on samples.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the P1 stub predictor (no Azure, no network) instead of the agent",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore the per-row prediction cache and re-run every row",
    )
    parser.add_argument(
        "--no-mismatches",
        action="store_true",
        help="suppress the per-row gold-vs-pred diff",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    gold = read_claims(SAMPLE_CLAIMS_CSV)

    if args.offline:
        predictor = predict
        title = "Evaluation Report (offline stub)"
        intro = (
            "Generated by the offline stub predictor (`--offline`). It validates "
            "the harness without Azure; numbers here are not the real agent."
        )
    else:
        predictor = _build_predictor(use_cache=not args.no_cache)
        title = "Evaluation Report — agent on sample_claims.csv"
        intro = (
            "Generated by the real tool-calling agent "
            "(`agent.loop.run_claim`) over the 20 labelled samples. The headline "
            "metric is the claim_status 3x3 confusion matrix; contradicted vs "
            "not_enough_information is the discrimination that earns the design."
        )

    predictions = [predictor(_input_only(row)) for row in gold]

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
    _print_field_confusion(report, "issue_type")
    _print_field_confusion(report, "severity")
    _print_risk_flags(report)
    _print_routing(report)
    if not args.no_mismatches:
        _print_mismatches(gold, predictions)
    REPORT_PATH.write_text(_render_report(report, intro, title), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
