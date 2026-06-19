from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


CLAIM_STATUS_LABELS = ["supported", "contradicted", "not_enough_information"]
EXACT_MATCH_FIELDS = [
    "evidence_standard_met",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "valid_image",
    "severity",
]
COVERAGE_FIELDS = [
    "evidence_standard_met_reason",
    "claim_status_justification",
    "supporting_image_ids",
]
FIELD_CONFUSION_FIELDS = ["issue_type", "severity", "object_part"]


@dataclass(frozen=True)
class FieldAccuracy:
    field: str
    n: int
    correct: int
    accuracy: float


def normalize(value: Any) -> str:
    if value is None:
        return "none"
    normalized = re.sub(r"\s+", " ", str(value).strip().lower())
    return normalized if normalized else "none"


def _row_value(rows: list[dict[str, Any]], index: int, field: str) -> str:
    if index >= len(rows):
        return "none"
    return normalize(rows[index].get(field))


def _split_labels(value: Any) -> set[str]:
    normalized = normalize(value)
    if normalized == "none":
        return set()
    labels = {normalize(part) for part in normalized.split(";")}
    return {label for label in labels if label != "none"}


def _canonical_multilabel(value: Any) -> str:
    labels = sorted(_split_labels(value))
    return ";".join(labels) if labels else "none"


def _prf(precision_den: int, recall_den: int, tp: int) -> dict[str, float]:
    precision = tp / precision_den if precision_den else 0.0
    recall = tp / recall_den if recall_den else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def exact_match_accuracy(
    gold: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    correct = 0
    for index, gold_row in enumerate(gold):
        gold_value = normalize(gold_row.get(field))
        pred_value = _row_value(pred, index, field)
        if field == "risk_flags":
            gold_value = _canonical_multilabel(gold_value)
            pred_value = _canonical_multilabel(pred_value)
        if gold_value == pred_value:
            correct += 1
    result = FieldAccuracy(
        field=field,
        n=len(gold),
        correct=correct,
        accuracy=correct / len(gold) if gold else 0.0,
    )
    return asdict(result)


def claim_status_confusion(
    gold: list[dict[str, Any]],
    pred: list[dict[str, Any]],
) -> dict[str, Any]:
    pred_values = [_row_value(pred, index, "claim_status") for index in range(len(gold))]
    include_other = any(value not in CLAIM_STATUS_LABELS for value in pred_values)
    columns = CLAIM_STATUS_LABELS + (["<other>"] if include_other else [])
    matrix = {label: {column: 0 for column in columns} for label in CLAIM_STATUS_LABELS}

    for index, gold_row in enumerate(gold):
        gold_label = normalize(gold_row.get("claim_status"))
        pred_label = pred_values[index]
        pred_bucket = pred_label if pred_label in CLAIM_STATUS_LABELS else "<other>"
        if gold_label in matrix:
            matrix[gold_label][pred_bucket] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for label in CLAIM_STATUS_LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[row_label][label] for row_label in CLAIM_STATUS_LABELS) - tp
        fn = sum(matrix[label].values()) - tp
        scores = _prf(tp + fp, tp + fn, tp)
        per_class[label] = {
            "support": sum(matrix[label].values()),
            **scores,
        }

    macro_f1 = (
        sum(float(scores["f1"]) for scores in per_class.values())
        / len(CLAIM_STATUS_LABELS)
        if CLAIM_STATUS_LABELS
        else 0.0
    )
    return {
        "labels": CLAIM_STATUS_LABELS,
        "columns": columns,
        "matrix": matrix,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "contra_as_nei": matrix["contradicted"]["not_enough_information"],
        "nei_as_contra": matrix["not_enough_information"]["contradicted"],
    }


def field_confusion(
    gold: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    """Generic normalized confusion matrix for one categorical field."""
    gold_values = [normalize(row.get(field)) for row in gold]
    pred_values = [_row_value(pred, index, field) for index in range(len(gold))]
    labels = sorted(set(gold_values) | set(pred_values))
    matrix = {label: {column: 0 for column in labels} for label in labels}

    for gold_value, pred_value in zip(gold_values, pred_values):
        matrix[gold_value][pred_value] += 1

    per_value: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[row_label][label] for row_label in labels) - tp
        fn = sum(matrix[label].values()) - tp
        per_value[label] = {
            "support": sum(matrix[label].values()),
            **_prf(tp + fp, tp + fn, tp),
        }

    return {
        "field": field,
        "labels": labels,
        "matrix": matrix,
        "per_value": per_value,
    }


def multilabel_prf(
    gold: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    field: str = "risk_flags",
) -> dict[str, Any]:
    gold_sets: list[set[str]] = []
    pred_sets: list[set[str]] = []
    labels: set[str] = set()
    for index, gold_row in enumerate(gold):
        gold_labels = _split_labels(gold_row.get(field))
        pred_labels = _split_labels(pred[index].get(field) if index < len(pred) else None)
        gold_sets.append(gold_labels)
        pred_sets.append(pred_labels)
        labels.update(gold_labels)
        labels.update(pred_labels)

    per_label: dict[str, dict[str, float | int]] = {}
    micro_tp = micro_fp = micro_fn = 0
    for label in sorted(labels):
        tp = sum(1 for g, p in zip(gold_sets, pred_sets) if label in g and label in p)
        fp = sum(1 for g, p in zip(gold_sets, pred_sets) if label not in g and label in p)
        fn = sum(1 for g, p in zip(gold_sets, pred_sets) if label in g and label not in p)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        per_label[label] = {
            "support": tp + fn,
            **_prf(tp + fp, tp + fn, tp),
        }

    micro = _prf(micro_tp + micro_fp, micro_tp + micro_fn, micro_tp)
    macro = {
        metric: (
            sum(float(scores[metric]) for scores in per_label.values()) / len(per_label)
            if per_label
            else 0.0
        )
        for metric in ("precision", "recall", "f1")
    }
    return {
        "field": field,
        "labels": sorted(labels),
        "micro": micro,
        "macro": macro,
        "per_label": per_label,
    }


def routing_metrics(pred_meta: list[dict[str, Any]] | None) -> dict[str, int]:
    early_stop_count = 0
    reinspection_count = 0
    post_reinspection_flip_count = 0
    for row in pred_meta or []:
        routing = row.get("_routing") or {}
        if routing.get("early_stop"):
            early_stop_count += 1
        if routing.get("reinspected"):
            reinspection_count += 1
        if routing.get("label_flipped"):
            post_reinspection_flip_count += 1
    return {
        "early_stop_count": early_stop_count,
        "reinspection_count": reinspection_count,
        "post_reinspection_flip_count": post_reinspection_flip_count,
    }


def coverage_metrics(
    pred: list[dict[str, Any]],
    fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected_fields = fields or COVERAGE_FIELDS
    report = []
    for field in selected_fields:
        present = 0
        non_empty = 0
        non_none = 0
        for row in pred:
            value = normalize(row.get(field))
            if field in row:
                present += 1
            if value != "none":
                non_empty += 1
                non_none += 1
            elif str(row.get(field, "")).strip():
                non_empty += 1
        report.append(
            {
                "field": field,
                "n": len(pred),
                "present": present,
                "present_rate": present / len(pred) if pred else 0.0,
                "non_empty": non_empty,
                "non_empty_rate": non_empty / len(pred) if pred else 0.0,
                "non_none": non_none,
                "non_none_rate": non_none / len(pred) if pred else 0.0,
            }
        )
    return report


def score_all(gold: list[dict[str, Any]], pred: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_gold": len(gold),
        "n_pred": len(pred),
        "field_accuracy": [
            exact_match_accuracy(gold, pred, field) for field in EXACT_MATCH_FIELDS
        ],
        "claim_status_confusion": claim_status_confusion(gold, pred),
        "field_confusions": {
            field: field_confusion(gold, pred, field)
            for field in FIELD_CONFUSION_FIELDS
        },
        "risk_flags_prf": multilabel_prf(gold, pred, "risk_flags"),
        "coverage": coverage_metrics(pred),
        "routing": routing_metrics(pred),
    }


def _assert_close(actual: float, expected: float) -> None:
    if abs(actual - expected) > 1e-9:
        raise AssertionError(f"expected {expected}, got {actual}")


def _self_test() -> None:
    gold = [
        {"claim_status": "supported", "risk_flags": "none"},
        {"claim_status": "contradicted", "risk_flags": "claim_mismatch;user_history_risk"},
        {"claim_status": "not_enough_information", "risk_flags": "damage_not_visible"},
    ]
    pred = [
        {"claim_status": "supported", "risk_flags": "none", "_routing": {"early_stop": True}},
        {"claim_status": "not_enough_information", "risk_flags": "claim_mismatch"},
        {
            "claim_status": "not_enough_information",
            "risk_flags": "damage_not_visible;manual_review_required",
            "_routing": {"reinspected": True, "label_flipped": True},
        },
    ]
    report = score_all(gold, pred)
    confusion = report["claim_status_confusion"]
    assert confusion["matrix"]["contradicted"]["not_enough_information"] == 1
    assert confusion["matrix"]["not_enough_information"]["contradicted"] == 0
    _assert_close(confusion["per_class"]["supported"]["f1"], 1.0)
    _assert_close(confusion["per_class"]["contradicted"]["f1"], 0.0)
    _assert_close(confusion["per_class"]["not_enough_information"]["f1"], 2 / 3)
    _assert_close(confusion["macro_f1"], 5 / 9)
    risk = report["risk_flags_prf"]
    _assert_close(risk["micro"]["precision"], 2 / 3)
    _assert_close(risk["micro"]["recall"], 2 / 3)
    _assert_close(risk["micro"]["f1"], 2 / 3)
    assert report["routing"] == {
        "early_stop_count": 1,
        "reinspection_count": 1,
        "post_reinspection_flip_count": 1,
    }

    field_gold = [
        {"issue_type": "dent"},
        {"issue_type": "scratch"},
        {"issue_type": "dent"},
        {"issue_type": "unknown"},
    ]
    field_pred = [
        {"issue_type": "dent"},
        {"issue_type": "dent"},
        {"issue_type": "unknown"},
        {"issue_type": "unknown"},
    ]
    issue_confusion = field_confusion(field_gold, field_pred, "issue_type")
    assert issue_confusion["labels"] == ["dent", "scratch", "unknown"]
    assert issue_confusion["matrix"]["dent"]["dent"] == 1
    assert issue_confusion["matrix"]["dent"]["unknown"] == 1
    assert issue_confusion["matrix"]["scratch"]["dent"] == 1
    assert issue_confusion["matrix"]["unknown"]["unknown"] == 1
    _assert_close(issue_confusion["per_value"]["dent"]["precision"], 1 / 2)
    _assert_close(issue_confusion["per_value"]["dent"]["recall"], 1 / 2)
    _assert_close(issue_confusion["per_value"]["dent"]["f1"], 1 / 2)
    _assert_close(issue_confusion["per_value"]["unknown"]["precision"], 1 / 2)
    _assert_close(issue_confusion["per_value"]["unknown"]["recall"], 1.0)
    _assert_close(issue_confusion["per_value"]["unknown"]["f1"], 2 / 3)


if __name__ == "__main__":
    _self_test()
    print("metrics self-test passed")
