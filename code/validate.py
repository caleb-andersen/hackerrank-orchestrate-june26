"""Output validator: schema + invariants, FLAG-and-LOG, never silently overwrite.

Architecture decision #3: the validator surfaces the model's real error rate. It
performs only **minimal, loud normalization** — format-level fixes that cannot
change a judgment (trim whitespace; lowercase the two boolean columns; render an
empty multi-valued field as the literal "none"). Every such fix is recorded as a
flag. It then checks the output schema (column set/order, allowed vocabulary)
and the five cross-field invariants from PLAN.md §3, and reports every violation
WITHOUT correcting it. A silently "fixed" contradiction would hide exactly the
error we need to measure and defend.

`validate_row(row, claim_object)` returns `(normalized_row, flags)` where each
flag is a `(severity, code, message)` tuple. `severity` is "norm" (a
normalization we applied), "schema" (off-contract value), or "invariant" (a
cross-field rule broken). Callers log them; the row is written as-is apart from
the loud normalizations.
"""

from __future__ import annotations

from typing import Any

from agent.prompts import (
    CLAIM_STATUS_VALUES,
    ISSUE_TYPE_VALUES,
    OBJECT_PART_VALUES,
    QUALITY_RISK_FLAGS,
    RISK_FLAG_VALUES,
    REVIEW_TRIGGER_FLAGS,
    SEVERITY_VALUES,
    INVALIDATING_RISK_FLAGS,
)
from config import OUTPUT_COLUMNS

Flag = tuple[str, str, str]  # (severity, code, message)

_BOOL_FIELDS = ["evidence_standard_met", "valid_image"]
_MULTI_FIELDS = ["risk_flags", "supporting_image_ids"]


def _normalize(row: dict[str, str], flags: list[Flag]) -> dict[str, str]:
    """Apply only safe, loud, format-level fixes; record each one."""
    out = dict(row)

    for column in OUTPUT_COLUMNS:
        value = out.get(column, "")
        stripped = value.strip() if isinstance(value, str) else str(value)
        if stripped != value:
            flags.append(("norm", "trim", f"{column}: trimmed surrounding whitespace"))
        out[column] = stripped

    for field in _BOOL_FIELDS:
        value = out.get(field, "")
        low = value.lower()
        if low != value:
            flags.append(("norm", "bool_case", f"{field}: lowercased to {low!r}"))
        out[field] = low

    for field in _MULTI_FIELDS:
        if not out.get(field, ""):
            flags.append(("norm", "empty_to_none", f"{field}: empty -> 'none'"))
            out[field] = "none"

    return out


def _risk_set(value: str) -> set[str]:
    if not value or value.strip().lower() == "none":
        return set()
    return {part.strip() for part in value.split(";") if part.strip()}


def _check_schema(row: dict[str, str], claim_object: str, flags: list[Flag]) -> None:
    missing = [c for c in OUTPUT_COLUMNS if c not in row]
    if missing:
        flags.append(("schema", "missing_columns", f"missing columns: {missing}"))
    extra = [c for c in row if c not in OUTPUT_COLUMNS and not c.startswith("_")]
    if extra:
        flags.append(("schema", "extra_columns", f"unexpected columns: {extra}"))

    if row.get("claim_status") not in CLAIM_STATUS_VALUES:
        flags.append(("schema", "bad_claim_status",
                      f"claim_status={row.get('claim_status')!r} not in vocab"))
    if row.get("issue_type") not in ISSUE_TYPE_VALUES:
        flags.append(("schema", "bad_issue_type",
                      f"issue_type={row.get('issue_type')!r} not in vocab"))
    if row.get("severity") not in SEVERITY_VALUES:
        flags.append(("schema", "bad_severity",
                      f"severity={row.get('severity')!r} not in vocab"))
    for field in _BOOL_FIELDS:
        if row.get(field) not in ("true", "false"):
            flags.append(("schema", "bad_bool",
                          f"{field}={row.get(field)!r} is not true/false"))

    # object_part: validate against the per-object vocab (union as last resort).
    allowed_parts = OBJECT_PART_VALUES.get(claim_object)
    part = row.get("object_part")
    if allowed_parts is not None and part not in allowed_parts:
        flags.append(("schema", "bad_object_part",
                      f"object_part={part!r} not valid for claim_object={claim_object!r}"))

    for flag in _risk_set(row.get("risk_flags", "")):
        if flag not in RISK_FLAG_VALUES:
            flags.append(("schema", "bad_risk_flag", f"risk_flag {flag!r} not in vocab"))


def _check_invariants(row: dict[str, str], flags: list[Flag]) -> None:
    status = row.get("claim_status")
    evidence = row.get("evidence_standard_met")
    valid = row.get("valid_image")
    severity = row.get("severity")
    issue = row.get("issue_type")
    supporting = row.get("supporting_image_ids", "none")
    has_support = supporting.strip().lower() != "none" and bool(supporting.strip())
    risks = _risk_set(row.get("risk_flags", ""))

    is_nei = status == "not_enough_information"

    # Invariant 1: the NEI cluster moves together.
    if is_nei:
        if evidence != "false":
            flags.append(("invariant", "nei_evidence",
                          "claim_status=not_enough_information but evidence_standard_met!=false"))
        if has_support:
            flags.append(("invariant", "nei_support",
                          "not_enough_information must have supporting_image_ids=none"))
        if severity != "unknown":
            flags.append(("invariant", "nei_severity",
                          "not_enough_information must have severity=unknown"))
    else:
        # Invariant 2: supported/contradicted need evidence=true and >=1 image.
        if evidence != "true":
            flags.append(("invariant", "decided_evidence",
                          f"claim_status={status} requires evidence_standard_met=true"))
        if not has_support:
            flags.append(("invariant", "decided_support",
                          f"claim_status={status} requires >=1 supporting image"))

    # Invariant 4: issue_type=none <-> severity=none; real damage -> severity!=none.
    if issue == "none" and severity != "none" and not is_nei:
        flags.append(("invariant", "none_issue_severity",
                      "issue_type=none requires severity=none"))
    if issue not in ("none", "unknown") and severity == "none":
        flags.append(("invariant", "damage_severity",
                      f"issue_type={issue} (real damage) but severity=none"))

    # Invariant 5: valid_image=false should be justified by an invalidating flag.
    if valid == "false" and not (risks & set(INVALIDATING_RISK_FLAGS)):
        # cropped/obstructed to the point of being unusable also justifies it.
        if "cropped_or_obstructed" not in risks:
            flags.append(("invariant", "valid_false_unjustified",
                          "valid_image=false without manipulation/non_original/"
                          "cropped_or_obstructed risk flag"))
    # Blur/wrong-angle alone must not invalidate an image.
    if valid == "false" and risks and risks.issubset(set(QUALITY_RISK_FLAGS)) \
            and not (risks & {"cropped_or_obstructed"}):
        flags.append(("invariant", "valid_false_quality_only",
                      "valid_image=false justified only by blur/angle quality flags"))

    # Review-trigger flags should be paired with manual_review_required.
    if (risks & set(REVIEW_TRIGGER_FLAGS)) and "manual_review_required" not in risks:
        flags.append(("invariant", "missing_manual_review",
                      "review-trigger risk flag present without manual_review_required"))


def validate_row(
    row: dict[str, Any],
    claim_object: str | None = None,
) -> tuple[dict[str, str], list[Flag]]:
    """Normalize (loudly), then flag schema + invariant violations. No overwrite."""
    flags: list[Flag] = []
    routing = row.get("_routing")
    clean = {k: v for k, v in row.items() if not k.startswith("_")}
    obj = claim_object if claim_object is not None else clean.get("claim_object", "")

    normalized = _normalize(clean, flags)
    _check_schema(normalized, obj, flags)
    _check_invariants(normalized, flags)

    if routing is not None:
        normalized["_routing"] = routing
    return normalized, flags


def _self_test() -> None:
    # A clean supported row -> no schema/invariant flags.
    good = {
        "user_id": "u", "image_paths": "images/test/case_001/img_1.jpg",
        "user_claim": "c", "claim_object": "car",
        "evidence_standard_met": "true",
        "evidence_standard_met_reason": "ok", "risk_flags": "none",
        "issue_type": "dent", "object_part": "rear_bumper",
        "claim_status": "supported", "claim_status_justification": "img_1",
        "supporting_image_ids": "img_1", "valid_image": "true", "severity": "medium",
    }
    _, gflags = validate_row(good)
    assert not [f for f in gflags if f[0] != "norm"], gflags

    # NEI cluster broken + missing manual review -> invariant flags.
    bad = dict(good)
    bad.update({
        "claim_status": "not_enough_information", "evidence_standard_met": "true",
        "supporting_image_ids": "img_1", "severity": "medium",
        "risk_flags": "user_history_risk", "issue_type": "unknown",
    })
    _, bflags = validate_row(bad)
    codes = {code for _, code, _ in bflags}
    assert "nei_evidence" in codes, codes
    assert "nei_support" in codes, codes
    assert "nei_severity" in codes, codes
    assert "missing_manual_review" in codes, codes

    # Loud normalization: lowercases TRUE and empties -> none.
    messy = dict(good)
    messy.update({"valid_image": "TRUE", "risk_flags": ""})
    norm, mflags = validate_row(messy)
    assert norm["valid_image"] == "true"
    assert norm["risk_flags"] == "none"
    assert any(code == "bool_case" for _, code, _ in mflags)


if __name__ == "__main__":
    _self_test()
    print("validate self-test passed")
