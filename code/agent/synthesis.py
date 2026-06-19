"""Synthesis: the final structured decision step of the agent.

The reasoner ends the tool loop by calling a single `submit_decision` tool whose
arguments ARE the ten judgment fields. Forcing the answer through a typed tool
schema (built from the canonical vocab in `prompts.py`) means the model can only
emit allowed values, and termination is explicit rather than guessed from prose.

This module owns:
  - `SUBMIT_DECISION_TOOL`: the OpenAI function/tool schema.
  - `decision_to_row(args, row)`: turn the tool arguments + the 4 passthrough
    input columns into the exact 14-column output dict (booleans lowercased,
    multi-valued fields joined with ';' or 'none').
  - `fallback_row(row, reason)`: the safe `not_enough_information` +
    `manual_review_required` row used when the loop cannot produce a decision.
"""

from __future__ import annotations

from typing import Any

from agent.prompts import (
    ALL_OBJECT_PARTS,
    CLAIM_STATUS_VALUES,
    ISSUE_TYPE_VALUES,
    RISK_FLAG_VALUES,
    SEVERITY_VALUES,
)
from config import OUTPUT_COLUMNS

INPUT_PASSTHROUGH = ["user_id", "image_paths", "user_claim", "claim_object"]

# Risk flags the model selects from (drop "none"; "none" is represented by an
# empty selection and rendered as the literal "none" string on output).
_RISK_FLAG_CHOICES = [flag for flag in RISK_FLAG_VALUES if flag != "none"]

SUBMIT_DECISION_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_decision",
        "description": (
            "Submit the final structured decision for this claim. Call exactly "
            "once, after inspecting the image(s) and consulting the evidence "
            "requirement and user history. All values must obey the hard "
            "invariants stated in the system prompt."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "evidence_standard_met": {
                    "type": "boolean",
                    "description": (
                        "true if the image SET is sufficient to evaluate the "
                        "claim (claimed object/part visible clearly enough)."
                    ),
                },
                "evidence_standard_met_reason": {
                    "type": "string",
                    "description": "One or two sentences justifying the evidence decision.",
                },
                "risk_flags": {
                    "type": "array",
                    "items": {"type": "string", "enum": _RISK_FLAG_CHOICES},
                    "description": (
                        "Risk flags that apply. Empty array means no risk. Pair "
                        "manual_review_required with claim_mismatch, "
                        "possible_manipulation, non_original_image, "
                        "user_history_risk, or wrong_object."
                    ),
                },
                "issue_type": {
                    "type": "string",
                    "enum": ISSUE_TYPE_VALUES,
                    "description": "The issue type VISIBLE in the supporting image(s).",
                },
                "object_part": {
                    "type": "string",
                    "enum": ALL_OBJECT_PARTS,
                    "description": "The object part VISIBLE in the supporting image(s).",
                },
                "claim_status": {
                    "type": "string",
                    "enum": CLAIM_STATUS_VALUES,
                },
                "claim_status_justification": {
                    "type": "string",
                    "description": "One or two sentences, image-grounded, citing image ids.",
                },
                "supporting_image_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Image ids that ground the decision (e.g. ['img_2']). "
                        "Empty only when claim_status=not_enough_information."
                    ),
                },
                "valid_image": {
                    "type": "boolean",
                    "description": (
                        "true if the image set is usable/trustworthy for "
                        "automated review. false only for manipulation, "
                        "non-original/screenshot, or unusable crop."
                    ),
                },
                "severity": {
                    "type": "string",
                    "enum": SEVERITY_VALUES,
                },
            },
            "required": [
                "evidence_standard_met",
                "evidence_standard_met_reason",
                "risk_flags",
                "issue_type",
                "object_part",
                "claim_status",
                "claim_status_justification",
                "supporting_image_ids",
                "valid_image",
                "severity",
            ],
        },
    },
}


def _bool_str(value: Any) -> str:
    """Render a JSON boolean (or truthy string) as the lowercase 'true'/'false'."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return "true" if str(value).strip().lower() == "true" else "false"


def _join_multi(value: Any) -> str:
    """Join a list (or already-joined string) with ';'; empty -> 'none'."""
    if isinstance(value, str):
        items = [part.strip() for part in value.split(";") if part.strip()]
    elif isinstance(value, list):
        items = [str(part).strip() for part in value if str(part).strip()]
    else:
        items = []
    # de-dupe while preserving order
    seen: list[str] = []
    for item in items:
        if item.lower() != "none" and item not in seen:
            seen.append(item)
    return ";".join(seen) if seen else "none"


def decision_to_row(args: dict[str, Any], row: dict[str, str]) -> dict[str, str]:
    """Build the full 14-column output row from submit_decision args + inputs."""
    out = {column: row.get(column, "") for column in INPUT_PASSTHROUGH}
    out["evidence_standard_met"] = _bool_str(args.get("evidence_standard_met"))
    out["evidence_standard_met_reason"] = str(
        args.get("evidence_standard_met_reason", "")
    ).strip()
    out["risk_flags"] = _join_multi(args.get("risk_flags"))
    out["issue_type"] = str(args.get("issue_type", "unknown")).strip() or "unknown"
    out["object_part"] = str(args.get("object_part", "unknown")).strip() or "unknown"
    out["claim_status"] = str(
        args.get("claim_status", "not_enough_information")
    ).strip()
    out["claim_status_justification"] = str(
        args.get("claim_status_justification", "")
    ).strip()
    out["supporting_image_ids"] = _join_multi(args.get("supporting_image_ids"))
    out["valid_image"] = _bool_str(args.get("valid_image"))
    out["severity"] = str(args.get("severity", "unknown")).strip() or "unknown"
    # Guarantee exact column set/order.
    return {column: out[column] for column in OUTPUT_COLUMNS}


def fallback_row(row: dict[str, str], reason: str) -> dict[str, str]:
    """Safe decision when the loop cannot produce one.

    Per PLAN.md: degrade to not_enough_information + manual_review_required, with
    the four NEI-cluster invariants satisfied (evidence false, no supporting
    image, severity unknown). valid_image=false because we could not complete an
    automated review.
    """
    args = {
        "evidence_standard_met": False,
        "evidence_standard_met_reason": (
            f"Automated review could not complete ({reason}); routed to manual review."
        ),
        "risk_flags": ["manual_review_required"],
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": (
            f"The system could not reach a grounded decision ({reason}), so the "
            "claim is sent for manual review."
        ),
        "supporting_image_ids": [],
        "valid_image": False,
        "severity": "unknown",
    }
    return decision_to_row(args, row)
