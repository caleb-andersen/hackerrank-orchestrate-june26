"""Deterministic user-history lookup for the agent tool loop."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config import USER_HISTORY_CSV
from io_utils import read_csv_dicts

_HISTORY_COLUMNS = [
    "user_id",
    "past_claim_count",
    "accept_claim",
    "manual_review_claim",
    "rejected_claim",
    "last_90_days_claim_count",
    "history_flags",
    "history_summary",
]
_HISTORY_BY_USER: dict[str, dict[str, str]] | None = None


def _int_value(value: Any) -> int:
    try:
        return int(str(value).strip() or "0")
    except ValueError:
        return 0


def _safe_row(user_id: str) -> dict[str, str]:
    return {
        "user_id": user_id,
        "past_claim_count": "0",
        "accept_claim": "0",
        "manual_review_claim": "0",
        "rejected_claim": "0",
        "last_90_days_claim_count": "0",
        "history_flags": "none",
        "history_summary": "no history on record",
    }


def _history_by_user() -> dict[str, dict[str, str]]:
    global _HISTORY_BY_USER
    if _HISTORY_BY_USER is None:
        index: dict[str, dict[str, str]] = {}
        for row in read_csv_dicts(USER_HISTORY_CSV):
            user_id = str(row.get("user_id", "")).strip()
            if not user_id:
                continue
            index[user_id] = {
                column: str(row.get(column, "")).strip()
                for column in _HISTORY_COLUMNS
            }
        _HISTORY_BY_USER = index
    return _HISTORY_BY_USER


def _risk_signal(row: dict[str, str]) -> tuple[bool, str]:
    flags = row.get("history_flags", "").strip().lower()
    past = _int_value(row.get("past_claim_count"))
    manual = _int_value(row.get("manual_review_claim"))
    rejected = _int_value(row.get("rejected_claim"))
    recent = _int_value(row.get("last_90_days_claim_count"))

    reasons: list[str] = []
    if flags and flags != "none":
        reasons.append(f"history flags: {row.get('history_flags', 'none')}")
    if rejected > 0:
        reasons.append(f"{rejected} prior rejected claim(s)")
    # Simple documented rule: manual review is meaningful at two or more prior
    # reviews, or at least 40% of a non-trivial history.
    if manual >= 2 or (past >= 3 and manual / past >= 0.40):
        reasons.append(f"{manual}/{past} prior claims needed manual review")
    if recent >= 5:
        reasons.append(f"{recent} claims in the last 90 days")

    if not reasons:
        return False, "no notable risk"
    return True, "; ".join(reasons)


def get_user_history(user_id: str) -> dict:
    """Return one flat user-history row plus derived risk helper fields."""
    key = (user_id or "").strip()
    found = key in _history_by_user()
    row = dict(_history_by_user().get(key, _safe_row(key)))
    suggests_risk, note = _risk_signal(row)
    if not found:
        note = "no history on record"
    row["suggests_user_history_risk"] = suggests_risk
    row["risk_note"] = note
    return row


def _self_test() -> None:
    low = get_user_history("user_001")
    assert low["suggests_user_history_risk"] is False, low
    assert low["risk_note"] == "no notable risk", low
    risky = get_user_history("user_005")
    assert risky["suggests_user_history_risk"] is True, risky
    missing = get_user_history("user_missing_for_self_test")
    assert missing["history_flags"] == "none", missing
    assert missing["suggests_user_history_risk"] is False, missing
    assert missing["risk_note"] == "no history on record"


if __name__ == "__main__":
    _self_test()
    print("history_lookup self-test passed")
