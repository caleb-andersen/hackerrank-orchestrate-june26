from __future__ import annotations

import csv
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

EVAL_ROOT = Path(__file__).resolve().parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from agent_predictor import _load_cache, _prompt_fingerprint, _row_key  # noqa: E402
from config import SAMPLE_CLAIMS_CSV  # noqa: E402
from io_utils import read_claims  # noqa: E402
from metrics import EXACT_MATCH_FIELDS  # noqa: E402


OUTPUT_PATH = EVAL_ROOT / "sample_predictions.csv"
MISSING = "MISSING"


def _fieldnames() -> list[str]:
    fields = ["user_id"]
    for field in EXACT_MATCH_FIELDS:
        fields.extend([f"gold_{field}", f"pred_{field}"])
    return fields


def dump_predictions(path: str | Path = OUTPUT_PATH) -> Path:
    gold_rows = read_claims(SAMPLE_CLAIMS_CSV)
    cache = _load_cache()
    fingerprint = _prompt_fingerprint()
    out_path = Path(path)

    rows: list[dict[str, str]] = []
    for gold_row in gold_rows:
        prediction = cache.get(_row_key(gold_row, fingerprint))
        out_row = {"user_id": gold_row.get("user_id", "")}
        for field in EXACT_MATCH_FIELDS:
            out_row[f"gold_{field}"] = gold_row.get(field, "")
            out_row[f"pred_{field}"] = (
                str(prediction.get(field, MISSING)) if prediction else MISSING
            )
        rows.append(out_row)

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main() -> int:
    out_path = dump_predictions()
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
