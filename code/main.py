"""Entry point: run the evidence-review agent over a claims CSV -> output.csv.

    python code/main.py                 # dataset/claims.csv  -> output.csv
    python code/main.py --limit 2       # smoke-test the first 2 rows
    python code/main.py --input dataset/sample_claims.csv --output /tmp/out.csv

Pipeline per row (file order — never case numbers, decision #8):
    resolve images -> run the tool loop (agent/loop.run_claim) -> validate
    (validate.validate_row, flag-and-log) -> collect. Then write all rows at
    once via io_utils.write_output (exact 14-column order).

The image inspector (tools.inspect_image) and the two CSV lookups are injected
into the loop. The lookups are Codex's P2-CX deliverables
(`tools.evidence_lookup`, `tools.history_lookup`); until those land this module
falls back to clearly-labelled minimal stand-ins so the pipeline still runs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def _load_dotenv() -> None:
    """Minimal .env loader (mirrors probe_azure.py); real env vars win.

    Must run BEFORE importing config so env-driven settings (endpoint, model
    deployment names, api version) are visible.
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


_load_dotenv()

from agent.loop import run_claim  # noqa: E402
from config import (  # noqa: E402
    CLAIMS_CSV,
    DATASET_ROOT,
    INSPECTION_MODEL,
    OUTPUT_COLUMNS,
    OUTPUT_CSV,
    SYNTHESIS_MODEL,
    azure_client,
)
from io_utils import read_claims, resolve_images, write_output  # noqa: E402
from tools.inspect_image import inspect_image  # noqa: E402
from validate import validate_row  # noqa: E402


# --------------------------------------------------------------------------- #
# Lookup wiring: prefer Codex's P2-CX modules; fall back to minimal stand-ins.
# --------------------------------------------------------------------------- #
try:
    from tools.evidence_lookup import get_evidence_requirement as _evidence_lookup
    _EVIDENCE_SOURCE = "tools.evidence_lookup"
except ImportError:
    from io_utils import read_csv_dicts
    from config import EVIDENCE_REQUIREMENTS_CSV

    def _evidence_lookup(claim_object: str, issue_family: str) -> dict:
        """FALLBACK (Codex P2-CX pending): return matching requirement rows."""
        rows = read_csv_dicts(EVIDENCE_REQUIREMENTS_CSV)
        obj = (claim_object or "").strip().lower()
        matches = [
            r for r in rows
            if r.get("claim_object", "").lower() in (obj, "all")
        ]
        return {"claim_object": claim_object, "issue_family": issue_family,
                "candidates": matches}
    _EVIDENCE_SOURCE = "fallback"

try:
    from tools.history_lookup import get_user_history as _history_lookup
    _HISTORY_SOURCE = "tools.history_lookup"
except ImportError:
    from io_utils import read_csv_dicts as _read_csv_dicts2
    from config import USER_HISTORY_CSV

    _HISTORY_INDEX: dict[str, dict] = {}

    def _history_lookup(user_id: str) -> dict:
        """FALLBACK (Codex P2-CX pending): return the user_history.csv row."""
        if not _HISTORY_INDEX:
            for r in _read_csv_dicts2(USER_HISTORY_CSV):
                _HISTORY_INDEX[r.get("user_id", "")] = r
        return _HISTORY_INDEX.get(
            (user_id or "").strip(),
            {"user_id": user_id, "history_flags": "none",
             "history_summary": "no history on record"},
        )
    _HISTORY_SOURCE = "fallback"


def _make_inspect_fn(row: dict, client, model: str):
    """Bind inspect_image to this row's resolved images (image_id -> path)."""
    id_to_path = {image_id: path for image_id, path in resolve_images(row["image_paths"])}
    claim_object = row.get("claim_object", "")

    def inspect(image_id: str) -> dict:
        path = id_to_path.get(image_id)
        if path is None:
            return {"image_id": image_id, "readable": False,
                    "notes": f"unknown image id {image_id!r}"}
        return inspect_image(image_id, path, claim_object, client=client, model=model)

    return inspect


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the evidence-review agent.")
    parser.add_argument("--input", type=Path, default=CLAIMS_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N rows (smoke test)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    claims = read_claims(args.input)
    if args.limit is not None:
        claims = claims[: args.limit]

    print(f"[main] input={args.input}  rows={len(claims)}")
    print(f"[main] synthesis={SYNTHESIS_MODEL}  inspection={INSPECTION_MODEL}")
    print(f"[main] evidence_lookup={_EVIDENCE_SOURCE}  history_lookup={_HISTORY_SOURCE}")

    client = azure_client()

    output_rows: list[dict] = []
    flag_total = {"norm": 0, "schema": 0, "invariant": 0}
    routing_total = {"early_stop": 0, "reinspected": 0, "label_flipped": 0}
    synth_calls = inspect_calls = 0
    start = time.time()

    for index, row in enumerate(claims):
        available_ids = [image_id for image_id, _ in resolve_images(row["image_paths"])]
        decision = run_claim(
            row,
            client=client,
            synth_model=SYNTHESIS_MODEL,
            inspect_fn=_make_inspect_fn(row, client, INSPECTION_MODEL),
            evidence_fn=_evidence_lookup,
            history_fn=_history_lookup,
            available_image_ids=available_ids,
        )
        routing = decision.get("_routing", {})
        for key in routing_total:
            routing_total[key] += int(bool(routing.get(key)))
        synth_calls += int(routing.get("synth_calls", 0))
        inspect_calls += int(routing.get("inspect_calls", 0))

        clean, flags = validate_row(decision, row.get("claim_object"))
        for severity, code, message in flags:
            flag_total[severity] = flag_total.get(severity, 0) + 1
            if severity != "norm":
                print(f"  [validate] row {index} {row.get('user_id')}: "
                      f"{severity}/{code}: {message}")
        clean.pop("_routing", None)
        output_rows.append({c: clean.get(c, "") for c in OUTPUT_COLUMNS})

    write_output(output_rows, args.output)
    elapsed = time.time() - start

    print(f"\n[main] wrote {args.output}  ({len(output_rows)} rows)")
    print(f"[main] model calls: synthesis={synth_calls} inspection={inspect_calls}")
    print(f"[main] routing: {routing_total}")
    print(f"[main] validation flags: {flag_total}")
    print(f"[main] elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
