"""Offline full-test-set cost/runtime projection from a live sample snapshot.

This script never calls Azure. It scales the live, instrumented sample run in
`operational_snapshot.json` linearly by row count, using the already-computed
per-model costs in the snapshot. The scaling assumption is intentionally simple:
average calls, tokens, cost, and latency per labelled sample row are assumed to
hold for each row in `claims.csv`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config import CLAIMS_CSV  # noqa: E402
from io_utils import read_claims  # noqa: E402


SNAPSHOT_PATH = EVAL_ROOT / "operational_snapshot.json"
REPORT_PATH = EVAL_ROOT / "cost_projection.md"

# Conservative planning defaults for one Azure deployment. Override by setting
# ASSUMED_AZURE_TPM / ASSUMED_AZURE_RPM before running this script.
ASSUMED_TPM = int(os.environ.get("ASSUMED_AZURE_TPM", "100000"))
ASSUMED_RPM = int(os.environ.get("ASSUMED_AZURE_RPM", "60"))


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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _read_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _project_model(counts: dict[str, Any], scale: float) -> dict[str, float]:
    prompt_tokens = _safe_int(counts.get("prompt_tokens"))
    completion_tokens = _safe_int(counts.get("completion_tokens"))
    return {
        "calls": _safe_int(counts.get("calls")) * scale,
        "prompt_tokens": prompt_tokens * scale,
        "completion_tokens": completion_tokens * scale,
        "total_tokens": (prompt_tokens + completion_tokens) * scale,
        "cost_usd": _safe_float(counts.get("cost_usd")) * scale,
    }


def build_projection(
    snapshot_doc: dict[str, Any],
    *,
    full_rows: int,
    assumed_tpm: int = ASSUMED_TPM,
    assumed_rpm: int = ASSUMED_RPM,
) -> dict[str, Any]:
    sample_rows = _safe_int(snapshot_doc.get("n_rows"))
    if sample_rows <= 0:
        raise ValueError("snapshot n_rows must be positive")

    snapshot = snapshot_doc.get("snapshot", {})
    models = snapshot.get("models", {})
    totals = snapshot.get("totals", {})
    timings = snapshot.get("timings_seconds", {})
    routing = snapshot_doc.get("routing", {})

    if not isinstance(models, dict) or not isinstance(totals, dict):
        raise ValueError("snapshot must contain snapshot.models and snapshot.totals")

    scale = full_rows / sample_rows if sample_rows else 0.0
    projected_models = {
        model: _project_model(counts, scale)
        for model, counts in sorted(models.items())
        if isinstance(counts, dict)
    }
    projected_totals = _project_model(totals, scale)
    projected_wall = _safe_float(timings.get("wall")) * scale
    projected_minutes = projected_wall / 60 if projected_wall > 0 else 0.0
    projected_tokens = projected_totals["total_tokens"]
    projected_calls = projected_totals["calls"]
    # The current runner is serial, so the sustained average from the live
    # sample run is also our best estimate of peak TPM/RPM for a full fresh run.
    tokens_per_minute = (
        projected_tokens / projected_minutes if projected_minutes else 0.0
    )
    requests_per_minute = (
        projected_calls / projected_minutes if projected_minutes else 0.0
    )

    tpm_headroom = assumed_tpm / tokens_per_minute if tokens_per_minute else float("inf")
    rpm_headroom = assumed_rpm / requests_per_minute if requests_per_minute else float("inf")
    throttle_expected = tokens_per_minute > assumed_tpm or requests_per_minute > assumed_rpm

    return {
        "chosen_synth_model": snapshot_doc.get("chosen_synth_model", "unknown"),
        "inspect_model": snapshot_doc.get("inspect_model", "unknown"),
        "sample_rows": sample_rows,
        "full_rows": full_rows,
        "scale": scale,
        "models": projected_models,
        "totals": projected_totals,
        "routing": {
            "synth_calls": _safe_int(routing.get("synth_calls")) * scale,
            "inspect_calls": _safe_int(routing.get("inspect_calls")) * scale,
            "images_available": _safe_int(routing.get("images_available")) * scale,
            "early_stop": _safe_int(routing.get("early_stop")) * scale,
            "reinspected": _safe_int(routing.get("reinspected")) * scale,
            "label_flipped": _safe_int(routing.get("label_flipped")) * scale,
        },
        "runtime": {
            "wall_seconds": projected_wall,
            "seconds_per_claim": projected_wall / full_rows if full_rows else 0.0,
        },
        "headroom": {
            "assumed_tpm": assumed_tpm,
            "assumed_rpm": assumed_rpm,
            "estimated_tokens_per_minute": tokens_per_minute,
            "estimated_requests_per_minute": requests_per_minute,
            "tpm_headroom_multiplier": tpm_headroom,
            "rpm_headroom_multiplier": rpm_headroom,
            "throttling_expected": throttle_expected,
        },
    }


def render_projection(projection: dict[str, Any]) -> str:
    model_rows = []
    for model, counts in projection["models"].items():
        model_rows.append(
            [
                model,
                round(counts["calls"], 1),
                round(counts["prompt_tokens"]),
                round(counts["completion_tokens"]),
                round(counts["total_tokens"]),
                round(counts["cost_usd"], 4),
            ]
        )
    totals = projection["totals"]
    model_rows.append(
        [
            "TOTAL",
            round(totals["calls"], 1),
            round(totals["prompt_tokens"]),
            round(totals["completion_tokens"]),
            round(totals["total_tokens"]),
            round(totals["cost_usd"], 4),
        ]
    )

    routing = projection["routing"]
    runtime = projection["runtime"]
    headroom = projection["headroom"]
    throttling_note = (
        "Throttling/backoff is expected under the assumed limits."
        if headroom["throttling_expected"]
        else "No throttling is expected under the assumed limits."
    )

    routing_rows = [
        ["synthesis calls", round(routing["synth_calls"], 1)],
        ["inspection calls / images processed", round(routing["inspect_calls"], 1)],
        ["images available", round(routing["images_available"], 1)],
        ["rows early-stopped", round(routing["early_stop"], 1)],
        ["rows re-inspected", round(routing["reinspected"], 1)],
        ["rows where re-inspection flipped label", round(routing["label_flipped"], 1)],
    ]
    runtime_rows = [
        ["projected wall-clock seconds", round(runtime["wall_seconds"], 1)],
        ["projected wall-clock minutes", round(runtime["wall_seconds"] / 60, 2)],
        ["seconds per claim", round(runtime["seconds_per_claim"], 2)],
        ["estimated peak tokens / minute", round(headroom["estimated_tokens_per_minute"])],
        ["estimated peak requests / minute", round(headroom["estimated_requests_per_minute"], 1)],
        ["assumed TPM limit", headroom["assumed_tpm"]],
        ["assumed RPM limit", headroom["assumed_rpm"]],
        ["TPM headroom multiplier", round(headroom["tpm_headroom_multiplier"], 2)],
        ["RPM headroom multiplier", round(headroom["rpm_headroom_multiplier"], 2)],
    ]

    return "\n".join(
        [
            "# Full-Test-Set Cost Projection",
            "",
            "Projection source: `operational_snapshot.json`, scaled linearly from "
            f"{projection['sample_rows']} live sample rows to "
            f"{projection['full_rows']} `claims.csv` rows. The assumption is "
            "linear in rows: per-row calls, tokens, cost, and latency from the "
            "sample run are representative of the full test set.",
            "",
            f"Synthesis model: `{projection['chosen_synth_model']}`; inspection "
            f"model: `{projection['inspect_model']}`.",
            "",
            "## Projected Calls, Tokens, Cost",
            "",
            _markdown_table(
                [
                    "model",
                    "calls",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "cost_usd",
                ],
                model_rows,
            ),
            "",
            "Costs use the per-model `cost_usd` already present in the live "
            "snapshot; no pricing lookup or network call is performed here.",
            "",
            "## Projected Routing",
            "",
            _markdown_table(["metric", "projected value"], routing_rows),
            "",
            "## Runtime And TPM/RPM Headroom",
            "",
            _markdown_table(["metric", "value"], runtime_rows),
            "",
            throttling_note,
            "",
            "Cache effect: with a warm `InspectionCache` and unchanged prediction "
            "cache, repeated evaluation runs should avoid re-inspecting unchanged "
            "images and can reduce live calls toward zero. This projection is for "
            "a fresh, cache-cold full run.",
            "",
        ]
    )


def write_projection_report(projection: dict[str, Any], path: Path) -> None:
    path.write_text(render_projection(projection), encoding="utf-8")


def _self_test() -> None:
    fake = {
        "chosen_synth_model": "gpt-5.4",
        "inspect_model": "gpt-4.1",
        "n_rows": 2,
        "snapshot": {
            "models": {
                "gpt-4.1": {
                    "calls": 3,
                    "prompt_tokens": 1000,
                    "completion_tokens": 100,
                    "cost_usd": 0.01,
                },
                "gpt-5.4": {
                    "calls": 4,
                    "prompt_tokens": 2000,
                    "completion_tokens": 200,
                    "cost_usd": 0.05,
                },
            },
            "totals": {
                "calls": 7,
                "prompt_tokens": 3000,
                "completion_tokens": 300,
                "cost_usd": 0.06,
            },
            "timings_seconds": {"wall": 60.0},
        },
        "routing": {
            "synth_calls": 4,
            "inspect_calls": 3,
            "images_available": 3,
            "early_stop": 0,
            "reinspected": 1,
            "label_flipped": 0,
        },
    }
    projection = build_projection(fake, full_rows=4, assumed_tpm=10000, assumed_rpm=30)
    assert projection["scale"] == 2.0, projection
    assert projection["totals"]["calls"] == 14, projection
    assert projection["totals"]["total_tokens"] == 6600, projection
    assert projection["totals"]["cost_usd"] == 0.12, projection
    assert projection["runtime"]["wall_seconds"] == 120, projection
    assert projection["headroom"]["estimated_tokens_per_minute"] == 3300, projection
    assert projection["headroom"]["estimated_requests_per_minute"] == 7, projection
    rendered = render_projection(projection)
    assert "Full-Test-Set Cost Projection" in rendered
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = Path(tmp_dir) / "cost_projection.md"
        write_projection_report(projection, out)
        assert out.read_text(encoding="utf-8").startswith("# Full-Test-Set")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project sample operational costs to the full claims.csv set."
    )
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--claims", type=Path, default=CLAIMS_CSV)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.self_test:
        _self_test()
        print("cost_projection self-test passed")
        return 0

    snapshot = _read_snapshot(args.snapshot)
    if snapshot is None:
        print(
            f"{args.snapshot} not found or unreadable; run evaluation/bakeoff.py "
            "first to create operational_snapshot.json."
        )
        return 0

    try:
        full_rows = len(read_claims(args.claims))
        projection = build_projection(
            snapshot,
            full_rows=full_rows,
            assumed_tpm=ASSUMED_TPM,
            assumed_rpm=ASSUMED_RPM,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"could not build projection: {type(exc).__name__}: {exc}")
        return 0

    rendered = render_projection(projection)
    print(rendered)
    write_projection_report(projection, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
