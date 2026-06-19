"""Synthesis-model bake-off: gpt-4.1 vs gpt-5.4 on the labelled samples (P4).

Architecture decision #6 puts the *expensive* reasoner only on the single
synthesis/decision call per claim, while cheap vision (gpt-4.1) does the
high-volume per-image inspection. Decision #5 says that tier must EARN its
place: if dropping the synthesis model down to the cheap one barely moves the
metric that matters (the contradicted-vs-not_enough_information discrimination),
the design should collapse to one model.

This script measures exactly that. It runs the *same* agent over
`sample_claims.csv` twice, varying ONLY the synthesis model; inspection stays
`gpt-4.1` in both runs. It reports accuracy + real call/token/cost/latency side
by side and prints a data-driven verdict.

Live: makes real Azure calls with the prediction cache OFF, so every row is a
fresh, fully instrumented call and the operational numbers are complete (cached
rows make no live call and would undercount tokens/cost).

Writes:
  - code/evaluation/bakeoff_report.md          (committed: the decision artifact)
  - code/evaluation/operational_snapshot.json  (gitignored: feeds evaluation_report.md)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
# Put EVAL_DIR ahead of CODE_ROOT so `import main` resolves to
# evaluation/main.py, not code/main.py (both define a top-level `main` module).
for _p in (str(CODE_ROOT), str(EVAL_DIR)):
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from config import INSPECTION_MODEL, SAMPLE_CLAIMS_CSV, SYNTHESIS_MODEL  # noqa: E402
from io_utils import read_claims  # noqa: E402
from main import _input_only, _load_dotenv, _markdown_table  # noqa: E402
from metrics import score_all  # noqa: E402

# The two probed deployments on this Azure resource. Inspection is held fixed at
# the cheap vision model; only the synthesis/decision model varies.
CANDIDATE_SYNTH_MODELS = ["gpt-4.1", "gpt-5.4"]
FIXED_INSPECT_MODEL = INSPECTION_MODEL  # gpt-4.1

BAKEOFF_REPORT_PATH = EVAL_DIR / "bakeoff_report.md"
OPERATIONAL_SNAPSHOT_PATH = EVAL_DIR / "operational_snapshot.json"

# Verdict rule (documented so it is defensible): the expensive reasoner only
# earns its place if it improves the headline discrimination by a margin that is
# meaningful on 20 samples. macro-F1 is the headline; we also refuse to collapse
# if the cheap model regresses the contradicted<->NEI cells (the actual task).
MACRO_F1_EARNS_MARGIN = 0.03


def _run_one(synth_model: str, gold: list[dict[str, str]], client: Any) -> dict[str, Any]:
    """One full sample run for a given synthesis model; returns scored + ops."""
    from agent_predictor import make_predictor  # lazy: needs the SDK
    from instrument import Instrument

    inst = Instrument()
    predictor = make_predictor(
        client,
        use_cache=False,  # fresh, fully instrumented calls
        synth_model=synth_model,
        inspect_model=FIXED_INSPECT_MODEL,
        instrument=inst,
    )

    with inst.track("wall"):
        predictions = [predictor(_input_only(row)) for row in gold]

    report = score_all(gold, predictions)
    snapshot = inst.snapshot()
    routing = _aggregate_routing(predictions)
    return {
        "synth_model": synth_model,
        "inspect_model": FIXED_INSPECT_MODEL,
        "report": report,
        "snapshot": snapshot,
        "routing": routing,
        "n_rows": len(gold),
    }


def _aggregate_routing(predictions: list[dict[str, Any]]) -> dict[str, int]:
    """Sum the per-row routing into run totals (model-agnostic call counts)."""
    totals = {
        "synth_calls": 0,
        "inspect_calls": 0,
        "images_available": 0,
        "early_stop": 0,
        "reinspected": 0,
        "label_flipped": 0,
    }
    for pred in predictions:
        routing = pred.get("_routing", {})
        totals["synth_calls"] += int(routing.get("synth_calls", 0))
        totals["inspect_calls"] += int(routing.get("inspect_calls", 0))
        totals["images_available"] += int(routing.get("images_available", 0))
        totals["early_stop"] += int(bool(routing.get("early_stop")))
        totals["reinspected"] += int(bool(routing.get("reinspected")))
        totals["label_flipped"] += int(bool(routing.get("label_flipped")))
    return totals


def _headline(run: dict[str, Any]) -> dict[str, float]:
    confusion = run["report"]["claim_status_confusion"]
    per_class = confusion["per_class"]
    acc = next(
        item["accuracy"]
        for item in run["report"]["field_accuracy"]
        if item["field"] == "claim_status"
    )
    return {
        "claim_status_accuracy": acc,
        "macro_f1": confusion["macro_f1"],
        "supported_f1": per_class["supported"]["f1"],
        "contradicted_f1": per_class["contradicted"]["f1"],
        "nei_f1": per_class["not_enough_information"]["f1"],
        "contra_as_nei": confusion["contra_as_nei"],
        "nei_as_contra": confusion["nei_as_contra"],
        "risk_micro_f1": run["report"]["risk_flags_prf"]["micro"]["f1"],
    }


def _ops(run: dict[str, Any]) -> dict[str, float]:
    totals = run["snapshot"]["totals"]
    wall = run["snapshot"]["timings_seconds"].get("wall", 0.0)
    n = max(run["n_rows"], 1)
    return {
        "calls": totals["calls"],
        "prompt_tokens": totals["prompt_tokens"],
        "completion_tokens": totals["completion_tokens"],
        "total_tokens": totals["prompt_tokens"] + totals["completion_tokens"],
        "cost_usd": totals["cost_usd"],
        "wall_seconds": wall,
        "sec_per_claim": wall / n,
        "cost_per_claim": totals["cost_usd"] / n,
    }


def _verdict(runs_by_model: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """Data-driven recommendation. Returns (chosen_model, rationale)."""
    cheap, premium = "gpt-4.1", "gpt-5.4"
    if cheap not in runs_by_model or premium not in runs_by_model:
        only = next(iter(runs_by_model))
        return only, "only one model ran; no comparison made"

    h_cheap = _headline(runs_by_model[cheap])
    h_prem = _headline(runs_by_model[premium])
    delta = h_prem["macro_f1"] - h_cheap["macro_f1"]
    # The cheap model "regresses the task" if it confuses contradicted<->NEI
    # more than the premium model does.
    cheap_cross = h_cheap["contra_as_nei"] + h_cheap["nei_as_contra"]
    prem_cross = h_prem["contra_as_nei"] + h_prem["nei_as_contra"]

    o_cheap, o_prem = _ops(runs_by_model[cheap]), _ops(runs_by_model[premium])
    cost_mult = (
        o_prem["cost_usd"] / o_cheap["cost_usd"] if o_cheap["cost_usd"] else float("inf")
    )

    earns = delta >= MACRO_F1_EARNS_MARGIN or prem_cross < cheap_cross
    if earns:
        return premium, (
            f"gpt-5.4 synthesis improves macro-F1 by {delta:+.3f} "
            f"(>= {MACRO_F1_EARNS_MARGIN:.2f} margin) and/or tightens the "
            f"contradicted<->NEI cells ({prem_cross} vs {cheap_cross}); the "
            f"reasoner tier earns its {cost_mult:.1f}x cost (decision #6)."
        )
    # Collapse to the cheap model. Distinguish "cheap is actually better/equal"
    # from "cheap is marginally worse but not worth the cost" so the prose is
    # never self-contradictory.
    if delta <= 0:
        margin = (
            f"matches gpt-5.4" if abs(delta) < 1e-9
            else f"beats gpt-5.4 by {-delta:.3f} macro-F1"
        )
        return cheap, (
            f"gpt-4.1 synthesis {margin} at {cost_mult:.1f}x lower cost, with "
            f"equal contradicted<->NEI confusion ({cheap_cross} vs {prem_cross}); "
            "the expensive reasoner tier does not earn its place — collapse the "
            "synthesis tier to the cheap model (decision #5)."
        )
    return cheap, (
        f"gpt-4.1 synthesis trails gpt-5.4 by only {delta:.3f} macro-F1 "
        f"(< {MACRO_F1_EARNS_MARGIN:.2f} margin) with no worse "
        f"contradicted<->NEI confusion ({cheap_cross} vs {prem_cross}) at "
        f"{cost_mult:.1f}x lower cost; collapse the synthesis tier to the cheap "
        "model (decision #5)."
    )


def _render_report(runs_by_model: dict[str, dict[str, Any]], chosen: str, rationale: str) -> str:
    models = list(runs_by_model)
    headline_rows = []
    for label, key in [
        ("claim_status accuracy", "claim_status_accuracy"),
        ("macro-F1 (headline)", "macro_f1"),
        ("supported F1", "supported_f1"),
        ("contradicted F1", "contradicted_f1"),
        ("not_enough_information F1", "nei_f1"),
        ("contradicted->NEI cells", "contra_as_nei"),
        ("NEI->contradicted cells", "nei_as_contra"),
        ("risk_flags micro-F1", "risk_micro_f1"),
    ]:
        headline_rows.append(
            [label] + [_headline(runs_by_model[m])[key] for m in models]
        )

    ops_rows = []
    for label, key in [
        ("model calls (synth+inspect)", "calls"),
        ("prompt tokens", "prompt_tokens"),
        ("completion tokens", "completion_tokens"),
        ("total tokens", "total_tokens"),
        ("est. cost (USD)", "cost_usd"),
        ("wall-clock (s)", "wall_seconds"),
        ("sec / claim", "sec_per_claim"),
        ("cost / claim (USD)", "cost_per_claim"),
    ]:
        ops_rows.append([label] + [_ops(runs_by_model[m])[key] for m in models])

    routing_rows = []
    for label, key in [
        ("synthesis calls", "synth_calls"),
        ("inspection calls (images processed)", "inspect_calls"),
        ("rows that early-stopped", "early_stop"),
        ("rows re-inspected", "reinspected"),
        ("rows where re-inspection flipped the label", "label_flipped"),
    ]:
        routing_rows.append([label] + [runs_by_model[m]["routing"][key] for m in models])

    header = ["metric"] + [f"synth={m}" for m in models]
    n_rows = runs_by_model[models[0]]["n_rows"]
    return "\n".join(
        [
            "# Synthesis Model Bake-off — gpt-4.1 vs gpt-5.4",
            "",
            "Same agent, same prompts, same cheap-vision inspector "
            f"(`{FIXED_INSPECT_MODEL}`) — only the synthesis/decision model "
            "varies. Run live over the "
            f"{n_rows} labelled samples with the prediction cache OFF, so the "
            "token/cost/latency numbers below are real and complete.",
            "",
            f"**Recommendation: `{chosen}` for synthesis.** {rationale}",
            "",
            "## Accuracy",
            "",
            "The headline is macro-F1 on claim_status, and specifically the "
            "contradicted<->not_enough_information cells — that discrimination "
            "is the task (decisions #3-#5). `supported` is comparatively easy.",
            "",
            _markdown_table(header, headline_rows),
            "",
            "## Operational (real, instrumented)",
            "",
            _markdown_table(header, ops_rows),
            "",
            "Cost uses the assumed per-1K-token `PRICING` in `code/instrument.py` "
            "(planning estimate, not a billing source of truth).",
            "",
            "## Routing",
            "",
            _markdown_table(header, routing_rows),
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    from config import azure_client  # lazy: needs the SDK

    gold = read_claims(SAMPLE_CLAIMS_CSV)
    client = azure_client()

    runs_by_model: dict[str, dict[str, Any]] = {}
    for synth_model in CANDIDATE_SYNTH_MODELS:
        print(f"== running bake-off: synthesis={synth_model}, "
              f"inspection={FIXED_INSPECT_MODEL} ==")
        run = _run_one(synth_model, gold, client)
        runs_by_model[synth_model] = run
        h, o = _headline(run), _ops(run)
        print(
            f"   macro-F1={h['macro_f1']:.3f} "
            f"claim_status_acc={h['claim_status_accuracy']:.3f} "
            f"cross(contra<->NEI)={h['contra_as_nei'] + h['nei_as_contra']} "
            f"cost=${o['cost_usd']:.4f} wall={o['wall_seconds']:.1f}s"
        )

    chosen, rationale = _verdict(runs_by_model)
    print(f"\n== verdict: synthesis={chosen} ==\n{rationale}\n")

    BAKEOFF_REPORT_PATH.write_text(
        _render_report(runs_by_model, chosen, rationale), encoding="utf-8"
    )
    print(f"Wrote {BAKEOFF_REPORT_PATH}")

    # Persist the chosen model's operational snapshot for evaluation_report.md.
    chosen_run = runs_by_model[chosen]
    OPERATIONAL_SNAPSHOT_PATH.write_text(
        json.dumps(
            {
                "chosen_synth_model": chosen,
                "inspect_model": chosen_run["inspect_model"],
                "n_rows": chosen_run["n_rows"],
                "snapshot": chosen_run["snapshot"],
                "routing": chosen_run["routing"],
                "verdict": rationale,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OPERATIONAL_SNAPSHOT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
