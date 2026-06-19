"""Real-agent predictor for the evaluation harness (P3+).

This is the drop-in replacement for the P1 stub `predict`: it runs the actual
tool-calling agent (`agent.loop.run_claim`) over one input row and returns the
14 output columns plus the `_routing` block the metrics consume. It is what lets
`evaluation/main.py` produce a *real* confusion matrix instead of a placeholder.

Two practical concerns this module handles:

  - **Cost / iteration speed.** Scoring is run repeatedly while tuning prompts.
    A JSON cache keyed by (input row + model names + a hash of the live prompts)
    means an unchanged row is never re-inspected, but editing any prompt
    automatically invalidates every cached row so a re-run reflects the change.
  - **Reuse, not duplication.** The lookups and the image inspector are the same
    concrete implementations `code/main.py` wires into the agent; we import them
    directly (they exist as of P2) rather than re-deriving the wiring.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from agent.loop import run_claim  # noqa: E402
from agent.prompts import (  # noqa: E402
    INSPECTION_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
)
from config import INSPECTION_MODEL, SYNTHESIS_MODEL  # noqa: E402
from io_utils import resolve_images  # noqa: E402
from tools.evidence_lookup import get_evidence_requirement  # noqa: E402
from tools.history_lookup import get_user_history  # noqa: E402
from tools.inspect_image import inspect_image  # noqa: E402

CACHE_PATH = Path(__file__).resolve().parent / ".pred_cache.json"

# Bump when the loop/synthesis *logic* changes in a way the prompt hash alone
# would not catch (e.g. re-inspection policy in loop.py). Forces a cache miss.
_LOGIC_VERSION = "p3.1"

INPUT_COLUMNS = ["user_id", "image_paths", "user_claim", "claim_object"]


def _prompt_fingerprint(
    synth_model: str = SYNTHESIS_MODEL,
    inspect_model: str = INSPECTION_MODEL,
) -> str:
    """Hash the live prompts + the models actually in use so any edit (or a
    bake-off model swap) busts the cache. The model names are arguments, not the
    config defaults, so a gpt-4.1-synthesis run and a gpt-5.4-synthesis run get
    distinct fingerprints and never share cached rows.
    """
    blob = "\x00".join(
        [
            _LOGIC_VERSION,
            synth_model,
            inspect_model,
            SYNTHESIS_SYSTEM_PROMPT,
            INSPECTION_SYSTEM_PROMPT,
        ]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _row_key(row: dict[str, str], fingerprint: str) -> str:
    payload = json.dumps(
        {c: row.get(c, "") for c in INPUT_COLUMNS},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256((fingerprint + "\x00" + payload).encode("utf-8")).hexdigest()


def _load_cache() -> dict[str, Any]:
    if CACHE_PATH.is_file():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _make_inspect_fn(
    row: dict[str, str], client: Any, model: str, instrument: Any = None
) -> Callable[[str], dict]:
    """Bind inspect_image to this row's resolved images (image_id -> path)."""
    id_to_path = {image_id: path for image_id, path in resolve_images(row["image_paths"])}
    claim_object = row.get("claim_object", "")

    def inspect(image_id: str) -> dict:
        path = id_to_path.get(image_id)
        if path is None:
            return {
                "image_id": image_id,
                "readable": False,
                "notes": f"unknown image id {image_id!r}",
            }
        return inspect_image(
            image_id, path, claim_object, client=client, model=model,
            instrument=instrument,
        )

    return inspect


def make_predictor(
    client: Any,
    *,
    use_cache: bool = True,
    synth_model: str = SYNTHESIS_MODEL,
    inspect_model: str = INSPECTION_MODEL,
    instrument: Any = None,
) -> Callable[[dict[str, str]], dict[str, str]]:
    """Return a `predict(row) -> dict` closure backed by the real agent.

    Caching is per-row and keyed on the input + live prompt fingerprint, so
    tuning the prompts re-runs every row, while re-scoring identical config
    costs nothing.

    `instrument` (optional) accumulates real token/cost/latency across the run
    for the P4 operational report. NOTE: cached rows make no live call, so they
    contribute nothing to the instrument — pass `use_cache=False` (as the
    bake-off does) when you need a complete operational snapshot.

    The cache key already includes both model names (via the prompt
    fingerprint), so swapping `synth_model` is a clean cache miss — gpt-4.1 and
    gpt-5.4 synthesis runs never collide.
    """
    fingerprint = _prompt_fingerprint(synth_model, inspect_model)
    cache = _load_cache() if use_cache else {}

    def predict(row: dict[str, str]) -> dict[str, str]:
        key = _row_key(row, fingerprint)
        if use_cache and key in cache:
            return cache[key]

        available_ids = [image_id for image_id, _ in resolve_images(row["image_paths"])]
        decision = run_claim(
            row,
            client=client,
            synth_model=synth_model,
            inspect_fn=_make_inspect_fn(row, client, inspect_model, instrument),
            evidence_fn=get_evidence_requirement,
            history_fn=get_user_history,
            available_image_ids=available_ids,
            instrument=instrument,
        )
        if use_cache:
            cache[key] = decision
            _save_cache(cache)
        return decision

    return predict
