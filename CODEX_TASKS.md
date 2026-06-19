# Codex task specs — [CX] work for the Evidence-Review build

These are the **mechanical, well-specced** tasks delegated to Codex (Claude Code
owns the judgment-heavy [CC] work). Run each in Codex, then **capture the Codex
session** and merge it into the chat transcript (Codex does not auto-log to
`~/hackerrank_orchestrate/log.txt`; label its section clearly).

Conventions for every task below:
- Python 3.10+. Standard library preferred; only add deps already in the plan
  (`openai`, `pydantic`, `tenacity`). No network calls in IO/lookup/audit code.
- **Secrets from env vars only.** Never hardcode keys.
- Cross-platform paths via `pathlib`. Write files UTF-8.
- `DATASET_ROOT = <repo>/dataset`. **CSV `image_paths` omit the `dataset/` prefix**
  (they look like `images/test/case_001/img_1.jpg`) — always resolve as
  `DATASET_ROOT / image_path`.
- **Image ID = filename stem** (`img_1.jpg` -> `img_1`).
- Ignore non-image files (e.g. `.DS_Store`); accept `.jpg/.jpeg/.png` only.

---

## P0-CX-1 — `.gitignore` (repo root)

Create `.gitignore` that excludes, at minimum:
```
__pycache__/
*.pyc
.env
.venv/
venv/
node_modules/
output.csv
code/azure_probe_report.json
.DS_Store
# never commit the transcript log (it lives in ~/hackerrank_orchestrate anyway)
log.txt
```
Do **not** ignore `PLAN.md`, `CODEX_TASKS.md`, or `code/**` source.

---

## P0-CX-2 — `code/config.py`

A single import point for paths + model + run config. No secrets inside.

Required contents:
- `REPO_ROOT`, `DATASET_ROOT`, `IMAGES_ROOT`, `OUTPUT_CSV = REPO_ROOT/"output.csv"`.
- Dataset file paths: `CLAIMS_CSV`, `SAMPLE_CLAIMS_CSV`, `USER_HISTORY_CSV`,
  `EVIDENCE_REQUIREMENTS_CSV`.
- `OUTPUT_COLUMNS` = the exact 14-name list in order (see io_utils spec).
- Azure wiring read from env (`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`,
  `AZURE_OPENAI_API_VERSION`), plus model deployment names read from env with
  safe fallbacks: `INSPECTION_MODEL` (default `gpt-4o`), `SYNTHESIS_MODEL`
  (default `gpt-4o`). These get pinned after `probe_azure.py` runs.
- Loop caps: `MAX_LOOP_ITERS = 8`, `MAX_RETRIES = 4`.
- A `def azure_client()` factory returning a configured `AzureOpenAI` client
  (import lazily so non-LLM code can import config without `openai` installed).

---

## P0-CX-3 — `code/io_utils.py`

Pure IO + path helpers. The exact output schema lives here.

```python
OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
    "issue_type", "object_part", "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]
INPUT_COLUMNS = ["user_id", "image_paths", "user_claim", "claim_object"]
```

Functions:
- `read_claims(path) -> list[dict]` — read input CSV **in file order** with
  `csv.DictReader`. Preserve order; return one dict per row. (Claims contain
  `|`, commas, quotes — rely on the csv module, do not hand-split.)
- `resolve_images(image_paths: str) -> list[tuple[str, Path]]` — split on `;`,
  strip, map each to `(image_id, DATASET_ROOT/relpath)`; skip blanks. Do **not**
  silently drop a path whose file is missing — return the Path and let callers
  check `.is_file()` (the audit/agent decide what to do).
- `image_ids(image_paths: str) -> list[str]` — just the stems, in order.
- `write_output(rows: list[dict], path=OUTPUT_CSV)` — `csv.DictWriter` with
  `fieldnames=OUTPUT_COLUMNS`, `extrasaction="raise"`, quote all fields
  (`csv.QUOTE_ALL`) to match the sample files. Booleans must already be the
  lowercase strings `"true"`/`"false"`; lists already joined with `;` or `none`.
- `read_csv_dicts(path) -> list[dict]` — generic helper used by lookups/metrics.

Add a `__main__` self-test that round-trips `dataset/sample_claims.csv`
(read then write to a temp file) and asserts column order is preserved.

---

## P0-CX-4 — `code/data_audit.py`

A read-only audit script that fails loudly if any assumption breaks. Print a
report and exit non-zero on any hard failure.

Checks:
1. `claims.csv` row count == 44; print it.
2. Every `image_paths` entry resolves to an existing file under `DATASET_ROOT`;
   list any missing. Print total images referenced and per-row image counts
   (expect 1–3).
3. Every test `user_id` exists in `user_history.csv`; list any missing
   (expected: none, but report gracefully).
4. Report duplicate `user_id`s across claim rows (expected: `user_045`×3 and
   six users ×2) — this confirms per-row history lookups matter.
5. Confirm case-folder **non-contiguity**: derive the case number from each
   `image_paths` and show the gaps (missing case_002, 009, ... up to 056).
6. Dump the **observed** distinct values that appear in `sample_claims.csv`
   for `issue_type`, `object_part`, `risk_flags`, `severity`, `claim_status`,
   and diff them against the allowed vocab in `problem_statement.md` (hardcode
   the allowed lists from the problem statement as constants in the script).

Output a short PASS/FAIL summary at the end.

---

## P1-CX-1 — `code/evaluation/metrics.py`

Pure, deterministic scoring. **No network, no model calls, no `openai` import.**
Stdlib only (plus `io_utils.read_csv_dicts` for loading). Operates on two
aligned lists of dicts — `gold` (from `sample_claims.csv`) and `pred` (whatever
the predictor returned) — matched **by file order** (architecture decision #8;
do not match on `user_id`, it has duplicates). Every function returns a plain
dict/dataclass so `main.py` can render and serialize it; **printing/formatting
lives in `main.py`, not here.**

Implement:

- `normalize(value) -> str` — trim, lowercase, collapse internal whitespace.
  Treat empty / missing / `"none"` consistently so scoring is robust to casing.
- `exact_match_accuracy(gold, pred, field) -> dict` — per-field exact-match
  over the **categorical** fields:
  `evidence_standard_met`, `risk_flags` (see multilabel note), `issue_type`,
  `object_part`, `claim_status`, `valid_image`, `severity`.
  Return `{field, n, correct, accuracy}`. (Free-text fields
  `evidence_standard_met_reason`, `claim_status_justification`,
  `supporting_image_ids` are **not** exact-match scored — report coverage/
  non-empty rate only.)
- `claim_status_confusion(gold, pred) -> dict` — **the headline metric.** Build
  the full **3×3 confusion matrix** over the fixed label order
  `["supported", "contradicted", "not_enough_information"]` (rows = gold,
  cols = pred; include an `"<other>"` bucket only if a prediction is off-vocab).
  Return the matrix **plus** per-class precision / recall / F1 and macro-F1.
  **Call out the contradicted ↔ not_enough_information cells explicitly** in the
  return dict (e.g. `contra_as_nei`, `nei_as_contra`): distinguishing
  *contradicted* from *not_enough_information* is the actual task — `supported`
  is comparatively easy, so a high overall accuracy that hides confusion between
  those two classes is a failing system. This metric is what proves the agent
  earns its complexity (decisions #3, #4, #5).
- `multilabel_prf(gold, pred, field="risk_flags") -> dict` — split on `;`,
  strip, drop the `none` sentinel to the empty set, dedupe to a set per row.
  Compute **micro and macro** precision / recall / F1 over the label set, plus
  per-label support. (Same helper must work for any multi-valued field.)
- `routing_metrics(pred_meta) -> dict` — consume optional per-row routing
  metadata the predictor may attach (key `_routing` per row, absent for the
  stub): `early_stop` (bool), `reinspected` (bool), and `label_flipped`
  (claim_status changed after re-inspection). Return counts:
  `early_stop_count`, `reinspection_count`, `post_reinspection_flip_count`,
  with safe zeros when metadata is absent. This is how decision #5 ("collapse
  the design if routing is flat") gets measured.
- `score_all(gold, pred) -> dict` — orchestrate the above into one report dict.

Add a `__main__` self-test that fabricates a tiny gold/pred pair (including one
contradicted-as-NEI error) and asserts the confusion matrix and F1 numbers are
what you compute by hand.

---

## P1-CX-2 — `code/evaluation/main.py` (stub predictor)

The evaluation entry point (problem statement requires an `evaluation/` folder).
Wires the harness together **before the real agent exists** (decision #4: eval
harness first).

- Define the predictor interface explicitly: **`predict(row: dict) -> dict`**
  returning the 14 output columns for one input row. For P1 ship a **stub**
  `predict` (clearly named/commented as a placeholder) — e.g. constant or
  trivial heuristic values drawn only from the **allowed vocab** in
  `problem_statement.md`. The stub must **not** import `openai` or hit the
  network; the whole eval must run offline. The real agent will later satisfy
  this same signature as a drop-in.
- Read `dataset/sample_claims.csv` via `io_utils` **in file order**; split each
  row into the 4 input columns (the predictor only sees inputs) and keep the
  full row as gold.
- Run `predict` over every row, collect predictions in order, then call
  `metrics.score_all(gold, pred)`.
- Print a readable report to stdout: per-field accuracy table, the
  `claim_status` 3×3 confusion matrix (labelled axes), per-class + macro F1 with
  the contradicted/NEI confusion called out, `risk_flags` micro/macro P/R/F1,
  and the routing counts.
- Also write `code/evaluation/evaluation_report.md` — for P1 a skeleton with the
  metric tables filled from this run plus **placeholder** sections for the
  operational analysis (model calls, token usage, images processed, cost,
  latency, TPM/RPM) the problem statement requires; these get real numbers once
  instrumentation (P2/P4) lands.
- Exit 0 always in P1 (it's a measurement tool, not a gate). A pass/fail
  threshold can be added later once the real agent's baseline is known.

Keep `metrics.py` import-clean: `main.py` does all I/O and presentation,
`metrics.py` does pure computation, so metrics stay unit-testable.

---

## Later phases (specs to be expanded when we reach them)

- **P2-CX** `code/tools/evidence_lookup.py`, `code/tools/history_lookup.py`:
  pure data lookups with an issue→family mapping for evidence requirements.
- **P2/P4-CX** `code/instrument.py`: call/token/cost/latency counters + a
  per-image inspection cache keyed by image path/hash.
- **P5-CX** schema/column-order validator pass over the final `output.csv`.

Full specs for these will be appended here before each phase.
