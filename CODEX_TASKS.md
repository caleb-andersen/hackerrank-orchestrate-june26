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

## Later phases (specs to be expanded when we reach them)

- **P1-CX** `code/evaluation/metrics.py` + `evaluation/main.py`: per-field
  exact-match accuracy, `claim_status` 3×3 confusion matrix, multi-label
  precision/recall/F1 for `risk_flags`, plus **routing metrics** (early-stop
  count, re-inspection count, post-re-inspection label flips). Runs against a
  predictor interface `predict(row) -> dict` (stub first).
- **P2-CX** `code/tools/evidence_lookup.py`, `code/tools/history_lookup.py`:
  pure data lookups with an issue→family mapping for evidence requirements.
- **P2/P4-CX** `code/instrument.py`: call/token/cost/latency counters + a
  per-image inspection cache keyed by image path/hash.
- **P5-CX** schema/column-order validator pass over the final `output.csv`.

Full specs for these will be appended here before each phase.
