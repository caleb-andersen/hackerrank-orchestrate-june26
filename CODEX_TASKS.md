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

## P2-CX — agent tool lookups + instrumentation

Claude Code owns the P2 judgment pieces (prompts, `inspect_image`, the loop,
synthesis, validator) and has already wired these three modules into the agent
via **dependency injection** — `code/main.py` imports them and passes them into
`agent/loop.run_claim`. Your job is to make the real implementations behind the
**exact signatures below**. While they are absent, `main.py` runs on labelled
fallbacks (`[main] evidence_lookup=fallback history_lookup=fallback`); landing
your versions flips that to the module names. **Do not change the signatures or
return-dict keys** without updating this file and `agent/loop.py`.

These are tool results fed to the reasoner as JSON, so keep returns **small and
flat** (the model reads them). Pure stdlib + `io_utils.read_csv_dicts`. No
network. Resolve dataset paths from `config` (`EVIDENCE_REQUIREMENTS_CSV`,
`USER_HISTORY_CSV`). Load each CSV **once** and cache in a module-level dict.

### P2-CX-1 — `code/tools/evidence_lookup.py`

```python
def get_evidence_requirement(claim_object: str, issue_family: str) -> dict
```
- Map `(claim_object, issue_family)` to the closest row(s) in
  `evidence_requirements.csv`. `issue_family` is a free-ish hint from the model
  (one of `agent.prompts.EVIDENCE_FAMILY_HINTS`, or a raw `issue_type` like
  `scratch`/`crack`/`water_damage`). Build a small keyword→`requirement_id` map
  per object so e.g. car+`dent or scratch`→`REQ_CAR_BODY_PANEL`,
  car+`crack`/`broken`/`missing`→`REQ_CAR_GLASS_LIGHT_MIRROR`,
  laptop+`screen`/`keyboard`/`trackpad`→`REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD`,
  package+`crushed`/`torn`/`seal`→`REQ_PACKAGE_EXTERIOR`, etc. Match on
  substrings so the model's phrasing need not be exact.
- Return:
  ```python
  {
    "claim_object": claim_object,
    "issue_family": issue_family,
    "matched": {            # the best-matching specific requirement, or None
        "requirement_id": ..., "applies_to": ..., "minimum_image_evidence": ...,
    },
    "always_applies": [     # the object-agnostic baselines, same shape as matched
        REQ_GENERAL_OBJECT_PART, REQ_GENERAL_MULTI_IMAGE, REQ_REVIEW_TRUST,
    ],
  }
  ```
  If nothing object-specific matches, `matched=None` and rely on
  `always_applies`. Never raise — unknown object ⇒ `matched=None`.
- `__main__` self-test: assert a car dent maps to `REQ_CAR_BODY_PANEL` and that
  `always_applies` always contains the three general rows.

### P2-CX-2 — `code/tools/history_lookup.py`

```python
def get_user_history(user_id: str) -> dict
```
- Index `user_history.csv` by `user_id` (load once). Return a flat dict with:
  `user_id`, `past_claim_count`, `accept_claim`, `manual_review_claim`,
  `rejected_claim`, `last_90_days_claim_count`, `history_flags`,
  `history_summary`, **plus** two derived helper keys:
  - `"suggests_user_history_risk"`: bool — true when `history_flags` is not
    `none`/empty, OR `rejected_claim` > 0, OR `manual_review_claim` is a
    meaningful fraction of `past_claim_count`. (Pick a simple, documented rule.)
  - `"risk_note"`: one short string summarizing why (or `"no notable risk"`).
  The reasoner decides `user_history_risk`; you only surface the signal — do not
  decide the flag for it.
- Missing `user_id` ⇒ return a safe default
  (`{"user_id": user_id, "history_flags": "none",
  "suggests_user_history_risk": False, "risk_note": "no history on record", ...}`),
  never raise (test rows are all present, but `claims.csv` must not crash on a
  gap — see `data_audit.py` check #3).
- `__main__` self-test: a known low-risk user ⇒ `suggests_user_history_risk`
  False; a fabricated unknown user ⇒ safe default.

### P2-CX-3 — `code/instrument.py`  (start in P2, finish in P4)

Call/token/cost/latency accounting + a per-image inspection cache. Optional for
the loop to function, required for `evaluation_report.md`'s operational
analysis.
- A small `Instrument` object (or module-level singleton) with:
  - `record_call(model: str, usage)` — accept an OpenAI `usage` object/dict;
    accumulate `prompt_tokens`, `completion_tokens`, call count **per model**.
  - `snapshot() -> dict` — totals per model + grand totals, plus a `cost_usd`
    estimate from a `PRICING` dict (read assumed $/1K in/out per model from a
    module constant; document the assumption).
  - timing helpers (context manager `track()` or start/stop) for wall-clock.
- `InspectionCache`: keyed by **absolute image path** (optionally + a content
  hash via `hashlib` so identical bytes share a result). `get(key)` /
  `set(key, observation)`. Intended to wrap `tools.inspect_image.inspect_image`
  so duplicate inspections of the same file (re-runs, repeated tool calls) cost
  nothing. Keep it in-memory; a JSON-on-disk layer is a nice-to-have.
- Wiring note: `agent/loop.run_claim` already returns per-row `synth_calls` /
  `inspect_calls` in `_routing`; `instrument` adds the token/cost layer. Coordinate
  the integration point with Claude Code (likely `main.py` passing the
  `usage` from each response into `record_call`).

## P3-CX — sample-iteration diagnostics

Context: Claude Code owns the P3 judgment work (reading the claim_status
confusion matrix and tuning the contradicted-vs-NEI / supported boundary in the
prompts). That landed: the real agent now runs inside the eval harness
(`evaluation/agent_predictor.py`, prompt-fingerprinted per-row cache), and
`evaluation/main.py` prints a per-row gold-vs-pred mismatch diff. As of the P3
tuning pass: **claim_status 16/20, macro-F1 0.73, and the contradicted/NEI
confusion cells are both 0** — the remaining misses are supported↔contradicted
vision close-calls, not reasoning-boundary errors.

Your P3-CX work is the **mechanical diagnostics** that make the next iteration
(and the P4 operational report) legible. Pure, deterministic, stdlib only. **Do
not** add any post-hoc overwrite of the agent's fields — the validator
flags-and-logs, it never silently rewrites (architecture decision #4). These are
read-only measurement helpers.

### P3-CX-1 — `field_confusion` in `code/evaluation/metrics.py`

Add a generic per-value confusion helper so we can see *where* the lower-scoring
categorical fields miss (issue_type=0.45, severity=0.35 right now — is the model
systematically defaulting to `unknown`, or is it noise?).

```python
def field_confusion(gold: list[dict], pred: list[dict], field: str) -> dict
```
- Matched **by file order** (same as every other metric — never by `user_id`).
- Use the existing `normalize()` for both sides.
- Return a small, JSON-serializable dict:
  - `field`
  - `labels`: sorted union of gold+pred normalized values actually seen.
  - `matrix`: `{gold_value: {pred_value: count}}` over those labels.
  - `per_value`: `{value: {support, precision, recall, f1}}` (treat each distinct
    gold value as a one-vs-rest class; same math as `claim_status_confusion`).
- **No printing here** — `metrics.py` stays pure. Extend its `__main__`
  self-test with a tiny hand-checked fixture for one field.
- Optionally surface it from `score_all` under a `field_confusions` key for
  `["issue_type", "severity", "object_part"]`, so the report can render them.

### P3-CX-2 — render the field confusions in `code/evaluation/main.py`

- Add a `_print_field_confusion(report, field)` that prints the matrix Claude
  Code's `_print_confusion` already models (labelled axes), for `issue_type` and
  `severity`. Presentation only; all computation stays in `metrics.py`.
- Add the same tables to `_render_report` under a new
  `## Categorical Field Confusions` section. Keep the existing report sections
  and the `intro`/`title` parameters intact (P3 added those).
- Do **not** change the default predictor path, the cache, or the mismatch diff
  — only add the new section.

### P3-CX-3 — `code/evaluation/dump_predictions.py`

A tiny convenience writer (the cache already holds predictions; this just makes
them human-diffable). Reads `dataset/sample_claims.csv` (gold) in file order,
reads the cached real-agent predictions via `agent_predictor` (cache only — do
**not** trigger Azure calls; if a row is uncached, write its cells as `MISSING`),
and writes `code/evaluation/sample_predictions.csv` with, per row and in file
order: `user_id`, then for each scored field a `gold_<field>` / `pred_<field>`
column pair. Gitignored already. Pure stdlib + `io_utils`. `__main__` runs it.

---

## P4-CX — operational reporting: persistence + cost projection

Context: Claude Code owns the P4 judgment + live work and it has landed. The
`Instrument` (token/cost/latency) is now threaded through the real call path —
`tools/inspect_image.py`, `agent/loop.run_claim`, and
`evaluation/agent_predictor.make_predictor` all take an optional `instrument=`
and call `instrument.record_call(model, response.usage)` on every synthesis /
inspection call. The **bake-off** (`code/evaluation/bakeoff.py`) runs the agent
over the samples once per synthesis model (`gpt-4.1` vs `gpt-5.4`, inspection
fixed at `gpt-4.1`), captures real call/token/cost/latency, writes
`bakeoff_report.md` (committed) with the model-tiering recommendation, and dumps
the chosen model's snapshot to `code/evaluation/operational_snapshot.json`
(gitignored). `evaluation/main.py` renders that snapshot into the
`## Operational Analysis` section of `evaluation_report.md`.

**Important — you cannot run the live pieces.** This environment has no `openai`
package and no Azure creds, so do **not** try to run `bakeoff.py` or the default
`evaluation/main.py`. Your P4-CX work is entirely **offline, deterministic,
stdlib-only**, and must not require a live run to self-test (fabricate a snapshot
dict in your `__main__`). It is additive — do not change the loop, the predictor,
the bake-off, or the instrument wiring CC added.

### P4-CX-1 — persistence helpers in `code/instrument.py`

You own `instrument.py`. Add JSON persistence so snapshots survive across
processes and the inspection cache can be reused between runs (the spec for this
module always called the on-disk cache layer a nice-to-have; P4 is when it pays
off — a warm `InspectionCache` means re-runs re-inspect zero images).

- `Instrument.to_json() -> str` / `Instrument.save(path)` — serialize the
  accumulated `_calls` and `_timings` (not just the computed `snapshot()`; keep
  enough to keep accumulating after a reload).
- `Instrument.load(path) -> Instrument` (classmethod) / `from_json(text)` —
  reconstruct an `Instrument` whose `record_call`/`snapshot` keep working.
- `Instrument.merge(other) -> None` — fold another Instrument's counts/timings
  in (so several partial runs can be combined into one operational total).
- `InspectionCache.save(path)` / `InspectionCache.load(path, use_content_hash=…)`
  — JSON-on-disk layer keyed exactly as today (abs path + optional content
  hash). A missing/corrupt file loads to an empty cache, never raises.
- Keep the existing `record_call`/`snapshot`/`track`/`key_for` behavior and the
  module-level `instrument` / `inspection_cache` singletons unchanged.
- Extend `__main__`: round-trip an Instrument (record → save → load → snapshot
  equal), a merge (two Instruments → summed totals), and a cache save/load.
- **Do not** change `PRICING` values without a cited source; they are documented
  planning estimates and CC's reports lean on them.

### P4-CX-2 — `code/evaluation/cost_projection.py`

A standalone, offline projection from the 20-sample live run to the full
`claims.csv` (44 rows) — the "what will the real test set cost / how long / will
it fit TPM-RPM" analysis the problem-statement operational section wants. CC's
`evaluation_report.md` carries a one-line linear estimate; this is the rigorous
version it points to.

- Read `code/evaluation/operational_snapshot.json` (written by `bakeoff.py`). If
  it is absent, print a clear `run evaluation/bakeoff.py first` message and exit
  0 — never crash, never hit the network.
- Read the full test-set row count from `claims.csv` via `io_utils`
  (`config.CLAIMS_CSV`) — **do not hardcode 44**; data_audit already asserts it.
  Read the sample row count from the snapshot (`n_rows`).
- Project, with a clearly documented scaling assumption (linear in rows; per-row
  token/cost/latency averages from the sample run): total calls, prompt/
  completion/total tokens, cost (per model and grand total, using the snapshot's
  already-computed per-model cost), wall-clock runtime.
- Add a **TPM/RPM headroom** check: from the projected tokens and runtime,
  estimate peak tokens-per-minute and requests-per-minute and compare against
  configurable assumed deployment limits (module constants with a documented
  default, e.g. `ASSUMED_TPM`, `ASSUMED_RPM`); report headroom / whether
  throttling/backoff is expected and the cache's mitigating effect.
- Print a readable table and **append/write** a `## Full-Test-Set Cost
  Projection` section to `evaluation_report.md` *or* write a sibling
  `cost_projection.md` — pick the sibling file to avoid racing CC's
  `evaluation_report.md` writer (cleaner; CC's report can link it).
- Pure stdlib + `io_utils`. `__main__` runs it; the self-test path must work off
  a fabricated snapshot dict so it passes with no live artifacts present.

---

## P5-CX — final submission gates: whole-file output validator + transcript merge

Context: Claude Code owns the P5 judgment/owner work (final live run on
`claims.csv`, submission hygiene, `code/README.md`, the transcript narrative and
interview prep) and is doing it in parallel — your P5-CX tasks do **not** block
on the final `output.csv` existing, and CC's work does not block on you. Both are
independent; the only hard gate is the user's live Azure run, which neither of us
can do.

Your P5-CX work is the **last automated gate before submission**: a standalone
whole-file validator over `output.csv`, plus the transcript log-merge helper.
Pure, deterministic, **stdlib only**, **no network, no `openai` import**. Both
must self-test offline with **no live artifacts present** (fabricate inputs in
`__main__`; if `output.csv` is absent the validator prints a clear message and
exits 0, it never crashes).

### P5-CX-1 — `code/evaluation/validate_output.py`

A standalone, file-level gate over the emitted `output.csv`. This is **distinct
from `code/validate.py`** (CC-owned), which flags **one row at a time inside the
pipeline**. This script validates the **whole written file** as the evaluator
will see it, and is the final check the owner runs before zipping. **Reuse**
`validate.validate_row` for the per-row invariant pass — do **not** re-implement
the invariant logic; single source of truth lives in `validate.py`.

`def validate_output(output_path=config.OUTPUT_CSV, claims_path=config.CLAIMS_CSV) -> tuple[bool, list[str]]`
returning `(ok, problems)`. Checks, each appending a clear human-readable string
to `problems` (and `ok=False` on any **hard** failure):

1. **File exists.** If `output_path` is absent: print `run code/main.py first to
   produce output.csv`, return `(True, [])`, and `__main__` exits 0 (not a
   failure — there is simply nothing to gate yet).
2. **Exact header.** The CSV header equals `config.OUTPUT_COLUMNS` — same names,
   same order, no extras, no missing. Report the first divergence precisely
   (`col 5: got 'risk_flags' expected 'evidence_standard_met'`). Read with
   `csv.reader` for the raw header (don't let `DictReader` mask order/dupes).
3. **Row count.** Equals the `claims.csv` row count read via
   `io_utils.read_claims(claims_path)` — **do not hardcode 44**.
4. **Input fidelity (the "diff" gate).** For each row in file order, the four
   **input** columns (`user_id`, `image_paths`, `user_claim`, `claim_object`)
   must match `claims.csv` **exactly, row-for-row in file order** (architecture
   decision #8 — never match on `user_id`, it has duplicates). This catches a
   dropped/reordered/mangled row, which would silently misalign every prediction
   against the evaluator's gold. Report the row index and field of the first
   mismatch.
5. **Vocab + format**, per row (reuse the allowed sets from `agent.prompts`:
   `CLAIM_STATUS_VALUES`, `ISSUE_TYPE_VALUES`, `SEVERITY_VALUES`,
   `OBJECT_PART_VALUES[claim_object]`, `RISK_FLAG_VALUES`): `claim_status`,
   `issue_type`, `severity`, `object_part` in vocab; `evidence_standard_met` and
   `valid_image` are exactly `true`/`false` (lowercase); every `;`-split token of
   `risk_flags` is in `RISK_FLAG_VALUES` (or the field is `none`);
   `supporting_image_ids` is `none` or `;`-joined ids that all appear in that
   row's `image_paths` stems. No empty cells in the four free-text/decision
   fields that must always be populated (`evidence_standard_met_reason`,
   `claim_status_justification`).
6. **Per-row invariants.** Run `validate.validate_row(row, row['claim_object'])`
   for every row; collect any `schema`/`invariant` flags (ignore `norm`) into a
   count summary so the owner sees how many rows the model left internally
   inconsistent. These are **reported, not fixed** (decision #4) — they do not by
   themselves set `ok=False` unless you decide a threshold; default: report-only,
   and let `ok` be driven by checks 2–5. Document that choice in a comment.

`__main__`: run `validate_output()`, print a readable PASS/FAIL summary (counts
per check, first few problems), and **exit non-zero on `not ok`** so it can be a
CI/pre-submission gate. The self-test must also fabricate a tiny good and a tiny
bad `output.csv` in a temp dir and assert the bad one fails on the expected
check (e.g. a reordered header, a dropped row, an off-vocab `severity`).

### P5-CX-2 — `merge_codex_log.py` (repo root, **not** under `code/`)

A small dev utility (it is **not** part of the evaluable solution, so keep it at
repo root, not in `code/` — it must not end up in `code.zip`). The shared
transcript log at `~/hackerrank_orchestrate/log.txt` (Windows:
`%USERPROFILE%\hackerrank_orchestrate\log.txt`) is the submission's
`chat_transcript`. Codex does **not** auto-append to it, so this reconciles a
captured Codex session into the shared log per AGENTS.md §5.

- Resolve the log path cross-platform via `pathlib.Path.home()` (AGENTS.md §7).
  Never hardcode `/Users/...` or `C:\Users\...`.
- `append_codex_section(session_text: str, *, label: str, log_path=None)` —
  append `session_text` as a clearly delimited, labelled block:
  ```text
  ## [ISO-8601 TIMESTAMP] CODEX SESSION — <label>
  <session_text, verbatim, secrets redacted>
  ```
  Append-only; **never** rewrite/reorder/delete prior entries (AGENTS.md §2).
  Create the parent dir + file if missing. Write UTF-8 with `\n` line endings.
- **Idempotency:** before appending, check whether a block with the same `label`
  and an identical body already exists; if so, skip and report `already merged`.
- **Redaction:** before writing, scrub anything matching common secret shapes
  (`AZURE_OPENAI_API_KEY=...`, `sk-...`, bearer tokens, `api_key`/`apikey` lines)
  to `[REDACTED]` (AGENTS.md §5.4). Never write a key to the log.
- CLI: `python merge_codex_log.py --label "Codex P5-CX" path/to/codex_session.txt`
  (reads the file, redacts, appends). `__main__` self-test: append to a **temp**
  log file (never the real one), assert the section lands and a second identical
  append is a no-op, and assert a fake key is redacted.

When done, run both files' self-tests, then capture your Codex session and merge
it into `log.txt` (use P5-CX-2 itself, labelled `Codex P5-CX`).
