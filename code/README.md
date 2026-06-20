# Multi-Modal Evidence Review — Solution

One claim-verification **agent** with a real tool-calling loop. For each of the
44 rows in `dataset/claims.csv` it inspects the submitted images, consults the
evidence checklist and the user's claim history, and emits 14 columns of
structured judgment into repo-root `output.csv`.

**Design in one line:** images are primary truth, the conversation defines what
to check, history adds risk but never overrides clear visual evidence — so a
single reasoning agent (not a rules pipeline, not a multi-agent crew) makes one
judgment over one claim, calling tools when it needs a fact.

See `../PLAN.md` for the full build plan and the eight architecture decisions,
and `evaluation/evaluation_report.md` for metrics + the operational analysis.

---

## Layout

```text
code/
  main.py                  # ENTRY: agent over dataset/claims.csv -> ../output.csv
  config.py                # paths, model tiers, Azure env wiring, loop caps
  io_utils.py              # CSV read/write (exact 14-col order), image-path resolution
  validate.py              # per-row schema + invariant validator (FLAGS, never overwrites)
  instrument.py            # call/token/cost/latency counters + per-image inspection cache
  probe_azure.py           # one-off Azure deployment capability probe
  tools/
    inspect_image.py       # one vision call on a single image -> structured observation JSON
    evidence_lookup.py     # get_evidence_requirement(claim_object, issue_family)
    history_lookup.py      # get_user_history(user_id)
  agent/
    prompts.py             # system / inspection / synthesis prompts + allowed vocab
    loop.py                # tool-calling loop: routing, early-stop, re-inspection
    synthesis.py           # final 14-field structured decision
  evaluation/
    main.py                # ENTRY: agent over sample_claims.csv -> metrics + report
    metrics.py             # per-field acc, claim_status confusion, multilabel P/R/F1
    agent_predictor.py     # wraps the real agent as predict(row)->dict, prompt-fingerprint cache
    bakeoff.py             # synthesis-model bake-off (gpt-4.1 vs gpt-5.4)
    cost_projection.py     # 20-sample -> full-test-set cost/TPM/RPM projection
    validate_output.py     # whole-file output.csv gate (column order, row count, vocab)  [P5]
    evaluation_report.md   # metrics + operational analysis (committed)
    bakeoff_report.md      # model-tiering decision, with numbers (committed)
```

---

## Architecture

**One agent, run once per claim row.** The loop (`agent/loop.run_claim`) drives a
raw Azure OpenAI tool-calling conversation — no orchestration framework. Three
tools are dependency-injected into the loop so each is unit-testable and the loop
stays the single place routing happens:

- **`inspect_image(image_id)`** — one vision call on a *single* image returning
  rich structured JSON: object & part visible, damage type, image-quality flags
  (blur / glare / crop / wrong-angle), authenticity cues (manipulation /
  non-original / screenshot), and any text rendered *inside* the image.
  Per-image (not per-claim) inspection is what makes `supporting_image_ids` and
  per-image `valid_image` honest — the agent can cite `img_2` and not the blurry
  `img_1`.
- **`get_evidence_requirement(claim_object, issue_family)`** — lookup into
  `evidence_requirements.csv`. Grounds `evidence_standard_met`.
- **`get_user_history(user_id)`** — lookup into `user_history.csv`. Surfaces a
  risk signal; the reasoner decides the `user_history_risk` flag.

**Routing the loop must prove, not assume:** early-stop on clear single-image
claims; **re-inspect** when an observation conflicts with the claim or the user
carries history risk. A final **synthesis** call writes the 14 fields. Routing is
*measured* (`evaluation_report.md` → Routing Metrics): re-inspection fires on
9/20 samples; early-stop is currently flat and reported honestly rather than
claimed.

**Three independent decision axes** — `evidence_standard_met` · `valid_image` ·
`claim_status` — because "is the evidence sufficient", "is the image usable", and
"does the image support the claim" are genuinely different questions (a usable
image can still be insufficient; an image can support a claim yet be flagged
non-original).

**Validator flags, never overwrites** (`validate.py`). It applies only loud,
format-level normalization (trim, lowercase booleans, empty→`none`) and then
*reports* every schema/invariant violation without correcting it — a silently
"fixed" contradiction would hide the exact model error rate we need to measure.

**Model tiering** (pinned from `probe_azure.py`): the cheap vision model
(`gpt-4.1`) absorbs the high-volume per-image inspections; the stronger reasoner
(`gpt-5.4`) runs only the single synthesis/decision loop. The bake-off
(`bakeoff_report.md`) holds that tier to a measured bar.

**Failure is safe, never fatal:** any model/parse failure degrades a row to
`not_enough_information` + `manual_review_required` rather than raising, so a
throttle or a bad response slows the run but never corrupts `output.csv`.

---

## Setup

Python 3.10+ (developed on 3.13). One runtime dependency for the live path:

```bash
pip install openai            # AzureOpenAI client; pydantic/tenacity optional
```

All offline code (metrics, validators, audits, projections) runs on the stdlib
with no `openai` installed.

**Secrets come from environment variables only** — never hardcoded. Set them in
your shell or a repo-root `.env` (gitignored; `main.py` loads it before importing
`config`):

```bash
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_API_VERSION=2025-01-01-preview   # optional; this is the default
INSPECTION_MODEL=gpt-4.1                       # optional; default shown
SYNTHESIS_MODEL=gpt-5.4                         # optional; default shown
```

---

## Run

**Produce the final predictions** (writes repo-root `output.csv`, one row per
`claims.csv` row, in file order):

```bash
python code/main.py                          # all 44 rows -> ../output.csv
python code/main.py --limit 2                # smoke-test the first 2 rows
python code/main.py --input dataset/sample_claims.csv --output /tmp/out.csv
```

`main.py` prints model-call counts, routing totals, validation-flag counts, and
elapsed time as it runs.

**Gate the output before submitting** (offline, no Azure):

```bash
python code/evaluation/validate_output.py    # exact 14 cols in order, row count,
                                             # input fidelity vs claims.csv, vocab
```

---

## Evaluation

```bash
python code/evaluation/main.py               # agent over the 20 labelled samples
```

Prints per-field accuracy, the **claim_status 3×3 confusion matrix** (the
headline — `contradicted` vs `not_enough_information` is the discrimination that
earns the design), per-class + macro-F1, `risk_flags` micro/macro P/R/F1,
categorical field confusions, and routing counts; it (re)writes
`evaluation/evaluation_report.md`. Predictions are cached per row on an
inputs+prompt+model fingerprint, so re-scoring an unchanged config costs zero
Azure calls (use `--no-cache` for a fully live, complete-instrumentation run).

Supporting tools: `bakeoff.py` (synthesis-model comparison →
`bakeoff_report.md`), `cost_projection.py` (sample → full-test-set cost / TPM /
RPM, → `cost_projection.md`), `dump_predictions.py` (human-diffable
gold-vs-pred CSV).

**Headline results on the 20 samples:** claim_status 16/20, macro-F1 0.73, with
the contradicted↔NEI confusion cells at **0** (the remaining misses are
supported↔contradicted vision close-calls, not reasoning-boundary errors). Full
numbers and the operational analysis (calls, tokens, cost, latency, TPM/RPM,
caching) are in `evaluation/evaluation_report.md`.

---

## Conventions that the data forces

- **Iterate CSV rows in file order**, one output row per input row in the same
  order. Never loop over case numbers — case folders are non-contiguous (they
  reach `case_056` with gaps). Metrics match gold↔pred by file order, never by
  `user_id` (it has duplicates: `user_045`×3 and six users ×2).
- **Path prefix mismatch:** CSV `image_paths` look like `images/test/...`; the
  files live at `dataset/images/...`. Always resolve as `DATASET_ROOT / path`.
  Image ID = filename stem (`img_1.jpg` → `img_1`). Non-image junk
  (`.DS_Store`) is filtered.
- **Adversarial multilingual chats / in-image text** are treated as data, never
  instructions — surfaced via the `text_instruction_present` flag, never obeyed.
