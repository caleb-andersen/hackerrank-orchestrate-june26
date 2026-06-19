# Build Plan — HackerRank Orchestrate: Multi-Modal Evidence Review

## Context

For each of 44 damage-claim rows (`car` / `laptop` / `package`) we get 1–3 images, a short multilingual support chat, the user's claim history, and a minimum-evidence checklist. We emit 14 columns of structured judgment per claim into `output.csv`. **Images are primary truth; the conversation defines what to check; history adds risk but cannot override clear visual evidence.** We develop and self-grade against 20 labeled samples. This build is **one claim-verification agent with a real tool-calling loop**, and the loop's routing (early-stop, re-inspection) must be *measured*, not assumed. If the routing metrics come back ~zero, we collapse to a simpler design and defend that with data.

**Decisions:** Azure deployments probed (not assumed). Final synthesis = bake-off a stronger reasoner (gpt-5.x/o-series) vs gpt-4o, pick by sample accuracy. `output.csv` at repo root. Orchestration = raw OpenAI (Azure) SDK tool-calling loop, no framework. Execution = strict [CC]/[CX] split (Claude Code = judgment; Codex = mechanical).

---

## 1. Architecture

**One agent, run once per claim row, that calls real tools and makes real routing decisions.** Single-agent (no division of labor — one judgment over one claim), not a rules pipeline (the hard cases need reasoning over distinct facts vs the claim).

**Three tools:**
- `inspect_image(image_id)` → one vision call on a **single** image returning rich **structured** JSON: object & part visible, damage type if any, image-quality flags (blur / glare / crop-obstruction / wrong-angle), authenticity cues (manipulation / non-original / screenshot), and any **text rendered inside the image**.
- `get_evidence_requirement(claim_object, issue_family)` → lookup into `evidence_requirements.csv`. **Grounds** `evidence_standard_met`.
- `get_user_history(user_id)` → lookup into `user_history.csv`. Feeds `risk_flags`.

**Loop must do — and prove — real routing:** early-stop on clear single-image claims; re-inspect when an observation conflicts with the claim or the user carries history risk. A **synthesis step** writes the 14 fields. Per-image inspection makes `supporting_image_ids` and per-image validity honest (cite `img_2`, not the blurry `img_1`).

---

## 2. Data facts (design around these)

- **`claims.csv` = 44 rows**, case folders **non-contiguous** (reach `case_056`). Iterate CSV rows in file order; one output row per input row, same order. Never loop over case numbers.
- **Path prefix mismatch (critical):** CSV `image_paths` = `images/test/...`; files live at `dataset/images/...`. Resolve as `DATASET_ROOT / image_path`. Image ID = filename stem.
- **`.DS_Store`** junk → filter to `.jpg/.jpeg/.png`.
- **7 repeat users** (`user_045`×3; `user_004/018/034/040/041/042`×2). History lookup per row; handle missing-user gracefully.
- **Multilingual chats** (Hinglish, Spanish, romanized Chinese). Adversarial in-chat/in-image text → flag `text_instruction_present`, never obey.

---

## 3. Output contract & invariants (the validator)

14 columns, exact order: `user_id, image_paths, user_claim, claim_object, evidence_standard_met, evidence_standard_met_reason, risk_flags, issue_type, object_part, claim_status, claim_status_justification, supporting_image_ids, valid_image, severity`. Booleans lowercase. `risk_flags` & `supporting_image_ids` semicolon-joined or `none`. Use full allowed vocab from `problem_statement.md`.

**Three independent axes:** `evidence_standard_met` · `valid_image` · `claim_status`.

**Invariants (validator FLAGS + LOGS, never silently overwrites):**
1. `not_enough_information` ⟺ `evidence_standard_met=false` ⟺ `supporting_image_ids=none` ⟺ `severity=unknown`.
2. `supported`/`contradicted` both require `evidence_standard_met=true` and ≥1 supporting image (cited even when `valid_image=false` — `case_008`).
3. `contradicted` ≠ "no damage" — image disagrees with the claim (severity mismatch, wrong issue/part, no-damage-where-claimed, wrong object).
4. `severity=none` when `issue_type=none` (`case_014`, `case_020`).
5. `valid_image` distinct from `evidence_standard_met` (`case_008` evidence=true/valid=false; `case_006` evidence=false/valid=true). `valid=false` from manipulation/non-original/screenshot or cropped-unusable — not blur/wrong-angle alone.

---

## 4. Repo structure & entry points

```
code/
  main.py                  # ENTRY: agent over dataset/claims.csv -> /output.csv
  config.py                # model tiers, Azure env wiring, paths, max-iter caps   [CX]
  io_utils.py              # CSV read/write (exact order), image-path resolution    [CX]
  probe_azure.py           # Azure deployment capability probe                      [CC]
  tools/
    inspect_image.py       # vision call -> structured observation JSON             [CC]
    evidence_lookup.py     # get_evidence_requirement                               [CX]
    history_lookup.py      # get_user_history                                       [CX]
  agent/
    loop.py                # tool-calling loop: routing, early-stop, re-inspect     [CC]
    prompts.py             # system / inspection / synthesis prompts                [CC]
    synthesis.py           # final 14-field structured decision                     [CC]
  validate.py              # schema + invariant validator (flags/logs)              [CC logic / CX scaffold]
  instrument.py            # call/token/cost/latency counters + per-image cache     [CX]
  evaluation/
    main.py                # ENTRY: over dataset/sample_claims.csv, computes metrics
    metrics.py             # per-field acc, confusion matrix, multi-label P/R/F1    [CX]
    evaluation_report.md   # strategy comparison + operational analysis            [CC]
  README.md                                                                          [CC]
PLAN.md  ·  CODEX_TASKS.md  ·  .gitignore  ·  output.csv (generated, not committed)
```
Output: repo-root **`output.csv`**, one row per `claims.csv` row, exact 14-col order. Entry points per AGENTS.md §6.

---

## 5. Phases (see CODEX_TASKS.md for [CX] specs)

- **P0** scaffolding + data audit ([CX]) · Azure probe ([CC]) · PLAN.md ([CC]). **Commit 1.**
- **P1** eval harness FIRST vs samples: metrics + routing metrics ([CX]); metric bar ([CC]). **Commit 2.**
- **P2** agent core: prompts, inspect_image, loop, synthesis, validator ([CC]); lookups ([CX]). **Commit 3.**
- **P3** sweep harness over 20 samples ([CX]); tune valid_image boundary + contradicted-vs-NEI from the table, log every regression ([CC]/owner). **Commit 4.**
- **P4** dual-config runner → side-by-side accuracy/cost/latency table ([CX]); pick by accuracy + write evaluation_report.md & operational analysis ([CC]/owner). **Commit 5.**
- **P5** log-merge + schema/diff validators ([CX]); final run, submission hygiene, transcript narrative & interview prep ([CC]/owner). **Commit 6.**

---

## 6. Libraries (secrets via env only)
`openai` (AzureOpenAI) · stdlib `csv`/`base64`/`pathlib` · `pydantic` (validate structured output) · `tenacity` (retry/backoff). Env: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, deployment names.

## 7. Risks
Model/region unavailable → probe gates P2, fall back to gpt-4o. Routing flat → collapse + defend with numbers. Contradicted-vs-NEI → decide evidence-sufficiency before support/contradict. valid_image boundary → tune on samples. Injection → guardrail + `text_instruction_present`. ~15.5h to deadline (2026-06-20 11:00 IST) → harness-first, commit often. Cost trivial but instrument everything (the *reasoning* is graded).
