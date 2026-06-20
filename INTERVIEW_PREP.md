# Interview Prep & Transcript Narrative

Owner-facing notes for the post-submission AI-judge interview. Not part of the
evaluable solution (it stays at repo root, out of `code.zip`). Everything here is
grounded in committed artifacts: `PLAN.md`, `code/evaluation/evaluation_report.md`,
`code/evaluation/bakeoff_report.md`, and the eight architecture decisions logged
in `~/hackerrank_orchestrate/log.txt`.

---

## 30-second pitch

> For each of 44 damage claims I run **one verification agent** with a real
> tool-calling loop. It inspects each submitted image individually, checks the
> evidence-requirement and the user's history through tools, and writes 14
> columns of structured judgment. Images are primary truth; the conversation
> says what to check; history adds risk but can't override what the image shows.
> I built the **evaluation harness first** and graded every change against 20
> labelled samples — the headline metric is the contradicted-vs-not-enough-info
> discrimination, which is the actual hard part of the task.

---

## The development narrative (how it was built)

1. **P0 — ground truth before code.** Probed the Azure resource
   (`probe_azure.py`) instead of assuming model availability — found `gpt-5.4`
   (reasoner) and `gpt-4.1` (vision). Audited the data (`data_audit.py`): 44
   rows, non-contiguous case folders (reach `case_056` with gaps), the
   `images/test` vs `dataset/images` path-prefix mismatch, 7 repeat users,
   multilingual adversarial chats. These facts drove every later design choice.
2. **P1 — eval harness first.** Wrote `metrics.py` + a stub predictor before the
   real agent, so the day-one question was always "did the number move", not "does
   it run". Headline metric chosen deliberately: the claim_status 3×3 confusion
   matrix, with the contradicted↔NEI cells called out.
3. **P2 — the agent.** Prompts, per-image `inspect_image`, the tool loop with
   routing, synthesis, and the flag-not-overwrite validator. Lookups injected as
   tools.
4. **P3 — tune against the table.** Read the confusion matrix and tuned the
   contradicted-vs-NEI / supported boundary in the prompts; logged every
   regression. Landed claim_status 16/20, macro-F1 0.73, contradicted↔NEI cells 0.
5. **P4 — operational truth.** Threaded `instrument.py` (tokens/cost/latency)
   through the real call path and ran a live model **bake-off** to decide the
   tiering with numbers, not vibes.
6. **P5 — submission gates.** Whole-file `output.csv` validator, transcript merge,
   final run, README, this prep.

**Execution split:** strict [CC]/[CX] — Claude Code owns judgment-heavy work
(prompts, loop, synthesis, validator logic, tuning, reports); Codex owns
mechanical, well-specced work (IO, lookups, metrics, audits, projections,
the output validator). Specs live in `CODEX_TASKS.md`.

---

## The eight architecture decisions (be ready to defend each)

1. **Single agent + tool loop, not rules or multi-agent.** One goal — one
   judgment over one claim — so no division of labor to justify a crew. A
   rules pipeline can't weigh distinct facts in an image against a claim's
   semantics and severity; that needs reasoning.
2. **Per-image inspection returning structured JSON.** Protects synthesis from a
   lossy summary — the image can be re-examined when an observation conflicts
   with the claim — and makes `supporting_image_ids` honest (cite the clear
   image, not the blurry one). Per-image judgement, not per-claim.
3. **Three independent axes + a validator that flags rather than overwrites.**
   `evidence_standard_met`, `valid_image`, `claim_status` answer different
   questions and can legitimately diverge (`case_008`: evidence true / valid
   false; `case_006`: evidence false / valid true).
4. **Validator flags-and-logs, never silently rewrites.** A silently corrected
   contradiction hides the real model error rate — the exact thing we measure
   and defend. Loud, format-only normalization is the one exception.
5. **Routing must be measured; collapse the design if it's flat.** Early-stop and
   re-inspection counts are reported. Re-inspection fires 9/20; early-stop is
   currently 0 and I say so rather than claiming a benefit I can't show.
6. **Two model tiers, held to a measured bar.** Cheap vision for high-volume
   inspection, the reasoner for the single synthesis call. The bake-off is the
   evidence the tier earned its place.
7. **Injection handled as data.** Multilingual / in-image instructions are
   surfaced via `text_instruction_present` and never obeyed.
8. **File-order iteration, never case numbers / `user_id`.** Folders are
   non-contiguous and `user_id` repeats; output is one row per input row in file
   order, and metrics align gold↔pred by position.

---

## Likely judge questions → answers

**"Why not a multi-agent system?"** No division of labor exists — it's one
judgment over one claim. Multiple agents would add coordination cost and failure
surface with nothing to parallelize. Decision #1.

**"Why an LLM agent over a deterministic rule pipeline?"** The hard rows need
semantic weighing of image facts against the claim and its severity (a "dent" of
the wrong severity on the wrong part is *contradicted*, not *supported*). Rules
can gate format and evidence sufficiency — and I do use them, in the validator —
but not the core judgment.

**"What's your headline metric and why?"** Macro-F1 on claim_status, and
specifically the contradicted↔not_enough_information cells. `supported` is easy;
distinguishing "the image disagrees with the claim" from "the image can't tell
us" *is* the task. Those cells are currently 0 — the model isn't confusing the
two classes; its misses are supported↔contradicted vision close-calls.

**"How did you choose your models?"** Probed availability first (no assumptions),
then ran a live bake-off (`bakeoff_report.md`): `gpt-4.1` synthesis scored
slightly higher (18/20 vs 17/20, macro-F1 0.893 vs 0.852) at ~2.1× lower cost.
But that gap is ~1 row, inside the agent's run-to-run sampling variance (a
separate cached gpt-5.4 run scored 16/20). So the honest reading is "the
expensive reasoner is *not demonstrably better*," and I keep `gpt-5.4` on the
single synthesis call as cheap insurance for the harder unseen 44-row set, where
the absolute cost difference (~$0.02/claim) is immaterial. The bake-off stays in
the record as proof the tier was held to a bar (decisions #5, #6).

**"Cost / latency / rate limits?"** From a live, un-cached 20-sample run: 69
calls, ~156K tokens, ~$0.73, ~257s (~12.9s/claim). Projected full 44-row run
~$1.6 and ~9.5 min. Vision inspections (cheap tier) are the call volume;
synthesis (reasoner) is the token/cost. Mitigations: per-row prediction cache +
path+content-hash inspection cache (re-runs cost zero), Azure SDK 429 backoff,
and a `MAX_LOOP_ITERS` bound. `cost_projection.py` does the TPM/RPM headroom
check. Full table in `evaluation_report.md`.

**"How do you prevent prompt injection from the chat or images?"** Treated as
data, never instructions; flagged `text_instruction_present`. Decision #7.

**"How do you know you're not overfitting to the 20 samples / hardcoding?"** No
file-specific labels anywhere — the validator and audit enforce that. The agent
only sees the 4 input columns per row; gold is never passed to `predict`. Tuning
was on prompts and boundaries, not per-case answers, and every regression was
logged.

**"What would you do with more time?"** Push the supported↔contradicted vision
close-calls (the remaining claim_status misses) and the lower-scoring
`severity`/`issue_type` fields — likely a second inspection pass on conflict
rows and tighter severity rubrics in the inspection prompt. And get a real
early-stop signal working, or formally retire it per decision #5.

---

## Submission checklist (P5 close-out)

- [ ] `python code/main.py` run with live Azure creds → repo-root `output.csv`
      (44 rows). **Requires the owner's creds — the only step no agent can do.**
- [ ] `python code/evaluation/validate_output.py` passes (Codex P5-CX-1 gate).
- [ ] `output.csv`: exact 14 columns in order, 44 rows, input columns match
      `claims.csv` row-for-row, all values in allowed vocab.
- [ ] `code.zip`: includes `code/` (with `README.md`, prompts/configs) and the
      `evaluation/` folder; excludes `.venv`, `__pycache__`, `.env`,
      `output.csv`, runtime snapshots (see `.gitignore`).
- [ ] `chat_transcript` = `~/hackerrank_orchestrate/log.txt`, with the Codex
      session merged in (Codex P5-CX-2).
- [ ] `evaluation/evaluation_report.md` carries the operational analysis.
