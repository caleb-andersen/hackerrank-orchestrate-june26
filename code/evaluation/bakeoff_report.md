# Synthesis Model Bake-off — gpt-4.1 vs gpt-5.4

Same agent, same prompts, same cheap-vision inspector (`gpt-4.1`) — only the synthesis/decision model varies. Run live over the 20 labelled samples with the prediction cache OFF, so the token/cost/latency numbers below are real and complete.

**Recommendation: `gpt-4.1` for synthesis.** gpt-4.1 synthesis beats gpt-5.4 by 0.041 macro-F1 at 2.1x lower cost, with equal contradicted<->NEI confusion (0 vs 0); the expensive reasoner tier does not earn its place — collapse the synthesis tier to the cheap model (decision #5).

> Caveat: this is a 20-sample result and the macro-F1 gap is ~1 row of claim_status (18/20 vs 17/20); risk_flags micro-F1 actually favours gpt-5.4 (0.677 vs 0.593). The defensible reading is that the expensive reasoner is **not demonstrably better** here, so the two-model tier is not justified by the data — not that gpt-4.1 is decisively superior.

**Owner decision: ship `gpt-5.4` for synthesis.** The data-driven recommendation above is to collapse to gpt-4.1, but the ~1-row gap is inside the agent's run-to-run sampling variance (an independent cached gpt-5.4 run scored 16/20, vs 17/20 here — same config), so the measurement does not reliably separate the two models. We keep the stronger reasoner on the single synthesis call (decision #6) as insurance for the harder, unseen 44-row test set, where a reasoning edge is most likely to matter and the absolute cost (~$0.02/claim difference) is immaterial. The bake-off stays in the record as the evidence the tier was held to a measured bar (decision #5).

## Accuracy

The headline is macro-F1 on claim_status, and specifically the contradicted<->not_enough_information cells — that discrimination is the task (decisions #3-#5). `supported` is comparatively easy.

| metric | synth=gpt-4.1 | synth=gpt-5.4 |
| --- | --- | --- |
| claim_status accuracy | 0.900 | 0.850 |
| macro-F1 (headline) | 0.893 | 0.852 |
| supported F1 | 0.929 | 0.889 |
| contradicted F1 | 0.750 | 0.667 |
| not_enough_information F1 | 1.000 | 1.000 |
| contradicted->NEI cells | 0 | 0 |
| NEI->contradicted cells | 0 | 0 |
| risk_flags micro-F1 | 0.593 | 0.677 |

## Operational (real, instrumented)

| metric | synth=gpt-4.1 | synth=gpt-5.4 |
| --- | --- | --- |
| model calls (synth+inspect) | 69 | 69 |
| prompt tokens | 143555 | 147070 |
| completion tokens | 8339 | 8799 |
| total tokens | 151894 | 155869 |
| est. cost (USD) | 0.354 | 0.735 |
| wall-clock (s) | 236.267 | 252.216 |
| sec / claim | 11.813 | 12.611 |
| cost / claim (USD) | 0.018 | 0.037 |

Cost uses the assumed per-1K-token `PRICING` in `code/instrument.py` (planning estimate, not a billing source of truth).

## Routing

| metric | synth=gpt-4.1 | synth=gpt-5.4 |
| --- | --- | --- |
| synthesis calls | 40 | 40 |
| inspection calls (images processed) | 29 | 29 |
| rows that early-stopped | 0 | 0 |
| rows re-inspected | 9 | 9 |
| rows where re-inspection flipped the label | 0 | 0 |
