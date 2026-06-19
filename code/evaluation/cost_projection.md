# Full-Test-Set Cost Projection

Projection source: `operational_snapshot.json`, scaled linearly from 20 live sample rows to 44 `claims.csv` rows. The assumption is linear in rows: per-row calls, tokens, cost, and latency from the sample run are representative of the full test set.

Synthesis model: `gpt-5.4`; inspection model: `gpt-4.1`.

## Projected Calls, Tokens, Cost

| model | calls | prompt_tokens | completion_tokens | total_tokens | cost_usd |
| --- | --- | --- | --- | --- | --- |
| gpt-4.1 | 63.800 | 77717 | 8136 | 85853 | 0.221 |
| gpt-5.4 | 88.000 | 245595 | 11070 | 256665 | 1.394 |
| TOTAL | 151.800 | 323312 | 19206 | 342518 | 1.615 |

Costs use the per-model `cost_usd` already present in the live snapshot; no pricing lookup or network call is performed here.

## Projected Routing

| metric | projected value |
| --- | --- |
| synthesis calls | 88.000 |
| inspection calls / images processed | 63.800 |
| images available | 63.800 |
| rows early-stopped | 0.000 |
| rows re-inspected | 19.800 |
| rows where re-inspection flipped label | 0.000 |

## Runtime And TPM/RPM Headroom

| metric | value |
| --- | --- |
| projected wall-clock seconds | 565.800 |
| projected wall-clock minutes | 9.430 |
| seconds per claim | 12.860 |
| estimated peak tokens / minute | 36325 |
| estimated peak requests / minute | 16.100 |
| assumed TPM limit | 100000 |
| assumed RPM limit | 60 |
| TPM headroom multiplier | 2.750 |
| RPM headroom multiplier | 3.730 |

No throttling is expected under the assumed limits.

Cache effect: with a warm `InspectionCache` and unchanged prediction cache, repeated evaluation runs should avoid re-inspecting unchanged images and can reduce live calls toward zero. This projection is for a fresh, cache-cold full run.
