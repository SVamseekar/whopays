# Efficiency log

A running record of the `efficiency-research-loop` passes: where the compute goes, what was made
cheaper, what was rejected and why, and what is queued. Numbers come from logged runs
(`docs/RUN_LOG.md`, `docs/DECISIONS.md`); savings are marked *projected* until a run measures them.

## Savings so far

| Date | Change | Validity | Saving | Source |
| --- | --- | --- | --- | --- |
| 2026-11-12 | Belebele prefill reuse (900 questions share 488 passages) | Exact; verified on a Modal T4 at `b820e74` | 30.6–33.0% of tokens removed (en, te, zh, bn) | DECISIONS 2026-11-12 |
| 2026-11-13 | Modal fan-out for the replication grid (T4 ×≤10) | Same code path as Kaggle | Remaining 141 runs finished for $1.61 in about 1.9 summed T4-hours | RUN_LOG entry 14 |
| 2026-11-13 | One bulk HF commit per Modal grid; background interval pushes | Upload only | Removes HTTP 429 waits | DECISIONS 2026-11-13 |
| 2026-11-14 | Batched MGSM eviction (batch 8, per-row sink and positions) | Numeric (FP16 batch shape only; fp32 rows equal); T4 batch gate PASS, 0 correctness flips (RUN_LOG 17, 18) | About 40 of 54 T4-hours, *projected*; the T4 gate measured eviction 4.4–6.4× faster per problem at batch 8 (1.5B, en and te). Also removes the batch-shape confound in the KV contrast | DECISIONS 2026-11-14 |
| 2026-11-14 | Kaggle grids: workers never push; the parent commits finished runs in bulk every 15 min; skipped runs re-sent in one commit | Upload only | About 10.7 idle GPU-hours on the 1.5B ParityQuant grid (11 hour-long HTTP 429 waits), *projected* to drop to near zero | RUN_LOG 16 |
| 2026-11-13 | 4-bit KV cache: per-token HQQ groups, batched MGSM generation (batch 8) | Numeric change, new run IDs (`KV_QUANT_RECIPE = 2`) | About 68 T4-hours, *projected*; T4 check of batch 8 versus batch 1 pending | DECISIONS 2026-11-13 |

## Backlog

| Lever | Candidate | Why it might pay | Status |
| --- | --- | --- | --- |
| Cheaper method | Per-language sensitivity proxy (Fisher or Hessian trace from calibration gradients) validated against the measured 1.5B probe | The 500-sentence probe took about 5.2 h on Kaggle; 3B would cost more. A validated proxy could also be a contribution | Not started; needs `researching-design-choices` and a K4 search |
| Cheaper method | Warm-start the 3B profile from the 1.5B profile (probe only the blocks that rank differently) | Fewer probe passes at 3B | Not started |
| Idle hardware | Fix the Modal driver hang after `run_jobs.map` | Each grid currently needs a manual `push_results.remote` | Open (RUN_LOG open items) |
| Cheaper measurement | Plan the 693 remaining audit runs and 308 speed runs by logged per-cell rates; route by cost (Modal versus Kaggle) | Largest remaining compute | Not started |

## Passes

### 2026-11-14: after stopping ParityQuant, before the audit
- Remaining work: 693 audit runs (about 61 T4-hours as planned 2026-11-06, plus the 2026-11-13 MGSM re-plan), 308 speed runs (about 25 T4-hours, batch 1 by design), 44 exploratory `evict_share` runs, two gates. Modal 8.40 USD; Kaggle quota resets weekly.
- Largest single cost: MGSM eviction cells at batch 1, 19.7 h (1.5B) + 34.3 h (3B), projected from logged rates.
- Candidates. Repeated work: KV policies already share each quantized model (`group_by_model`); speed-grid runs never overlap audit IDs (batch 1 against batch 8), so no reuse without changing the speed protocol, rejected. Idle hardware: bulk HF commits (done today); Modal for the short gate. Cheaper measurement: **batched eviction (done)**; stopping generation once the final answer is extracted, rejected (changes the preregistered 256-token stopping rule). Cheaper method: none applies to the audit (no method). Statistical: fewer calibration or seed replicates, rejected (third amendment fixes them; would cut a preregistered cell).
- Acted: batched eviction, tested on CPU; the gate run measures the real rate before re-slicing the audit by logged rates.
- Next in queue: re-slice `configs/audit_slice_*` by logged per-run rates once the gate gives the batched MGSM rates; 3B GPTQ time is still unmeasured.

### 2026-11-14: after the 1.5B ParityQuant run
- Measured: 10.8 h wall for 19.69 summed job-hours on T4×2, against a projection of about 6 h wall. The gap is HTTP 429 waits on per-run commits: 11 waits of 3,065–3,596 s, about 10.7 GPU-hours. GPU 0's own work took about 4.5 of its 9.3 h.
- Fixed: `hub_sync.BulkPush` and `run_parallel(push_interval=900)`; `whopays.run` re-sends skipped runs in one commit. Commits per grid drop from about one per run to about one per 15 minutes.
- Per-run rates confirmed (no 429): Belebele 62–75 s for Latin, Han, Kana and Thai; 106–123 s for Bengali; 137–158 s for Telugu; about 730–810 s for English including GPTQ. Held-out FLORES 12–55 s.
- Not run: 3B ParityQuant. It waits on the Belebele investigation (DECISIONS 2026-11-14).

### 2026-11-14: before the 1.5B ParityQuant launch
- Cost picture (projected from logged rates, DECISIONS 2026-11-12): per quantized 1.5B model about
  20 T4-min GPTQ, 13 Belebele (18.7 less the logged 31% prefill saving), 4 held-out FLORES
  (7.2 for all of devtest, about half the items). 18 models: about 11 T4-hours, about 6 h wall
  on Kaggle T4 x2.
- Rejected, Modal fan-out: at about 0.85 USD per T4-hour (logged: 1.61 USD for 1.9 T4-hours)
  the grid costs about 9 USD, above the 8.40 left.
- Rejected, reuse of the replication grid's uniform W3/W4 models for the uniform comparators:
  the code path is the same (`resolve_bitmap` expands an int and an all-equal block map to the
  same per-layer widths), but held-out FLORES would come from runs batched over all of devtest,
  so batch composition differs (near-tie FP16 numerics) in the baseline arm of the H4
  comparison. Saving about 2.5 of 11 T4-hours on free Kaggle time does not justify that caveat.
- Launched on Kaggle at `2a7caef`.

### 2026-11-14: H4 unit change
Held-out FLORES evaluation adds one NLL job per bit map (about 9 T4-minutes, projected). The
per-sentence conversion reuses the existing profile: no new probe run. Two candidates added to
the backlog (uniform-map reuse, Modal for the ParityQuant grid).

### 2026-11-13: seeded
Log created from the savings already recorded in DECISIONS and RUN_LOG. Next pass: before the
ParityQuant 1.5B evaluation launch.
