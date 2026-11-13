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
| 2026-11-13 | 4-bit KV cache: per-token HQQ groups, batched MGSM generation (batch 8) | Numeric change, new run IDs (`KV_QUANT_RECIPE = 2`) | About 68 T4-hours, *projected*; T4 check of batch 8 versus batch 1 pending | DECISIONS 2026-11-13 |

## Backlog

| Lever | Candidate | Why it might pay | Status |
| --- | --- | --- | --- |
| Cheaper method | Per-language sensitivity proxy (Fisher or Hessian trace from calibration gradients) validated against the measured 1.5B probe | The 500-sentence probe took about 5.2 h on Kaggle; 3B would cost more. A validated proxy could also be a contribution | Not started; needs `researching-design-choices` and a K4 search |
| Cheaper method | Warm-start the 3B profile from the 1.5B profile (probe only the blocks that rank differently) | Fewer probe passes at 3B | Not started |
| Idle hardware | Fix the Modal driver hang after `run_jobs.map` | Each grid currently needs a manual `push_results.remote` | Open (RUN_LOG open items) |
| Cheaper measurement | Plan the 693 remaining audit runs and 308 speed runs by logged per-cell rates; route by cost (Modal versus Kaggle) | Largest remaining compute | Not started |

## Passes

### 2026-11-13: seeded
Log created from the savings already recorded in DECISIONS and RUN_LOG. Next pass: before the
ParityQuant 1.5B evaluation launch.
