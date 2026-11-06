# Autonomous implementation status

Updated 2026-11-06: K1 passed on Kaggle. The next step is the user's decision on the compute
re-plan (80.1 projected GPU-hours against 60). Notebooks 02-07 and analysis exist, and all
research code runs on CPU fixtures. Downstream notebooks check K1 from the results dataset.

Earlier on 2026-11-03: H2 now has item-clustered GEE p-values, fractional budgets have
a sensitivity-free `matched_uniform` comparator, and profile resume pins code, model
and library identity. These are CPU specifications, not measurements.
[repository/research review](REVIEW_2026-10-31.md) records the 31 Oct inventory.
GPU acceptance gates after K1 remain unverified.

Started 2026-10-22 on `codex/complete-research-pipeline`.

Run `pytest -q` for the current count; download-dependent tests skip without `WHOPAYS_SLOW=1`.
These checks are software verification, not research results.

## Execution order

1. Quantization, calibration, loader and sensitivity profiling; CPU verification.
2. KV policies and exact speculative decoding; task integration.
3. Statistical estimators and allocator; planted-effect and budget tests.
4. Experiment grids, resumable orchestration and generated runners.
5. Reporting, dashboard, preregistration and an evidence-pending manuscript draft.
6. Whole-repository verification and Git commits.

## External execution status

K1 passed on 2026-11-06 (Kaggle T4; DECISIONS 2026-11-06). Before that, the 31 Oct T4 run
finished 33 FP16 baselines, then lm-eval ran out of memory. No CUDA runtime is available locally. On 2026-10-22,
computer-use access to Chrome returned “Computer Use permissions are not granted”.
No Kaggle CLI or Kaggle API environment credentials were available. HF_TOKEN was
not present in the process environment. No GPU experiment was submitted.

GPU gates, measured throughput, replication, sensitivity diversity, ParityQuant
benefit and release regeneration timing remain evidence requirements. Implementing
a runner does not complete its GPU gate. No research result will be fabricated.

## Decisions

- Proceed with independently testable implementation while K1 is pending. Do not
  launch downstream research runs or mark GPU acceptance checks complete.
- Preserve the untracked, user-provided AGENTS.md and existing main branch.
- Use free GPU resources only; do not purchase compute or accept new account terms.
