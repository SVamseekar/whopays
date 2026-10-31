# Autonomous implementation status

Updated 2026-10-31: [repository/research review](REVIEW_2026-10-31.md) records the current
software inventory, repaired KV/speculative paths, protocol amendment, prior-art check and
K1 handoff. The original baseline count below describes the earlier session, not current
verification. K1 and all subsequent GPU acceptance gates remain unverified.

Started 2026-10-22 on `codex/complete-research-pipeline`.

The existing M0 baseline suite passes locally (104 passed, 5 download-dependent tests skipped).
These checks are software verification, not research results.

## Execution order

1. Quantization, calibration, loader and sensitivity profiling; CPU verification.
2. KV policies and exact speculative decoding; task integration.
3. Statistical estimators and allocator; planted-effect and budget tests.
4. Experiment grids, resumable orchestration and generated runners.
5. Reporting, dashboard, preregistration and an evidence-pending manuscript draft.
6. Whole-repository verification and Git commits.

## External execution status

K1 remains pending. No CUDA runtime is available locally. On 2026-10-22,
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
