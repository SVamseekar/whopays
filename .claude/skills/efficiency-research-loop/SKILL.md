---
name: efficiency-research-loop
description: Use in WhoPays at every checkpoint — after ingesting a run, before launching any GPU job or grid, when planning the next milestone or task, when a run takes longer or costs more than planned, while waiting on a running GPU job, and when the user asks "what next". Keeps searching for ways to make the research cheaper, faster or statistically stronger without losing novelty.
---

# Efficiency research loop

## Overview

Efficiency research never finishes. Every checkpoint runs one bounded pass: measure where the
cost is, look for a cheaper or stronger way, check it keeps the science valid and the
contribution novel, act on the best one, log the rest. The research idea itself (the audit
grid, H1–H4, K1–K4, ParityQuant's worst-language objective) is never cut to save compute.

## A pass

1. **Measure.** From `docs/RUN_LOG.md` and `docs/EFFICIENCY_LOG.md`: the remaining work (jobs
   not started, T4-hours estimated from logged rates), money left (Modal), Kaggle hours left this
   week, and the slowest step per cell. Also the biggest scientific weakness: noise-limited
   gates, wide intervals, underpowered tests. Logged numbers only.
2. **Generate candidates** across all five levers. Write at least one per lever before judging:
   - *Repeated work*: reuse quantized weights, FP16 baselines, shared prefill, cached
     calibration statistics, cross-run caches on the Volume.
   - *Idle hardware*: second Kaggle T4, Modal fan-out, overlapping CPU work (solver, stats) with
     GPU runs, background uploads.
   - *Cheaper measurement*: batching, NLL or log-likelihood scoring in place of generation,
     early stopping of hopeless cells, adaptive or sequential sample sizes, smaller pilots on
     0.5B before 1.5B or 3B.
   - *Cheaper method*: proxies for per-language layer sensitivity (Fisher, Hessian trace, weight
     or activation statistics) checked against the measured probe; faster solvers; reusing the
     1.5B profile to warm-start 3B.
   - *Statistical efficiency*: paired designs, variance reduction, stratified items, fewer
     seeds where seed variance is logged as small. Same power for less compute.
3. **Research the top 2–3.** Three or more searches each, in one turn: arXiv or ACL Anthology,
   the library's source at the pinned version, known failure modes. Choices that change a
   protocol or method go through `researching-design-choices`; fixes for something broken go
   through `investigate-then-research`.
4. **Validity gate.** Pass only if one holds:
   - outputs are bit-identical (show the test), or
   - the numeric change is bounded and checked on a logged sample, results change run IDs
     (`RunConfig.canonical()`), and any preregistered change has a numbered amendment written
     before the affected result is seen.
   Never choose a saving because it moves a seen result towards a pass.
5. **Novelty gate.** Read the contribution list in `docs/RESEARCH_CONTEXT.md`.
   - Infrastructure changes pass freely.
   - Method changes must keep all of: worst-language (minimax) objective, languages as groups,
     dense multilingual LLMs, calibration-only data, the price-of-parity curve. A change that
     turns ParityQuant into an existing method (FairQuant, Colla-Q, SliM-LLM, CoopQ and others in
     that file) is rejected or re-scoped.
   - A new cheap technique that is itself unpublished (for example a validated sensitivity
     proxy per language) is logged as a candidate contribution, and checked with a quick K4
     search.
6. **Act.** Implement the best candidate that passes both gates when it costs under half a day
   and the saving exceeds its cost. Tests first. Otherwise put it on the backlog with the reason.
7. **Record** in `docs/EFFICIENCY_LOG.md`: date, checkpoint, cost picture, candidates per lever,
   what was done, what was rejected and why, measured saving once a run confirms it, backlog.
   Commit by exact path and push.
8. **Report** in one or two lines: the saving found, done or backlogged, the next idea in the
   queue.

## Bounds

- One pass: about 30 minutes of agent time, longer only for a candidate worth more than 5
  T4-hours. A pass never blocks a GPU launch that is ready.
- The loop has no end state. "Nothing left to save" is never a valid conclusion; when the top
  candidates are all done, look at the next milestone's costs.
- Never trade away a preregistered cell, hypothesis, gate, language or model to save compute
  without asking the user.

## Common mistakes

- Optimising what is cheap to optimise instead of the largest remaining cost.
- Claiming a saving before a logged run measures it.
- Changing numerics silently: no run-ID change, no amendment.
- Adopting a published method wholesale inside ParityQuant without the novelty gate.
- Generating candidates for one lever only.
