---
name: researching-design-choices
description: Use in WhoPays when a method, metric, protocol or engineering choice is open and nothing has failed — for example which degradation metric H4 should use, which ParityQuant solver or baseline to adopt, how to bootstrap a statistic, which calibration set or bit budgets to pick, or whether a new preprint changes a claim. Not for failed gates or wrong-looking results (use investigate-then-research).
---

# Researching design choices

## Overview

A choice is settled by evidence, not by the first idea. Write the options down, read this
repo's constraints, search wide in the literature, then pick one option and record it so a
reviewer can check it later. Ported from the maxx-pay `researching-before-deciding` skill and
adapted to a preregistered research project.

## When to use

- Two or more defensible methods, and the PRD, `PREREGISTRATION.md` and `docs/DECISIONS.md`
  do not already pick one.
- A choice a reviewer could challenge: metric, estimator, baseline, solver, dataset, budget.
- A new paper may overlap a novelty claim (K4).

Not for: facts readable in the code; choices already in `docs/DECISIONS.md` (say so if you
reopen one); failed runs or gates (`investigate-then-research`); naming and style.

## The recipe

1. **State the choice** in one sentence. List the options before any search.
2. **Read local constraints.** `docs/PRD.md`, `PREREGISTRATION.md` (amendments included),
   `docs/DECISIONS.md`, `docs/RESEARCH_CONTEXT.md`, the code path. Also: T4 FP16 only, the
   GPU-hour and Modal budget, which results are already seen. List what rules options out.
3. **Search wide.** At least 8 searches in one turn, each from a different angle: the method's
   name, arXiv and ACL Anthology, the strongest competing method, multilingual or low-resource
   variants, known failure modes and critiques, library source or docs, the opposite position,
   the most recent preprints. Use `extended` mode for niche or recent questions.
4. **Read deep.** Open 6–10 sources in full. Cite at least 12 distinct sources from at least 3
   types: peer-reviewed papers, preprints, library source or official docs, benchmark or
   dataset cards, practitioner writing. Include at least one source that argues against the
   chosen option, if one exists.
5. **Decide against the repo.** For each option: what the sources say, what local constraints
   allow, the cost in T4-hours or Modal dollars. Pick one. Name what would reverse it.
6. **Check protocol impact.** If the choice changes a preregistered metric, test, threshold or
   gate, it needs a numbered amendment in `PREREGISTRATION.md` written **before** the affected
   result is seen. If a related result was already seen, disclose that in the amendment. Never
   pick an option because it makes a seen result pass.
7. **Record.**
   - Full note: `docs/decisions/YYYY-MM-DD-<slug>.md` with: choice, options (as written in
     step 1), constraints, sources (title, URL, the exact claim or number used), decision,
     rejected options and why, what would reverse it.
   - One dated paragraph at the top of `docs/DECISIONS.md` linking the note.
   - Cite the source in the docstring of any constant taken from a paper.
   - Commit by exact path, then push.
8. **Tell the user** the decision in two lines, plus the note path. Ask first only when the
   choice cuts scope, costs GPU money beyond the plan, or changes a preregistered hypothesis.

## Quick reference

| Gate | Minimum |
|---|---|
| Options written before searching | yes |
| Parallel searches | 8, distinct angles |
| Sources read in full | 6–10 |
| Distinct sources cited | 12, from 3+ types |
| Dissenting source | 1, if one exists |
| Prereg amendment if protocol changes | before the result is seen |

## Common mistakes

- One search, three abstracts, called research.
- Choosing first and searching for support afterwards.
- Citing a paper from its abstract when the number used is in the body.
- Letting a paper override a local constraint (T4, budget, preregistration).
- Changing a metric after seeing the result without a disclosed amendment.
- Treating an adjacent method (FairQuant, Colla-Q) as absent because its domain differs.
