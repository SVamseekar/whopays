---
name: investigate-then-research
description: Use in WhoPays when a GPU run, gate (K1–K4, FR-3, FR-5), test or result is unsatisfying, fails, or looks wrong, before proposing any fix. Inspects the codebase first, then researches how the field handles it, citing every source, then implements and records the decision. Triggers on "FR-3 failed", "results look off", "why is this worse", "gate failed", pasted Kaggle/Colab errors or tables that miss a "Done when" check.
---

# Investigate, then research, then fix

Never jump from a symptom to a fix. Work through these steps in order and show the evidence for each.

## 1. Pin the symptom

- Quote the exact evidence: the log line, the table rows or the error string. Give numbers with units.
- Name the check it fails: the PRD "Done when" row, the kill gate (K1–K4) or the test. Give the limit and the observed value.
- Note what passed. Partial passes narrow the cause; for example, "Latin-script languages within 2%, non-Latin not".

## 2. Inspect our codebase first

- Trace the code path that produced the number, from the notebook cell through `src/whopays/` to the computation.
- Read the defaults that shape it: sample counts, token budgets, batch sizes, seeds, dtype and settings. Compute their real magnitude (for example, "128 sentences ≈ 3,500 tokens"), not the nominal one.
- Check `docs/PRD.md`, `PREREGISTRATION.md` and `docs/DECISIONS.md` for what was promised and why.
- If the result is compared with a library, read that library's source at the pinned version, not only its docs. Use `pip download <pkg>==<ver> --no-deps --no-binary :all:` into the scratchpad. List each setting where its defaults differ from ours.
- Rank hypotheses. For each, list the evidence for and against, and what pattern it predicts across languages or settings. Prefer the hypothesis that explains the whole pattern.
- Reproduce cheaply where possible: CPU tests or a 0.5B smoke run on MPS. These are pipeline checks only; never report their numbers as results.

## 3. Research how the field does it

- Follow `.claude/skills/researching-design-choices/references/literature-protocol.md`:
  scholarly API first, snowball, both sides searched, deciding sources read in full,
  evidence graded, citations checked, corpus saved to `docs/literature/`.
- Library behaviour: read the source at the pinned version, plus its issue tracker for the
  symptom.
- Answer three questions: what is standard practice, what are the known failure modes, and what
  does each option cost on a T4 (GPU-hours)?

## 4. Decide

- Give one recommendation, the main alternatives and why each was rejected, and the cost in GPU-hours.
- Protocol impact: if the change follows a seen result, it is a disclosed amendment. Never change a threshold or gate to make a result pass. Never relabel a failure as a pass.
- If the decision belongs to the user (scope, budget, a scientific cut), ask one question with the trade-offs. Otherwise act.

## 5. Implement

- Tests first (`superpowers:test-driven-development`). For refactors, keep a characterization test that proves identical outputs.
- Changes that alter results must change run IDs (`RunConfig.canonical()`). Unaffected runs must keep their IDs; add a regression test for logged IDs.
- Regenerate `configs/` (`python -m whopays.cli grid`, slices) and notebooks (`python scripts/build_notebooks.py`).
- Run `pytest -q` and `ruff check src tests`. Report failures by name.

## 6. Record

- `docs/DECISIONS.md`: a dated entry covering the problem (with numbers), what the field does (inline links), the decision, the rejected options, limitations and what would reverse it.
- `PREREGISTRATION.md`: a new numbered amendment when the protocol changes. Disclose that it came after the result and which runs it supersedes.
- `docs/BUILD_PLAN.md`: a note under the task, plus the next GPU check.
- Code docstrings: cite the source behind any constant taken from the literature.
- Commit by exact path only (never `git add docs` or `git add .`), then push.

## 7. Report to the user

Keep it short. Cover the cause, the evidence, what changed, the next GPU step with a time estimate, and a final **Sources** list of markdown links.
