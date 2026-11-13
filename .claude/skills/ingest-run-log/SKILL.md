---
name: ingest-run-log
description: Use in WhoPays when a GPU run finishes or its output arrives — a Kaggle or Modal run launched by Claude completes, the user says "it's done" or "notebook finished", or sends `.log`/`download*.txt` files, a downloaded `.ipynb`, or pasted log text. Condenses the log with the bundled parser, checks it against the PRD "Done when" and kill gates, confirms results reached the HF dataset, updates the plan and decision docs with logged numbers only, commits, and reports the verdict and next step.
---

# Ingest a run log

## 0. Get the output yourself

- Kaggle: `python scripts/kaggle_run.py fetch <notebook>` puts the log and executed notebook in
  `runs/kaggle/<notebook>/<timestamp>/`.
- Modal: the driver output, plus results on HF (section 4).
- Only ask the user for files when neither works.

## 1. Read the files with the parser, not by scrolling

```bash
source .venv/bin/activate
python .claude/skills/ingest-run-log/scripts/parse_run.py ~/Downloads/<file>.log ~/Downloads/"download (1).txt"
```

The parser skips files identical to one already read (Kaggle's `.log` and `download*.txt` are usually the same file). It prints:
- the environment block, including `code_version`;
- secrets that were missing;
- job counts by status and task, with timings;
- failed jobs;
- every table row and verdict dict;
- errors;
- whether the notebook reached its end.

For pasted text, save it to the scratchpad first, then run the parser on that file. Read raw lines around anything the parser flags (`sed -n`, `grep -n -A20 Traceback`) before drawing conclusions.

## 2. Check the run is the one intended

- `code_version` matches the commit given in the handoff, or a later one. An older SHA means a stale imported notebook.
- `platform` and GPU are as expected (T4). If `HF_TOKEN` is reported missing, results were not synced.
- Job counts match the notebook's job list. Finished-before runs show as `skipped`.
- "reached end: False" means the session died or a cell raised. Find the last progress line and the error.

## 3. Judge against the gate

- Look up the "Done when" for this notebook's FR in `docs/PRD.md`, any kill gate (K1–K4) and the relevant `PREREGISTRATION.md` amendment.
- State plainly the observed value, the limit, and PASS, FAIL or INCOMPLETE. Use the verdict the code printed; do not recompute a looser one.
- Show **all 11 languages** in every per-language table, never only the extremes. Build it with `python -m whopays.cli langtable <long.csv> --value <col> --by <cols> [--where col=value] [--interval low high]`; it uses the canonical order and prints `missing` for absent languages.
- Label every result **confirmatory** (a preregistered test, as amended before the data was seen) or **exploratory** (anything else, including tests added by an amendment after the data was seen).
- Note partial patterns (which languages fail and which pass). Note anything that would be read differently under a stricter reading of the rule.
- On FAIL, or an unexplained result, switch to the `investigate-then-research` skill before proposing fixes. On a kill gate failure, say so and propose the PRD's pivot.

## 4. Confirm the results reached the HF dataset

`hf_fs find hf://datasets/Heavy-Dervish1205/whopays-results --name DONE` (Hugging Face connector), compared with the run IDs. If the local `.env` token works (test it without printing it), load rows with `whopays.io.results.load_runs` for anything the log did not print.

## 5. Record, with logged numbers only

- `docs/RUN_LOG.md`: always append an entry, even for a failed, stopped or superseded run. Give the date and UTC times, platform, commit, notebook, rows written (run IDs or counts), the outcome and the log path. Refresh the snapshot with `python scripts/run_inventory.py`.
- `docs/BUILD_PLAN.md`: tick the task on PASS. Otherwise add a dated note under it: run date, platform, `code_version`, the key numbers, the verdict and the next GPU check.
- `docs/DECISIONS.md`: a dated entry when the result changes a plan (re-plan, cut, amendment). Include the environment (torch, transformers, GPU, commit).
- `docs/IMPLEMENTATION_STATUS.md`: the first paragraph if milestone status changed.
- Every number must come from this log or the results dataset. Never round in a way that flips a verdict. Never put numbers from the M1 Mac into these tables.
- Commit by exact path (never `git add docs` or `git add .`), then push.

## 6. Report

Keep it short:
- verdict table: check, observed value, limit, result;
- the 11-language table for the main metric, labelled confirmatory or exploratory;
- run time and cost (Modal dollars, Kaggle hours);
- anything odd;
- what was updated;
- one `efficiency-research-loop` pass result, in a line;
- the next step: the next run via the `launch-gpu-run` skill, or a decision the user must make.
