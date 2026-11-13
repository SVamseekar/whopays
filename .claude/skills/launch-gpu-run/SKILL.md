---
name: launch-gpu-run
description: Use in WhoPays whenever work needs a GPU — "run notebook 03", "next GPU run", "launch the grid", a fix that needs GPU verification, or a job list in configs/ to execute. Covers choosing Kaggle or Modal, launching from the M1, monitoring, fetching outputs and the known failure modes of both. Manual Kaggle or Colab steps only when a launch needs the user.
---

# Launch a GPU run

Claude launches, monitors and fetches GPU runs from the M1. The user only acts when a token or
account step is needed. Before launching, run one pass of `efficiency-research-loop` if none
was done since the last run.

## 1. Pick the platform

| Job | Platform | Why |
| --- | --- | --- |
| Job list (`configs/*.yaml`) that fans out by model group | Modal (`modal_run.py jobs`) | Up to 10 T4s in parallel; logged cost $1.61 for 141 replication runs (RUN_LOG 14) |
| One long sequential notebook (probe, single model) | Kaggle (`kaggle_run.py push`) | Free, 12 h per session, about 30 GPU-hours a week |
| Short check under about 30 min | Modal (`modal_run.py notebook` or `fr3`) | No queue, no weekly quota |
| Interactive debugging | Colab (manual, section 6) | User drives |

Check money and quota first: `modal billing report` and the Kaggle hours already used this
week (RUN_LOG). Say the choice and the estimated cost in one line. Over 11 h on Kaggle: split.

## 2. Before launching

- Work is committed and pushed; note the SHA. Notebooks regenerated with
  `python scripts/build_notebooks.py` and committed.
- **Estimate from logged evidence only**: per-run rates in `docs/DECISIONS.md` and RUN_LOG, plus
  finished runs on HF (`hf_fs find hf://datasets/Heavy-Dervish1205/whopays-results --name DONE`),
  which skip in about 5 s. Label unmeasured steps. Give a range and the riskiest step.
- `modal_run.py jobs <spec> --dry-run` to see the plan.
- Partial runs from an older commit are refused by the resume guard. Inspect the folder (no
  `DONE`, not on HF) before deleting it from the Volume; never delete a finished run.

## 3. Launch

```bash
source .venv/bin/activate
# Kaggle: one kernel, souramarti/whopays-runner, T4 x2, background run
python scripts/kaggle_run.py push <notebook> [--ref <full sha>]
python scripts/kaggle_run.py wait            # or status
python scripts/kaggle_run.py fetch <notebook>   # log + executed notebook into runs/kaggle/

# Modal: --ref goes BEFORE the subcommand
python scripts/modal_run.py --ref <sha> jobs configs/<spec>.yaml --chunk-minutes 60
python scripts/modal_run.py --ref <sha> notebook <notebook>
```

- Run long launches in the background and monitor finished run IDs (count unique IDs, not log
  lines; lines repeat).
- `kaggle_run.py push` refuses on a dirty tree. Other agents leave uncommitted files: pass
  `--ref <full sha>` of the pushed commit instead of cleaning their work.
- HF access from the M1: the `.env` `HF_TOKEN` is invalid. Prefix with `env -u HF_TOKEN` so
  the cached `hf` login is used.

## 4. Known failures

| Symptom | Cause | Fix |
| --- | --- | --- |
| Modal driver stays in `run_jobs.map` after Modal shows 0 tasks; nothing pushed | Unknown (open item) | Kill the driver; run `modal_app.push_results.remote(ids)` with the finished IDs (one commit) |
| HTTP 429 from HF | Too many commits | Default is one bulk commit per grid; do not use `--push-each` for grids |
| `refusing to resume` | Partial folder from another commit | Section 2 rule, then rerun |
| `secret HF_TOKEN not available (ConnectionError)` on Kaggle | First secret source failed | Harmless if the private `whopays-secrets` dataset fallback worked; check results reached HF |
| `OutOfMemoryError` | GPU memory | Read the last 20 lines; investigate, do not retry blindly |
| `Invalid user token` | HF token revoked | User makes a new Write token; update Modal secret `whopays-hf`, Kaggle dataset `whopays-secrets`, local `.env` |

## 5. After it finishes

Fetch outputs, then follow `ingest-run-log`. Record Modal cost from `modal billing report` in
the RUN_LOG entry and in `docs/EFFICIENCY_LOG.md` when it confirms a projected saving.

## 6. Manual fallback (only when the user must act)

Numbered, plain English steps (not caveman: order matters).

**Kaggle:** 1. Create → New Notebook. 2. File → Import Notebook,
`https://github.com/SVamseekar/whopays/blob/main/notebooks/<NAME>.ipynb`. 3. Session options:
GPU T4 x2, Internet On. 4. Add-ons → Secrets → tick `HF_TOKEN`. 5. Run the first cell only;
expect `platform: kaggle | GPU: Tesla T4, Tesla T4`, no missing-token line, `code_version` at
the pushed SHA or later. 6. Save Version → Save & Run All. 7. Send back the log path.

**Colab:** 1. Open `https://colab.research.google.com/github/SVamseekar/whopays/blob/main/notebooks/<NAME>.ipynb`.
2. Runtime → T4 GPU. 3. Secrets: `HF_TOKEN` with notebook access. 4. Runtime → Run all; keep
the tab open. 5. File → Download .ipynb; send back the path. Sessions die after about 3 h.
