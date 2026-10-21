# %% [markdown]
# # 01 · FP16 baselines and the K1 gate
#
# 1. **Baselines:** FP16 Belebele, MGSM and FLORES NLL in 11 languages for each model in
#    `MODELS`. Runs are resumable and synced to the HF results dataset, so a dead session
#    loses at most one flush (50 items).
# 2. **K1 gate:** lm-evaluation-harness on the K1 model (1.5B), Belebele and MGSM, all 11
#    languages, compared item by item with our runs. The gate passes when, for each task, the
#    mean |ours − lm-eval| over languages is ≤ 1 point.
# 3. **Throughput:** measured GPU time, projected to the full benchmark and compared with the
#    PRD's assumptions. More than 2× slower means re-planning the compute budget.
#
# **Colab (about 2–2.5 h):** keep `MODELS` as the 1.5B only. **Kaggle:** add the 0.5B and 3B
# (Save & Run All). Re-running skips finished work, including lm-eval's cached requests.
#
# **Send back:** download the `.ipynb` after the run (File → Download → .ipynb).

# %%
from whopays.grids import K1_MODEL, M0_MODELS

MODELS = [K1_MODEL]  # Kaggle: list(M0_MODELS)
RUN_LMEVAL = True
# Smoke tests only: lm-eval takes Belebele items in file order and we sort them by ID, so
# with a limit the two sides score different Belebele items and the comparison is empty.
LIMIT = None
print("models:", MODELS, "| all M0 models:", list(M0_MODELS))

# %% [markdown]
# ## 1 · FP16 baselines

# %%
import pandas as pd

import whopays
from whopays.config import load_jobs
from whopays.env import results_root
from whopays.grids import m0_fp16

JOBS = load_jobs(m0_fp16(MODELS, limit=LIMIT))
results = whopays.run(JOBS)
status = pd.DataFrame([r.__dict__ for r in results])
print(status.groupby(["task", "status"]).size().unstack(fill_value=0))
print(status[status.status == "failed"][["run_id", "task", "lang", "error"]].to_string())

# %% [markdown]
# ## 2 · K1 gate: lm-evaluation-harness on the same items

# %%
from whopays import crosscheck
from whopays.io import hub_sync
from whopays.io.results import load_runs
from whopays.languages import CODES

LMEVAL_DIR = results_root() / f"lmeval-{K1_MODEL.split('/')[-1]}"
if RUN_LMEVAL:
    if not LMEVAL_DIR.exists():  # resume a previous session's request cache from the hub
        hub_sync.pull_runs(results_root(), [LMEVAL_DIR.name])
    crosscheck.run_lmeval(K1_MODEL, ["belebele", "mgsm"], CODES, LMEVAL_DIR, limit=LIMIT)
    hub_sync.push_run(LMEVAL_DIR)

ours = load_runs(results_root(), [c for c in JOBS if c.model == K1_MODEL])
table = crosscheck.compare(ours, crosscheck.read_samples(LMEVAL_DIR))
pd.set_option("display.width", 160)
print(table.to_string(index=False))
print()
print(crosscheck.k1_verdict(table).to_string(index=False))

# %% [markdown]
# ## 3 · Throughput vs the PRD's assumptions

# %%
from whopays import throughput

rows = load_runs(results_root(), JOBS)
print(throughput.summary(rows).to_string(index=False))

# MGSM health per language: no-number outputs and outputs cut at the 256-token cap.
mg = rows[rows.task == "mgsm"]
if not mg.empty:
    print()
    print(mg.groupby(["model", "lang"])[["correct", "correct_ascii", "extraction_failed",
                                         "hit_max_tokens"]].mean().round(3).to_string())
