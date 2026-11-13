# %% [markdown]
# # 02 · Uniform replication (FR-3)
#
# 1. **FR-3 first (about 1 h):** our W4 fake quantization against GPTQModel quantizing the same
#    model on the same calibration windows (128 × 2,048 Wikipedia tokens, multilingual, seed 0).
#    Pass: mean FLORES devtest perplexity gap ≤ 2%. Qwen's official GPTQ-Int4 checkpoint is
#    printed as a secondary comparison. **A failure stops the notebook** before the grid.
# 2. **Replication grid (about 8 h):** uniform W8/W4/W3 × calibration × 3 seeds on 1.5B, on
#    Belebele and FLORES devtest.
# 3. **Gap direction:** W4 against FP16 on Belebele, per language and seed.
#
# **Needs a K1 pass.** Run on Kaggle (Save & Run All). If a session ends, run it again: finished
# runs skip. **Send back:** the printed tables, or the downloaded `.ipynb`.

# %%
import pandas as pd

import whopays
from whopays import replicate
from whopays.config import load_jobs
from whopays.env import results_root
from whopays.grids import K1_MODEL, replication_grid
from whopays.io.results import load_runs
from whopays.parallel import run_parallel

table, verdict, official = replicate.fr3_check(K1_MODEL)
print(table.round(3).to_string(index=False))
print(verdict)
print("\nSecondary, descriptive: Qwen's official GPTQ-Int4 (undisclosed calibration, symmetric)")
print(official.round(3).to_string(index=False))
if verdict["FR3"] != "PASS":
    raise SystemExit(f"FR-3 {verdict['FR3']}: stopping before the replication grid")

# %% [markdown]
# ## Replication grid

# %%
JOBS = load_jobs(replication_grid())
run_parallel(JOBS)  # one worker per T4; outputs identical to a single-process run
results = whopays.run(JOBS, stop_on_error=True)  # checks every run is finished (all "skipped")
status = pd.DataFrame([r.__dict__ for r in results])
print(status.status.value_counts())

# %% [markdown]
# ## Per-language gap direction (fifth amendment, 2026-11-13)
#
# The published direction (lower-resource and non-Latin-script languages hurt more) is judged
# on two floor-free measures, seeds averaged per item, with paired 95% bootstrap intervals:
# FLORES relative NLL increase (primary) and Belebele above-chance retention of calibrated
# accuracy. Raw Belebele points are printed alongside, as preregistered.

# %%
def _rows(task, **match):
    keep = [c for c in JOBS if c.task == task
            and all(getattr(c, k) == v for k, v in match.items())]
    return load_runs(results_root(), keep)


for weight in ("w4_gptq", "w3_gptq"):
    for calib in ("en", "multi"):
        print(f"\n=== {weight}, calibration {calib} (3 seeds) ===")
        nll = replicate.nll_gap_direction(_rows("flores_nll", weight="fp16"),
                                          _rows("flores_nll", weight=weight, calib=calib))
        print("FLORES relative NLL increase (primary):")
        print(nll.round(2).to_string(index=False))
        kept = replicate.retention_gap_direction(_rows("belebele", weight="fp16"),
                                                 _rows("belebele", weight=weight, calib=calib))
        print("Belebele above-chance retention, calibrated accuracy:")
        print(kept.round(3).to_string(index=False))
        raw = pd.concat([replicate.gap_direction(_rows("belebele", weight="fp16"),
                                                 _rows("belebele", weight=weight, calib=calib,
                                                       seed=s)).assign(seed=s)
                         for s in (0, 1, 2)])
        print("Raw Belebele points (preregistered, floor-limited):")
        print(raw.groupby("lang")[["change_pts", "gap_vs_english_pts"]].mean().round(2)
              .to_string())
