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
from whopays.gates import require_k1
from whopays.grids import K1_MODEL, replication_grid
from whopays.io.results import load_runs
from whopays.parallel import run_parallel

require_k1(results_root())
OURS = replicate.fr3_jobs(K1_MODEL)
whopays.run(OURS, stop_on_error=True)
reference = load_runs(results_root(), replicate.run_reference(K1_MODEL))
table, verdict = replicate.fr3_verdict(load_runs(results_root(), OURS), reference)
print(table.round(3).to_string(index=False))
print(verdict)

official = load_runs(results_root(), replicate.run_reference(K1_MODEL, kind="official"))
print("\nSecondary, descriptive: Qwen's official GPTQ-Int4 (undisclosed calibration, symmetric)")
print(replicate.fr3_verdict(load_runs(results_root(), OURS), official)[0]
      .round(3).to_string(index=False))
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
# ## Per-language gap direction: W4 against FP16 on Belebele

# %%
def _belebele(**match):
    keep = [c for c in JOBS if c.task == "belebele"
            and all(getattr(c, k) == v for k, v in match.items())]
    return load_runs(results_root(), keep)


for calib in ("en", "multi"):
    for seed in (0, 1, 2):
        print(f"calibration: {calib}, seed {seed}")
        print(replicate.gap_direction(_belebele(weight="fp16"),
                                      _belebele(weight="w4_gptq", calib=calib, seed=seed))
              .round(2).to_string(index=False))
