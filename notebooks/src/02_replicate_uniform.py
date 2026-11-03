# %% [markdown]
# # 02 · Uniform replication (FR-3)
#
# Uniform W8/W4/W3 × calibration × 3 seeds on 1.5B, on Belebele and FLORES devtest. Then a real
# GPTQ W4 checkpoint, for the FR-3 check: W4 fake-quant perplexity within 2% of the checkpoint.
#
# **Needs a K1 pass.** Run on Kaggle (Save & Run All). If a session ends, run it again: finished
# runs skip. **Send back:** the printed tables, or the downloaded `.ipynb`.

# %%
import pandas as pd

import whopays
from whopays.config import load_jobs
from whopays.env import results_root
from whopays.gates import require_k1
from whopays.grids import K1_MODEL, replication_grid

require_k1(results_root())
JOBS = load_jobs(replication_grid())
results = whopays.run(JOBS, stop_on_error=True)
status = pd.DataFrame([r.__dict__ for r in results])
print(status.status.value_counts())

# %% [markdown]
# ## FR-3: W4 fake quantization against the GPTQ checkpoint

# %%
from whopays import replicate
from whopays.io.results import load_runs

reference = load_runs(results_root(), replicate.run_reference(K1_MODEL))
w4 = [c for c in JOBS if c.weight == "w4_gptq" and c.calib == "multi" and c.seed == 0
      and c.task == "flores_nll"]
table, verdict = replicate.fr3_verdict(load_runs(results_root(), w4), reference)
print(table.round(3).to_string(index=False))
print(verdict)

# %% [markdown]
# ## Per-language gap direction: W4 against FP16 on Belebele

# %%
def _belebele(**match):
    keep = [c for c in JOBS if c.task == "belebele"
            and all(getattr(c, k) == v for k, v in match.items())]
    return load_runs(results_root(), keep)


for calib in ("en", "multi"):
    print("calibration:", calib)
    print(replicate.gap_direction(_belebele(weight="fp16"),
                                  _belebele(weight="w4_gptq", calib=calib, seed=0))
          .round(2).to_string(index=False))
