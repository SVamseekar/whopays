# %% [markdown]
# # 07 · 7B confirmation (exploratory)
#
# FP16 and W4 on Belebele and the first 30 MGSM items. ParityQuant at 3.5 bits follows if the
# probe finishes. **Kaggle accelerator: GPU T4×2**, because 7B needs both GPUs in FP16.
#
# **Needs a K1 pass.** The preregistration labels these runs exploratory.
# **Send back:** the status counts, the K2 line and the last projection.

# %%
import pandas as pd

import whopays
from whopays.config import load_jobs
from whopays.env import results_root
from whopays.gates import require_k1
from whopays.grids import QWEN, scale_grid

require_k1(results_root())
JOBS = load_jobs(scale_grid())
status = pd.DataFrame([r.__dict__ for r in whopays.run(JOBS)])
print(status.groupby(["task", "status"]).size().unstack(fill_value=0))

# %%
from whopays.parityquant.probe import k2_verdict, run_probe, solve

MODEL = QWEN["7B"]
rows = run_probe(MODEL, limit=40)  # exploratory and costly: fewer sentences than 1.5B
print(k2_verdict(rows).to_string())

# %%
allocation = solve(rows, 3.5, "minimax")
parity = load_jobs([{"model": MODEL, "task": "belebele", "lang": "all", "weight": "bitmap",
                     "calib": "multi", "seed": 0, "bitmap": allocation.bit_map, "budget": 3.5}])
status = pd.DataFrame([r.__dict__ for r in whopays.run(parity)])
print(status.groupby("status").size())
