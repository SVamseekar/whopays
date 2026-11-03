# %% [markdown]
# # 05 · Exactness gate and speed grid (FR-5)
#
# 1. **Gate:** speculative output must equal plain greedy, token for token, across 110 prompts
#    (10 per language) under every cache policy, running past EOS so flushes occur. This is the
#    T4 evidence compressed-cache block verification waits for.
# 2. **Speed grid:** first 30 MGSM items per language, draft off and on, batch size 1.
#
# **Needs a K1 pass.** **Send back:** the gate verdict and the tables.

# %%
import pandas as pd

import whopays
from whopays.config import load_jobs
from whopays.decoding.exactness import run_gate
from whopays.env import results_root
from whopays.gates import require_k1
from whopays.grids import K1_MODEL, speed_grid, speed_subset

require_k1(results_root())
gate, state = run_gate(K1_MODEL)
print(gate.to_string(index=False))
print("exactness gate:", state)
if state == "FAIL":
    raise SystemExit("Speculative output diverged from greedy: a harness bug. Do not run the grid.")

# %%
JOBS = load_jobs(speed_grid() + speed_subset())
results = whopays.run(JOBS)
status = pd.DataFrame([r.__dict__ for r in results])
print(status.groupby("status").size())

# %%
from whopays.io.results import load_runs

rows = load_runs(results_root(), JOBS)
summary = rows.groupby(["model", "weight", "kv", "spec", "lang"], as_index=False).agg(
    items=("item_id", "size"), wall_s=("wall_time_s", "sum"), output=("output_tokens", "sum"))
summary["tokens_per_s"] = summary.output / summary.wall_s
print(summary.round(2).to_string(index=False))
