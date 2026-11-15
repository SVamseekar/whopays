# %% [markdown]
# # 04 · Audit grid, one or more slices
#
# Runs one job list from `configs/`. Each slice is sized to fit one Kaggle session and is
# resumable. Run on Kaggle with Save & Run All, once per slice.
#
# **Needs a K1 pass and the committed preregistration.** Set `SLICES`, then run. Slices may be
# combined when logged rates show they fit one session together; run IDs do not change.
# **Send back:** the status counts. Results reach the results dataset on their own.

# %%
import pandas as pd

import whopays
from whopays.config import load_jobs
from whopays.env import results_root, source_root
from whopays.gates import require_k1
from whopays.parallel import run_parallel

SLICES = (5, 6)
require_k1(results_root())
CONFIGS = source_root() / "configs"
JOBS = [job for s in SLICES for job in load_jobs(CONFIGS / f"audit_slice_{s:02d}.yaml")]
if 0 in SLICES:  # exploratory retention-matched eviction cell (seventh amendment)
    JOBS += load_jobs(CONFIGS / "exploratory_evict_share.yaml")
run_parallel(JOBS, check=False)  # one worker per T4
results = whopays.run(JOBS)  # skips finished runs, retries failures, gives the status table
status = pd.DataFrame([r.__dict__ for r in results])
print(status.groupby(["task", "status"]).size().unstack(fill_value=0))
print(status[status.status == "failed"][["run_id", "task", "lang", "error"]].to_string())
