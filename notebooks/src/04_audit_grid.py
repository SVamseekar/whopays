# %% [markdown]
# # 04 · Audit grid, one slice
#
# Runs one job list from `configs/`. Each slice is sized to fit one Kaggle session and is
# resumable. Run on Kaggle with Save & Run All, once per slice.
#
# **Needs a K1 pass and the committed preregistration.** Set `SLICE`, then run.
# **Send back:** the status counts. Results reach the results dataset on their own.

# %%
import pandas as pd

import whopays
from whopays.env import results_root, source_root
from whopays.gates import require_k1

SLICE = 0
require_k1(results_root())
JOBS = source_root() / "configs" / f"audit_slice_{SLICE:02d}.yaml"
results = whopays.run(JOBS)
status = pd.DataFrame([r.__dict__ for r in results])
print(status.groupby(["task", "status"]).size().unstack(fill_value=0))
print(status[status.status == "failed"][["run_id", "task", "lang", "error"]].to_string())
