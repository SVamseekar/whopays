# %% [markdown]
# # 06 · ParityQuant evaluation (K3 gate)
#
# Evaluates bit maps solved locally by `python -m whopays.cli parityquant`, which refuses a K2
# failure. The job lists are committed under `configs/`. Set `SIZE` and run on Kaggle.
#
# **Needs a K1 pass.** The verdict is computed locally from the results dataset, in `analysis`.
# **Send back:** the status counts.

# %%
import pandas as pd

import whopays
from whopays.env import results_root, source_root
from whopays.gates import require_k1

SIZE = "1.5B"
require_k1(results_root())
results = whopays.run(source_root() / "configs" / f"parityquant_{SIZE}.yaml")
status = pd.DataFrame([r.__dict__ for r in results])
print(status.groupby("status").size())
print(status[status.status == "failed"][["run_id", "lang", "error"]].to_string())
