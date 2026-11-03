# whopays: standalone

# %% [markdown]
# # Analysis (M1 Air, CPU only)
#
# Pulls the results dataset and rebuilds every hypothesis table and figure. It needs no GPU, and
# never produces a run. Run it with `HF_TOKEN` set, from an environment with `pip install -e .`.
# Pass `ALLOCATIONS` once `python -m whopays.cli parityquant` has solved a model.

# %%
import json
from pathlib import Path

from whopays.io import hub_sync
from whopays.report.analysis import timed_regenerate

ROOT, OUT = Path("results"), Path("reports")
hub_sync.pull_runs(ROOT)
ALLOCATIONS = {}  # {"Qwen/Qwen2.5-1.5B-Instruct": json.loads(Path("configs/...json").read_text())}
print(timed_regenerate(ROOT, OUT, allocations=ALLOCATIONS))

# %%
print(json.loads((OUT / "summary.json").read_text()))
