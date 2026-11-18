# %% [markdown]
# # 08 · ParityQuant v2 controllability gate (exploratory)
#
# Uniform W3 GPTQ on Qwen2.5-1.5B with all 128 × 2,048 calibration windows from Telugu, or
# from Bengali, seeds 0 and 1, on Belebele and FLORES devtest (88 runs, 4 quantized models).
# The multilingual seeds 0–2 from the replication grid are the paired comparison.
# Gate (DECISIONS 2026-11-18): does the target language's held-out NLL increase fall below every
# multilingual seed by more than the multilingual seed spread? Other languages are expected to
# get worse; their numbers are reported, not gated.
#
# Run on Kaggle (Save & Run All, T4 x2). Resumable. **Send back:** the printed tables.

# %%
import whopays
from whopays.config import load_jobs
from whopays.env import results_root, source_root
from whopays.grids import K1_MODEL, replication_grid
from whopays.io.hub_sync import pull_runs
from whopays.io.results import load_runs
from whopays.parallel import run_parallel
from whopays.parityquant import controllability as ct

JOBS = load_jobs(source_root() / "configs" / "pq2_controllability.yaml")
run_parallel(JOBS, check=False)  # one worker per T4
whopays.run(JOBS, stop_on_error=True)

# %%
REFERENCE = [c for c in load_jobs(replication_grid())
             if c.weight == "fp16" or (c.weight == "w3_gptq" and c.calib == "multi")]
pull_runs(results_root(), [c.run_id for c in REFERENCE])
rows = load_runs(results_root(), JOBS + REFERENCE)
increases = ct.nll_increases(rows, K1_MODEL)
print(increases.pivot_table(index=["calib", "seed"], columns="lang", values="increase").round(1))
print(ct.gate(increases).round(2).to_string(index=False))
acc = ct.belebele_accuracy(rows, K1_MODEL)
print(acc.pivot_table(index=["weight", "calib", "seed"], columns="lang",
                      values="calibrated").round(1))
