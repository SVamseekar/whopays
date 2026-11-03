# %% [markdown]
# # 03 · Layer sensitivity probe (FR-12, K2 gate)
#
# Per block × bits × language NLL increase on FLORES devtest for 1.5B. The other blocks stay at
# reference precision. K2 passes when the minimum pairwise Spearman correlation of 4-bit
# sensitivity vectors is at most 0.9. If K2 fails, the PRD drops M4 and extends the audit.
#
# **Needs a K1 pass.** Every row is durable, so re-running resumes. Each finished block prints
# the projected total: if it is above the PRD's 3 h, tell Claude Code before continuing.
# **Send back:** the K2 line and the last projection.

# %%
from whopays.env import results_root
from whopays.gates import require_k1
from whopays.grids import K1_MODEL
from whopays.parityquant.probe import k2_verdict, run_probe

require_k1(results_root())
LIMIT = 100  # sentences per language, split into two halves; DECISIONS 2026-11-03
rows = run_probe(K1_MODEL, limit=LIMIT)

# %%
print(k2_verdict(rows).to_string())
