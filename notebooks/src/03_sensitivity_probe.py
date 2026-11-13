# %% [markdown]
# # 03 · Layer sensitivity probe (FR-12, K2 gate)
#
# Per block × bits × language NLL increase on FLORES devtest for 1.5B. The other blocks stay at
# reference precision. K2 passes when the minimum pairwise Spearman correlation of 4-bit
# sensitivity vectors is at most 0.9. If K2 fails, the PRD drops M4 and extends the audit.
#
# **Needs a K1 pass.** Every row is durable, so re-running resumes. Each finished block prints
# the projected total (about 7 h expected at 500 sentences; tell Claude Code if it exceeds 11 h).
# **Send back:** the K2 line and the last projection.

# %%
from whopays.env import results_root
from whopays.gates import require_k1
from whopays.grids import K1_MODEL
from whopays.parityquant.probe import k2_verdict, run_probe

require_k1(results_root())
LIMIT = 500  # sentences per language, two halves; 100 was noise-limited (DECISIONS 2026-11-13)
rows = run_probe(K1_MODEL, limit=LIMIT)

# %%
print(k2_verdict(rows).to_string())
