"""FR-3 replication checks against a real GPTQ checkpoint, and the published gap direction.

The reference runs through the normal job path in an isolated venv, so its rows carry the
same provenance as ours and GPTQModel stays out of the main environment.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from whopays import crosscheck
from whopays.config import RunConfig
from whopays.env import results_root
from whopays.grids import write_jobs
from whopays.languages import CODES

GPTQMODEL_VERSION = "7.6.0"
TOLERANCE_PCT = 2.0
COLUMNS = ["lang", "ours", "reference", "abs_pct", "flag"]


def reference_checkpoint(model: str) -> str:
    return f"{model.removesuffix('-Instruct')}-Instruct-GPTQ-Int4"


def reference_jobs(model: str) -> list[RunConfig]:
    return [RunConfig(model=reference_checkpoint(model), task="flores_nll", lang=lang,
                      weight="prequant") for lang in CODES]


def install() -> Path:
    return crosscheck.install(Path("/tmp/venv-gptq"), package=f"gptqmodel=={GPTQMODEL_VERSION}")


def run_reference(model: str, *, root: Path | None = None, sync: bool = True) -> list[RunConfig]:
    """Score the checkpoint on FLORES devtest. Finished runs skip, as for any job list."""
    root = Path(root) if root is not None else results_root()
    jobs = reference_jobs(model)
    path = write_jobs(jobs, root / "reference-jobs.yaml")
    command = [str(install()), "-m", "whopays.cli", "run", str(path), "--root", str(root)]
    if not sync:
        command.append("--no-sync")
    print("reference environment python:", command[0], "| caller:", sys.executable)
    subprocess.run(command, check=True, env={**os.environ, "USE_TF": "0"})
    return jobs


def perplexity(rows: pd.DataFrame) -> dict[str, float]:
    """Token-weighted perplexity per language."""
    totals = rows.groupby("lang")[["nll_sum", "n_tokens"]].sum()
    return {lang: float(np.exp(r.nll_sum / r.n_tokens)) for lang, r in totals.iterrows()}


def fr3_verdict(ours: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Perplexity agreement on identical items; the K1-style mean decides, large gaps flag."""
    mine, theirs = perplexity(ours), perplexity(reference)
    same_items = (set(mine) == set(theirs) and bool(mine) and all(
        set(ours[ours.lang == lang].item_id) == set(reference[reference.lang == lang].item_id)
        for lang in mine))
    table = pd.DataFrame([{"lang": lang, "ours": mine[lang], "reference": theirs[lang],
                           "abs_pct": 100 * abs(mine[lang] / theirs[lang] - 1)}
                          for lang in sorted(set(mine) & set(theirs))])
    table["flag"] = table["abs_pct"] > TOLERANCE_PCT
    mean = float(table.abs_pct.mean())
    state = ("INCOMPLETE" if not same_items else "PASS" if mean <= TOLERANCE_PCT else "FAIL")
    return table[COLUMNS], {"mean_abs_pct": mean, "max_abs_pct": float(table.abs_pct.max()),
                            "FR3": state}


def gap_direction(fp16: pd.DataFrame, treated: pd.DataFrame) -> pd.DataFrame:
    """Paired accuracy change per language, relative to English's change."""
    paired = fp16[["lang", "item_id", "correct"]].merge(
        treated[["lang", "item_id", "correct"]], on=["lang", "item_id"], suffixes=("_fp16", "_q"),
        validate="one_to_one")
    delta = 100 * (paired.correct_q.astype(float) - paired.correct_fp16.astype(float))
    change = delta.groupby(paired.lang).mean()
    gap = change - change["en"]
    return pd.DataFrame({"lang": change.index, "change_pts": change.to_numpy(),
                         "gap_vs_english_pts": gap.to_numpy(),
                         "worse_than_english": (gap < 0).to_numpy()})
