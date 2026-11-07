"""FR-3 replication checks against GPTQModel, and the published gap direction.

FR-3 asks whether our fake-quantized GPTQ matches a production GPTQ implementation. The primary
reference (fourth amendment, 2026-11-07) is GPTQModel quantizing the same model on the same
calibration windows with the same classic settings, so a gap measures the implementation and
not calibration data. Qwen's official GPTQ-Int4 checkpoint was the first reference; its
calibration data are undisclosed and it is symmetric, so it stays as a descriptive secondary
comparison. On 2026-11-06 it differed from our 128-sentence calibration by 7.5% on average.

References run through the normal job path in an isolated venv, so their rows carry the same
provenance as ours and GPTQModel stays out of the main environment.
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
# GPTQModel 7.6 defaults differ from classic GPTQ: act_group_aware (GAR reordering) is on for
# GPTQ, sym is True and damp_percent is method-dependent. Pin every setting to match
# quant/gptq.py: asymmetric, group 128, no act-order, damping 0.01, true-sequential.
GPTQMODEL_CONFIG = {"bits": 4, "group_size": 128, "sym": False, "desc_act": False,
                    "act_group_aware": False, "damp_percent": 0.01, "static_groups": False,
                    "mse": 0.0, "true_sequential": True}
FR3_CALIB = {"calib": "multi", "seed": 0, "calib_source": "wikipedia_128x2048"}
TOLERANCE_PCT = 2.0
COLUMNS = ["lang", "ours", "reference", "abs_pct", "flag"]


def reference_checkpoint(model: str) -> str:
    return f"{model.removesuffix('-Instruct')}-Instruct-GPTQ-Int4"


def fr3_jobs(model: str) -> list[RunConfig]:
    """Our W4 on FLORES devtest, calibrated exactly as the GPTQModel reference."""
    return [RunConfig(model=model, task="flores_nll", lang=lang, weight="w4_gptq", **FR3_CALIB)
            for lang in CODES]


def reference_jobs(model: str, kind: str = "gptqmodel") -> list[RunConfig]:
    if kind == "gptqmodel":
        return [RunConfig(model=model, task="flores_nll", lang=lang, weight="w4_gptqmodel",
                          **FR3_CALIB) for lang in CODES]
    if kind == "official":
        return [RunConfig(model=reference_checkpoint(model), task="flores_nll", lang=lang,
                          weight="prequant") for lang in CODES]
    raise ValueError("reference kind must be 'gptqmodel' or 'official'")


def quantize_with_gptqmodel(cfg: RunConfig, tokenizer, cache: Path | None = None):
    """Quantize cfg.model with GPTQModel on our calibration windows; reuse a finished result.

    Returns the checkpoint directory and the calibration provenance.
    """
    import json

    from whopays.quant.calibration import batches_for

    cache = Path(cache or os.environ.get("WHOPAYS_GPTQMODEL_CACHE", "/tmp/whopays-gptqmodel"))
    out = cache / cfg.model_key()
    if (out / "DONE").exists():
        return out, json.loads((out / "calibration.json").read_text())
    from gptqmodel import GPTQModel, QuantizeConfig

    batches, provenance = batches_for(tokenizer, cfg.calib_source, cfg.calib, cfg.seed)
    settings = {**GPTQMODEL_CONFIG, "group_size": cfg.group_size}
    model = GPTQModel.load(cfg.model, quantize_config=QuantizeConfig(**settings))
    model.quantize([{"input_ids": b["input_ids"][0].tolist(),
                     "attention_mask": [1] * b["input_ids"].shape[1]} for b in batches],
                   batch_size=1)
    model.save(str(out))
    (out / "calibration.json").write_text(json.dumps(provenance) + "\n")
    (out / "DONE").write_text(json.dumps({"gptqmodel": settings}) + "\n")
    return out, provenance


def install() -> Path:
    return crosscheck.install(Path("/tmp/venv-gptq"), package=f"gptqmodel=={GPTQMODEL_VERSION}")


def run_reference(model: str, *, kind: str = "gptqmodel", root: Path | None = None,
                  sync: bool = True) -> list[RunConfig]:
    """Score a reference on FLORES devtest. Finished runs skip, as for any job list."""
    root = Path(root) if root is not None else results_root()
    jobs = reference_jobs(model, kind)
    path = write_jobs(jobs, root / f"reference-{kind}-jobs.yaml")
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
