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
from whopays.io.results import load_runs
from whopays.languages import CODES

GPTQMODEL_VERSION = "7.6.0"
# GPTQModel 7.6 defaults differ from classic GPTQ: act_group_aware (GAR reordering) is on for
# GPTQ, sym is True and damp_percent is method-dependent. Pin every setting to match
# quant/gptq.py: asymmetric, group 128, no act-order, damping 0.01, true-sequential.
# format gptq_v2: the v1 "gptq" layout stores zero points minus one, which overflows for
# asymmetric groups (GPTQModel 7.6.0, quantization/config.py "v2 format fixed sym = False
# quantization"; utils/model.py "v1 checkpoint format with sym=False ... will overflow").
GPTQMODEL_CONFIG = {"bits": 4, "group_size": 128, "sym": False, "desc_act": False,
                    "act_group_aware": False, "damp_percent": 0.01, "static_groups": False,
                    "mse": 0.0, "true_sequential": True, "format": "gptq_v2"}
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


def quantize_with_gptqmodel(cfg: RunConfig, tokenizer, cache: Path | None = None,
                            dtype=None):
    """Quantize cfg.model with GPTQModel on our calibration windows; reuse a finished result.

    Runs in ``dtype`` (FP16 by default, as every other T4 run), not the config's bfloat16.

    Returns the checkpoint directory and the calibration provenance.
    """
    import json

    from whopays.quant.calibration import batches_for

    default = Path.home() / ".cache" / "whopays" / "gptqmodel"
    cache = Path(cache or os.environ.get("WHOPAYS_GPTQMODEL_CACHE", default))
    _private_dir(cache)
    out = cache / cfg.model_key()
    if (out / "DONE").exists():
        done = json.loads((out / "DONE").read_text())
        if done.get("weights") != _weights_digest(out):
            raise ValueError(f"cached GPTQModel weights in {out} changed since quantization; "
                             "delete the folder to re-quantize")
        return out, json.loads((out / "calibration.json").read_text())
    import torch
    from gptqmodel import GPTQModel, QuantizeConfig

    dtype = dtype or torch.float16

    batches, provenance = batches_for(tokenizer, cfg.calib_source, cfg.calib, cfg.seed)
    settings = {**GPTQMODEL_CONFIG, "group_size": cfg.group_size}
    model = GPTQModel.load(cfg.model, quantize_config=QuantizeConfig(**settings),
                           dtype=dtype)
    model.quantize([{"input_ids": b["input_ids"][0].tolist(),
                     "attention_mask": [1] * b["input_ids"].shape[1]} for b in batches],
                   batch_size=1)
    model.save(str(out))
    (out / "calibration.json").write_text(json.dumps(provenance) + "\n")
    (out / "DONE").write_text(json.dumps({"gptqmodel": settings, "dtype": str(dtype),
                                          "weights": _weights_digest(out)}) + "\n")
    return out, provenance


def _private_dir(path: Path) -> None:
    """Create or accept a cache folder only this user can write; loaded weights come from it."""
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise PermissionError(f"{path} is owned by another user")
    if info.st_mode & 0o022:
        raise PermissionError(f"{path} is writable by group or others")


def _weights_digest(folder: Path) -> str:
    """SHA-256 over the checkpoint's safetensors files, in name order."""
    import hashlib

    digest = hashlib.sha256()
    files = sorted(folder.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no safetensors weights in {folder}")
    for f in files:
        digest.update(f.name.encode())
        with f.open("rb") as handle:
            for chunk in iter(lambda: handle.read(2**20), b""):
                digest.update(chunk)
    return digest.hexdigest()


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


def fr3_check(model: str, *, root: Path | None = None):
    """FR-3 end to end: K1 gate, our W4 runs, the GPTQModel reference (primary) and Qwen's
    official GPTQ-Int4 checkpoint (secondary, descriptive). Returns (table, verdict, official
    table). Notebook 02 and the Modal runner both call this."""
    import whopays
    from whopays import gates

    root = Path(root) if root is not None else results_root()
    gates.require_k1(root)
    ours_jobs = fr3_jobs(model)
    whopays.run(ours_jobs, stop_on_error=True)
    ours = load_runs(root, ours_jobs)
    table, verdict = fr3_verdict(ours, load_runs(root, run_reference(model, root=root)))
    official, _ = fr3_verdict(ours, load_runs(root, run_reference(model, kind="official",
                                                                   root=root)))
    return table, verdict, official


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


# Fifth amendment (2026-11-13): replication direction is judged on measures without a floor.
# Raw Belebele points cannot show the published direction at 1.5B, where low-resource languages
# start a few points above the 25% chance level (DECISIONS 2026-11-13).
BELEBELE_CHANCE = 0.25


def _shared_bootstrap(n_items: int, statistic, n_resamples: int, seed: int):
    """Percentile interval of `statistic(index)` under one item draw shared by all languages."""
    rng = np.random.default_rng(seed)
    draws = np.asarray([statistic(rng.integers(0, n_items, n_items))
                        for _ in range(n_resamples)])
    return np.nanquantile(draws, [0.025, 0.975], axis=0)


def _seed_mean_panel(rows: pd.DataFrame, value: str) -> pd.DataFrame:
    """item × language panel, averaging repeated runs (calibration seeds) per item."""
    panel = rows.groupby(["item_id", "lang"])[value].mean().unstack("lang")
    panel = panel.sort_index().sort_index(axis=1)
    if panel.isna().any().any():
        raise ValueError("incomplete item/language panel")
    return panel.astype(float)


def nll_gap_direction(fp16: pd.DataFrame, treated: pd.DataFrame, *, n_resamples: int = 2000,
                      seed: int = 0) -> pd.DataFrame:
    """Relative increase in total FLORES NLL per language, minus English's, with paired CIs.

    NLL has no floor, so a language the model barely knows cannot hide its degradation the
    way near-chance accuracy does. Sentences are parallel, so one bootstrap draw is shared by
    every language. Several treated runs (seeds) are averaged per sentence.
    """
    base, treat = _seed_mean_panel(fp16, "nll_sum"), _seed_mean_panel(treated, "nll_sum")
    if not base.index.equals(treat.index) or not base.columns.equals(treat.columns):
        raise ValueError("FP16 and treated rows must cover identical sentences and languages")
    a, b, langs = base.to_numpy(), treat.to_numpy(), list(base.columns)
    en = langs.index("en")

    def statistic(index):
        rel = 100 * (b[index].sum(0) / a[index].sum(0) - 1)
        return rel - rel[en]

    rel = 100 * (b.sum(0) / a.sum(0) - 1)
    low, high = _shared_bootstrap(len(a), statistic, n_resamples, seed)
    return pd.DataFrame({"lang": langs, "rel_increase_pct": rel,
                         "gap_vs_english_pct": rel - rel[en], "low": low, "high": high,
                         "worse_than_english": low > 0})


def _calibrated_correct(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-item correctness after removing each run's mean preference for every letter.

    The PRD's letter-bias mitigation (risk register: "also report calibrated accuracy"),
    the run-level version of prior debiasing (Zheng et al., ICLR 2024, arXiv 2309.03882).
    """
    run_key = "run_id" if "run_id" in rows else None
    out = []
    for _, group in rows.groupby(([run_key] if run_key else []) + ["lang"]):
        lp = np.stack(group["letter_logprobs"].map(np.asarray).to_numpy())
        pred = (lp - lp.mean(axis=0, keepdims=True)).argmax(axis=1)
        out.append(group[["item_id", "lang"]].assign(
            correct=(pred == group["gold"].to_numpy()).astype(float)))
    return pd.concat(out, ignore_index=True)


def retention_gap_direction(fp16: pd.DataFrame, treated: pd.DataFrame, *,
                            chance: float = BELEBELE_CHANCE, n_resamples: int = 2000,
                            seed: int = 0) -> pd.DataFrame:
    """Share of above-chance calibrated Belebele accuracy kept, minus English's share.

    retention = (acc_q − chance) / (acc_fp16 − chance). It is undefined (NaN) for a language
    with no headroom above chance. Chance-adjusted accuracy is a common multiple-choice
    correction (for example arXiv 2502.12896). A negative gap means the language keeps less of
    its ability than English. Several treated runs (seeds) are averaged per item.
    """
    base = _seed_mean_panel(_calibrated_correct(fp16), "correct")
    treat = _seed_mean_panel(_calibrated_correct(treated), "correct")
    if not base.index.equals(treat.index) or not base.columns.equals(treat.columns):
        raise ValueError("FP16 and treated rows must cover identical items and languages")
    a, b, langs = base.to_numpy(), treat.to_numpy(), list(base.columns)
    en = langs.index("en")

    def retention(index):
        head = a[index].mean(0) - chance
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(head > 1e-12, (b[index].mean(0) - chance) / head, np.nan)

    def statistic(index):
        r = retention(index)
        return r - r[en]

    point = retention(np.arange(len(a)))
    low, high = _shared_bootstrap(len(a), statistic, n_resamples, seed)
    return pd.DataFrame({"lang": langs, "fp16_calibrated_acc": a.mean(0),
                         "treated_calibrated_acc": b.mean(0), "retention": point,
                         "gap_vs_english": point - point[en], "low": low, "high": high,
                         "worse_than_english": high < 0})
