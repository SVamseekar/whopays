"""ParityQuant v2 controllability gate (exploratory, DECISIONS 2026-11-18).

Question: does putting calibration weight on one language move that language's W3 loss by more
than GPTQ seed noise? Telugu-only and Bengali-only calibration are compared with the logged
multilingual seeds 0–2 of the replication grid. A one-language set is expected to hurt other
languages, so the worst of 11 is reported but is not the gate.

Measure: mean NLL increase per sentence against FP16 on held-out FLORES devtest (items after
the first 500 by item ID, the H4 set). The gate passes for a target language when every
target-only seed lies below every multilingual seed by more than the multilingual seed spread
(max − min). Belebele raw and letter-prior-calibrated accuracy are reported beside it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HELD_OUT_FROM = 500


def _held_out(rows: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for _lang, group in rows.groupby("lang"):
        ids = sorted(group.item_id.unique())[HELD_OUT_FROM:]
        keep.append(group[group.item_id.isin(ids)])
    return pd.concat(keep)


def nll_increases(rows: pd.DataFrame, model: str) -> pd.DataFrame:
    """Mean held-out NLL increase per sentence, per (calib, seed, lang)."""
    flores = _held_out(rows[(rows.model == model) & (rows.task == "flores_nll")])
    fp16 = flores[flores.weight == "fp16"].set_index(["lang", "item_id"]).nll_sum
    quant = flores[flores.weight == "w3_gptq"]
    records = []
    for (calib, seed, lang), group in quant.groupby(["calib", "seed", "lang"]):
        delta = group.set_index(["lang", "item_id"]).nll_sum - fp16.reindex(
            group.set_index(["lang", "item_id"]).index)
        if delta.isna().any():
            raise ValueError(f"FP16 reference missing items for {lang}")
        records.append({"calib": calib, "seed": int(seed), "lang": lang,
                        "increase": float(delta.mean()), "n_items": int(len(delta))})
    return pd.DataFrame(records)


def belebele_accuracy(rows: pd.DataFrame, model: str) -> pd.DataFrame:
    """Raw and letter-prior-calibrated accuracy (%), per (weight, calib, seed, lang)."""
    from whopays.tasks.belebele import calibrated_accuracy

    bele = rows[(rows.model == model) & (rows.task == "belebele")]
    records = []
    for (weight, calib, seed, lang), g in bele.groupby(["weight", "calib", "seed", "lang"]):
        lp = np.stack(g.letter_logprobs.map(np.asarray).to_numpy())
        gold = g.gold.to_numpy()
        records.append({"weight": weight, "calib": calib, "seed": int(seed), "lang": lang,
                        "raw": 100 * float((lp.argmax(1) == gold).mean()),
                        "calibrated": 100 * calibrated_accuracy(lp, gold)})
    return pd.DataFrame(records)


def gate(increases: pd.DataFrame, targets=("te", "bn")) -> pd.DataFrame:
    """Per target language: multilingual seed range, target-only seeds, and the verdict."""
    out = []
    for lang in targets:
        multi = increases[(increases.calib == "multi") & (increases.lang == lang)].increase
        own = increases[(increases.calib == lang) & (increases.lang == lang)].increase
        if len(multi) < 2 or not len(own):
            out.append({"lang": lang, "verdict": "INCOMPLETE"})
            continue
        spread = float(multi.max() - multi.min())
        margin = float(multi.min() - own.max())
        worst_own = increases[increases.calib == lang].groupby("seed").increase.max()
        worst_multi = increases[increases.calib == "multi"].groupby("seed").increase.max()
        out.append({"lang": lang, "multi_min": float(multi.min()),
                    "multi_max": float(multi.max()), "spread": spread,
                    "own_mean": float(own.mean()), "own_max": float(own.max()),
                    "margin": margin, "worst11_own": float(worst_own.mean()),
                    "worst11_multi": float(worst_multi.mean()),
                    "verdict": "CONTROLLABLE" if margin > spread else "NOT_CONTROLLABLE"})
    return pd.DataFrame(out)
