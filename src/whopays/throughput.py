"""Measured T4 throughput vs the PRD's planning assumptions (BUILD_PLAN M0 gate).

Rows log `wall_time_s` per item (batch time / batch size for batched tasks), so a run's
summed wall time is its GPU time without model loading. Times are projected to the full
benchmark size, so a run with `limit` still answers "how long would the whole grid take?".
"""

from __future__ import annotations

import re

import pandas as pd

FULL_ITEMS = {"belebele": 9_900, "mgsm": 2_750, "flores_nll": 11_132}  # 11 languages each
# PRD §Compute plan, minutes for all 11 languages at FP16 on a T4.
ASSUMED_MIN = {("belebele", "1.5B"): 15, ("belebele", "3B"): 30,
               ("mgsm", "1.5B"): 30, ("mgsm", "3B"): 60}
REPLAN_RATIO = 2.0  # PRD: re-plan the compute budget if more than 2x slower than assumed


def model_size(model: str) -> str:
    m = re.search(r"(\d+(?:\.\d+)?B)\b", model)
    return m.group(1) if m else model


def summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Per model × task: items, GPU minutes, projection to the full benchmark, vs assumption.

    `rows` needs model, task, item_id, wall_time_s; output_tokens is used when present.
    """
    out = []
    for (model, task), g in rows.groupby(["model", "task"], sort=False):
        seconds = float(g["wall_time_s"].sum())
        size = model_size(model)
        projected = seconds / len(g) * FULL_ITEMS[task] / 60 if task in FULL_ITEMS else None
        assumed = ASSUMED_MIN.get((task, size))
        ratio = projected / assumed if assumed else None
        rec = {"model": size, "task": task, "items": len(g), "gpu_min": round(seconds / 60, 1),
               "items_per_s": round(len(g) / seconds, 2) if seconds else None,
               "projected_full_min": round(projected, 1) if projected is not None else None,
               "assumed_min": assumed,
               "ratio_vs_assumed": round(ratio, 2) if ratio is not None else None,
               "replan": ratio is not None and ratio > REPLAN_RATIO}
        if "output_tokens" in g and g["output_tokens"].notna().any():
            rec["out_tokens_per_s"] = round(float(g["output_tokens"].sum()) / seconds, 1)
        out.append(rec)
    return pd.DataFrame(out)
