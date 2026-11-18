"""Depth breakdown of the K2 sensitivity probe (exploratory, DECISIONS 2026-11-18).

Per block: cost of demoting 4 → 3 bits and gain of promoting 4 → 8 bits, as the per-token NLL
increase of that block quantized alone (mean of the probe's two halves), for en, ru, bn and te,
with depth-band sums and the Spearman correlation of the 4 → 3 cost profiles.

    python scripts/probe_depth.py [results/probe-6f52db8b4007/profile.jsonl]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LANGS = ["en", "ru", "bn", "te"]
BANDS = ([-1, 2, 9, 17, 22, 27], ["0-2", "3-9", "10-17", "18-22", "23-27"])


def main(path: Path) -> None:
    r = pd.read_json(path, lines=True)
    r["base"] = r.lang.str.split("#").str[0]
    r["block"] = r.layer.str.split(".").str[2].astype(int)
    s = r.groupby(["base", "bits", "block"]).sensitivity.mean().unstack("bits")
    cost43 = (s[3] - s[4]).unstack("base")
    gain84 = (s[4] - s[8]).unstack("base")
    table = pd.concat({**{f"cost43_{x}": cost43[x] for x in LANGS},
                       **{f"gain84_{x}": gain84[x] for x in LANGS}}, axis=1)
    print("nats per token × 1000\n" + (1000 * table).round(1).to_string())
    for x in LANGS:
        print(f"{x}: cheapest to demote {list(cost43[x].sort_values().index[:10])}; "
              f"largest promote gain {list(gain84[x].sort_values(ascending=False).index[:5])}")
    rho = cost43.corr(method="spearman")
    print("\nSpearman of 4→3 cost profiles\n" + rho.loc[LANGS, LANGS].round(2).to_string())
    print("min over all pairs:", round(rho.where(~np.eye(len(rho), dtype=bool)).min().min(), 3))
    band = cost43[LANGS].groupby(pd.cut(cost43.index, *BANDS), observed=True).sum()
    print("\n4→3 cost by depth band (× 1000)\n" + (1000 * band).round(1).to_string())
    print("\nshare of summed 4→3 cost\n" + (band / cost43[LANGS].sum()).round(2).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1
         else ROOT / "results" / "probe-6f52db8b4007" / "profile.jsonl")
