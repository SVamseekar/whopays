"""ParityQuant v1 Belebele: raw versus letter-prior-calibrated accuracy gaps (exploratory).

Compares each sensitivity-using map (minimax, average) with uniform multilingual quantization
at the same budget. Calibration (`tasks.belebele.calibrated_accuracy`) is applied to both sides;
differences are paired by item and averaged over the map's GPTQ seeds. Intervals resample items
only (2,000 draws), not GPTQ seeds. Reads runs already under results/ (scripts/parity_eval.py
pulls them). DECISIONS 2026-11-18.

    python scripts/pq_letter_bias.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LANGS = ["en", "de", "fr", "es", "ru", "zh", "ja", "th", "bn", "sw", "te"]


def _load():
    from whopays.config import load_jobs
    from whopays.io.results import is_done, load_runs

    jobs = [c for c in load_jobs(ROOT / "configs" / "parityquant_1.5B.yaml")
            if c.task == "belebele" and is_done(ROOT / "results" / c.run_id)]
    rows = load_runs(ROOT / "results", jobs)
    names = {(json.dumps(a["bit_map"], sort_keys=True), a["budget"], a["calib"]): a["method"]
             for a in json.loads((ROOT / "configs" / "parityquant_1.5B.allocations.json")
                                 .read_text())}
    rows["method"] = [names[(json.dumps(json.loads(b) if isinstance(b, str) else b,
                                        sort_keys=True), float(g), c)]
                      for b, g, c in zip(rows.bitmap, rows.budget, rows.calib, strict=True)]
    return rows


def _arrays(frame):
    frame = frame.sort_values("item_id")
    return (frame.item_id.to_numpy(), np.stack(frame.letter_logprobs.map(np.asarray)),
            frame.gold.to_numpy())


def _correct(lp, gold, calibrated):
    x = lp - lp.mean(axis=0, keepdims=True) if calibrated else lp
    return (x.argmax(axis=1) == gold).astype(float)


def main() -> None:
    rows = _load()
    rng = np.random.default_rng(0)
    out = []
    for budget in sorted(rows.budget.unique()):
        for lang in LANGS:
            sub = rows[(rows.budget == budget) & (rows.lang == lang)]
            ids, lp_u, gold = _arrays(sub[sub.method == "uniform_multi"])
            for method in ("minimax", "average"):
                for calibrated in (False, True):
                    base = _correct(lp_u, gold, calibrated)
                    diffs = []
                    for _, g in sub[sub.method == method].groupby("seed"):
                        i2, lp, g2 = _arrays(g)
                        assert (i2 == ids).all()
                        diffs.append(_correct(lp, g2, calibrated) - base)
                    d = np.mean(diffs, axis=0)
                    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                    out.append({"budget": budget, "lang": lang, "method": method,
                                "calibrated": calibrated, "seeds": len(diffs),
                                "diff": 100 * d.mean(), "lo": 100 * np.percentile(boot, 2.5),
                                "hi": 100 * np.percentile(boot, 97.5)})
    table = pd.DataFrame(out)
    for (budget, method), t in table.groupby(["budget", "method"]):
        t = t.assign(v=[f"{r.diff:+.1f} [{r.lo:+.1f}, {r.hi:+.1f}]" for r in t.itertuples()])
        print(f"\n{budget} bits, {method} minus uniform multilingual "
              f"(points; {t.seeds.iloc[0]} seed(s))")
        print(t.pivot(index="lang", columns="calibrated", values="v").reindex(LANGS)
              .to_string())


if __name__ == "__main__":
    main()
