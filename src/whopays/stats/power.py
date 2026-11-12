"""Power of the preregistered tests, before GPU time is spent on them.

Two questions:

1. **Per-cell comparisons.** What is the smallest paired accuracy change (treated − FP16, same
   items) a language × configuration comparison can detect? For a paired binary outcome the
   variance of the mean difference is about d / n, with d the discordance rate (share of
   items whose correctness flips either way), so
   MDE = (z_{1-α/2} + z_{power}) · sqrt(d / n)   (Connor, 1987, Biometrics 43:207; the
   normal approximation to McNemar's test). Benjamini–Hochberg makes the effective α smaller,
   so this is a lower bound.
2. **H1.** H1 is a Spearman correlation over only 10 non-English languages, with an item
   bootstrap interval (`report.hypotheses.h1`). Power is simulated: plant a per-language loss
   that grows with token premium, draw paired item outcomes, run the exact H1 code, count
   how often the preregistered rule calls it supported.

Planning inputs (discordance, base accuracy, premium) should come from logged FP16 runs and,
for discordance, from earlier quantized runs used for planning only. None of the outputs of
this module is a result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def paired_mde(n_items: int, discordance: float, *, alpha: float = 0.05,
               power: float = 0.8) -> float:
    """Smallest detectable paired accuracy difference (as a proportion, two-sided)."""
    if n_items < 1 or not 0 < discordance <= 1:
        raise ValueError("need n_items >= 1 and 0 < discordance <= 1")
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * np.sqrt(discordance / n_items))


def _paired_draw(rng, n: int, base: float, discordance: float, loss: float):
    """FP16 and treated correctness with mean FP16 accuracy `base`, flips `discordance`,
    and mean loss `loss` = P(10) − P(01)."""
    p10, p01 = (discordance + loss) / 2, (discordance - loss) / 2
    p11 = base - p10
    if min(p10, p01, p11, 1 - base - p01) < 0:
        raise ValueError(f"infeasible cell: base={base}, discordance={discordance}, loss={loss}")
    cell = rng.choice(4, size=n, p=[p11, p10, p01, 1 - p11 - p10 - p01])
    return np.isin(cell, (0, 1)), np.isin(cell, (0, 2))  # fp16 correct, treated correct


def simulate_paired_power(n_items: int, discordance: float, delta: float, *,
                          base: float = 0.7, alpha: float = 0.05, n_sims: int = 2000,
                          seed: int = 0) -> float:
    """Share of simulated cells where a two-sided paired z-test rejects at `alpha`."""
    rng = np.random.default_rng(seed)
    crit = norm.ppf(1 - alpha / 2)
    hits = 0
    for _ in range(n_sims):
        fp16, treated = _paired_draw(rng, n_items, base, discordance, delta)
        diff = fp16.astype(float) - treated.astype(float)
        se = diff.std(ddof=1) / np.sqrt(n_items)
        hits += se > 0 and abs(diff.mean()) / se > crit
    return hits / n_sims


def simulate_h1_power(premium, base_accuracy, *, slope: float, n_items: int,
                      discordance: float, n_sims: int = 200, n_resamples: int = 500,
                      seed: int = 0) -> float:
    """Share of simulations in which preregistered H1 is called supported.

    `premium` and `base_accuracy` are per non-English language. The planted loss is
    slope × (premium − 1), English has premium 1 and no loss.
    """
    from whopays.report.hypotheses import h1

    premium, base_accuracy = np.asarray(premium, float), np.asarray(base_accuracy, float)
    if premium.shape != base_accuracy.shape:
        raise ValueError("premium and base_accuracy need one value per language")
    langs = ["en"] + [f"l{i}" for i in range(len(premium))]
    prem = np.concatenate([[1.0], premium])
    base = np.concatenate([[base_accuracy.max()], base_accuracy])
    loss = slope * (prem - 1)
    items = [f"q{i:04d}" for i in range(n_items)]
    rng = np.random.default_rng(seed)
    supported = 0
    for _ in range(n_sims):
        fp16_rows, treated_rows = [], []
        for lang, b, p, lo in zip(langs, base, prem, loss, strict=True):
            fp16, treated = _paired_draw(rng, n_items, b, discordance, lo)
            tokens = np.full(n_items, round(100 * p))
            fp16_rows.append(pd.DataFrame({"item_id": items, "lang": lang, "correct": fp16,
                                           "input_tokens": tokens}))
            treated_rows.append(pd.DataFrame({"item_id": items, "lang": lang,
                                              "correct": treated, "input_tokens": tokens}))
        result = h1(pd.concat(fp16_rows), pd.concat(treated_rows), n_resamples=n_resamples,
                    seed=int(rng.integers(2**31)))
        supported += result["supported"]
    return supported / n_sims
