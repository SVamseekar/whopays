"""Preregistered H1-H4 from per-item rows. Every function refuses incomplete designs.

Rows come from `report.analysis.load_parquet`. Filter to one model, task and calibration seed
before calling, because seed repetitions are not independent questions.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from whopays.languages import LANGUAGES
from whopays.stats.bootstrap import (
    benjamini_hochberg,
    paired_binary_pvalues,
    paired_frame_bootstrap,
)
from whopays.stats.mixed import interaction_tests
from whopays.stats.oaxaca import oaxaca
from whopays.stats.shapley import shapley_values

MEAN_COST_LIMIT = 0.01  # one percentage point (PREREGISTRATION H4)


def _panel(rows: pd.DataFrame, column: str = "correct") -> pd.DataFrame:
    if rows.duplicated(["item_id", "lang"]).any():
        raise ValueError("duplicate item/language rows; filter to one configuration")
    panel = rows.pivot(index="item_id", columns="lang", values=column).sort_index().sort_index(
        axis=1)
    if panel.isna().any().any():
        raise ValueError("incomplete item/language panel")
    return panel.astype(float)


def _rho(x, y) -> float | None:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def h1(fp16: pd.DataFrame, treated: pd.DataFrame, *, n_resamples: int = 2000, seed: int = 0):
    """Spearman of per-language loss with token premium and with FP16 accuracy, item bootstrap.

    Loss is FP16 accuracy minus treated accuracy, so H1 predicts positive correlation with
    premium and negative correlation with base accuracy. Falsified if both intervals hold 0.
    """
    base, treat = _panel(fp16), _panel(treated)
    tokens = _panel(fp16, "input_tokens")
    if not base.index.equals(treat.index) or not base.columns.equals(treat.columns):
        raise ValueError("FP16 and treated rows must cover identical items and languages")
    if "en" not in tokens:
        raise ValueError("token premium needs English items")
    premium_items = tokens.div(tokens["en"], axis=0).drop(columns="en")
    gap = (base - treat)[premium_items.columns]
    accuracy = base[premium_items.columns]

    def statistics(index):
        loss = gap.iloc[index].mean().to_numpy()
        return (_rho(premium_items.iloc[index].mean().to_numpy(), loss),
                _rho(accuracy.iloc[index].mean().to_numpy(), loss))

    point = statistics(np.arange(len(gap)))
    rng = np.random.default_rng(seed)
    draws = [statistics(rng.integers(len(gap), size=len(gap))) for _ in range(n_resamples)]
    result = {}
    excludes_zero = []
    for position, name in enumerate(("token_premium", "fp16_accuracy")):
        values = np.array([d[position] if d[position] is not None else 0.0 for d in draws])
        low, high = np.quantile(values, [0.025, 0.975])
        result[name] = {"rho": point[position], "low": float(low), "high": float(high)}
        excludes_zero.append(bool(low > 0 or high < 0) and point[position] is not None)
    result["supported"] = any(excludes_zero)
    return result


def h2(rows: pd.DataFrame, weight: str, kv: str, **kwargs):
    """Item-clustered weight × KV interaction tests for one model, task and seed."""
    cells = {(0, 0): ("fp16", "full"), (1, 0): (weight, "full"), (0, 1): ("fp16", kv),
             (1, 1): (weight, kv)}
    frames = []
    for (w, k), (level_w, level_k) in cells.items():
        cell = rows[(rows.weight == level_w) & (rows.kv == level_k) & ~rows.spec.astype(bool)]
        if cell.empty:
            raise ValueError(f"missing factorial cell {level_w}/{level_k}")
        frames.append(pd.DataFrame({"item_id": cell.item_id, "language": cell.lang,
                                    "correct": cell.correct.astype(int), "W": w, "K": k}))
    return interaction_tests(pd.concat(frames, ignore_index=True), **kwargs)


def h3(fp16: pd.DataFrame, treated: pd.DataFrame, *, n_resamples: int = 2000, seed: int = 0,
       reference: str = "en") -> pd.DataFrame:
    """English-versus-language Oaxaca on item degradation. Share is undefined for a zero gap.

    Covariates: the language's token count, item length (the English token count of the
    parallel item, so it carries no premium) and the FP16 gold log-probability.
    """
    tokens, logprob = _panel(fp16, "input_tokens"), _panel(fp16, "gold_logprob")
    loss = _panel(fp16) - _panel(treated)
    if reference not in loss:
        raise ValueError("reference language is missing")
    length = tokens[reference].to_numpy()

    def design(lang):
        return np.column_stack([tokens[lang].to_numpy(), length, logprob[lang].to_numpy()])

    rows = []
    for lang in loss.columns:
        if lang == reference:
            continue
        parts = oaxaca(design(reference), loss[reference].to_numpy(), design(lang),
                       loss[lang].to_numpy(), paired=True, n_resamples=n_resamples, seed=seed)
        gap, endowment = parts["gap"]["estimate"], parts["endowment"]["estimate"]
        share = endowment / gap if abs(gap) > 1e-12 else float("nan")
        rows.append({"lang": lang, "gap": gap, "gap_low": parts["gap"]["low"],
                     "gap_high": parts["gap"]["high"], "endowment": endowment,
                     "endowment_low": parts["endowment"]["low"],
                     "endowment_high": parts["endowment"]["high"],
                     "treatment": parts["treatment"]["estimate"],
                     "treatment_low": parts["treatment"]["low"],
                     "treatment_high": parts["treatment"]["high"], "endowment_share": share})
    return pd.DataFrame(rows)


def h3_supported(table: pd.DataFrame) -> bool:
    """Endowment below half of the gap for at least 3 non-Latin-script languages."""
    below = [lang for lang, share in zip(table.lang, table.endowment_share, strict=True)
             if LANGUAGES[lang].script != "Latin" and share == share and share < 0.5]
    return len(below) >= 3


def h4_budget(rows: pd.DataFrame, *, n_resamples: int = 2000, seed: int = 0,
              baselines: tuple[str, ...] | None = None) -> dict:
    """One budget: minimax against the best baseline, with item-paired percentile intervals.

    `rows` holds item_id, lang, correct and method, including `fp16`, `minimax`, and `average`.
    The best baseline is reselected inside each resample, so its choice is not treated as known.
    """
    methods = {m: _panel(g).to_numpy() for m, g in rows.groupby("method")}
    for required in ("fp16", "minimax", "average"):
        if required not in methods:
            raise ValueError(f"H4 needs the {required} method")
    names = list(baselines or [m for m in methods if m not in ("fp16", "minimax")])
    if len({v.shape for v in methods.values()}) != 1:
        raise ValueError("H4 methods must cover identical items and languages")

    def estimate(index):
        reference = methods["fp16"][index].mean(axis=0)
        worst = {m: float((reference - methods[m][index].mean(axis=0)).max())
                 for m in ["minimax", *names]}
        mean = {m: float(methods[m][index].mean()) for m in ["minimax", "average", *names]}
        best = min(names, key=lambda m: worst[m])
        return (worst[best] - worst["minimax"], mean[best] - mean["minimax"],
                mean["average"] - mean["minimax"], best, worst["minimax"], worst[best])

    n_items = methods["fp16"].shape[0]
    point = estimate(np.arange(n_items))
    rng = np.random.default_rng(seed)
    draws = [estimate(rng.integers(n_items, size=n_items))[:3] for _ in range(n_resamples)]
    low, high = np.quantile(np.array(draws), [0.025, 0.975], axis=0)
    return {"improvement": point[0], "improvement_low": float(low[0]),
            "improvement_high": float(high[0]), "mean_cost": point[1],
            "mean_cost_low": float(low[1]), "mean_cost_high": float(high[1]),
            "price_of_parity": point[2], "price_low": float(low[2]), "price_high": float(high[2]),
            "best_baseline": point[3], "worst_minimax": point[4], "worst_baseline": point[5]}


def k3_verdict(budgets: list[dict]) -> str:
    """H4 holds unless the CI includes 0, or the mean cost is above 1 point, at 2+ budgets."""
    failing = sum(not (b["improvement_low"] > 0) or b["mean_cost"] > MEAN_COST_LIMIT
                  for b in budgets)
    return "NEGATIVE_RESULT" if failing >= 2 else "POSITIVE"


def language_config_table(rows: pd.DataFrame, *, n_resamples: int = 2000, seed: int = 0,
                          q: float = 0.05) -> pd.DataFrame:
    """Δ accuracy against FP16 per language × config, paired CI, McNemar p, BH over the family."""
    factors = [c for c in ("model", "task", "weight", "kv", "calib", "seed") if c in rows]
    plain = rows[~rows.spec.astype(bool)]
    baseline = plain[(plain.weight == "fp16") & (plain.kv == "full")]
    records = []
    for config, group in plain.groupby(factors, sort=True):
        settings = dict(zip(factors, config, strict=True))
        if settings["weight"] == "fp16" and settings["kv"] == "full":
            continue
        reference = baseline[(baseline.model == settings["model"])
                             & (baseline.task == settings["task"])]
        ci = paired_frame_bootstrap(reference, group, language="lang", n_resamples=n_resamples,
                                    seed=seed)
        a, b = _panel(reference), _panel(group)
        pvalues = paired_binary_pvalues(a.to_numpy().astype(int), b.to_numpy().astype(int))
        for i, lang in enumerate(a.columns):
            records.append({**settings, "lang": lang, "delta": float(ci.estimate[i]),
                            "low": float(ci.low[i]), "high": float(ci.high[i]),
                            "pvalue": float(pvalues[i])})
    table = pd.DataFrame(records)
    if table.empty:
        return table
    table["p_bh"] = benjamini_hochberg(table.pvalue.to_numpy())
    table["reject"] = table.p_bh <= q
    return table


def attribution(rows: pd.DataFrame, lang: str, weight: str, kv: str, metric: str = "correct"):
    """Shapley contribution of weight, KV and speculation over one eight-cell sub-grid."""
    players = {"W": ("weight", weight, "fp16"), "K": ("kv", kv, "full"), "S": ("spec", True, False)}
    frame = rows[rows.lang == lang]
    values = {}
    for size in range(4):
        for coalition in combinations(players, size):
            cell = frame
            for name, (column, on, off) in players.items():
                cell = cell[cell[column] == (on if name in coalition else off)]
            if cell.empty:
                raise ValueError(f"missing factorial cell {coalition}")
            values[frozenset(coalition)] = float(cell[metric].astype(float).mean())
    return shapley_values(values)
