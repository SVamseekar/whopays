"""Turn a stored profile into ParityQuant bit maps, job lists and the H4 comparison.

Four methods per budget: minimax and average at multilingual calibration, and the
sensitivity-free uniform comparator at English and at multilingual calibration. Uniform means
one width when the budget is an allowed width, otherwise `matched_uniform`.

Sixth amendment: sensitivities are in nats per parallel sentence, every bit map is scored on
Belebele and on the FLORES devtest sentences the probe did not use, H4's worst-language
degradation is the held-out NLL increase per sentence, and its mean cost is Belebele accuracy.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from whopays.config import ALLOWED_BITS, canonical_json
from whopays.grids import write_jobs
from whopays.parityquant.probe import PROBE_LIMIT, solve, to_sentence_units
from whopays.report.hypotheses import h4_budget, k3_verdict

BUDGETS = (3.0, 3.5, 4.0)


def _methods(budget: float) -> list[tuple[str, str, str]]:
    uniform = "uniform" if budget in ALLOWED_BITS else "matched_uniform"
    return [("minimax", "minimax", "multi"), ("average", "average", "multi"),
            ("uniform_en", uniform, "en"), ("uniform_multi", uniform, "multi")]


def build_plan(rows, model: str, *, budgets: Sequence[float] = BUDGETS,
               seeds: Sequence[int] = (0, 1, 2), task: str = "belebele",
               tokens_per_sentence: dict[str, float] | None = None,
               held_out_offset: int = PROBE_LIMIT) -> dict:
    """Allocations and jobs. Minimax replicates over `seeds`; other methods use seed 0.

    With `tokens_per_sentence` (keyed like the profile's `lang`), sensitivities are converted to
    nats per sentence before solving. Each bit map also gets a FLORES NLL job on the devtest
    items after `held_out_offset`, the sentences the profile did not see.
    """
    rows = list(rows)
    unit = "nats_per_token"
    if tokens_per_sentence is not None:
        rows, unit = to_sentence_units(rows, tokens_per_sentence), "nats_per_sentence"
    allocations, jobs = [], []
    for budget in budgets:
        for method, objective, calib in _methods(budget):
            result = solve(rows, budget, objective)
            allocations.append({"method": method, "objective": objective, "calib": calib,
                                "budget": budget, "average_bits": result.average_bits,
                                "bit_map": result.bit_map,
                                "language_losses": list(result.language_losses),
                                "unit": unit})
            for seed in (seeds if method == "minimax" else (0,)):
                common = {"model": model, "lang": "all", "weight": "bitmap", "calib": calib,
                          "seed": seed, "bitmap": result.bit_map, "budget": budget}
                jobs.append({**common, "task": task})
                jobs.append({**common, "task": "flores_nll", "offset": held_out_offset})
    return {"allocations": allocations, "jobs": jobs}


def language_increases(rows: pd.DataFrame, allocations: Sequence[dict], model: str, *,
                       task: str = "flores_nll") -> pd.DataFrame:
    """Mean held-out NLL increase per sentence against FP16, per budget, method and language.

    Long format (budget, method, lang, increase, n_items) for `whopays langtable`.
    """
    selected = rows[(rows.model == model) & (rows.task == task)]
    records = []
    for budget in sorted({a["budget"] for a in allocations}):
        labelled = label_methods(selected, allocations, budget)
        held_out = set(labelled.loc[labelled.method != "fp16", "item_id"])
        labelled = labelled[labelled.item_id.isin(held_out)]
        fp16 = labelled[labelled.method == "fp16"].set_index(["lang", "item_id"]).nll_sum
        for method, group in labelled[labelled.method != "fp16"].groupby("method"):
            delta = group.set_index(["lang", "item_id"]).nll_sum - fp16.reindex(
                group.set_index(["lang", "item_id"]).index)
            for lang, values in delta.groupby(level="lang"):
                records.append({"budget": budget, "method": method, "lang": lang,
                                "increase": float(values.mean()), "n_items": int(len(values))})
    return pd.DataFrame(records)


def bits_to_parity(worst: pd.DataFrame, *, method: str = "minimax",
                   baseline: str = "uniform_multi") -> pd.DataFrame:
    """Extra uniform bits needed to match `method`'s worst-language degradation at each budget.

    `worst` has budget, method and worst columns. The uniform curve is interpolated linearly
    between measured budgets; a target outside the measured range gives NaN rather than an
    extrapolation.
    """
    import numpy as np

    curve = worst[worst.method == baseline].sort_values("budget")
    bits, values = curve.budget.to_numpy(float), curve.worst.to_numpy(float)
    records = []
    for budget, target in worst[worst.method == method][["budget", "worst"]].itertuples(
            index=False):
        match = np.nan
        for i in range(len(bits) - 1):
            low, high = sorted((values[i], values[i + 1]))
            if low <= target <= high and values[i] != values[i + 1]:
                frac = (values[i] - target) / (values[i] - values[i + 1])
                match = bits[i] + frac * (bits[i + 1] - bits[i])
                break
        records.append({"budget": budget, "uniform_bits_needed": match,
                        "extra_bits": match - budget})
    return pd.DataFrame(records)


def compare_units(rows, tokens_per_sentence: dict[str, float], *,
                  budgets: Sequence[float] = BUDGETS, objective: str = "minimax") -> pd.DataFrame:
    """CPU robustness check: how the unit of the sensitivities changes the solved bit maps.

    Solves under per-sentence (the sixth amendment's unit), per-token and relative (divided by
    the FP16 per-token NLL) units, and reports for each the share of parameters whose width
    matches the per-sentence map and the language it ranks worst, judged in per-sentence units.
    """
    rows = list(rows)
    units = {"sentence": to_sentence_units(rows, tokens_per_sentence), "token": rows,
             "relative": [{**r, "sensitivity": r["sensitivity"] / r["baseline_nll"]}
                          for r in rows]}
    sentence_rows = units["sentence"]
    records = []
    for budget in budgets:
        reference = solve(sentence_rows, budget, objective)
        for unit, unit_rows in units.items():
            result = solve(unit_rows, budget, objective)
            params = {r["layer"]: r["parameters"] for r in rows}
            total = sum(params.values())
            same = sum(params[layer] for layer, bits in result.bit_map.items()
                       if reference.bit_map[layer] == bits)
            losses = _sentence_losses(sentence_rows, result.bit_map)
            records.append({"budget": budget, "unit": unit, "same_width_share": same / total,
                            "worst_lang": max(losses, key=losses.get),
                            "worst_nats_per_sentence": max(losses.values())})
    return pd.DataFrame(records)


def _sentence_losses(sentence_rows, bit_map) -> dict[str, float]:
    """Additive per-sentence loss of a bit map, per language (halves averaged)."""
    from whopays.parityquant.probe import merge_halves

    losses: dict[str, float] = {}
    for row in merge_halves(sentence_rows):
        if bit_map[row["layer"]] == row["bits"]:
            losses[row["lang"]] = losses.get(row["lang"], 0.0) + row["sensitivity"]
    return losses


def write_plan(plan: dict, directory, name: str) -> tuple[Path, Path]:
    directory = Path(directory)
    jobs = write_jobs(plan["jobs"], directory / f"parityquant_{name}.yaml")
    allocations = directory / f"parityquant_{name}.allocations.json"
    allocations.write_text(json.dumps(plan["allocations"], indent=2, sort_keys=True) + "\n")
    return jobs, allocations


def label_methods(rows: pd.DataFrame, allocations: Sequence[dict], budget: float) -> pd.DataFrame:
    """Rows of one budget at seed 0, tagged by allocation, plus the FP16 reference."""
    frames = []
    fp16 = rows[rows.weight == "fp16"]
    for column, value in (("kv", "full"), ("spec", False)):
        if column in fp16:
            fp16 = fp16[fp16[column] == value]
    frames.append(fp16.assign(method="fp16"))
    for allocation in allocations:
        if allocation["budget"] != budget:
            continue
        match = rows[(rows.weight == "bitmap") & (rows.budget == budget) & (rows.seed == 0)
                     & (rows.calib == allocation["calib"])
                     & (rows.bitmap == canonical_json(allocation["bit_map"]))]
        frames.append(match.assign(method=allocation["method"]))
    return pd.concat(frames, ignore_index=True)


def parity_report(rows: pd.DataFrame, allocations: Sequence[dict], model: str, *,
                  primary: str = "flores_nll", accuracy_task: str = "belebele",
                  n_resamples: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, str]:
    """One H4 row per budget, and the K3 verdict over all budgets.

    `primary="flores_nll"` (sixth amendment): improvement and worst-language figures come from
    the held-out FLORES NLL increase per sentence (only items present in the bit-map runs, so the
    FP16 rows for probe sentences are dropped); `mean_cost` is `accuracy_task` mean accuracy of
    that primary best baseline minus minimax's. The accuracy task's own worst-language figures,
    best baseline reselected among all, are kept with an `acc_` prefix. `primary="belebele"`
    is the original protocol, accuracy only.
    """
    def select(task):
        return rows[(rows.model == model) & (rows.task == task)]

    accuracy_rows = select(accuracy_task)
    records = []
    for budget in sorted({a["budget"] for a in allocations}):
        accuracy_labelled = label_methods(accuracy_rows, allocations, budget)
        accuracy = h4_budget(accuracy_labelled, n_resamples=n_resamples, seed=seed)
        if primary == accuracy_task:
            records.append({"budget": budget, **accuracy})
            continue
        labelled = label_methods(select(primary), allocations, budget)
        held_out = set(labelled.loc[labelled.method != "fp16", "item_id"])
        labelled = labelled[labelled.item_id.isin(held_out)]
        nll = h4_budget(labelled, n_resamples=n_resamples, seed=seed, column="nll_sum",
                        higher_is_better=False)
        # Mean accuracy cost against the baseline minimax beat on the primary measure.
        paired = h4_budget(accuracy_labelled, n_resamples=n_resamples, seed=seed,
                           baselines=(nll["best_baseline"],))
        cost = {k: paired[k] for k in ("mean_cost", "mean_cost_low", "mean_cost_high")}
        records.append({"budget": budget, **nll,
                        "nll_mean_cost": nll["mean_cost"], **cost,
                        **{f"acc_{k}": v for k, v in accuracy.items()}})
    return pd.DataFrame(records), k3_verdict(records)
