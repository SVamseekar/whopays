"""Turn a stored profile into ParityQuant bit maps, job lists and the H4 comparison.

Four methods per budget: minimax and average at multilingual calibration, and the
sensitivity-free uniform comparator at English and at multilingual calibration. Uniform means
one width when the budget is an allowed width, otherwise `matched_uniform`.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from whopays.config import ALLOWED_BITS, canonical_json
from whopays.grids import write_jobs
from whopays.parityquant.probe import solve
from whopays.report.hypotheses import h4_budget, k3_verdict

BUDGETS = (3.0, 3.5, 4.0)


def _methods(budget: float) -> list[tuple[str, str, str]]:
    uniform = "uniform" if budget in ALLOWED_BITS else "matched_uniform"
    return [("minimax", "minimax", "multi"), ("average", "average", "multi"),
            ("uniform_en", uniform, "en"), ("uniform_multi", uniform, "multi")]


def build_plan(rows, model: str, *, budgets: Sequence[float] = BUDGETS,
               seeds: Sequence[int] = (0, 1, 2), task: str = "belebele") -> dict:
    """Allocations and jobs. Minimax replicates over `seeds`; other methods use seed 0."""
    rows = list(rows)
    allocations, jobs = [], []
    for budget in budgets:
        for method, objective, calib in _methods(budget):
            result = solve(rows, budget, objective)
            allocations.append({"method": method, "objective": objective, "calib": calib,
                                "budget": budget, "average_bits": result.average_bits,
                                "bit_map": result.bit_map,
                                "language_losses": list(result.language_losses)})
            for seed in (seeds if method == "minimax" else (0,)):
                jobs.append({"model": model, "task": task, "lang": "all", "weight": "bitmap",
                             "calib": calib, "seed": seed, "bitmap": result.bit_map,
                             "budget": budget})
    return {"allocations": allocations, "jobs": jobs}


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
                  task: str = "belebele", n_resamples: int = 2000,
                  seed: int = 0) -> tuple[pd.DataFrame, str]:
    """One H4 row per budget, and the K3 verdict over all budgets."""
    selected = rows[(rows.model == model) & (rows.task == task)]
    records = []
    for budget in sorted({a["budget"] for a in allocations}):
        labelled = label_methods(selected, allocations, budget)
        records.append({"budget": budget, **h4_budget(labelled, n_resamples=n_resamples,
                                                      seed=seed)})
    return pd.DataFrame(records), k3_verdict(records)
