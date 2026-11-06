"""Job lists for each experiment, built in code so notebooks stay one-liners.

Each function returns plain dicts for `whopays.run()`. `lang: "all"` expands to the 11
languages there.
"""

from __future__ import annotations

from collections.abc import Sequence

QWEN = {size: f"Qwen/Qwen2.5-{size}-Instruct" for size in ("0.5B", "1.5B", "3B", "7B")}
K1_MODEL = QWEN["1.5B"]
M0_MODELS = (QWEN["1.5B"], QWEN["0.5B"], QWEN["3B"])  # K1 model first: it gates everything
M0_TASKS = ("belebele", "mgsm", "flores_nll")


def m0_fp16(models: Sequence[str] = M0_MODELS, tasks: Sequence[str] = M0_TASKS,
            limit: int | None = None) -> list[dict]:
    """FP16 baselines (BUILD_PLAN M0): every model × task × language, full cache, no draft."""
    return [{"model": m, "task": t, "lang": "all", "weight": "fp16", "kv": "full",
             "limit": limit} for m in models for t in tasks]


def _weight_cells():
    yield {"weight": "fp16"}
    yield {"weight": "w8_rtn"}
    for weight in ("w4_gptq", "w3_gptq"):
        for calib in ("en", "multi"):
            for seed in (0, 1, 2):
                yield {"weight": weight, "calib": calib, "seed": seed}


def _primary_cells():
    """The PRD's 4 weight levels, without calibration or seed replication."""
    yield {"weight": "fp16"}
    yield {"weight": "w8_rtn"}
    for weight in ("w4_gptq", "w3_gptq"):
        yield {"weight": weight, "calib": "multi", "seed": 0}


def replication_grid(models: Sequence[str] = (QWEN["1.5B"],),
                     limit: int | None = None) -> list[dict]:
    """Uniform replication, including paired FP16 reference runs."""
    return [{"model": m, "task": t, "lang": "all", "limit": limit, **w}
            for m in models for w in _weight_cells() for t in ("belebele", "flores_nll")]


def audit_grid(models: Sequence[str] = (QWEN["1.5B"], QWEN["3B"]),
               tasks: Sequence[str] = ("belebele", "mgsm"),
               limit: int | None = None,
               replicate: Sequence[str] = (QWEN["1.5B"],)) -> list[dict]:
    """12 quality cells per model and task.

    Calibration language and seed replication run on Belebele only (second amendment) and
    only for models in `replicate` (third amendment, 2026-11-06: 1.5B).
    """
    jobs = []
    for m in models:
        for t in tasks:
            cells = (_weight_cells() if t == "belebele" and m in replicate
                     else _primary_cells())
            jobs += [{"model": m, "task": t, "lang": "all", "kv": kv, "limit": limit, **w}
                     for w in cells for kv in ("full", "quant4", "evict")]
    return jobs


def speed_grid(models: Sequence[str] = (QWEN["1.5B"],)) -> list[dict]:
    """Paired first 30 MGSM items per language: 12 cells × draft off/on = the PRD's 24."""
    return [{"model": m, "task": "mgsm", "lang": "all", "limit": 30, "kv": kv, "spec": spec,
             "batch_size": 1, **w}
            for m in models for w in _primary_cells() for kv in ("full", "quant4", "evict")
            for spec in (False, True)]


def speed_subset(models: Sequence[str] = (QWEN["3B"],)) -> list[dict]:
    """The second model's 4 speed cells: FP16 and W4, full cache, draft off/on."""
    return [{"model": m, "task": "mgsm", "lang": "all", "limit": 30, "kv": "full", "spec": spec,
             "batch_size": 1, **w}
            for m in models for w in _primary_cells() if w["weight"] in ("fp16", "w4_gptq")
            for spec in (False, True)]


def scale_grid(models: Sequence[str] = (QWEN["7B"],)) -> list[dict]:
    """7B confirmation (exploratory): FP16 and W4 on Belebele and the first 30 MGSM items."""
    return [{"model": m, "task": t, "lang": "all", "limit": limit, **w}
            for m in models for w in _primary_cells() if w["weight"] in ("fp16", "w4_gptq")
            for t, limit in (("belebele", None), ("mgsm", 30))]


FULL_ITEMS_PER_LANGUAGE = {"belebele": 900, "mgsm": 250, "flores_nll": 1012}


def measured_rates(rows) -> dict[tuple, float]:
    """Seconds per item by (model, task), or (model, task, "spec") for speculative rows."""
    frame = rows.copy()
    spec = frame["spec"].fillna(False).astype(bool) if "spec" in frame else False
    frame["_spec"] = spec
    rates = {}
    for (model, task, is_spec), group in frame.groupby(["model", "task", "_spec"]):
        key = (model, task, "spec") if is_spec else (model, task)
        rates[key] = float(group["wall_time_s"].sum() / len(group))
    return rates


def project_costs(jobs, rates: dict[tuple, float], *, assume: dict | None = None,
                  load_seconds: dict[str, float] | None = None) -> dict[str, float]:
    """Projected seconds per run from measured rates, refusing unmeasured combinations.

    `assume` maps an unmeasured model to (measured model, factor) and is an explicit planning
    assumption. `load_seconds` by weight is charged once per loaded model.
    """
    from whopays.config import load_jobs

    assume, load_seconds = assume or {}, load_seconds or {}
    costs: dict[str, float] = {}
    loaded: set[str] = set()
    for job in load_jobs(jobs):
        base, factor = assume.get(job.model, (job.model, 1.0))
        key = (base, job.task, "spec") if job.spec else (base, job.task)
        if key not in rates:
            raise ValueError(f"no measured rate for {key}")
        items = job.limit or FULL_ITEMS_PER_LANGUAGE[job.task]
        cost = items * rates[key] * factor
        if job.model_key() not in loaded:
            loaded.add(job.model_key())
            cost += load_seconds.get(job.weight, 0.0)
        costs[job.run_id] = cost
    return costs


def compute_plan(jobs, costs: dict[str, float]):
    """GPU-hours by model and task, with a TOTAL row, for the M0 re-plan."""
    import pandas as pd

    from whopays.config import load_jobs

    frame = pd.DataFrame([{"model": j.model, "task": j.task, "hours": costs[j.run_id] / 3600}
                          for j in load_jobs(jobs)])
    table = frame.groupby(["model", "task"], as_index=False).agg(runs=("hours", "size"),
                                                                 hours=("hours", "sum"))
    total = pd.DataFrame([{"model": "TOTAL", "task": "", "runs": int(table.runs.sum()),
                           "hours": float(table.hours.sum())}])
    return pd.concat([table, total], ignore_index=True)


def slice_jobs(jobs, seconds_by_run: dict[str, float], max_hours: float = 10) -> list[list]:
    """Pack jobs using measured/projected per-run seconds, refusing unknown costs.

    Include calibration/loading overhead in supplied costs. A single oversize run
    must be resumed across sessions; it cannot be represented as a bounded slice.
    """
    import math

    from whopays.config import load_jobs
    from whopays.run import group_by_model

    if not math.isfinite(max_hours) or not 0 < max_hours <= 10:
        raise ValueError("max_hours must be positive and at most 10")
    result, current, elapsed = [], [], 0.0
    for job in group_by_model(load_jobs(jobs)):
        if job.run_id not in seconds_by_run:
            raise ValueError(f"missing measured cost for {job.run_id}")
        cost = seconds_by_run[job.run_id]
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError("costs must be positive finite seconds")
        if cost > max_hours * 3600:
            raise ValueError(f"run {job.run_id} exceeds the session budget")
        if current and elapsed + cost > max_hours * 3600:
            result.append(current)
            current, elapsed = [], 0.0
        current.append(job)
        elapsed += cost
    if current:
        result.append(current)
    return result


def write_jobs(jobs, path):
    """Write a reproducible YAML job list; run hashes are independent of the path."""
    from pathlib import Path

    import yaml

    from whopays.config import load_jobs

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"jobs": [j.model_dump(mode="json")
                                            for j in load_jobs(jobs)]}, sort_keys=True))
    return path


def write_slices(jobs, seconds_by_run: dict[str, float], directory, name: str,
                 max_hours: float = 10) -> list:
    """Write <name>_slice_NN.yaml job lists, each within one Kaggle session's budget."""
    from pathlib import Path

    return [write_jobs(part, Path(directory) / f"{name}_slice_{i:02d}.yaml")
            for i, part in enumerate(slice_jobs(jobs, seconds_by_run, max_hours))]
