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


def replication_grid(models: Sequence[str] = (QWEN["1.5B"],),
                     limit: int | None = None) -> list[dict]:
    """Uniform replication, including paired FP16 reference runs."""
    return [{"model": m, "task": t, "lang": "all", "limit": limit, **w}
            for m in models for w in _weight_cells() for t in ("belebele", "flores_nll")]


def audit_grid(models: Sequence[str] = (QWEN["1.5B"], QWEN["3B"]),
               tasks: Sequence[str] = ("belebele", "mgsm"),
               limit: int | None = None) -> list[dict]:
    """12 factorial cells, with calibration/seed replication for GPTQ cells."""
    return [{"model": m, "task": t, "lang": "all", "kv": kv, "limit": limit, **w}
            for m in models for w in _weight_cells() for kv in ("full", "quant4", "evict")
            for t in tasks]


def speed_grid(models: Sequence[str] = (QWEN["1.5B"],)) -> list[dict]:
    """Paired first 30 MGSM items per language, with/without draft verification.

    The full primary-model grid is emitted; second-model confirmation can select
    a preregistered subset rather than silently expanding the compute budget.
    """
    return [{**j, "spec": spec, "batch_size": 1}
            for j in audit_grid(models, tasks=("mgsm",), limit=30) for spec in (False, True)]


def slice_jobs(jobs, seconds_by_run: dict[str, float], max_hours: float = 10) -> list[list]:
    """Pack jobs using measured/projected per-run seconds, refusing unknown costs.

    Include calibration/loading overhead in supplied costs. A single oversize run
    must be resumed across sessions; it cannot be represented as a bounded slice.
    """
    import math

    from whopays.config import load_jobs

    if not math.isfinite(max_hours) or not 0 < max_hours <= 10:
        raise ValueError("max_hours must be positive and at most 10")
    result, current, elapsed = [], [], 0.0
    for job in load_jobs(jobs):
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
