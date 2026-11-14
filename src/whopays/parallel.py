"""Run one job list on several GPUs at once: one worker process per GPU.

Kaggle's "GPU T4 x2" session gives two T4s, but `run()` uses one device. Here the job list is
split into shards of whole model groups (a group shares one set of weights, so quantization
still happens once), each shard goes to a `whopays.cli run` process pinned to one GPU with
`CUDA_VISIBLE_DEVICES`, and the processes run side by side.

Numerics do not change: each worker runs the same code on one T4 with the same batch sizes,
so outputs and run IDs are the same as in a single-process run. Each run folder is written
by exactly one worker. Speed measurements (notebook 05) must not use this, because workers
would share the host CPU and PCIe bus.

Shards are balanced by a rough cost model (minutes on a T4) taken from the logged
notebook-01 throughput (DECISIONS 2026-11-06) plus a GPTQ quantization allowance. The model
only has to rank groups; it never feeds a result.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path

import yaml

from whopays.config import RunConfig, load_jobs
from whopays.run import group_by_model

# FP16 GPU-minutes per language for 1.5B on a T4 (notebook 01: Belebele 18.7, MGSM 68.1,
# FLORES 7.2 minutes over 11 languages). 3B counts double, as in the PRD compute plan.
MINUTES_PER_LANG = {"belebele": 18.7 / 11, "mgsm": 68.1 / 11, "flores_nll": 7.2 / 11}
FULL_ITEMS = {"belebele": 900, "mgsm": 250, "flores_nll": 1012}
QUANT_MINUTES = {"w8_rtn": 1.0, "fp16": 0.0}  # anything else (GPTQ, bit maps): GPTQ_MINUTES
GPTQ_MINUTES = 20.0  # 128 x 2,048-token calibration on 1.5B; rough, from the 2026-11-07 run


def _scale(model: str) -> float:
    return 2.0 if "3B" in model else 4.0 if "7B" in model else 1.0


def job_cost(cfg: RunConfig) -> float:
    per_lang = MINUTES_PER_LANG.get(cfg.task, 5.0)
    frac = min(1.0, cfg.limit / FULL_ITEMS[cfg.task]) if cfg.limit and cfg.task in FULL_ITEMS \
        else 1.0
    return per_lang * frac * _scale(cfg.model)


def group_costs(jobs: Iterable[RunConfig]) -> dict[str, float]:
    """Projected minutes per model group: quantize once, then every job of the group."""
    costs: dict[str, float] = {}
    for cfg in jobs:
        key = cfg.model_key()
        if key not in costs:
            costs[key] = QUANT_MINUTES.get(cfg.weight, GPTQ_MINUTES) * _scale(cfg.model)
        costs[key] += job_cost(cfg)
    return costs


def shard_jobs(jobs: Iterable[RunConfig], n: int) -> list[list[RunConfig]]:
    """Split into at most `n` shards of whole model groups, largest group first (LPT)."""
    jobs = group_by_model(list(jobs))
    costs = group_costs(jobs)
    n = max(1, min(n, len(costs)))
    load = [0.0] * n
    owner: dict[str, int] = {}
    for key in sorted(costs, key=costs.get, reverse=True):
        i = load.index(min(load))
        owner[key] = i
        load[i] += costs[key]
    shards: list[list[RunConfig]] = [[] for _ in range(n)]
    for cfg in jobs:
        shards[owner[cfg.model_key()]].append(cfg)
    return [s for s in shards if s]


def chunk_jobs(jobs: Iterable[RunConfig], target_minutes: float = 60.0) -> list[list[RunConfig]]:
    """Cut each model group into chunks of about `target_minutes` of evaluation, for many GPUs.

    Each chunk loads (and, for GPTQ, quantizes) its weights again. That costs minutes but lets
    one large group, such as 3B FP16 with three KV policies and MGSM, spread over many GPUs.
    Quantization is deterministic for a given model key, calibration and software stack, and
    every run records a weights digest so that can be checked afterwards.
    """
    chunks: list[list[RunConfig]] = []
    current: list[RunConfig] = []
    work = 0.0
    key = None
    for cfg in group_by_model(list(jobs)):
        cost = job_cost(cfg)
        if current and (cfg.model_key() != key or work + cost > target_minutes):
            chunks.append(current)
            current, work = [], 0.0
        current.append(cfg)
        work += cost
        key = cfg.model_key()
    if current:
        chunks.append(current)
    return chunks


def visible_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def worker_count(jobs: list[RunConfig], gpus: int | None = None) -> int:
    """One worker per GPU, except models that need both GPUs (7B is split across a T4 x2)."""
    gpus = visible_gpus() if gpus is None else gpus
    if gpus < 2 or any("7B" in c.model for c in jobs):
        return 1
    return gpus


def _pump(stream, prefix: str) -> None:
    for line in stream:
        print(f"{prefix}{line}", end="", flush=True)


def run_parallel(jobs, *, gpus: int | None = None, root: str | Path | None = None,
                 sync: bool = True, workdir: str | Path | None = None,
                 check: bool = True, push_interval: float = 900.0) -> list[int]:
    """Run `jobs` across GPUs; returns worker exit codes.

    With `sync`, workers pull but never push; this process commits finished runs in bulk every
    `push_interval` seconds from a background thread, and once more at the end. One commit per
    run from two workers hit HTTP 429 about every 1.7 h on the 1.5B ParityQuant grid, and each
    429 slept a worker for an hour (RUN_LOG 16).

    A worker exits non-zero when any of its jobs failed. With `check` that raises; without it,
    callers follow with an in-process `whopays.run(jobs)`, which skips finished runs, retries
    failed ones and returns the per-job status table.

    With one worker this still runs in a subprocess, so the notebook kernel never holds a
    model and a CUDA error cannot poison it.
    """
    configs = load_jobs(jobs)
    shards = shard_jobs(configs, worker_count(configs, gpus))
    workdir = Path(workdir or tempfile.mkdtemp(prefix="whopays-shards-"))
    workdir.mkdir(parents=True, exist_ok=True)
    pusher = None
    if sync:
        from whopays.env import results_root
        from whopays.io.hub_sync import BulkPush

        pusher = BulkPush(Path(root) if root is not None else results_root(),
                          [c.run_id for c in configs], push_interval).start()
    procs, pumps = [], []
    for i, shard in enumerate(shards):
        path = workdir / f"shard_{i}.yaml"
        path.write_text(yaml.safe_dump({"jobs": [c.model_dump(mode="json") for c in shard]},
                                       sort_keys=True, allow_unicode=True))
        cmd = [sys.executable, "-m", "whopays.cli", "run", str(path)]
        if root is not None:
            cmd += ["--root", str(root)]
        cmd.append("--no-push" if sync else "--no-sync")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i), "PYTHONUNBUFFERED": "1"}
        print(f"worker {i}: GPU {i}, {len(shard)} jobs, "
              f"{len({c.model_key() for c in shard})} model groups")
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        pump = threading.Thread(target=_pump, args=(proc.stdout, f"[gpu{i}] "), daemon=True)
        pump.start()
        procs.append(proc)
        pumps.append(pump)
    codes = [p.wait() for p in procs]
    for pump in pumps:
        pump.join(timeout=5)
    if pusher is not None:
        pusher.close()
    failed = [i for i, code in enumerate(codes) if code != 0]
    if failed and check:
        raise RuntimeError(f"worker {', '.join(map(str, failed))} exited non-zero; "
                           "see its [gpuN] lines above. Re-run to resume.")
    return codes
