"""`run(jobs)`: the only entry point notebooks call.

For each job: restore its folder from the hub if missing locally, skip it if DONE, otherwise
load the model (reused across consecutive jobs with the same weights), stream task rows into
the resumable writer, compact to Parquet, mark DONE and push. A failing job is recorded and
the next one starts, so one bad cell does not waste a 12-hour background session.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from whopays.config import RunConfig, canonical_json, load_jobs
from whopays.env import env_info, results_root
from whopays.io import hub_sync
from whopays.io.results import ITEMS, ResultsWriter, is_done, mark_done, to_parquet
from whopays.tasks import get_task

log = logging.getLogger(__name__)

Loader = Callable[[RunConfig], Any]
# A run resumed under a different value of any of these would mix measurements in one file.
PINNED_ENV = ("code_version", "gpu", "torch", "transformers")
ROW_ENV = ("gpu", "torch", "transformers", "cuda", "code_version")


@dataclass
class JobResult:
    run_id: str
    task: str
    lang: str
    status: Literal["done", "skipped", "failed"]
    n_items: int = 0
    seconds: float = 0.0
    error: str | None = None


class _ModelCache:
    """Holds one loaded model; jobs are grouped so each model loads once."""

    def __init__(self, loader: Loader | None):
        self.loader = loader
        self.key: str | None = None
        self.bundle: Any = None

    def get(self, cfg: RunConfig) -> Any:
        key = cfg.model_key()
        if key != self.key:
            self.clear()
            if self.loader is None:
                from whopays.models.loader import load

                self.loader = load
            self.bundle = self.loader(cfg)
            self.key = key
        return self.bundle

    def clear(self) -> None:
        self.bundle, self.key = None, None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def group_by_model(configs: list[RunConfig]) -> list[RunConfig]:
    """Stable reorder so jobs sharing weights are adjacent, in order of first appearance."""
    order: dict[str, int] = {}
    for cfg in configs:
        order.setdefault(cfg.model_key(), len(order))
    return sorted(configs, key=lambda c: order[c.model_key()])


def _check_meta(run_dir: Path, allow_env_change: bool) -> None:
    """Record the environment of every (re)start; refuse to mix environments in one run."""
    path = run_dir / "meta.json"
    now = {**env_info(), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    meta = json.loads(path.read_text()) if path.exists() else {"starts": []}
    if meta["starts"] and not allow_env_change:
        first = meta["starts"][0]
        changed = {k: (first.get(k), now[k]) for k in PINNED_ENV if first.get(k) != now[k]}
        if changed:
            raise RuntimeError(
                f"run {run_dir.name} started under a different environment {changed}; "
                "resuming would mix measurements. Re-run with the original code/GPU, delete "
                "the folder, or pass allow_env_change=True."
            )
    meta["starts"].append(now)
    path.write_text(json.dumps(meta, indent=2) + "\n")


def _run_one(cfg: RunConfig, root: Path, models: _ModelCache, *, sync: bool, flush_every: int,
             sync_interval: float, allow_env_change: bool) -> JobResult:
    run_id = cfg.run_id
    run_dir = root / run_id
    pending_restore = root / f".restore-{run_id}"
    if pending_restore.exists() and not sync:
        raise RuntimeError("incomplete remote restore; retry with sync=True before running")
    if sync and (not run_dir.exists() or pending_restore.exists()):
        # Persist this before downloading: an interrupted partial download is not local truth.
        root.mkdir(parents=True, exist_ok=True)
        pending_restore.touch()
        restored = hub_sync.pull_runs(root, [run_id])
        if not restored and run_dir.exists():
            raise RuntimeError("remote progress is unknown after incomplete restore")
        pending_restore.unlink()
    if is_done(run_dir):
        if sync:
            hub_sync.ThrottledPush(run_dir, sync_interval)(force=True)
        n = json.loads((run_dir / "DONE").read_text())["n_items"]
        return JobResult(run_id, cfg.task, cfg.lang, "skipped", n_items=n)

    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        previous = RunConfig.from_yaml(config_path)
        if previous.canonical() != cfg.canonical():
            raise ValueError("run config differs from stored canonical configuration")
        if previous.batch_size != cfg.batch_size:
            raise ValueError("batch_size cannot change on resume: it affects FP16 outputs")
    elif (run_dir / ITEMS).exists():
        raise ValueError("cannot resume results without their original config.yaml")
    _check_meta(run_dir, allow_env_change)
    config_path.write_text(cfg.to_yaml())
    task = get_task(cfg.task)
    bundle = models.get(cfg)
    calibration = (getattr(bundle, "quantization", None) or {}).get("calibration")
    if calibration:  # released with the results (Williams & Aletras, 2024)
        (run_dir / "calibration.json").write_text(json.dumps(calibration, indent=1) + "\n")

    info = env_info()
    provenance = cfg.model_dump(mode="json", exclude={"tag"})
    provenance["bitmap"] = canonical_json(cfg.bitmap) if cfg.bitmap is not None else None
    provenance["config_json"] = canonical_json(cfg.canonical())
    stamp = {**provenance, "run_id": run_id, "task": cfg.task, "lang": cfg.lang,
             **{k: info[k] for k in ROW_ENV}}
    if calibration and "sha256" in calibration:
        stamp["calib_sha256"] = calibration["sha256"]
    push = hub_sync.ThrottledPush(run_dir, sync_interval) if sync else None
    with ResultsWriter(run_dir / ITEMS, flush_every, on_flush=push) as writer:
        for row in task(cfg, bundle, writer.done_ids()):
            writer.write({**row, **stamp})
    n_items = len(writer.done_ids())
    to_parquet(run_dir / ITEMS)
    mark_done(run_dir, n_items)
    if push:
        push(force=True)
    return JobResult(run_id, cfg.task, cfg.lang, "done", n_items=n_items)


def run(
    jobs: str | Path | Iterable[RunConfig | dict],
    *,
    root: str | Path | None = None,
    loader: Loader | None = None,
    sync: bool = True,
    flush_every: int = 50,
    sync_interval: float = 600.0,
    stop_on_error: bool = False,
    allow_env_change: bool = False,
) -> list[JobResult]:
    """Run a job list (YAML path, configs or dicts). Finished runs are skipped.

    Jobs are reordered so those sharing weights run back to back (quantizing is expensive).
    """
    configs = group_by_model(load_jobs(jobs))
    root = Path(root) if root is not None else results_root()
    models = _ModelCache(loader)
    results: list[JobResult] = []
    try:
        for cfg in configs:
            start = time.monotonic()
            try:
                res = _run_one(cfg, root, models, sync=sync, flush_every=flush_every,
                               sync_interval=sync_interval, allow_env_change=allow_env_change)
            except Exception as exc:
                if stop_on_error:
                    raise
                log.exception("job %s (%s/%s) failed", cfg.run_id, cfg.task, cfg.lang)
                res = JobResult(cfg.run_id, cfg.task, cfg.lang, "failed",
                                error=f"{type(exc).__name__}: {exc}")
            res.seconds = round(time.monotonic() - start, 1)
            results.append(res)
            print(f"[{res.status:>7}] {res.run_id} {res.task:<10} {res.lang} "
                  f"items={res.n_items} {res.seconds}s" + (f"  {res.error}" if res.error else ""))
    finally:
        models.clear()
    return results
