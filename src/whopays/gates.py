"""Execution guards for the PRD kill criteria: downstream runs need a logged K1 pass."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from whopays.io import hub_sync

log = logging.getLogger(__name__)


class GateError(RuntimeError):
    """A required research gate has no logged pass."""


def k1_table(root: Path) -> pd.DataFrame:
    """K1 verdict rebuilt from stored baselines and lm-eval samples; never a cached claim."""
    from whopays import crosscheck
    from whopays.config import load_jobs
    from whopays.grids import K1_MODEL, m0_fp16
    from whopays.io.results import load_runs

    root = Path(root)
    jobs = load_jobs(m0_fp16([K1_MODEL]))
    reference = root / f"lmeval-{K1_MODEL.split('/')[-1]}"
    wanted = [reference.name, *[job.run_id for job in jobs]]
    if any(not (root / name).exists() for name in wanted):
        hub_sync.pull_runs(root, wanted)
    samples = crosscheck.read_samples(reference) if reference.exists() else pd.DataFrame()
    if samples.empty:
        raise GateError("K1 has no lm-eval samples; run notebook 01 first")
    return crosscheck.k1_verdict(crosscheck.compare(load_runs(root, jobs), samples))


def require_k1(root: Path, override: str | None = None) -> bool:
    """Return True on a logged K1 pass. Otherwise raise, unless an override reason is given.

    An override never counts as a pass: it is written to gate_overrides.jsonl and must not be
    used for results. It exists for small debugging slices.
    """
    root = Path(root)
    try:
        table = k1_table(root)
        passed = set(table.K1) == {"PASS"}
        detail = table.to_string(index=False)
    except GateError as exc:
        passed, detail = False, str(exc)
    if passed:
        return True
    if not override:
        raise GateError(f"K1 not passed; PRD says stop all experiments until it is.\n{detail}")
    log.warning("K1 not passed; continuing only for: %s", override)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "gate_overrides.jsonl").open("a") as f:
        f.write(json.dumps({"gate": "K1", "reason": override,
                            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    return False
