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


def require_k2(rows) -> bool:
    """Stop ParityQuant on a K2 failure: the PRD drops M4 and extends the audit instead."""
    from whopays.parityquant.probe import k2_verdict

    verdict = k2_verdict(rows)
    if verdict.K2 != "PASS":
        raise GateError(f"K2 failed: minimum Spearman {verdict.min_spearman:.3f} above the "
                        "threshold. PRD: drop M4 and extend the audit with Llama-3.2.")
    if verdict.noise_limited:
        log.warning("K2 passed but split-half reliability is low (%.3f); differences between "
                    "languages may be sampling noise", verdict.split_half_min)
    return True


BATCH_GATE_MAX_FLIPS = 2  # per (cache policy, language) out of BATCH_GATE_N problems


def batch_gate(model: str, *, langs=("en", "te"), n: int = 16, batch_size: int = 8,
               kv_policies=("full", "quant4", "evict"), loader=None,
               items=None) -> tuple[pd.DataFrame, str]:
    """Compare batched against batch-1 MGSM generation per cache policy on one GPU.

    Batching changes FP16 numerics only (rows are equal in fp32; tests/test_mgsm.py), so
    near-tied greedy choices may move. The full cache is the reference: its batched cells are
    already accepted. PASS when no policy and language flips the correctness of more than
    BATCH_GATE_MAX_FLIPS of `n` problems. Also returns seconds per problem for planning.
    Nothing is written to the results dataset.
    """
    from whopays.config import RunConfig
    from whopays.datasets import mgsm as load_mgsm
    from whopays.models.loader import load
    from whopays.tasks import mgsm

    bundle = (loader or load)(RunConfig(model=model, task="mgsm", lang=langs[0]))
    records = []
    for lang in langs:
        problems = (items or load_mgsm)(lang)[:n]
        for kv in kv_policies:
            out = {}
            for size in (1, batch_size):
                cfg = RunConfig(model=model, task="mgsm", lang=lang, kv=kv, batch_size=size,
                                limit=n)
                start = time.perf_counter()
                out[size] = list(mgsm.score_items(cfg, bundle, problems, set()))
                out[f"t{size}"] = (time.perf_counter() - start) / len(problems)
            one, many = out[1], out[batch_size]
            pairs = list(zip(one, many, strict=True))
            records.append({
                "lang": lang, "kv": kv, "n": len(one),
                "same_text": sum(a["output_text"] == b["output_text"] for a, b in pairs),
                "correct_b1": sum(r["correct"] for r in one),
                "correct_batched": sum(r["correct"] for r in many),
                "flips": sum(a["correct"] != b["correct"] for a, b in pairs),
                "s_per_problem_b1": round(out["t1"], 3),
                "s_per_problem_batched": round(out[f"t{batch_size}"], 3),
            })
    table = pd.DataFrame(records)
    verdict = "PASS" if (table.flips <= BATCH_GATE_MAX_FLIPS).all() else "FAIL"
    return table, verdict
