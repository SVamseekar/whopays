"""Command line: `python -m whopays.cli {run,hash,smoke,grid,plan} ...`."""

from __future__ import annotations

import argparse
import logging
import sys

from whopays.config import load_jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="whopays")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run a job-list YAML")
    p_run.add_argument("jobs")
    p_run.add_argument("--root", default=None, help="results root (default: platform default)")
    p_run.add_argument("--no-sync", action="store_true", help="do not pull/push the HF dataset")

    p_hash = sub.add_parser("hash", help="print run IDs for a job-list YAML")
    p_hash.add_argument("jobs")

    p_smoke = sub.add_parser("smoke", help="end-to-end check: small model, few Belebele items")
    p_smoke.add_argument("--device", default=None, help="cuda / mps / cpu (default: best)")
    p_smoke.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p_smoke.add_argument("--n", type=int, default=20, help="items per language")
    p_smoke.add_argument("--langs", nargs="+", default=["en", "te"])

    p_grid = sub.add_parser("grid", help="write job lists for the experiment grids")
    p_grid.add_argument("--out", default="configs")

    p_plan = sub.add_parser("plan", help="GPU-hours and slices from measured FP16 baselines")
    p_plan.add_argument("--root", default="results")
    p_plan.add_argument("--out", default="configs")
    p_plan.add_argument("--load-minutes", type=float, default=0.0,
                        help="measured time to quantize one GPTQ model; 0 until notebook 02")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.cmd == "hash":
        for cfg in load_jobs(args.jobs):
            print(cfg.run_id, cfg.task, cfg.lang, cfg.model, cfg.weight, cfg.kv)
        return 0

    if args.cmd == "grid":
        return write_grids(args.out)

    if args.cmd == "plan":
        return plan(args.root, args.out, args.load_minutes)

    if args.cmd == "smoke":
        return smoke(args.model, args.langs, args.n, args.device)

    from whopays.run import run

    results = run(args.jobs, root=args.root, sync=not args.no_sync)
    return int(any(r.status == "failed" for r in results))


def write_grids(out: str) -> int:
    from pathlib import Path

    from whopays import grids

    specs = {"replication": grids.replication_grid(), "audit": grids.audit_grid(),
             "speed": grids.speed_grid() + grids.speed_subset()}
    for name, jobs in specs.items():
        jobs = load_jobs(jobs)
        path = grids.write_jobs(jobs, Path(out) / f"{name}.yaml")
        print(f"{path}: {len(jobs)} runs")
    return 0


def plan(root: str, out: str, load_minutes: float) -> int:
    """Print the M0 re-plan table and write audit slices from logged 1.5B baselines."""
    from whopays import grids
    from whopays.io.results import load_runs

    rows = load_runs(root, load_jobs(grids.m0_fp16([grids.K1_MODEL])))
    if rows.empty:
        raise SystemExit(f"no baselines under {root}; pull the results dataset first")
    # Planning assumption until 3B baselines exist: the PRD's 2x from 1.5B to 3B.
    assume = {grids.QWEN["3B"]: (grids.K1_MODEL, 2.0)}
    load = {w: load_minutes * 60 for w in ("w4_gptq", "w3_gptq")}
    jobs = load_jobs(grids.audit_grid())
    costs = grids.project_costs(jobs, grids.measured_rates(rows), assume=assume,
                                load_seconds=load)
    print(grids.compute_plan(jobs, costs).round(1).to_string(index=False))
    paths = grids.write_slices(jobs, costs, out, "audit")
    print(f"wrote {len(paths)} slices to {out}")
    return 0


def smoke(model: str, langs: list[str], n: int, device: str | None) -> int:
    """Run Belebele end to end into a fresh temporary folder. Never a reportable number."""
    import tempfile
    import time
    from functools import partial

    import pandas as pd

    from whopays.models.loader import load
    from whopays.run import run

    root = tempfile.mkdtemp(prefix="whopays-smoke-")
    jobs = [{"model": model, "task": "belebele", "lang": lang, "limit": n, "tag": "smoke"}
            for lang in langs]
    start = time.monotonic()
    results = run(jobs, root=root, loader=partial(load, device=device), sync=False)
    for r in results:
        if r.status == "done":
            df = pd.read_parquet(f"{root}/{r.run_id}/items.parquet")
            print(f"{r.lang}: acc={df.correct.mean():.2f} (n={len(df)}, smoke only) "
                  f"median_input_tokens={int(df.input_tokens.median())} "
                  f"device={df.gpu.iloc[0]}")
    print(f"total {time.monotonic() - start:.0f}s, results in {root}")
    return int(any(r.status != "done" for r in results))


if __name__ == "__main__":
    sys.exit(main())
