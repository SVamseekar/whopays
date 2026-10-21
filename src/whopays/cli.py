"""Command line: `python -m whopays.cli {run,hash} ...`."""

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

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.cmd == "hash":
        for cfg in load_jobs(args.jobs):
            print(cfg.run_id, cfg.task, cfg.lang, cfg.model, cfg.weight, cfg.kv)
        return 0

    if args.cmd == "smoke":
        return smoke(args.model, args.langs, args.n, args.device)

    from whopays.run import run

    results = run(args.jobs, root=args.root, sync=not args.no_sync)
    return int(any(r.status == "failed" for r in results))


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
