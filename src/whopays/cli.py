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

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "hash":
        for cfg in load_jobs(args.jobs):
            print(cfg.run_id, cfg.task, cfg.lang, cfg.model, cfg.weight, cfg.kv)
        return 0

    from whopays.run import run

    results = run(args.jobs, root=args.root, sync=not args.no_sync)
    return int(any(r.status == "failed" for r in results))


if __name__ == "__main__":
    sys.exit(main())
