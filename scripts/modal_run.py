"""Run WhoPays work on Modal T4s from the M1.

    python scripts/modal_run.py env                          # smoke test: print the stack
    python scripts/modal_run.py notebook 03_sensitivity_probe
    python scripts/modal_run.py jobs configs/audit.yaml --chunk-minutes 60

`jobs` fans a job list out over up to 10 T4s in two phases. Phase 1 runs the first chunk of
every quantized model group, which quantizes it once and stores the weights in the Volume's
cache. Phase 2 runs every other chunk, loading those weights instead of quantizing again.
Outputs are the same as a single-process run: each chunk uses the same code, batch sizes and
(verified by digest) the same quantized weights.

The commit must be on GitHub (the image installs it). Needs `modal token new` and the secret
`whopays-hf` holding HF_TOKEN.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def _plan(spec: str, chunk_minutes: float) -> tuple[list, list]:
    from whopays.config import load_jobs
    from whopays.parallel import chunk_jobs

    chunks = chunk_jobs(load_jobs(spec), chunk_minutes)
    first, rest, seen = [], [], set()
    for chunk in chunks:
        key = chunk[0].model_key()
        quantized = chunk[0].weight in {"w4_gptq", "w3_gptq", "bitmap"}
        (first if quantized and key not in seen else rest).append(chunk)
        seen.add(key)
    return first, rest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env")
    n = sub.add_parser("notebook")
    n.add_argument("name")
    j = sub.add_parser("jobs")
    j.add_argument("spec")
    j.add_argument("--chunk-minutes", type=float, default=60.0)
    j.add_argument("--no-sync", action="store_true")
    j.add_argument("--dry-run", action="store_true", help="print the plan only")
    parser.add_argument("--ref", help="commit to install (default: HEAD, must be on origin)")
    args = parser.parse_args()

    if args.cmd == "jobs" and args.dry_run:
        first, rest = _plan(args.spec, args.chunk_minutes)
        print(f"phase 1: {len(first)} chunks (quantize once), phase 2: {len(rest)} chunks")
        return 0

    from kaggle_run import _head_on_origin

    os.environ["WHOPAYS_SHA"] = args.ref or _head_on_origin()
    import modal
    import modal_app

    with modal.enable_output(), modal_app.app.run():
        # Fail fast: a container that cannot start would otherwise be retried, and billed,
        # until the local process is stopped.
        if args.cmd == "env":
            print(modal_app.environment.spawn().get(timeout=20 * 60))
            return 0
        if args.cmd == "notebook":
            return modal_app.notebook.spawn(args.name).get(timeout=12 * 3600)
        first, rest = _plan(args.spec, args.chunk_minutes)
        failed = 0
        for phase, chunks in (("quantize", first), ("evaluate", rest)):
            print(f"{phase}: {len(chunks)} chunks")
            payload = [[c.model_dump(mode="json") for c in chunk] for chunk in chunks]
            for results in modal_app.run_jobs.map(payload, kwargs={"sync": not args.no_sync},
                                                  return_exceptions=True):
                if isinstance(results, Exception):
                    failed += 1
                    print(f"chunk failed: {results!r}")
                    continue
                for r in results:
                    print(f"[{r['status']:>7}] {r['run_id']} {r['task']:<10} {r['lang']} "
                          f"{r['seconds']}s" + (f"  {r['error']}" if r.get("error") else ""))
                    failed += r["status"] == "failed"
        print("failed:", failed)
        return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
