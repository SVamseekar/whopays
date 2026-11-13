"""Compute the H4 / K3 verdict and the per-language tables for one ParityQuant job list.

    python scripts/parity_eval.py 1.5B            # pulls runs from HF into results/
    python scripts/parity_eval.py 1.5B --no-pull  # uses runs already under results/

Writes docs/figures/parityquant_<size>_{h4,languages,bits_to_parity}.csv and prints the H4
table, the K3 verdict and all-language tables. Uses the cached `hf auth login` token; an
invalid `HF_TOKEN` in the environment is ignored. Runs on the M1 (statistics only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("size", help="model size in the job-list name, e.g. 1.5B")
    parser.add_argument("--root", default=str(ROOT / "results"))
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()

    os.environ.pop("HF_TOKEN", None)
    from huggingface_hub import get_token

    from whopays.config import load_jobs
    from whopays.grids import QWEN, m0_fp16
    from whopays.io.hub_sync import pull_runs
    from whopays.io.results import is_done, load_runs
    from whopays.parityquant.plan import (
        bits_to_parity,
        language_increases,
        parity_report,
    )
    from whopays.report.langtable import language_table

    model = QWEN[args.size]
    jobs = load_jobs(ROOT / "configs" / f"parityquant_{args.size}.yaml")
    allocations = json.loads(
        (ROOT / "configs" / f"parityquant_{args.size}.allocations.json").read_text())
    fp16 = load_jobs(m0_fp16(models=(model,), tasks=("belebele", "flores_nll")))
    configs = [*jobs, *fp16]
    if not args.no_pull:
        pull_runs(Path(args.root), [c.run_id for c in configs], token=get_token())
    missing = [c for c in configs if not is_done(Path(args.root) / c.run_id)]
    if missing:
        print(f"{len(missing)} of {len(configs)} runs are not finished; verdict not computed")
        return 1
    rows = load_runs(Path(args.root), configs)

    table, verdict = parity_report(rows, allocations, model, n_resamples=args.resamples)
    increases = language_increases(rows, allocations, model)
    worst = increases.groupby(["budget", "method"]).increase.max().rename("worst").reset_index()
    curve = bits_to_parity(worst)

    out = ROOT / "docs" / "figures"
    table.to_csv(out / f"parityquant_{args.size}_h4.csv", index=False)
    increases.to_csv(out / f"parityquant_{args.size}_languages.csv", index=False)
    curve.to_csv(out / f"parityquant_{args.size}_bits_to_parity.csv", index=False)

    print(table.to_string(index=False))
    print("K3:", verdict)
    for budget, frame in increases.groupby("budget"):
        print(f"\nHeld-out NLL increase per sentence (nats), budget {budget}")
        print(language_table(frame, "increase", ["method"]))
    print("\nBits to parity\n" + curve.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
