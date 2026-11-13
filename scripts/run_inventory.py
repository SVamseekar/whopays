"""Inventory of the results dataset, for docs/RUN_LOG.md.

    python scripts/run_inventory.py            # sessions + grid completeness, as markdown

Downloads only `config.yaml`, `meta.json` and `DONE` from every run folder of
`Heavy-Dervish1205/whopays-results` (a few MB), groups runs into sessions by first start
date, platform and commit, and checks each job list in `configs/` against the finished runs.
Uses the cached `hf auth login` token; an invalid `HF_TOKEN` in the environment is ignored.
"""

from __future__ import annotations

import collections
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = "Heavy-Dervish1205/whopays-results"
GRIDS = ("replication", "audit", "speed")


def fetch(dest: Path) -> list[dict]:
    os.environ.pop("HF_TOKEN", None)
    from huggingface_hub import snapshot_download

    snapshot_download(REPO, repo_type="dataset", local_dir=dest, max_workers=16,
                      allow_patterns=["*/config.yaml", "*/meta.json", "*/DONE"])
    rows = []
    for d in sorted(p for p in dest.iterdir() if (p / "meta.json").exists()):
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        starts = json.loads((d / "meta.json").read_text())["starts"]
        done = (d / "DONE").exists()
        rows.append({
            "run_id": d.name, "first": starts[0]["time_utc"], "last": starts[-1]["time_utc"],
            "starts": len(starts), "platform": starts[0]["platform"],
            "code": starts[0]["code_version"][:7], "done": done,
            "n_items": json.loads((d / "DONE").read_text())["n_items"] if done else None,
            **cfg,
        })
    return rows


def sessions(rows: list[dict]) -> str:
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["first"][:10], r["platform"], r["code"])].append(r)
    out = ["| First start (UTC) | Last start (UTC) | Platform | Commit | Runs | Done | Cells |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for key in sorted(groups):
        v = groups[key]
        cells = collections.Counter(
            f"{r['model'].split('-')[1]} {r['task']} {r['weight']}"
            + (f" {r['calib_source']}" if r.get("calib_source") else "") for r in v)
        out.append(f"| {min(r['first'] for r in v)[:16]} | {max(r['first'] for r in v)[11:16]} "
                   f"| {key[1]} | `{key[2]}` | {len(v)} | {sum(r['done'] for r in v)} | "
                   + "; ".join(f"{k} ×{n}" for k, n in sorted(cells.items())) + " |")
    return "\n".join(out)


def completeness(rows: list[dict]) -> str:
    sys.path.insert(0, str(ROOT / "src"))
    from whopays.config import load_jobs

    by_id = {r["run_id"]: r for r in rows}
    out = ["| Job list | Runs | Done | Partial | Not started |", "| --- | --- | --- | --- | --- |"]
    for name in GRIDS:
        path = ROOT / "configs" / f"{name}.yaml"
        if not path.exists():
            continue
        ids = [j.run_id for j in load_jobs(path)]
        done = sum(1 for i in ids if i in by_id and by_id[i]["done"])
        partial = sum(1 for i in ids if i in by_id and not by_id[i]["done"])
        out.append(f"| `configs/{name}.yaml` | {len(ids)} | {done} | {partial} | "
                   f"{len(ids) - done - partial} |")
    return "\n".join(out)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        rows = fetch(Path(tmp))
    print(f"{len(rows)} run folders, {sum(r['done'] for r in rows)} with DONE\n")
    print(sessions(rows), "\n")
    print(completeness(rows))
    partial = [r for r in rows if not r["done"]]
    if partial:
        print("\nPartial runs (no DONE):")
        for r in partial:
            print(f"- `{r['run_id']}` {r['task']} {r['weight']} {r['lang']} "
                  f"calib {r['calib']} seed {r['seed']}, last start {r['last']} ({r['platform']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
