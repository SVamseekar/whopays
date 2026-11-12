"""Run a WhoPays notebook on Kaggle from the M1, without the web UI.

    python scripts/kaggle_run.py push 02_replicate_uniform   # build at HEAD, push, start a run
    python scripts/kaggle_run.py status
    python scripts/kaggle_run.py wait                        # poll until the run finishes
    python scripts/kaggle_run.py fetch 02_replicate_uniform  # log + executed notebook into runs/

All notebooks share one private Kaggle kernel, `<user>/whopays-runner`. Each push saves a new
version holding the chosen notebook and runs it in the background ("Save & Run All") on GPU
T4 x2 with internet on. One kernel means the secret below is attached once, and only one run is
live at a time. The setup cell installs the repo at the pinned commit, so that commit must
already be on GitHub; `push` refuses otherwise.

The Kaggle API cannot attach secrets. Attach `HF_TOKEN` once to `whopays-runner` in the web UI
(Add-ons -> Secrets); later versions keep it. Without it, results stay in the kernel output and
`fetch --results` downloads them.

Needs the Kaggle CLI (`uv tool install kaggle --with six`) and an API token in
`~/.kaggle/access_token` or `KAGGLE_API_TOKEN`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

STAGING = ROOT / "build" / "kaggle"
RUNS = ROOT / "runs" / "kaggle"
ACCELERATOR = "NvidiaTeslaT4"  # GPU T4 x2
FINISHED = {"complete", "error", "cancelAcknowledged", "cancelRequested"}


def _run(*cmd: str, check: bool = True) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"{' '.join(cmd[:3])} failed:\n{proc.stdout}{proc.stderr}")
    return proc.stdout + proc.stderr


def username() -> str:
    """Kaggle account the CLI is authenticated as."""
    out = _run("kaggle", "config", "view")
    match = re.search(r"^\s*-?\s*username:\s*(\S+)", out, re.MULTILINE)
    if not match or match.group(1) in ("None", ""):
        raise SystemExit("Kaggle CLI is not authenticated. Save a token to ~/.kaggle/access_token "
                         "(https://www.kaggle.com/settings -> API -> Create New Token).")
    return match.group(1)


KERNEL = "whopays-runner"


def kernel_ref() -> str:
    return f"{username()}/{KERNEL}"


def _head_on_origin() -> str:
    """Full SHA of HEAD, after checking GitHub has it and the notebook sources are committed."""
    sha = _run("git", "-C", str(ROOT), "rev-parse", "HEAD").strip()
    _run("git", "-C", str(ROOT), "fetch", "--quiet", "origin")
    remote = _run("git", "-C", str(ROOT), "branch", "-r", "--contains", sha).strip()
    if not remote:
        raise SystemExit(f"HEAD {sha[:7]} is not on origin. Push it first: the Kaggle setup cell "
                         "installs from GitHub.")
    dirty = _run("git", "-C", str(ROOT), "status", "--porcelain", "--", "notebooks/src",
                 "src", "requirements.lock", "pyproject.toml").strip()
    if dirty:
        raise SystemExit(f"Uncommitted changes would not reach Kaggle:\n{dirty}")
    return sha


def stage(notebook: str, ref: str, user: str) -> Path:
    """Write the notebook pinned to `ref` plus kernel-metadata.json into build/kaggle/<slug>/."""
    from build_notebooks import SRC, build

    source = SRC / f"{notebook}.py"
    if not source.exists():
        raise SystemExit(f"no notebook source {source.relative_to(ROOT)}")
    slug = KERNEL
    folder = STAGING / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{slug}.ipynb").write_text(build(source, ref))
    metadata = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return folder


def push(notebook: str, ref: str | None, accelerator: str, no_run: bool) -> None:
    user = username()
    ref = ref or _head_on_origin()
    folder = stage(notebook, ref, user)
    cmd = ["kaggle", "kernels", "push", "-p", str(folder), "--accelerator", accelerator]
    if no_run:
        cmd.append("--no-run")
    out = _run(*cmd)
    print(out.strip())
    if "error" in out.lower() and "successfully" not in out.lower():
        raise SystemExit(1)
    print(f"pushed {notebook} to {user}/{KERNEL} at {ref[:12]} on {accelerator}")
    print(f"https://www.kaggle.com/code/{user}/{KERNEL}")


def status() -> str:
    out = _run("kaggle", "kernels", "status", kernel_ref())
    match = re.search(r'status "?(?:KernelWorkerStatus\.)?([A-Za-z_]+)"?', out)
    state = match.group(1) if match else out.strip()
    return {"COMPLETE": "complete", "ERROR": "error", "RUNNING": "running",
            "QUEUED": "queued", "CANCEL_ACKNOWLEDGED": "cancelAcknowledged",
            "CANCEL_REQUESTED": "cancelRequested"}.get(state, state)


def wait(every: int, limit_h: float) -> str:
    deadline = time.time() + limit_h * 3600
    last = None
    while True:
        state = status()
        if state != last:
            print(f"{datetime.now():%H:%M:%S} {state}", flush=True)
            last = state
        if state in FINISHED:
            return state
        if time.time() > deadline:
            raise SystemExit(f"still {state} after {limit_h} h")
        time.sleep(every)


def fetch(notebook: str, with_results: bool) -> Path:
    """Download the latest run's log, executed notebook and (optionally) results."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    dest = RUNS / notebook / stamp
    dest.mkdir(parents=True, exist_ok=True)
    ref = kernel_ref()
    pattern = r".*" if with_results else r"^[^/]*\.(log|ipynb|txt)$"
    _run("kaggle", "kernels", "output", ref, "-p", str(dest), "-o", "-q",
         "--file-pattern", pattern, check=False)
    logs = _run("kaggle", "kernels", "logs", ref, check=False)
    (dest / "kernel_logs.txt").write_text(logs)
    (dest / "status.txt").write_text(status() + "\n")
    print(dest)
    for path in sorted(dest.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(dest)}  {path.stat().st_size:,} B")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("notebook")
    p.add_argument("--ref", help="commit to install (default: HEAD, must be on origin)")
    p.add_argument("--accelerator", default=ACCELERATOR)
    p.add_argument("--no-run", action="store_true", help="save a version without running it")
    sub.add_parser("status")
    w = sub.add_parser("wait")
    w.add_argument("--every", type=int, default=300, help="seconds between polls")
    w.add_argument("--limit-h", type=float, default=12.5)
    f = sub.add_parser("fetch")
    f.add_argument("notebook")
    f.add_argument("--results", action="store_true", help="also download results/ files")
    args = parser.parse_args()

    if args.cmd == "push":
        push(args.notebook, args.ref, args.accelerator, args.no_run)
    elif args.cmd == "status":
        print(status())
    elif args.cmd == "wait":
        print(wait(args.every, args.limit_h))
    elif args.cmd == "fetch":
        fetch(args.notebook, args.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
