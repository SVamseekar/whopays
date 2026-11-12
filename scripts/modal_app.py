"""Modal app: WhoPays on Modal T4s, with the same software stack as the Kaggle runs.

Do not run this file directly; `scripts/modal_run.py` sets WHOPAYS_SHA and drives it.

Stack parity with Kaggle (DECISIONS 2026-11-06 log): Tesla T4, Python 3.13, torch 2.15.0+cu132,
transformers 5.19.0 from requirements.lock. The image installs the repository at one pushed
commit, and WHOPAYS_CODE_VERSION carries that commit into every result row, exactly as the
notebook setup cell does.

Storage: the Volume `whopays-results` is the results root (/results), so finished runs outlive
any container, and it holds the quantized-weights cache (/results/.quant-cache), so each model
key is quantized once for all containers.
"""

from __future__ import annotations

import os
import subprocess
import sys

import modal

REPO = "SVamseekar/whopays"
SHA = os.environ.get("WHOPAYS_SHA", "")
TORCH = "torch==2.15.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu132"
GPU = "T4"

app = modal.App("whopays")
volume = modal.Volume.from_name("whopays-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install(TORCH, index_url=TORCH_INDEX)
    .run_commands(
        "git init -q /root/whopays",
        f"git -C /root/whopays remote add origin https://github.com/{REPO}.git",
        f"git -C /root/whopays fetch -q --depth 1 origin {SHA or 'main'}",
        "git -C /root/whopays checkout -q --force FETCH_HEAD",
        "pip install -q -c /root/whopays/requirements.lock '/root/whopays[gpu,stats]'",
    )
    .env({"WHOPAYS_CODE_VERSION": SHA, "WHOPAYS_SHA": SHA,
          "WHOPAYS_QUANT_CACHE": "/results/.quant-cache", "PYTHONUNBUFFERED": "1"})
)


# Function dependencies must be identical when the app is defined locally and inside the
# container, or every container fails at startup ("Function has 3 dependencies but container
# got 2 object ids") and Modal keeps starting new ones. So nothing here may depend on
# environment variables that differ between the M1 and the container.
COMMON = dict(image=image, gpu=GPU, volumes={"/results": volume},
              secrets=[modal.Secret.from_name("whopays-hf")], max_containers=10)


@app.function(timeout=20 * 60, **COMMON)
def environment() -> str:
    """Print the environment block every result row logs; used as a smoke test."""
    out = subprocess.run([sys.executable, "-c", "import whopays.env as e; e.print_env()"],
                         capture_output=True, text=True, cwd="/root/whopays")
    return out.stdout + out.stderr


@app.function(timeout=8 * 3600, **COMMON)
def run_jobs(jobs: list[dict], sync: bool = True) -> list[dict]:
    """Run one chunk of a job list; finished runs are skipped, so a retry resumes."""
    volume.reload()
    from whopays.run import run

    try:
        results = run(jobs, sync=sync)
    finally:
        volume.commit()
    return [r.__dict__ for r in results]


@app.function(timeout=12 * 3600, **COMMON)
def notebook(name: str) -> int:
    """Execute notebooks/src/<name>.py (percent-format notebooks are plain Python)."""
    volume.reload()
    try:
        return subprocess.run([sys.executable, f"notebooks/src/{name}.py"],
                              cwd="/root/whopays").returncode
    finally:
        volume.commit()


@app.function(timeout=6 * 3600, **COMMON)
def python(code: str) -> int:
    """Run a short Python snippet against the installed package, e.g. the FR-3 check."""
    volume.reload()
    try:
        return subprocess.run([sys.executable, "-c", code], cwd="/root/whopays").returncode
    finally:
        volume.commit()
