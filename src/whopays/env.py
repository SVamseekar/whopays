"""Where the code is running and what is attached: platform, device, secrets, paths, versions."""

from __future__ import annotations

import functools
import json
import logging
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

Platform = Literal["colab", "kaggle", "local"]


def platform() -> Platform:
    """Detect Colab, Kaggle or local from environment variables both services set."""
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "KAGGLE_URL_BASE" in os.environ:
        return "kaggle"
    return "local"


def device() -> str:
    """Best available torch device: cuda, then mps, then cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_secret(name: str) -> str | None:
    """Read a secret from the environment, Colab userdata or Kaggle secrets, in that order.

    Returns None when the secret is absent so callers can degrade (e.g. skip hub sync).
    """
    value = os.environ.get(name)
    if value:
        return value
    where = platform()
    try:
        if where == "colab":
            from google.colab import userdata

            return userdata.get(name) or None
        if where == "kaggle":
            from kaggle_secrets import UserSecretsClient

            return UserSecretsClient().get_secret(name) or None
    except Exception as exc:  # each service raises its own not-found / no-access errors
        log.warning("secret %s not available on %s (%s)", name, where, type(exc).__name__)
    return None


def results_root() -> Path:
    """Directory holding results/<run_id>/. WHOPAYS_RESULTS overrides the platform default."""
    override = os.environ.get("WHOPAYS_RESULTS")
    if override:
        return Path(override)
    return {
        "colab": Path("/content/whopays-results"),
        "kaggle": Path("/kaggle/working/results"),
        "local": Path.cwd() / "results",
    }[platform()]


def prepare_text_runtime() -> bool:
    """Remove a broken optional torchvision from an explicitly prepared text-only runtime.

    Notebook setup calls this after installing dependencies and before importing
    Transformers. Probe in a child process to keep failed native imports out of the
    notebook kernel. Healthy/absent torchvision and the platform torch are preserved.
    Returns whether torchvision was removed; previously loaded imports need a restart.
    """
    try:
        metadata.version("torchvision")
    except metadata.PackageNotFoundError:
        return False
    probe = subprocess.run([sys.executable, "-c", "import torchvision"],
                           capture_output=True, text=True, timeout=60)
    if probe.returncode == 0:
        return False
    log.warning("torchvision failed its import check; removing this optional vision package "
                "for WhoPays text-only evaluation (keeping the platform torch)")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"],
                   capture_output=True, text=True, check=True)
    if "transformers" in sys.modules or "torchvision" in sys.modules:
        raise RuntimeError("Removed broken torchvision. Restart the Colab/Kaggle session, "
                           "then Run all so Transformers detects the repaired environment.")
    return True


def code_version() -> str:
    """Commit the installed code came from, '+dirty' if the local checkout has edits.

    Order: WHOPAYS_CODE_VERSION (set by the notebook setup cell), pip's direct_url.json for
    VCS installs, `git rev-parse` next to the source, then the package version.
    """
    override = os.environ.get("WHOPAYS_CODE_VERSION")
    if override:
        return override
    try:
        direct = metadata.distribution("whopays").read_text("direct_url.json")
        commit = json.loads(direct or "{}").get("vcs_info", {}).get("commit_id")
        if commit:
            return commit
    except metadata.PackageNotFoundError:
        pass
    src = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=src, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown-" + metadata.version("whopays")


@functools.cache
def env_info() -> dict[str, str]:
    """Versions and hardware logged into every result row (golden rule 8)."""
    import torch
    import transformers

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else device()
    return {
        "platform": platform(),
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "none",
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
        "code_version": code_version(),
    }


def print_env() -> None:
    for key, value in env_info().items():
        print(f"{key:>13}: {value}")
