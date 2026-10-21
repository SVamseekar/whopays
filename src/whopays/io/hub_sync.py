"""Mirror run folders to a private Hugging Face dataset (FR-7).

Each run folder is stored at the repo root under its run ID, the same layout as the local
results root, so pulling into results_root() restores results/<run_id>/ directly. Every
function is a no-op returning False when no HF_TOKEN is available.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path

from whopays.env import get_secret

log = logging.getLogger(__name__)

DEFAULT_REPO = "SVamseekar/whopays-results"


def repo_id() -> str:
    return os.environ.get("WHOPAYS_HF_REPO", DEFAULT_REPO)


def _api(token: str):
    from huggingface_hub import HfApi

    return HfApi(token=token)


_created: set[str] = set()


def push_run(run_dir: Path, *, repo: str | None = None, token: str | None = None) -> bool:
    """Upload one run folder to <repo>/<run_id>/. Network errors propagate to the caller."""
    token = token or get_secret("HF_TOKEN")
    if not token:
        log.info("no HF_TOKEN; skipping push of %s", run_dir)
        return False
    repo, run_dir = repo or repo_id(), Path(run_dir)
    api = _api(token)
    if repo not in _created:
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        _created.add(repo)
    api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(run_dir),
        path_in_repo=run_dir.name,
        commit_message=f"sync {run_dir.name}",
    )
    return True


def pull_runs(
    dest: Path,
    run_ids: Iterable[str] | None = None,
    *,
    repo: str | None = None,
    token: str | None = None,
) -> bool:
    """Download run folders (all, or the given IDs) into dest/<run_id>/.

    Returns False without a token or when the repo does not exist yet.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    token = token or get_secret("HF_TOKEN")
    if not token:
        log.info("no HF_TOKEN; skipping pull")
        return False
    patterns = None if run_ids is None else [f"{rid}/*" for rid in run_ids]
    try:
        snapshot_download(
            repo or repo_id(),
            repo_type="dataset",
            local_dir=str(dest),
            allow_patterns=patterns,
            token=token,
        )
    except RepositoryNotFoundError:
        log.info("results repo %s does not exist yet", repo or repo_id())
        return False
    return True


class ThrottledPush:
    """Push a run folder at most once per `min_interval` seconds (HF limits commit rates).

    Failures are logged, not raised: the local copy is authoritative and the next push retries.
    """

    def __init__(self, run_dir: Path, min_interval: float = 600.0, clock=time.monotonic):
        self.run_dir = Path(run_dir)
        self.min_interval = min_interval
        self.clock = clock
        self._last: float | None = None

    def __call__(self, _path: Path | None = None, *, force: bool = False) -> bool:
        now = self.clock()
        if not force and self._last is not None and now - self._last < self.min_interval:
            return False
        self._last = now
        try:
            return push_run(self.run_dir)
        except Exception as exc:  # network, auth, rate limit: keep the GPU job running
            log.warning("hub push of %s failed: %s", self.run_dir.name, exc)
            return False
