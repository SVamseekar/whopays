"""Mirror run folders to a private Hugging Face dataset (FR-7).

Each run folder is stored at the repo root under its run ID, the same layout as the local
results root, so pulling into results_root() restores results/<run_id>/ directly. Every
function is a no-op returning False when no HF_TOKEN is available.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterable
from pathlib import Path

from whopays.env import get_secret

log = logging.getLogger(__name__)

DEFAULT_REPO = "Heavy-Dervish1205/whopays-results"


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


def push_runs(root: Path, run_ids: Iterable[str], *, repo: str | None = None,
              token: str | None = None) -> list[str]:
    """Upload many run folders under <root> in one commit; returns the IDs uploaded.

    The hub limits commits per hour per account, so a grid that commits once per run (and
    once more per flush) from several machines runs into HTTP 429 and hour-long waits.
    Missing folders are ignored. Without a token nothing is uploaded.
    """
    root = Path(root)
    present = [rid for rid in dict.fromkeys(run_ids) if (root / rid).is_dir()]
    token = token or get_secret("HF_TOKEN")
    if not token or not present:
        if not token:
            log.info("no HF_TOKEN; skipping push of %d runs", len(present))
        return []
    repo = repo or repo_id()
    api = _api(token)
    if repo not in _created:
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        _created.add(repo)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(root),
                      allow_patterns=[f"{rid}/*" for rid in present],
                      commit_message=f"sync {len(present)} runs")
    return present


def pull_runs(
    dest: Path,
    run_ids: Iterable[str] | None = None,
    *,
    repo: str | None = None,
    token: str | None = None,
) -> bool:
    """Download run folders (all, or the given IDs) into dest/<run_id>/.

    Returns False without a token or when the repo does not exist yet. Raises
    PermissionError on a rejected token: the hub reports 401 as "repository not found", and
    reading that as "no results yet" would restart finished runs from nothing.
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
    except RepositoryNotFoundError as exc:
        if getattr(exc.response, "status_code", None) == 401:
            raise PermissionError(f"HF_TOKEN rejected (401) for {repo or repo_id()}; "
                                  "replace the token before running") from exc
        log.info("results repo %s does not exist yet", repo or repo_id())
        return False
    return True


class BulkPush:
    """Upload finished runs of a grid in one commit every `interval` seconds, off the GPU path.

    Workers run with uploads off; this thread commits every run that is `is_done` and not yet
    uploaded. A rate-limited commit (huggingface_hub sleeps up to an hour on HTTP 429) delays
    only this thread. `close()` stops the loop and makes one final commit of what is left.
    Failures are logged, not raised: the local copy is authoritative and the next commit retries.
    """

    def __init__(self, root: Path, run_ids: Iterable[str], interval: float = 900.0,
                 push=None):
        self.root = Path(root)
        self.run_ids = list(dict.fromkeys(run_ids))
        self.interval = interval
        self.pushed: set[str] = set()
        self._push = push or push_runs
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def pending(self) -> list[str]:
        from whopays.io.results import is_done

        return [r for r in self.run_ids if r not in self.pushed and is_done(self.root / r)]

    def flush(self) -> list[str]:
        ids = self.pending()
        if not ids:
            return []
        try:
            done = self._push(self.root, ids)
        except Exception as exc:  # network, auth, rate limit: never stop the grid
            log.warning("bulk push of %d runs failed: %s", len(ids), exc)
            return []
        self.pushed.update(done)
        return done

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.flush()

    def start(self) -> BulkPush:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="bulk-push")
        self._thread.start()
        return self

    def close(self) -> list[str]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        return self.flush()


class ThrottledPush:
    """Push a run folder at most once per `min_interval` seconds (HF limits commit rates).

    Failures are logged, not raised: the local copy is authoritative and the next push retries.
    With `background=True`, interval pushes run in a daemon thread, so a rate-limited upload
    (huggingface_hub sleeps up to an hour on HTTP 429) never stalls the GPU; a push due while
    one is still running is skipped, not queued. `force=True` waits for any running upload and
    then pushes in the calling thread, so the final state is always uploaded.
    """

    def __init__(self, run_dir: Path, min_interval: float = 600.0, clock=time.monotonic,
                 background: bool = False):
        self.run_dir = Path(run_dir)
        self.min_interval = min_interval
        self.clock = clock
        self.background = background
        self._last: float | None = None
        self._thread: threading.Thread | None = None

    def _push(self) -> bool:
        try:
            return push_run(self.run_dir)
        except Exception as exc:  # network, auth, rate limit: keep the GPU job running
            log.warning("hub push of %s failed: %s", self.run_dir.name, exc)
            return False

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def __call__(self, _path: Path | None = None, *, force: bool = False) -> bool:
        now = self.clock()
        if not force and self._last is not None and now - self._last < self.min_interval:
            return False
        if self.background and not force:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._last = now
            self._thread = threading.Thread(target=self._push, daemon=True,
                                            name=f"push-{self.run_dir.name}")
            self._thread.start()
            return True
        self.wait()
        self._last = now
        return self._push()
