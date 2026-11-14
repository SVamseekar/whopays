from unittest import mock

import pytest

from whopays.io import hub_sync


@pytest.fixture(autouse=True)
def no_token(monkeypatch):
    for var in ("HF_TOKEN", "COLAB_RELEASE_TAG", "KAGGLE_KERNEL_RUN_TYPE", "WHOPAYS_HF_REPO"):
        monkeypatch.delenv(var, raising=False)
    hub_sync._created.clear()


def test_push_without_token_is_noop(tmp_path):
    with mock.patch.object(hub_sync, "_api") as api:
        assert hub_sync.push_run(tmp_path) is False
    api.assert_not_called()


def test_pull_without_token_is_noop(tmp_path):
    with mock.patch("huggingface_hub.snapshot_download") as dl:
        assert hub_sync.pull_runs(tmp_path) is False
    dl.assert_not_called()


def test_push_uploads_folder_under_run_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tok")
    run_dir = tmp_path / "abc123def456"
    run_dir.mkdir()
    with mock.patch.object(hub_sync, "_api") as api:
        assert hub_sync.push_run(run_dir) is True
        assert hub_sync.push_run(run_dir) is True
    client = api.return_value
    client.create_repo.assert_called_once_with(
        "Heavy-Dervish1205/whopays-results", repo_type="dataset", private=True, exist_ok=True
    )
    kwargs = client.upload_folder.call_args.kwargs
    assert kwargs["path_in_repo"] == "abc123def456"
    assert kwargs["repo_type"] == "dataset"


def test_repo_override(monkeypatch):
    monkeypatch.setenv("WHOPAYS_HF_REPO", "someone/else")
    assert hub_sync.repo_id() == "someone/else"


def test_pull_selected_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tok")
    with mock.patch("huggingface_hub.snapshot_download") as dl:
        assert hub_sync.pull_runs(tmp_path, ["r1", "r2"]) is True
    assert dl.call_args.kwargs["allow_patterns"] == ["r1/*", "r2/*"]
    assert dl.call_args.kwargs["local_dir"] == str(tmp_path)


def test_pull_missing_repo_returns_false(tmp_path, monkeypatch):
    from huggingface_hub.errors import RepositoryNotFoundError

    monkeypatch.setenv("HF_TOKEN", "tok")
    err = RepositoryNotFoundError("nope", response=mock.MagicMock(status_code=404))
    with mock.patch("huggingface_hub.snapshot_download", side_effect=err):
        assert hub_sync.pull_runs(tmp_path) is False


def test_throttled_push(tmp_path):
    now = [0.0]
    push = hub_sync.ThrottledPush(tmp_path, min_interval=600, clock=lambda: now[0])
    with mock.patch.object(hub_sync, "push_run", return_value=True) as pr:
        assert push() is True
        now[0] = 300
        assert push() is False
        assert push(force=True) is True
        now[0] = 1000
        assert push() is True
    assert pr.call_count == 3


def test_throttled_push_swallows_errors(tmp_path):
    push = hub_sync.ThrottledPush(tmp_path)
    with mock.patch.object(hub_sync, "push_run", side_effect=OSError("network down")):
        assert push() is False


def _hub_error(status):
    from huggingface_hub.errors import RepositoryNotFoundError

    response = mock.MagicMock(status_code=status)
    return RepositoryNotFoundError("hub refused", response=response)


def test_pull_rejected_token_raises_instead_of_reporting_missing_repo(tmp_path, monkeypatch):
    # The hub answers an invalid token with 401 wrapped as RepositoryNotFoundError. Treating
    # that as "no repo yet" would silently restart finished GPU runs from nothing.
    monkeypatch.setenv("HF_TOKEN", "expired")
    with (mock.patch("huggingface_hub.snapshot_download", side_effect=_hub_error(401)),
          pytest.raises(PermissionError, match="HF_TOKEN")):
        hub_sync.pull_runs(tmp_path)


def test_pull_absent_repo_with_valid_token_still_returns_false(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tok")
    with mock.patch("huggingface_hub.snapshot_download", side_effect=_hub_error(404)):
        assert hub_sync.pull_runs(tmp_path) is False


def test_background_push_does_not_block_and_skips_while_busy(tmp_path):
    import threading

    release, started = threading.Event(), threading.Event()

    def slow(_run_dir):
        started.set()
        release.wait(5)
        return True

    now = [0.0]
    push = hub_sync.ThrottledPush(tmp_path, min_interval=600, clock=lambda: now[0],
                                  background=True)
    with mock.patch.object(hub_sync, "push_run", side_effect=slow) as pr:
        assert push() is True          # started in a thread, returns at once
        assert started.wait(5)
        now[0] = 1000
        assert push() is False         # previous upload still running: skip, do not queue
        release.set()
        assert push(force=True) is True  # force waits for the thread, then pushes in-line
    assert pr.call_count == 2


def test_background_push_swallows_errors(tmp_path):
    push = hub_sync.ThrottledPush(tmp_path, background=True)
    with mock.patch.object(hub_sync, "push_run", side_effect=OSError("429")):
        push()
        push.wait()
        assert push(force=True) is False


def test_push_runs_is_one_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tok")
    for rid in ("r1", "r2", "r3"):
        (tmp_path / rid).mkdir()
    with mock.patch.object(hub_sync, "_api") as api:
        assert hub_sync.push_runs(tmp_path, ["r1", "r3", "missing"]) == ["r1", "r3"]
    client = api.return_value
    client.upload_folder.assert_called_once()
    kwargs = client.upload_folder.call_args.kwargs
    assert kwargs["folder_path"] == str(tmp_path)
    assert kwargs["allow_patterns"] == ["r1/*", "r3/*"]


def test_push_runs_without_token_is_noop(tmp_path):
    (tmp_path / "r1").mkdir()
    with mock.patch.object(hub_sync, "_api") as api:
        assert hub_sync.push_runs(tmp_path, ["r1"]) == []
    api.assert_not_called()


def _finished(root, run_id):
    d = root / run_id
    d.mkdir(parents=True)
    for name in ("DONE", "items.jsonl", "items.parquet", "config.yaml", "meta.json"):
        (d / name).write_text("{}" if name == "DONE" else "")
    return d


def test_bulk_push_commits_only_finished_runs_once(tmp_path, monkeypatch):
    from whopays.io import results

    monkeypatch.setattr(results, "is_done", lambda d: (d / "DONE").exists())
    calls = []
    push = hub_sync.BulkPush(tmp_path, ["a", "b", "c"], interval=3600,
                             push=lambda root, ids: calls.append(list(ids)) or list(ids))
    _finished(tmp_path, "a")
    (tmp_path / "b").mkdir()  # still running: no DONE
    assert push.flush() == ["a"]
    assert push.flush() == []  # nothing new: no empty commit
    _finished(tmp_path, "c")
    assert push.close() == ["c"]
    assert calls == [["a"], ["c"]]


def test_bulk_push_failure_is_retried_next_time(tmp_path, monkeypatch):
    from whopays.io import results

    monkeypatch.setattr(results, "is_done", lambda d: (d / "DONE").exists())
    attempts = []

    def flaky(root, ids):
        attempts.append(list(ids))
        if len(attempts) == 1:
            raise RuntimeError("429")
        return list(ids)

    push = hub_sync.BulkPush(tmp_path, ["a"], push=flaky)
    _finished(tmp_path, "a")
    assert push.flush() == []
    assert push.flush() == ["a"]
    assert attempts == [["a"], ["a"]]


def test_bulk_push_background_loop_flushes_on_interval(tmp_path, monkeypatch):
    import threading

    from whopays.io import results

    monkeypatch.setattr(results, "is_done", lambda d: (d / "DONE").exists())
    pushed = threading.Event()
    _finished(tmp_path, "a")
    push = hub_sync.BulkPush(tmp_path, ["a"], interval=0.01,
                             push=lambda root, ids: pushed.set() or list(ids)).start()
    assert pushed.wait(2)
    assert push.close() == []
