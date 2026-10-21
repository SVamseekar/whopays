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
        "SVamseekar/whopays-results", repo_type="dataset", private=True, exist_ok=True
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
    err = RepositoryNotFoundError("nope", response=mock.MagicMock())
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
