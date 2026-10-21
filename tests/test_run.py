import json
from unittest import mock

import pandas as pd
import pytest

from whopays import tasks
from whopays.config import RunConfig
from whopays.run import group_by_model, run

N = 120
BASE = {"model": "tiny", "task": "belebele"}


class Crash(Exception):
    pass


@pytest.fixture
def fake_task(monkeypatch):
    """A deterministic task whose crash point the test controls."""
    state = {"crash_at": None, "computed": 0}

    def task(cfg, bundle, skip):
        for i in range(N):
            item_id = f"{cfg.lang}-{i:03d}"
            if item_id in skip:
                continue
            if i == state["crash_at"]:
                raise Crash("GPU session died")
            state["computed"] += 1
            yield {"item_id": item_id, "correct": (i + len(bundle)) % 2 == 0}

    monkeypatch.setitem(tasks._REGISTRY, "belebele", task)
    return state


@pytest.fixture
def loader():
    return mock.Mock(side_effect=lambda cfg: f"model:{cfg.model}:{cfg.weight}")


def test_runs_to_completion_then_skips(tmp_path, fake_task, loader):
    [res] = run([{**BASE, "lang": "te"}], root=tmp_path, loader=loader, sync=False)
    assert (res.status, res.n_items) == ("done", N)
    run_dir = tmp_path / res.run_id
    assert {p.name for p in run_dir.iterdir()} == {
        "config.yaml", "meta.json", "items.jsonl", "items.parquet", "DONE"}
    df = pd.read_parquet(run_dir / "items.parquet")
    assert len(df) == N
    assert {"run_id", "lang", "gpu", "torch", "transformers", "code_version"} <= set(df.columns)
    assert (df["run_id"] == res.run_id).all()

    [again] = run([{**BASE, "lang": "te"}], root=tmp_path, loader=loader, sync=False)
    assert (again.status, again.n_items) == ("skipped", N)
    assert loader.call_count == 1  # finished runs never load a model


def test_crash_then_resume_matches_clean_run(tmp_path, fake_task, loader):
    [clean] = run([{**BASE, "lang": "sw"}], root=tmp_path / "clean", loader=loader, sync=False)

    fake_task["crash_at"] = 77
    [failed] = run([{**BASE, "lang": "sw"}], root=tmp_path / "x", loader=loader, sync=False)
    assert failed.status == "failed" and "GPU session died" in failed.error
    fake_task["crash_at"] = None
    fake_task["computed"] = 0
    [resumed] = run([{**BASE, "lang": "sw"}], root=tmp_path / "x", loader=loader, sync=False)
    assert resumed.status == "done"
    assert fake_task["computed"] == N - 77  # finished items are not recomputed

    a = (tmp_path / "clean" / clean.run_id / "items.jsonl").read_bytes()
    b = (tmp_path / "x" / resumed.run_id / "items.jsonl").read_bytes()
    assert a == b


def test_stop_on_error_raises(tmp_path, fake_task, loader):
    fake_task["crash_at"] = 3
    with pytest.raises(Crash):
        run([{**BASE, "lang": "en"}], root=tmp_path, loader=loader, sync=False,
            stop_on_error=True)


def test_model_loaded_once_per_weights(tmp_path, fake_task, loader):
    jobs = [
        {**BASE, "lang": "en"},
        {**BASE, "lang": "en", "weight": "w8_rtn"},
        {**BASE, "lang": "te"},  # same weights as the first job: grouped with it
        {**BASE, "lang": "te", "weight": "w8_rtn", "kv": "evict"},  # KV policy reuses weights
    ]
    results = run(jobs, root=tmp_path, loader=loader, sync=False)
    assert [r.status for r in results] == ["done"] * 4
    assert loader.call_count == 2
    assert [r.lang for r in results] == ["en", "te", "en", "te"]


def test_group_by_model_is_stable():
    cfgs = [RunConfig(**BASE, lang=lang, weight=w)
            for lang, w in [("en", "fp16"), ("de", "w8_rtn"), ("fr", "fp16"), ("es", "w8_rtn")]]
    assert [c.lang for c in group_by_model(cfgs)] == ["en", "fr", "de", "es"]


def test_resume_under_different_code_is_refused(tmp_path, fake_task, loader, monkeypatch):
    fake_task["crash_at"] = 10
    [r] = run([{**BASE, "lang": "ja"}], root=tmp_path, loader=loader, sync=False)
    fake_task["crash_at"] = None

    from whopays import run as run_mod

    other = {**run_mod.env_info(), "code_version": "different"}
    monkeypatch.setattr(run_mod, "env_info", lambda: other)
    [refused] = run([{**BASE, "lang": "ja"}], root=tmp_path, loader=loader, sync=False)
    assert refused.status == "failed" and "different environment" in refused.error

    [ok] = run([{**BASE, "lang": "ja"}], root=tmp_path, loader=loader, sync=False,
               allow_env_change=True)
    assert ok.status == "done"
    meta = json.loads((tmp_path / r.run_id / "meta.json").read_text())
    assert len(meta["starts"]) == 2  # the refused attempt is not recorded


def test_sync_pulls_missing_and_pushes_finished(tmp_path, fake_task, loader):
    from whopays.io import hub_sync

    with mock.patch.object(hub_sync, "pull_runs", return_value=False) as pull, \
         mock.patch.object(hub_sync, "push_run", return_value=True) as push:
        [res] = run([{**BASE, "lang": "bn"}], root=tmp_path, loader=loader)
    pull.assert_called_once_with(tmp_path, [res.run_id])
    assert push.call_count >= 1
    assert push.call_args.args[0] == tmp_path / res.run_id


def test_finished_run_restored_from_hub_is_skipped(tmp_path, fake_task, loader):
    from whopays.io import hub_sync

    cfg = RunConfig(**BASE, lang="th")

    def fake_pull(dest, run_ids):
        d = dest / run_ids[0]
        d.mkdir(parents=True)
        (d / "DONE").write_text('{"n_items": 900}\n')
        return True

    with mock.patch.object(hub_sync, "pull_runs", side_effect=fake_pull):
        [res] = run([cfg], root=tmp_path, loader=loader)
    assert (res.status, res.n_items) == ("skipped", 900)
    loader.assert_not_called()


def test_unknown_task_module_fails_cleanly(tmp_path, loader, monkeypatch):
    monkeypatch.setattr(tasks, "_REGISTRY", {})
    missing = ModuleNotFoundError("gone", name="whopays.tasks.flores_nll")
    monkeypatch.setattr(tasks.importlib, "import_module", mock.Mock(side_effect=missing))
    [res] = run([{**BASE, "task": "flores_nll", "lang": "en"}], root=tmp_path, loader=loader,
                sync=False)
    assert res.status == "failed" and "not implemented" in res.error
