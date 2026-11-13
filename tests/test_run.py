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

    import importlib

    run_mod = importlib.import_module("whopays.run")

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

    source = tmp_path / "source"
    run([cfg], root=source, loader=loader, sync=False)
    loader.reset_mock()

    def fake_pull(dest, run_ids):
        import shutil
        shutil.copytree(source / run_ids[0], dest / run_ids[0])
        return True

    with mock.patch.object(hub_sync, "pull_runs", side_effect=fake_pull), \
         mock.patch.object(hub_sync, "push_run", return_value=True):
        [res] = run([cfg], root=tmp_path, loader=loader)
    assert (res.status, res.n_items) == ("skipped", N)
    loader.assert_not_called()


def test_unknown_task_module_fails_cleanly(tmp_path, loader, monkeypatch):
    monkeypatch.setattr(tasks, "_REGISTRY", {})
    missing = ModuleNotFoundError("gone", name="whopays.tasks.flores_nll")
    monkeypatch.setattr(tasks.importlib, "import_module", mock.Mock(side_effect=missing))
    [res] = run([{**BASE, "task": "flores_nll", "lang": "en"}], root=tmp_path, loader=loader,
                sync=False)
    assert res.status == "failed" and "not implemented" in res.error


def test_resume_batch_change_refused_even_with_env_override(tmp_path, fake_task, loader):
    fake_task['crash_at'] = 10
    job = {**BASE, 'lang': 'en', 'batch_size': 8}
    run([job], root=tmp_path, loader=loader, sync=False)
    fake_task['crash_at'] = None
    [r] = run([{**job, 'batch_size': 4}], root=tmp_path, loader=loader,
              sync=False, allow_env_change=True)
    assert r.status == 'failed' and 'batch_size' in r.error


def test_parquet_contains_config_and_done_retries_push(tmp_path, fake_task, loader):
    from whopays.io import hub_sync
    job = {**BASE, 'lang': 'en'}
    [r] = run([job], root=tmp_path, loader=loader, sync=False)
    df = pd.read_parquet(tmp_path / r.run_id / 'items.parquet')
    assert {'model', 'weight', 'kv', 'spec', 'calib', 'seed', 'bitmap',
            'config_json', 'batch_size'} <= set(df.columns)
    with mock.patch.object(hub_sync, 'push_run', return_value=True) as push:
        [again] = run([job], root=tmp_path, loader=loader)
    assert again.status == 'skipped'
    push.assert_called_once()


def test_incomplete_done_and_failed_restore_fail_closed(tmp_path, fake_task, loader):
    from whopays.io import hub_sync
    job = {**BASE, 'lang': 'en'}
    cfg = RunConfig(**job)
    d = tmp_path / cfg.run_id
    d.mkdir()
    (d / 'DONE').write_text('{"n_items": 900}')
    [r] = run([job], root=tmp_path, loader=loader, sync=False)
    assert r.status == 'failed' and 'incomplete' in r.error
    with mock.patch.object(hub_sync, 'pull_runs', side_effect=ConnectionError('offline')):
        [r] = run([job], root=tmp_path / 'other', loader=loader)
    assert r.status == 'failed'
    loader.assert_not_called()


def test_partial_restore_cannot_be_used_after_failed_download(tmp_path, fake_task, loader):
    from whopays.io import hub_sync
    cfg = RunConfig(**BASE, lang='en')
    def interrupted(dest, run_ids):
        (dest / run_ids[0]).mkdir()
        raise ConnectionError('download interrupted')
    with mock.patch.object(hub_sync, 'pull_runs', side_effect=interrupted):
        [r] = run([cfg], root=tmp_path, loader=loader)
    assert r.status == 'failed'
    [r] = run([cfg], root=tmp_path, loader=loader, sync=False)
    assert r.status == 'failed' and 'incomplete remote restore' in r.error
    loader.assert_not_called()


def test_bitmap_provenance_is_json_string(tmp_path, fake_task, loader):
    cfg = RunConfig(**BASE, lang='en', weight='bitmap', calib='multi', bitmap={'layer': 4})
    [r] = run([cfg], root=tmp_path, loader=loader, sync=False)
    assert r.status == 'done'
    df = pd.read_parquet(tmp_path / r.run_id / 'items.parquet')
    assert json.loads(df.bitmap.iloc[0]) == {'layer': 4}
    assert json.loads(df.config_json.iloc[0]) == cfg.canonical()


def test_gptq_runs_release_their_calibration_provenance(tmp_path, monkeypatch):
    from whopays.models.loader import ModelBundle

    def task(cfg, bundle, skip):
        yield {"item_id": "x", "correct": True}

    monkeypatch.setitem(tasks._REGISTRY, "belebele", task)
    prov = {"source": "wikimedia/wikipedia", "sha256": "abc", "tokens": 262144}
    loader = mock.Mock(return_value=ModelBundle(model=None, tokenizer=None, device="cpu",
                                                dtype=None, quantization={"calibration": prov}))
    [res] = run([{**BASE, "lang": "te", "weight": "w4_gptq", "calib": "multi"}],
                root=tmp_path, loader=loader, sync=False)
    run_dir = tmp_path / res.run_id
    assert json.loads((run_dir / "calibration.json").read_text()) == prov
    row = pd.read_parquet(run_dir / "items.parquet").iloc[0]
    assert row.calib_sha256 == "abc" and row.calib_source == "wikipedia_128x2048"


def test_run_restored_from_hub_is_not_pushed_back(tmp_path, fake_task, loader):
    from whopays.io import hub_sync

    cfg = RunConfig(**BASE, lang="th")
    source = tmp_path / "source"
    run([cfg], root=source, loader=loader, sync=False)

    def fake_pull(dest, run_ids):
        import shutil
        shutil.copytree(source / run_ids[0], dest / run_ids[0])
        return True

    with mock.patch.object(hub_sync, "pull_runs", side_effect=fake_pull), \
         mock.patch.object(hub_sync, "push_run", return_value=True) as push:
        [res] = run([cfg], root=tmp_path / "dest", loader=loader)
    assert res.status == "skipped"
    push.assert_not_called()  # it is already on the hub: a push would only spend a commit


def test_push_false_pulls_but_never_pushes(tmp_path, fake_task, loader):
    from whopays.io import hub_sync

    with mock.patch.object(hub_sync, "pull_runs", return_value=False) as pull, \
         mock.patch.object(hub_sync, "push_run", return_value=True) as push:
        [res] = run([{**BASE, "lang": "bn"}], root=tmp_path, loader=loader, push=False)
        [again] = run([{**BASE, "lang": "bn"}], root=tmp_path, loader=loader, push=False)
    assert (res.status, again.status) == ("done", "skipped")
    pull.assert_called_once_with(tmp_path, [res.run_id])
    push.assert_not_called()
