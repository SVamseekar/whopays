"""Splitting a job list across GPUs: whole model groups per worker, balanced, nothing lost."""

import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
import yaml

from whopays import parallel
from whopays.config import RunConfig

WEIGHTS = [{"weight": "fp16"}, {"weight": "w8_rtn"},
           {"weight": "w4_gptq", "calib": "en", "seed": 0},
           {"weight": "w4_gptq", "calib": "multi", "seed": 0},
           {"weight": "w3_gptq", "calib": "en", "seed": 1}]


def _jobs(model="Qwen/Qwen2.5-1.5B-Instruct"):
    return [RunConfig(model=model, task=t, lang=lang, **w)
            for w in WEIGHTS for t in ("belebele", "flores_nll") for lang in ("en", "te")]


def test_shards_partition_jobs_and_keep_model_groups_whole():
    jobs = _jobs()
    shards = parallel.shard_jobs(jobs, 2)
    ids = [c.run_id for shard in shards for c in shard]
    assert sorted(ids) == sorted(c.run_id for c in jobs)
    owner = {}
    for i, shard in enumerate(shards):
        for cfg in shard:
            assert owner.setdefault(cfg.model_key(), i) == i


def test_shards_are_balanced_by_cost():
    shards = parallel.shard_jobs(_jobs(), 2)
    costs = [sum(parallel.job_cost(c) for c in s) for s in shards]
    assert max(costs) - min(costs) <= max(parallel.group_costs(_jobs()).values())


def test_gptq_groups_cost_more_than_fp16():
    jobs = _jobs()
    costs = parallel.group_costs(jobs)
    key = {c.weight: c.model_key() for c in jobs}
    assert costs[key["w4_gptq"]] > costs[key["fp16"]]


def test_never_more_shards_than_groups():
    jobs = [c for c in _jobs() if c.weight == "fp16"]
    assert len(parallel.shard_jobs(jobs, 4)) == 1


def test_models_spanning_two_gpus_are_not_split():
    jobs = _jobs("Qwen/Qwen2.5-7B-Instruct")
    assert parallel.worker_count(jobs, gpus=2) == 1
    assert parallel.worker_count(_jobs(), gpus=2) == 2
    assert parallel.worker_count(_jobs(), gpus=0) == 1


def test_run_parallel_launches_one_pinned_process_per_gpu(tmp_path):
    jobs = _jobs()
    procs = []

    def fake_popen(cmd, env, **kw):
        procs.append((cmd, env))
        proc = mock.Mock()
        proc.stdout = iter([])
        proc.wait.return_value = 0
        return proc

    with mock.patch.object(subprocess, "Popen", side_effect=fake_popen):
        codes = parallel.run_parallel(jobs, gpus=2, workdir=tmp_path)
    assert codes == [0, 0]
    assert sorted(env["CUDA_VISIBLE_DEVICES"] for _, env in procs) == ["0", "1"]
    listed = []
    for cmd, _ in procs:
        assert cmd[:4] == [sys.executable, "-m", "whopays.cli", "run"]
        doc = yaml.safe_load(Path(cmd[4]).read_text())
        listed += [RunConfig(**d).run_id for d in doc["jobs"]]
    assert sorted(listed) == sorted(c.run_id for c in jobs)


def test_run_parallel_raises_when_a_worker_fails(tmp_path):
    def fake_popen(cmd, env, **kw):
        proc = mock.Mock()
        proc.stdout = iter([])
        proc.wait.return_value = 1 if env["CUDA_VISIBLE_DEVICES"] == "1" else 0
        return proc

    with (mock.patch.object(subprocess, "Popen", side_effect=fake_popen),
          pytest.raises(RuntimeError, match="worker 1")):
        parallel.run_parallel(_jobs(), gpus=2, workdir=tmp_path)


def test_run_parallel_can_leave_failures_to_the_caller(tmp_path):
    def fake_popen(cmd, env, **kw):
        proc = mock.Mock()
        proc.stdout = iter([])
        proc.wait.return_value = 1
        return proc

    with mock.patch.object(subprocess, "Popen", side_effect=fake_popen):
        assert parallel.run_parallel(_jobs(), gpus=2, workdir=tmp_path, check=False) == [1, 1]
