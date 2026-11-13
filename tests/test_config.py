import pytest
import yaml
from pydantic import ValidationError

from whopays.config import RunConfig, canonical_json, load_jobs, stable_hash

BASE = {"model": "Qwen/Qwen2.5-1.5B-Instruct", "task": "belebele", "lang": "te"}


def test_hash_ignores_key_order():
    a = RunConfig(**BASE, weight="w4_gptq", calib="multi", seed=1)
    b = RunConfig(seed=1, calib="multi", weight="w4_gptq", lang="te", task="belebele",
                  model="Qwen/Qwen2.5-1.5B-Instruct")
    assert a.run_id == b.run_id


def test_hash_ignores_float_formatting():
    texts = ["budget: 3.5", "budget: 3.50", "budget: 3.500000000000001", "budget: 35e-1"]
    ids = {
        RunConfig(**BASE, weight="bitmap", calib="multi", bitmap={"0": 4},
                  **yaml.safe_load(t)).run_id
        for t in texts
    }
    assert len(ids) == 1


def test_canonical_json_normalises_numbers():
    assert canonical_json({"b": 4.0, "a": [1.5, 2]}) == canonical_json({"a": [1.50, 2.0], "b": 4})
    assert canonical_json({"x": 0.1 + 0.2}) == canonical_json({"x": 0.3})


def test_hash_is_pinned():
    # Guards against silent changes to canonicalisation: a changed hash orphans every result
    # folder. If this fails on purpose, bump SCHEMA_VERSION and log it in DECISIONS.md.
    assert RunConfig(**BASE).run_id == stable_hash(RunConfig(**BASE).canonical())
    assert len(RunConfig(**BASE).run_id) == 12
    assert RunConfig(**BASE).run_id == PINNED_FP16_TE


def test_irrelevant_fields_do_not_fork_the_hash():
    fp16 = RunConfig(**BASE).run_id
    assert RunConfig(**BASE, seed=7).run_id == fp16  # fp16 has no calibration sampling
    assert RunConfig(**BASE, group_size=64).run_id == fp16  # no quantization groups in fp16
    assert RunConfig(**BASE, kv_budget=512).run_id == fp16  # budget only matters for eviction
    assert RunConfig(**BASE, spec_k=8).run_id == fp16  # k only matters when speculative is on
    assert RunConfig(**BASE, batch_size=32, tag="debug").run_id == fp16  # execution detail


def test_relevant_fields_fork_the_hash():
    ids = {
        RunConfig(**BASE).run_id,
        RunConfig(**{**BASE, "lang": "en"}).run_id,
        RunConfig(**BASE, weight="w8_rtn").run_id,
        RunConfig(**BASE, weight="w4_gptq", calib="en").run_id,
        RunConfig(**BASE, weight="w4_gptq", calib="multi").run_id,
        RunConfig(**BASE, weight="w4_gptq", calib="multi", seed=1).run_id,
        RunConfig(**BASE, kv="evict").run_id,
        RunConfig(**BASE, kv="evict", kv_budget=512).run_id,
        RunConfig(**BASE, kv="quant4").run_id,
        RunConfig(**BASE, limit=20).run_id,
        RunConfig(**{**BASE, "task": "mgsm"}, spec=True).run_id,
        RunConfig(**{**BASE, "task": "mgsm"}).run_id,
    }
    assert len(ids) == 12


def test_validation():
    with pytest.raises(ValidationError):
        RunConfig(**{**BASE, "lang": "xx"})
    with pytest.raises(ValidationError):
        RunConfig(**BASE, weight="w4_gptq")  # GPTQ needs a calibration set
    with pytest.raises(ValidationError):
        RunConfig(**BASE, weight="w8_rtn", calib="en")  # RTN has no calibration
    with pytest.raises(ValidationError):
        RunConfig(**BASE, weight="bitmap", calib="multi")  # bitmap needs a bit map
    with pytest.raises(ValidationError):
        RunConfig(**BASE, weight="bitmap", calib="multi", bitmap={"0": 5})  # bits not in {2,3,4,8}
    with pytest.raises(ValidationError):
        RunConfig(**BASE, unknown_field=1)


def test_speculative_gets_default_draft():
    cfg = RunConfig(**{**BASE, "task": "mgsm"}, spec=True)
    assert cfg.draft_model == "Qwen/Qwen2.5-0.5B-Instruct"


def test_model_key_shared_across_tasks_and_languages():
    a = RunConfig(**BASE, weight="w4_gptq", calib="en")
    b = RunConfig(model=BASE["model"], task="mgsm", lang="sw", weight="w4_gptq", calib="en")
    c = RunConfig(**BASE, weight="w4_gptq", calib="en", seed=2)
    assert a.model_key() == b.model_key() != c.model_key()


def test_yaml_roundtrip(tmp_path):
    cfg = RunConfig(**BASE, weight="w3_gptq", calib="multi", seed=2)
    path = tmp_path / "c.yaml"
    path.write_text(cfg.to_yaml())
    assert RunConfig.from_yaml(path).run_id == cfg.run_id


def test_load_jobs_expands_defaults_and_languages(tmp_path):
    path = tmp_path / "jobs.yaml"
    path.write_text(
        "defaults:\n  model: Qwen/Qwen2.5-0.5B-Instruct\n  limit: 20\n"
        "jobs:\n  - {task: belebele, lang: [en, te]}\n  - {task: mgsm, lang: all}\n"
    )
    jobs = load_jobs(path)
    assert [(j.task, j.lang) for j in jobs[:2]] == [("belebele", "en"), ("belebele", "te")]
    assert len(jobs) == 2 + 11
    assert all(j.limit == 20 for j in jobs)


def test_load_jobs_accepts_objects_and_dicts():
    cfg = RunConfig(**BASE)
    assert load_jobs([cfg, dict(BASE)]) == [cfg, cfg]


PINNED_FP16_TE = "8531f57a9dfb"


def test_calibration_source_changes_only_calibrated_run_ids():
    from whopays.config import LEGACY_CALIB_SOURCE

    w4 = RunConfig(**BASE, weight="w4_gptq", calib="multi")
    legacy = RunConfig(**BASE, weight="w4_gptq", calib="multi",
                       calib_source=LEGACY_CALIB_SOURCE)
    assert w4.calib_source == "wikipedia_128x2048"
    assert w4.run_id != legacy.run_id and w4.model_key() != legacy.model_key()
    # Runs made before the field existed keep their IDs under the legacy value.
    assert "calib_source" not in legacy.canonical()
    assert RunConfig(**BASE, calib_source=LEGACY_CALIB_SOURCE).run_id == PINNED_FP16_TE
    assert RunConfig(**BASE, weight="w8_rtn").run_id == RunConfig(
        **BASE, weight="w8_rtn", calib_source=LEGACY_CALIB_SOURCE).run_id


def test_fp16_baselines_keep_their_logged_run_ids():
    from whopays.config import load_jobs
    from whopays.grids import K1_MODEL, m0_fp16

    ids = {c.run_id for c in load_jobs(m0_fp16([K1_MODEL]))}
    assert {"a0fc2d0018cc", "b3cbde1ac3c0", "2e3ea51a8f75"} <= ids


def test_quant4_cache_recipe_enters_only_quant4_run_ids():
    from whopays import config

    base = dict(model="Qwen/Qwen2.5-1.5B-Instruct", task="mgsm", lang="te")
    q = RunConfig(**base, kv="quant4").canonical()
    assert q["kv_recipe"] == config.KV_QUANT_RECIPE == 2  # recipe 2: HQQ axis 1 (per token)
    for kv in ("full", "evict"):
        assert "kv_recipe" not in RunConfig(**base, kv=kv).canonical()


def test_offset_changes_id_only_when_set():
    flores = {**BASE, "task": "flores_nll"}
    assert RunConfig(**flores).run_id == RunConfig(**flores, offset=0).run_id
    assert RunConfig(**flores, offset=500).run_id != RunConfig(**flores).run_id
    assert "offset" not in RunConfig(**flores).canonical()
    with pytest.raises(ValidationError, match="flores_nll only"):
        RunConfig(**BASE, offset=500)
