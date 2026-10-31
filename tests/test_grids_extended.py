import pytest

from whopays.config import RunConfig, load_jobs
from whopays.grids import audit_grid, replication_grid, slice_jobs, speed_grid


def test_audit_crosses_calibration_seeds_and_all_languages():
    jobs = load_jobs(audit_grid())
    assert len({j.run_id for j in jobs}) == len(jobs)
    for model in {j.model for j in jobs}:
        sub = [j for j in jobs if j.model == model and j.task == 'belebele' and j.lang == 'en']
        assert len(sub) == 3 * (2 + 2 * 2 * 3)
        assert {j.kv for j in sub} == {'full', 'quant4', 'evict'}
        assert {j.seed for j in sub if j.weight == 'w4_gptq'} == {0, 1, 2}


def test_speed_pairs_same_subset_and_configuration():
    jobs = load_jobs(speed_grid())
    assert all(j.task == 'mgsm' and j.limit == 30 for j in jobs)
    off = [j for j in jobs if not j.spec]
    on = [j for j in jobs if j.spec]
    assert len(off) == len(on)
    assert {j.model_copy(update={'spec': False, 'draft_model': None}).run_id for j in on} == {
        j.run_id for j in off}


def test_replication_has_reference_and_calibration_conditions():
    jobs = load_jobs(replication_grid())
    assert {j.weight for j in jobs} == {'fp16', 'w8_rtn', 'w4_gptq', 'w3_gptq'}
    assert all(j.kv == 'full' for j in jobs)


def test_slices_require_measured_costs_and_never_exceed_budget():
    jobs = [RunConfig(model='test', task='belebele', lang=x) for x in ['en', 'de', 'te']]
    costs = {j.run_id: 15000 for j in jobs}
    slices = slice_jobs(jobs, costs, max_hours=10)
    assert list(map(len, slices)) == [2, 1]
    with pytest.raises(ValueError, match='missing'):
        slice_jobs(jobs, {})
    with pytest.raises(ValueError, match='exceeds'):
        slice_jobs(jobs, costs, max_hours=1)
