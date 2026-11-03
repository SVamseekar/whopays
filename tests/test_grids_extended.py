import pandas as pd
import pytest

from whopays.config import RunConfig, load_jobs
from whopays.grids import (
    QWEN,
    audit_grid,
    compute_plan,
    measured_rates,
    project_costs,
    replication_grid,
    slice_jobs,
    speed_grid,
    speed_subset,
    write_slices,
)


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


def test_audit_replicates_calibration_only_on_belebele():
    jobs = [j for j in load_jobs(audit_grid()) if j.lang == 'en']
    for model in {j.model for j in jobs}:
        mgsm = [j for j in jobs if j.model == model and j.task == 'mgsm']
        assert len(mgsm) == 12  # PRD: 4 weights x 3 KV policies
        assert {(j.calib, j.seed) for j in mgsm if j.weight == 'w4_gptq'} == {('multi', 0)}


def test_speed_grid_has_the_prd_24_primary_cells():
    jobs = [j for j in load_jobs(speed_grid()) if j.lang == 'en']
    assert len(jobs) == 24
    assert {j.weight for j in jobs} == {'fp16', 'w8_rtn', 'w4_gptq', 'w3_gptq'}
    assert all(j.batch_size == 1 for j in jobs)


def test_second_model_speed_subset_has_four_cells():
    jobs = [j for j in load_jobs(speed_subset()) if j.lang == 'en']
    assert len(jobs) == 4 and {j.weight for j in jobs} == {'fp16', 'w4_gptq'}
    assert {j.kv for j in jobs} == {'full'} and {j.model for j in jobs} == {QWEN['3B']}


def _rates():
    frame = pd.DataFrame({'model': ['m'] * 4, 'task': ['belebele', 'belebele', 'mgsm', 'mgsm'],
                          'wall_time_s': [1.0, 3.0, 4.0, 4.0]})
    return measured_rates(frame)


def test_rates_are_seconds_per_item():
    assert _rates() == {('m', 'belebele'): 2.0, ('m', 'mgsm'): 4.0}


def test_costs_scale_with_items_and_honour_limit():
    jobs = load_jobs([{'model': 'm', 'task': 'belebele', 'lang': 'en'},
                      {'model': 'm', 'task': 'mgsm', 'lang': 'en', 'limit': 30}])
    costs = project_costs(jobs, _rates())
    assert costs[jobs[0].run_id] == 1800.0 and costs[jobs[1].run_id] == 120.0


def test_costs_refuse_an_unmeasured_model_unless_scaled_explicitly():
    jobs = load_jobs([{'model': 'other', 'task': 'belebele', 'lang': 'en'}])
    with pytest.raises(ValueError, match='no measured rate'):
        project_costs(jobs, _rates())
    costs = project_costs(jobs, _rates(), assume={'other': ('m', 2.0)})
    assert costs[jobs[0].run_id] == 3600.0


def test_speculative_cost_needs_its_own_measured_rate():
    jobs = load_jobs([{'model': 'm', 'task': 'mgsm', 'lang': 'en', 'limit': 30, 'spec': True}])
    with pytest.raises(ValueError, match='no measured rate'):
        project_costs(jobs, _rates())
    rates = {**_rates(), ('m', 'mgsm', 'spec'): 10.0}
    assert project_costs(jobs, rates)[jobs[0].run_id] == 300.0


def test_quantization_overhead_is_charged_once_per_loaded_model():
    jobs = load_jobs([{'model': 'm', 'task': 'belebele', 'lang': x, 'weight': 'w4_gptq',
                       'calib': 'multi'} for x in ('en', 'de')])
    costs = project_costs(jobs, _rates(), load_seconds={'w4_gptq': 600.0})
    assert sorted(costs.values()) == [1800.0, 2400.0]


def test_plan_totals_hours_by_model_and_task():
    jobs = load_jobs([{'model': 'm', 'task': 'belebele', 'lang': 'all'}])
    table = compute_plan(jobs, {j.run_id: 3600.0 for j in jobs})
    row = table[(table.model == 'm') & (table.task == 'belebele')].iloc[0]
    assert (row.runs, row.hours) == (11, 11.0)
    assert table.iloc[-1].model == 'TOTAL' and table.iloc[-1].hours == 11.0


def test_slices_keep_jobs_sharing_weights_together():
    a = [RunConfig(model='a', task='belebele', lang=x) for x in ('en', 'de')]
    b = [RunConfig(model='b', task='belebele', lang='en')]
    mixed = [a[0], b[0], a[1]]
    slices = slice_jobs(mixed, {j.run_id: 100.0 for j in mixed}, max_hours=10)
    assert [j.model for j in slices[0]] == ['a', 'a', 'b']


def test_write_slices_round_trips(tmp_path):
    jobs = load_jobs([{'model': 'm', 'task': 'belebele', 'lang': 'all'}])
    costs = {j.run_id: 3600.0 * 5 for j in jobs}
    paths = write_slices(jobs, costs, tmp_path, 'audit')
    assert [p.name for p in paths][:2] == ['audit_slice_00.yaml', 'audit_slice_01.yaml']
    assert [j.run_id for p in paths for j in load_jobs(p)] == [j.run_id for j in jobs]
    assert len(paths) == 6


def test_cli_writes_loadable_grids(tmp_path):
    from whopays.cli import main

    assert main(['grid', '--out', str(tmp_path)]) == 0
    assert {p.name for p in tmp_path.iterdir()} == {'replication.yaml', 'audit.yaml', 'speed.yaml'}
    audit = load_jobs(tmp_path / 'audit.yaml')
    assert {j.run_id for j in audit} == {j.run_id for j in load_jobs(audit_grid())}


def test_cli_plan_projects_audit_from_logged_baselines(tmp_path, capsys):
    from whopays.cli import main
    from whopays.grids import K1_MODEL, m0_fp16
    from whopays.io.results import ResultsWriter, mark_done, to_parquet

    for job in load_jobs(m0_fp16([K1_MODEL])):
        folder = tmp_path / 'results' / job.run_id
        folder.mkdir(parents=True)
        with ResultsWriter(folder / 'items.jsonl', 50) as writer:
            writer.write({'item_id': 'i0', 'wall_time_s': 0.1, 'task': job.task, 'lang': job.lang,
                          'run_id': job.run_id})
        to_parquet(folder / 'items.jsonl')
        mark_done(folder, 1)
    out = tmp_path / 'configs'
    assert main(['plan', '--root', str(tmp_path / 'results'), '--out', str(out)]) == 0
    printed = capsys.readouterr().out
    assert 'TOTAL' in printed and 'slices' in printed
    assert sorted(p.name for p in out.iterdir())[0] == 'audit_slice_00.yaml'
