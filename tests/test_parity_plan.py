"""ParityQuant plan, K2 stop rule, parity report, 7B loading and the release check."""
import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from whopays.config import canonical_json, load_jobs
from whopays.gates import GateError, require_k2
from whopays.languages import CODES
from whopays.parityquant.plan import build_plan, label_methods, parity_report, write_plan


def _profile(n_layers=8):
    rows = []
    for bit, scale in ((2, 4.0), (3, 2.0), (4, 1.0), (8, 0.1)):
        for j in range(n_layers):
            for k, lang in enumerate(('en', 'te')):
                value = scale * (j + 1 if k == 0 else n_layers - j)
                rows.append({'layer': f'model.layers.{j}', 'bits': bit, 'lang': lang,
                             'sensitivity': float(value), 'parameters': 100})
    return rows


def test_plan_has_four_methods_per_budget_and_meets_each_exactly():
    plan = build_plan(_profile(), 'm', budgets=(3.0, 3.5, 4.0), seeds=(0,))
    for budget in (3.0, 3.5, 4.0):
        entries = [a for a in plan['allocations'] if a['budget'] == budget]
        assert {a['method'] for a in entries} == {'minimax', 'average', 'uniform_en',
                                                  'uniform_multi'}
        assert all(a['average_bits'] == budget for a in entries)


def test_fractional_budget_uses_the_matched_uniform_comparator():
    plan = build_plan(_profile(), 'm', budgets=(3.5,), seeds=(0,))
    objectives = {a['method']: a['objective'] for a in plan['allocations']}
    assert objectives['uniform_en'] == objectives['uniform_multi'] == 'matched_uniform'
    whole = build_plan(_profile(), 'm', budgets=(4.0,), seeds=(0,))
    assert {a['objective'] for a in whole['allocations'] if 'uniform' in a['method']} == {'uniform'}


def test_plan_jobs_cover_all_languages_and_replicate_minimax_seeds():
    plan = build_plan(_profile(), 'm', budgets=(4.0,), seeds=(0, 1, 2))
    jobs = load_jobs(plan['jobs'])
    assert {j.lang for j in jobs} == set(CODES) and {j.weight for j in jobs} == {'bitmap'}
    en = [j for j in jobs if j.lang == 'en']
    assert len(en) == 3 + 3  # minimax on 3 seeds, other methods on seed 0
    assert {j.calib for j in en if j.bitmap == next(
        a['bit_map'] for a in plan['allocations'] if a['method'] == 'uniform_en')} >= {'en'}


def test_write_plan_round_trips(tmp_path):
    plan = build_plan(_profile(), 'm', budgets=(4.0,), seeds=(0,))
    jobs_path, alloc_path = write_plan(plan, tmp_path, '1.5B')
    assert jobs_path.name == 'parityquant_1.5B.yaml'
    assert len(load_jobs(jobs_path)) == len(load_jobs(plan['jobs']))
    assert json.loads(alloc_path.read_text()) == plan['allocations']


def _rows(plan, accuracy, n=300):
    frames = []
    uniform = np.random.default_rng(7).random(n)
    for allocation in plan['allocations']:
        for lang in ('en', 'te'):
            frames.append(pd.DataFrame({
                'item_id': [f'i{i}' for i in range(n)], 'lang': lang, 'model': 'm',
                'task': 'belebele', 'weight': 'bitmap', 'calib': allocation['calib'], 'seed': 0,
                'budget': allocation['budget'], 'bitmap': canonical_json(allocation['bit_map']),
                'correct': uniform < accuracy[allocation['method']][lang]}))
    for lang in ('en', 'te'):
        frames.append(pd.DataFrame({
            'item_id': [f'i{i}' for i in range(n)], 'lang': lang, 'model': 'm', 'task': 'belebele',
            'weight': 'fp16', 'calib': 'none', 'seed': 0, 'budget': None, 'bitmap': None,
            'correct': uniform < 0.9}))
    return pd.concat(frames, ignore_index=True)


ACCURACY = {'minimax': {'en': .85, 'te': .82}, 'average': {'en': .88, 'te': .6},
            'uniform_en': {'en': .86, 'te': .55}, 'uniform_multi': {'en': .86, 'te': .65}}


def test_label_methods_ties_rows_to_allocations():
    plan = build_plan(_profile(), 'm', budgets=(4.0,), seeds=(0,))
    # At a whole budget both uniform methods share a bit map, so calibration must separate them.
    labelled = label_methods(_rows(plan, ACCURACY), plan['allocations'], 4.0)
    assert set(labelled.method) == {'fp16', 'minimax', 'average', 'uniform_en', 'uniform_multi'}


def test_parity_report_recovers_planted_gain_and_verdict():
    # Hand-written maps: identical solved maps share one run, so they cannot differ in accuracy.
    allocations = [{'method': method, 'calib': 'en' if method == 'uniform_en' else 'multi',
                    'budget': budget, 'bit_map': {'model.layers.0': 2 + i}}
                   for budget in (3.0, 3.5, 4.0) for i, method in enumerate(ACCURACY)]
    table, verdict = parity_report(_rows({'allocations': allocations}, ACCURACY), allocations, 'm',
                                   n_resamples=100)
    assert set(table.budget) == {3.0, 3.5, 4.0}
    assert (table.best_baseline == 'uniform_multi').all()
    assert (table.improvement_low > 0).all()
    assert verdict == 'POSITIVE'


def test_k2_failure_stops_parityquant():
    identical = [{'layer': f'model.layers.{j}', 'bits': 4, 'lang': lang, 'sensitivity': float(j),
                  'parameters': 1} for j in range(6) for lang in ('en', 'te')]
    with pytest.raises(GateError, match='K2'):
        require_k2(identical)
    reverse = [{**row, 'sensitivity': -row['sensitivity'] if row['lang'] == 'te'
                else row['sensitivity']} for row in identical]
    assert require_k2(reverse) is True


def test_seven_billion_model_splits_across_two_gpus():
    from whopays.models import loader

    with (mock.patch('torch.cuda.device_count', return_value=2),
          mock.patch('transformers.AutoModelForCausalLM.from_pretrained') as load):
        loader._load_causal_lm('Qwen/Qwen2.5-7B-Instruct', 'cuda', 'fp16')
    assert load.call_args.kwargs['device_map'] == 'auto'
    load.return_value.to.assert_not_called()


def test_seven_billion_model_refuses_one_gpu_clearly():
    from whopays.models import loader

    with (mock.patch('torch.cuda.device_count', return_value=1),
          pytest.raises(ValueError, match='T4×2')):
        loader._load_causal_lm('Qwen/Qwen2.5-7B-Instruct', 'cuda', 'fp16')


def test_scale_grid_has_fp16_and_w4_with_an_mgsm_subset():
    from whopays.grids import QWEN, scale_grid

    jobs = [j for j in load_jobs(scale_grid()) if j.lang == 'en']
    assert {j.model for j in jobs} == {QWEN['7B']}
    assert {j.weight for j in jobs} == {'fp16', 'w4_gptq'}
    assert {(j.task, j.limit) for j in jobs} == {('belebele', None), ('mgsm', 30)}
