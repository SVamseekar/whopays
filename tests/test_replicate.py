"""FR-3 reference gate and the K1 execution guard. Planted rows only."""
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from whopays import gates, replicate
from whopays.grids import K1_MODEL
from whopays.languages import CODES


def _nll(rows_per_lang, nll_per_token, langs=('en', 'te')):
    return pd.DataFrame([{'lang': lang, 'item_id': f'i{i}', 'nll_sum': nll_per_token * 10,
                          'n_tokens': 10} for lang in langs for i in range(rows_per_lang)])


def test_perplexity_is_token_weighted():
    rows = pd.DataFrame([{'lang': 'en', 'item_id': 'a', 'nll_sum': 10.0, 'n_tokens': 10},
                         {'lang': 'en', 'item_id': 'b', 'nll_sum': 30.0, 'n_tokens': 10}])
    assert replicate.perplexity(rows)['en'] == pytest.approx(np.exp(2.0))


def test_fr3_passes_within_two_percent():
    table, verdict = replicate.fr3_verdict(_nll(3, 2.0), _nll(3, 2.01))
    assert verdict['FR3'] == 'PASS'
    assert verdict['mean_abs_pct'] == pytest.approx(100 * abs(np.exp(-0.01) - 1))
    assert set(table.lang) == {'en', 'te'}


def test_fr3_fails_beyond_two_percent():
    assert replicate.fr3_verdict(_nll(3, 2.0), _nll(3, 2.5))[1]['FR3'] == 'FAIL'


def test_fr3_refuses_mismatched_items():
    theirs = _nll(3, 2.0)
    theirs.loc[0, 'item_id'] = 'other'
    assert replicate.fr3_verdict(_nll(3, 2.0), theirs)[1]['FR3'] == 'INCOMPLETE'


def test_primary_reference_is_gptqmodel_on_our_calibration():
    jobs = replicate.reference_jobs(K1_MODEL)
    assert {j.weight for j in jobs} == {'w4_gptqmodel'} and {j.task for j in jobs} == {'flores_nll'}
    assert {j.lang for j in jobs} == set(CODES)
    assert {(j.model, j.calib, j.seed, j.calib_source) for j in jobs} == {
        (K1_MODEL, 'multi', 0, 'wikipedia_128x2048')}
    ours = replicate.fr3_jobs(K1_MODEL)
    assert {(j.weight, j.calib, j.seed, j.calib_source) for j in ours} == {
        ('w4_gptq', 'multi', 0, 'wikipedia_128x2048')}
    assert {j.model_key() for j in jobs} != {j.model_key() for j in ours}


def test_official_checkpoint_is_a_secondary_reference():
    jobs = replicate.reference_jobs(K1_MODEL, kind='official')
    assert {j.weight for j in jobs} == {'prequant'} and {j.task for j in jobs} == {'flores_nll'}
    assert {j.lang for j in jobs} == set(CODES)
    assert all(j.model.endswith('GPTQ-Int4') for j in jobs)
    with pytest.raises(ValueError):
        replicate.reference_jobs(K1_MODEL, kind='other')


def test_gptqmodel_settings_match_our_quantizer():
    # Classic GPTQ as in our quant/gptq.py: asymmetric, no act-order, no GPTQModel extras.
    cfg = replicate.GPTQMODEL_CONFIG
    assert cfg == {'bits': 4, 'group_size': 128, 'sym': False, 'desc_act': False,
                   'act_group_aware': False, 'damp_percent': 0.01, 'static_groups': False,
                   'mse': 0.0, 'true_sequential': True, 'format': 'gptq_v2'}


def test_gptqmodel_reference_ids_moved_off_the_bf16_v1_run():
    # The 2026-11-07 reference (bf16, v1 asymmetric zeros) is superseded; ours keep their IDs.
    ref = replicate.reference_jobs(K1_MODEL)
    assert {j.lang: j.run_id for j in ref}['en'] != '8e8544b892fa'
    assert {j.lang: j.run_id for j in replicate.fr3_jobs(K1_MODEL)}['en'] == '7f6a3aa0fa4b'
    official = replicate.reference_jobs(K1_MODEL, kind='official')
    assert {j.lang: j.run_id for j in official}['en'] == '31c65f4ac746'


def test_gptqmodel_quantization_is_cached_and_uses_our_windows(tmp_path, monkeypatch):
    import sys
    import types

    import torch

    from whopays.config import RunConfig

    calls = {}

    class FakeModel:
        def quantize(self, calibration, batch_size):
            calls['calibration'] = calibration
            calls['batch_size'] = batch_size

        def save(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / 'model.safetensors').write_text('x')

    fake = types.SimpleNamespace(
        QuantizeConfig=lambda **kw: calls.setdefault('config', kw),
        GPTQModel=types.SimpleNamespace(load=lambda name, quantize_config, dtype: (
            calls.setdefault('loaded', (name, dtype)), FakeModel())[1]))
    monkeypatch.setitem(sys.modules, 'gptqmodel', fake)
    batch = {'input_ids': torch.tensor([[5, 6, 7]])}
    prov = {'sha256': 'abc'}
    monkeypatch.setattr('whopays.quant.calibration.batches_for',
                        lambda tok, source, mode, seed: ([batch], prov))
    cfg = RunConfig(model=K1_MODEL, task='flores_nll', lang='en', weight='w4_gptqmodel',
                    calib='multi')
    path, got = replicate.quantize_with_gptqmodel(cfg, tokenizer=None, cache=tmp_path)
    assert got == prov and (path / 'DONE').exists()
    assert calls['calibration'] == [{'input_ids': [5, 6, 7], 'attention_mask': [1, 1, 1]}]
    assert calls['config'] == replicate.GPTQMODEL_CONFIG
    assert calls['loaded'] == (K1_MODEL, torch.float16)
    calls.clear()
    again, _ = replicate.quantize_with_gptqmodel(cfg, tokenizer=None, cache=tmp_path)
    assert again == path and not calls  # reused, not re-quantized
    (path / 'model.safetensors').write_text('tampered')
    with pytest.raises(ValueError, match='changed'):
        replicate.quantize_with_gptqmodel(cfg, tokenizer=None, cache=tmp_path)


def test_gptqmodel_cache_must_be_private(tmp_path):
    shared = tmp_path / 'shared'
    shared.mkdir()
    shared.chmod(0o777)
    with pytest.raises(PermissionError, match='writable'):
        replicate._private_dir(shared)
    replicate._private_dir(tmp_path / 'fresh')
    assert (tmp_path / 'fresh').stat().st_mode & 0o777 == 0o700


def test_reference_runs_in_an_isolated_environment(tmp_path):
    done = mock.MagicMock(returncode=0)
    with (mock.patch.object(replicate, 'install', return_value=Path('/venv/bin/python')),
          mock.patch.object(replicate.subprocess, 'run', return_value=done) as run):
        replicate.run_reference(K1_MODEL, root=tmp_path, sync=False)
    argv = run.call_args.args[0]
    assert argv[:4] == ['/venv/bin/python', '-m', 'whopays.cli', 'run']
    assert '--no-sync' in argv


def test_gap_direction_counts_languages_worse_than_english():
    def cell(acc_by_lang):
        return pd.DataFrame([{'lang': lang, 'item_id': f'i{i}', 'correct': i < 100 * acc}
                             for lang, acc in acc_by_lang.items() for i in range(100)])

    table = replicate.gap_direction(cell({'en': .9, 'te': .7, 'de': .8}),
                                    cell({'en': .85, 'te': .5, 'de': .78}))
    row = table.set_index('lang')
    assert row.loc['te', 'gap_vs_english_pts'] == pytest.approx(-15.0)
    assert bool(row.loc['te', 'worse_than_english'])
    assert not bool(row.loc['de', 'worse_than_english'])


def _write_k1(root, verdicts=('PASS', 'PASS')):
    return pd.DataFrame({'task': ['belebele', 'mgsm'], 'K1': list(verdicts)})


def test_guard_blocks_downstream_runs_without_k1_pass(tmp_path):
    with (mock.patch.object(gates, 'k1_table', return_value=_write_k1(tmp_path, ('PASS', 'FAIL'))),
          pytest.raises(gates.GateError, match='K1')):
        gates.require_k1(tmp_path)


def test_guard_allows_after_k1_pass(tmp_path):
    with mock.patch.object(gates, 'k1_table', return_value=_write_k1(tmp_path)):
        assert gates.require_k1(tmp_path) is True


def test_guard_blocks_when_reference_never_ran(tmp_path):
    with (mock.patch.object(gates.hub_sync, 'pull_runs', return_value=False),
          pytest.raises(gates.GateError, match='K1')):
        gates.require_k1(tmp_path)


def test_guard_override_is_explicit_and_logged(tmp_path, caplog):
    with mock.patch.object(gates, 'k1_table', return_value=_write_k1(tmp_path, ('FAIL', 'FAIL'))):
        assert gates.require_k1(tmp_path, override='debugging on a limit=5 slice') is False
    assert 'debugging' in caplog.text
    assert json.loads((tmp_path / 'gate_overrides.jsonl').read_text())['gate'] == 'K1'


def test_fr3_check_runs_ours_then_reference_and_reports(monkeypatch, tmp_path):
    import whopays
    from whopays import gates
    calls = []
    monkeypatch.setattr(gates, "require_k1", lambda root: calls.append("k1"))
    monkeypatch.setattr(whopays, "run", lambda jobs, **kw: calls.append(("ours", len(jobs))))
    monkeypatch.setattr(replicate, "run_reference",
                        lambda model, kind="gptqmodel", **kw: calls.append(kind) or [kind])
    rows = {"ours": _nll(3, 2.0), "gptqmodel": _nll(3, 2.01), "official": _nll(3, 2.2)}
    monkeypatch.setattr(replicate, "load_runs",
                        lambda root, jobs: rows[jobs[0] if isinstance(jobs[0], str) else "ours"])
    table, verdict, official = replicate.fr3_check("m", root=tmp_path)
    assert calls[0] == "k1" and calls[1][0] == "ours" and calls[2:] == ["gptqmodel", "official"]
    assert verdict["FR3"] == "PASS" and len(official) == len(table)


def _flores(scale_by_lang, n=200, seed=0, run='r'):
    rng = np.random.default_rng(seed)
    base = rng.uniform(20, 60, n)
    return pd.DataFrame([{'lang': lang, 'item_id': f's{i}', 'nll_sum': base[i] * s, 'run_id': run}
                         for lang, s in scale_by_lang.items() for i in range(n)])


def test_nll_gap_direction_flags_languages_that_degrade_more_than_english():
    fp16 = _flores({'en': 1.0, 'te': 1.0, 'de': 1.0})
    treated = _flores({'en': 1.10, 'te': 1.30, 'de': 1.10})
    row = replicate.nll_gap_direction(fp16, treated, n_resamples=200).set_index('lang')
    assert row.loc['en', 'rel_increase_pct'] == pytest.approx(10.0)
    assert row.loc['te', 'gap_vs_english_pct'] == pytest.approx(20.0)
    assert bool(row.loc['te', 'worse_than_english'])
    assert not bool(row.loc['de', 'worse_than_english'])


def test_nll_gap_direction_averages_seeds_per_sentence():
    fp16 = _flores({'en': 1.0, 'te': 1.0})
    treated = pd.concat([_flores({'en': 1.1, 'te': 1.2}, run='a'),
                         _flores({'en': 1.1, 'te': 1.4}, run='b')])
    row = replicate.nll_gap_direction(fp16, treated, n_resamples=50).set_index('lang')
    assert row.loc['te', 'rel_increase_pct'] == pytest.approx(30.0)


def _belebele(acc_by_lang, n=400, run='r', bias=0.0):
    """Letter log-probs with the gold letter highest for the first acc*n items; optional
    constant preference for letter A that calibration must remove."""
    rows = []
    for lang, acc in acc_by_lang.items():
        for i in range(n):
            gold = i % 4
            lp = np.full(4, -3.0)
            lp[gold if i < acc * n else (gold + 1) % 4] = -0.5
            lp[0] += bias
            rows.append({'lang': lang, 'item_id': f'q{i}', 'gold': gold,
                         'letter_logprobs': lp.tolist(), 'run_id': run})
    return pd.DataFrame(rows)


def test_retention_gap_direction_uses_above_chance_share():
    # en keeps 0.9-0.25 -> 0.8-0.25: retention 0.846; te keeps 0.35-0.25 -> 0.3-0.25: 0.5.
    # In raw points te loses less (5 vs 10) but retains a smaller share above chance.
    fp16 = _belebele({'en': 0.9, 'te': 0.35})
    treated = _belebele({'en': 0.8, 'te': 0.3})
    row = replicate.retention_gap_direction(fp16, treated, n_resamples=200).set_index('lang')
    assert row.loc['en', 'retention'] == pytest.approx(0.55 / 0.65, abs=1e-6)
    assert row.loc['te', 'retention'] == pytest.approx(0.5, abs=1e-6)
    assert row.loc['te', 'gap_vs_english'] < 0


def test_retention_gap_direction_removes_a_constant_letter_prior():
    clean = replicate.retention_gap_direction(_belebele({'en': 0.9, 'te': 0.6}),
                                              _belebele({'en': 0.8, 'te': 0.5}),
                                              n_resamples=20).set_index('lang')
    biased = replicate.retention_gap_direction(_belebele({'en': 0.9, 'te': 0.6}),
                                               _belebele({'en': 0.8, 'te': 0.5}, bias=1.0),
                                               n_resamples=20).set_index('lang')
    assert biased.loc['te', 'retention'] == pytest.approx(clean.loc['te', 'retention'], abs=0.02)


def test_retention_undefined_without_headroom_above_chance():
    table = replicate.retention_gap_direction(_belebele({'en': 0.9, 'te': 0.25}),
                                              _belebele({'en': 0.8, 'te': 0.25}),
                                              n_resamples=20).set_index('lang')
    assert np.isnan(table.loc['te', 'retention'])
