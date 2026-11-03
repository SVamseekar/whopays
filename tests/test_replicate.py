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


def test_reference_jobs_use_the_real_checkpoint_everywhere():
    jobs = replicate.reference_jobs(K1_MODEL)
    assert {j.weight for j in jobs} == {'prequant'} and {j.task for j in jobs} == {'flores_nll'}
    assert {j.lang for j in jobs} == set(CODES)
    assert all(j.model.endswith('GPTQ-Int4') for j in jobs)


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
