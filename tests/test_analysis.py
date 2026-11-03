"""Report regeneration from released Parquet, on synthetic rows only."""
import numpy as np
import pandas as pd
import pytest

from whopays.report import analysis

LANGS = ['en', 'de', 'ru', 'zh', 'th', 'bn', 'te']


def _write_run(root, run_id, *, weight='fp16', kv='full', calib='none', task='belebele',
               gpu='Tesla T4', drop=0.0, model='m', limit=None, n=120):
    rng = np.random.default_rng(sum(map(ord, weight + kv)))
    rows = []
    for lang_index, lang in enumerate(LANGS):
        noise = rng.random(n)
        for i in range(n):
            rows.append(dict(run_id=f'{run_id}{lang}', model=model, task=task, lang=lang,
                             item_id=f'i{i}', correct=bool(noise[i] < 0.8 - drop * lang_index),
                             weight=weight, kv=kv, spec=False, calib=calib, seed=0, limit=limit,
                             gpu=gpu,
                             input_tokens=(50 + i % 9) * (1 + 0.2 * lang_index)
                             + (i * lang_index) % 5,
                             gold_logprob=float(-noise[i]), bitmap=None, budget=None))
    frame = pd.DataFrame(rows)
    folder = root / run_id
    folder.mkdir(parents=True)
    frame.to_parquet(folder / 'items.parquet')


def _release(root):
    _write_run(root, 'a')
    _write_run(root, 'b', weight='w4_gptq', calib='multi', drop=0.03)
    _write_run(root, 'c', kv='evict', drop=0.02)
    _write_run(root, 'd', weight='w4_gptq', kv='evict', calib='multi', drop=0.05)


def test_research_rows_drop_local_and_reference_runs_visibly(tmp_path, capsys):
    _release(tmp_path)
    _write_run(tmp_path, 'smoke', gpu='mps', limit=20, n=20)
    _write_run(tmp_path, 'ref', weight='prequant')
    rows = analysis.research_rows(analysis.load_parquet(tmp_path))
    assert set(rows.gpu) == {'Tesla T4'} and 'prequant' not in set(rows.weight)
    assert 'excluded' in capsys.readouterr().out


def test_audit_report_writes_every_table_and_marks_pending(tmp_path):
    _release(tmp_path / 'release')
    out = tmp_path / 'out'
    summary = analysis.audit_report(tmp_path / 'release', out, n_resamples=30)
    assert (out / 'language_config.csv').exists()
    assert (out / 'h1.csv').exists() and (out / 'h2.csv').exists() and (out / 'h3.csv').exists()
    assert (out / 'inequality.csv').exists()
    table = pd.read_csv(out / 'language_config.csv')
    assert {'delta', 'low', 'high', 'p_bh'} <= set(table.columns)
    # A configuration without its full factorial is recorded, not silently skipped.
    assert 'w3_gptq/evict' in summary['pending']['h2']


def test_regeneration_refuses_a_release_with_no_research_rows(tmp_path):
    _write_run(tmp_path, 'smoke', gpu='mps', limit=20, n=20)
    with pytest.raises(ValueError, match='research'):
        analysis.audit_report(tmp_path, tmp_path / 'out', n_resamples=5)


def test_timed_regenerate_reports_the_ten_minute_requirement(tmp_path):
    _release(tmp_path / 'release')
    result = analysis.timed_regenerate(tmp_path / 'release', tmp_path / 'out', n_resamples=20)
    assert result['within_limit'] is True and result['seconds'] < 600
    assert result['artifacts'] >= 1
