"""K2 gate and sensitivity-probe plumbing. Planted vectors; no research measurements."""
import numpy as np
import pandas as pd
import pytest

from whopays.parityquant.probe import (
    PRIMARY_BITS,
    k2_verdict,
    merge_halves,
    probe_texts,
    sensitivity_tensor,
    solve,
)


def _rows(vectors, bits=PRIMARY_BITS, parameters=100):
    return [{'layer': f'model.layers.{j}', 'bits': bits, 'lang': lang, 'sensitivity': float(v),
             'parameters': parameters}
            for lang, vector in vectors.items() for j, v in enumerate(vector)]


def test_identical_rankings_fail_k2():
    v = np.arange(1, 9.0)
    table = k2_verdict(_rows({'en': v, 'de': 2 * v, 'te': v ** 2}))
    assert table.min_spearman == pytest.approx(1.0)
    assert table.K2 == 'FAIL'


def test_reversed_ranking_passes_k2_and_names_the_pair():
    v = np.arange(1, 9.0)
    table = k2_verdict(_rows({'en': v, 'de': v, 'te': v[::-1]}))
    assert table.min_spearman == pytest.approx(-1.0)
    assert table.K2 == 'PASS'
    assert set(table.min_pair) == {'en', 'te'} or 'te' in table.min_pair


def test_other_widths_do_not_decide_k2():
    # An 8-bit vector is mostly rounding noise; it must not make K2 pass on its own.
    v = np.arange(1, 9.0)
    rows = _rows({'en': v, 'de': v}) + _rows({'en': v, 'de': v[::-1]}, bits=8)
    assert k2_verdict(rows).K2 == 'FAIL'


def test_noise_ceiling_flags_differences_within_sampling_error():
    rng = np.random.default_rng(0)
    halves = {f'{lang}#{h}': rng.normal(size=12) for lang in ('en', 'de', 'te') for h in 'ab'}
    table = k2_verdict(_rows(halves))
    assert table.K2 == 'PASS'  # the PRD threshold stays literal
    assert bool(table.noise_limited)


def test_stable_language_differences_are_not_noise_limited():
    v = np.arange(1, 13.0)
    halves = {'en#a': v, 'en#b': v + 0.01 * np.sin(v), 'te#a': v[::-1], 'te#b': v[::-1]}
    table = k2_verdict(_rows(halves))
    assert table.split_half_min == pytest.approx(1.0)
    assert not bool(table.noise_limited)


def test_merge_halves_averages_each_language():
    merged = merge_halves(_rows({'en#a': [1.0], 'en#b': [3.0]}))
    assert [(r['lang'], r['sensitivity']) for r in merged] == [('en', 2.0)]


def test_tensor_refuses_a_partial_profile():
    rows = _rows({'en': [1.0, 2.0], 'de': [1.0]})
    with pytest.raises(ValueError, match='incomplete'):
        sensitivity_tensor(rows)


def test_tensor_orders_layers_numerically():
    rows = _rows({'en': np.arange(11.0)})
    layers, bits, langs, tensor, params = sensitivity_tensor(rows)
    assert layers[2] == 'model.layers.2' and layers[10] == 'model.layers.10'
    assert tensor[10, 0, 0] == 10.0
    assert params == [100] * 11 and bits == [PRIMARY_BITS] and langs == ['en']


def test_tensor_axes_are_layer_bit_language():
    rows = [{'layer': f'model.layers.{j}', 'bits': b, 'lang': lang, 'parameters': 1,
             'sensitivity': 100.0 * j + 10.0 * i + k}
            for j in range(3) for i, b in enumerate((2, 4)) for k, lang in enumerate(('de', 'en'))]
    _, bits, langs, tensor, _ = sensitivity_tensor(rows)
    assert (bits, langs) == ([2, 4], ['de', 'en']) and tensor.shape == (3, 2, 2)
    assert tensor[2, 1, 0] == 210.0 and tensor[1, 0, 1] == 101.0


def test_solve_meets_budget_and_returns_block_bitmap():
    rows = []
    for bit, scale in ((2, 4.0), (3, 2.0), (4, 1.0), (8, 0.1)):
        rows += _rows({'en': scale * np.array([1, 2, 3, 4.0]),
                       'te': scale * np.array([4, 3, 2, 1.0])}, bits=bit)
    result = solve(rows, 3.5, 'minimax')
    assert result.average_bits == 3.5
    assert set(result.bit_map) == {f'model.layers.{j}' for j in range(4)}


def test_probe_texts_are_deterministic_disjoint_halves(monkeypatch):
    def fake_flores(lang, split):
        assert split == 'devtest'  # calibration text must never appear here
        return [{'item_id': f'i{n:02d}', 'text': f'{lang}{n}'} for n in reversed(range(30))]

    monkeypatch.setattr('whopays.datasets.flores', fake_flores)
    texts = probe_texts(limit=10, languages=['en', 'te'])
    assert set(texts) == {'en#a', 'en#b', 'te#a', 'te#b'}
    assert texts['en#a'] == [f'en{n}' for n in (0, 2, 4, 6, 8)]
    assert texts['en#b'] == [f'en{n}' for n in (1, 3, 5, 7, 9)]
    assert not set(texts['en#a']) & set(texts['en#b'])


def test_run_probe_persists_and_resumes(tiny_model_dir, tmp_path, monkeypatch):
    from whopays.config import RunConfig
    from whopays.models.loader import load
    from whopays.parityquant.probe import load_profile, run_probe

    texts = {'en#a': ['the river'], 'en#b': ['the town'], 'de#a': ['Das'], 'de#b': ['Wetter']}
    monkeypatch.setattr('whopays.parityquant.probe.probe_texts', lambda **kw: texts)
    monkeypatch.setattr('whopays.parityquant.probe._calibration', lambda *a: None)

    def loader(cfg):
        assert isinstance(cfg, RunConfig) and cfg.weight == 'fp16'
        return load(cfg, 'cpu')

    kwargs = dict(model=str(tiny_model_dir), bits=(2,), method='rtn', root=tmp_path, sync=False,
                  loader=loader)
    first = run_probe(**kwargs)
    assert len(first) == 2 * 4
    assert run_probe(**kwargs) == first
    assert load_profile(tmp_path, str(tiny_model_dir), bits=(2,), method='rtn') == first
    assert isinstance(pd.DataFrame(first).sensitivity.iloc[0], float)
