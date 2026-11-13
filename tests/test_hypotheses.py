"""H1-H4 on synthetic rows with planted effects (golden rule 11). No research measurements."""
import numpy as np
import pandas as pd
import pytest

from whopays.report import hypotheses as hyp

LANGS = ['en', 'de', 'ru', 'zh', 'th', 'bn', 'te']
PREMIUM = {'en': 1.0, 'de': 1.2, 'ru': 1.5, 'zh': 1.8, 'th': 2.2, 'bn': 2.8, 'te': 3.5}


def _cell(acc, *, weight='fp16', kv='full', n=400, seed=0, tokens=None, calib='none'):
    """acc: language -> item probability of correct; items are shared across languages."""
    rng = np.random.default_rng(seed)
    uniform = rng.random(n)
    rows = []
    for lang, p in acc.items():
        for i in range(n):
            base = 100 * (tokens[lang] if tokens else PREMIUM[lang]) + i % 7
            rows.append(dict(model='m', task='belebele', lang=lang, item_id=f'i{i:03d}',
                             correct=bool(uniform[i] < p), weight=weight, kv=kv, spec=False,
                             calib=calib, seed=0, limit=None, gpu='Tesla T4', input_tokens=base,
                             gold_logprob=float(-uniform[i])))
    return pd.DataFrame(rows)


def test_h1_recovers_a_gap_that_grows_with_token_premium():
    # Shared uniforms make each language's lost items a nested set, so the gap is monotone.
    fp16 = _cell({lang: 0.9 for lang in LANGS})
    treated = _cell({lang: 0.9 - 0.08 * (PREMIUM[lang] - 1) for lang in LANGS}, weight='w4_gptq')
    result = hyp.h1(fp16, treated, n_resamples=200)
    assert result['token_premium']['rho'] == pytest.approx(1.0)
    assert result['token_premium']['low'] > 0
    assert result['supported'] is True


def test_h1_is_falsified_when_both_intervals_include_zero():
    fp16 = _cell({lang: 0.9 for lang in LANGS})
    treated = _cell(dict.fromkeys(LANGS, 0.9), weight='w4_gptq', seed=1)
    result = hyp.h1(fp16, treated, n_resamples=200)
    assert result['token_premium']['low'] <= 0 <= result['token_premium']['high']
    assert result['fp16_accuracy']['rho'] is None or result['supported'] is False


def _factorial(effect_te=0.0):
    """Independent item noise per cell, so outcomes vary in every cell; planted te interaction."""
    frames = []
    for w, weight in enumerate(('fp16', 'w4_gptq')):
        for k, kv in enumerate(('full', 'evict')):
            acc = {lang: 0.75 - 0.05 * w - 0.05 * k for lang in ('en', 'te')}
            acc['te'] -= effect_te * w * k
            frames.append(_cell(acc, weight=weight, kv=kv, seed=10 * w + k + 1))
    return pd.concat(frames, ignore_index=True)


def test_h2_detects_the_planted_interaction_only_where_planted():
    table = hyp.h2(_factorial(0.45), 'w4_gptq', 'evict')['tests'].set_index('language')
    assert bool(table.loc['te', 'reject'])
    assert not bool(table.loc['en', 'reject'])


def test_h2_refuses_a_missing_cell():
    rows = _factorial()
    rows = rows[~((rows.weight == 'w4_gptq') & (rows.kv == 'evict'))]
    with pytest.raises(ValueError, match='cell'):
        hyp.h2(rows, 'w4_gptq', 'evict')


def _oaxaca_rows(degrade, tokens):
    n = 300
    rng = np.random.default_rng(3)
    fp16, treated = [], []
    for lang in ('en', 'te'):
        for i in range(n):
            size = tokens[lang] * (1 + 0.5 * rng.random())
            base = dict(model='m', task='belebele', lang=lang, item_id=f'i{i:03d}', kv='full',
                        spec=False, limit=None, gpu='Tesla T4', seed=0,
                        input_tokens=size, gold_logprob=float(-rng.random()))
            lost = rng.random() < degrade(lang, size)
            fp16.append({**base, 'weight': 'fp16', 'calib': 'none', 'correct': True})
            treated.append({**base, 'weight': 'w4_gptq', 'calib': 'multi', 'correct': not lost})
    return pd.DataFrame(fp16), pd.DataFrame(treated)


def test_h3_assigns_a_token_driven_gap_to_endowment():
    fp16, treated = _oaxaca_rows(lambda lang, size: min(1.0, size / 800),
                                 {'en': 100, 'te': 400})
    row = hyp.h3(fp16, treated, n_resamples=50).set_index('lang').loc['te']
    assert row.endowment_share > 0.5


def test_h3_assigns_a_same_token_gap_to_treatment():
    fp16, treated = _oaxaca_rows(lambda lang, size: 0.5 if lang == 'te' else 0.05,
                                 {'en': 100, 'te': 100})
    row = hyp.h3(fp16, treated, n_resamples=50).set_index('lang').loc['te']
    assert abs(row.endowment_share) < 0.2


def test_h3_item_length_is_the_english_token_count_for_both_groups(monkeypatch):
    fp16, treated = _oaxaca_rows(lambda lang, size: 0.1, {'en': 100, 'te': 400})
    captured = {}

    def spy(x_reference, y_reference, x_group, y_group, **kwargs):
        captured.update(reference=x_reference, group=x_group)
        part = {'estimate': 1.0, 'low': 0.0, 'high': 2.0}
        return {'gap': part, 'endowment': part, 'treatment': part}

    monkeypatch.setattr(hyp, 'oaxaca', spy)
    hyp.h3(fp16, treated, n_resamples=2)
    english = hyp._panel(fp16, 'input_tokens')['en'].to_numpy()
    assert np.array_equal(captured['reference'][:, 1], english)
    assert np.array_equal(captured['group'][:, 1], english)
    assert not np.array_equal(captured['group'][:, 0], english)


def test_h3_verdict_counts_non_latin_languages_below_half():
    table = pd.DataFrame({'lang': ['ru', 'th', 'te', 'de', 'bn'],
                          'endowment_share': [0.1, 0.2, 0.4, 0.1, 0.9]})
    assert hyp.h3_supported(table) is True
    assert hyp.h3_supported(table[table.lang != 'te']) is False


def test_h3_undefined_share_does_not_count():
    table = pd.DataFrame({'lang': ['ru', 'th', 'te'],
                          'endowment_share': [0.1, 0.2, float('nan')]})
    assert hyp.h3_supported(table) is False


def _parity(worst_by_method, n=500):
    frames = []
    rng = np.random.default_rng(5)
    uniform = rng.random(n)
    for method, acc in worst_by_method.items():
        for lang, p in acc.items():
            frames.append(pd.DataFrame({'item_id': [f'i{i:03d}' for i in range(n)], 'lang': lang,
                                        'correct': uniform < p, 'method': method}))
    return pd.concat(frames, ignore_index=True)


def _budget_rows(minimax_te):
    return _parity({'fp16': {'en': .9, 'te': .9},
                    'minimax': {'en': .84, 'te': minimax_te},
                    'average': {'en': .88, 'te': .6},
                    'uniform': {'en': .85, 'te': .65}})


def test_h4_recovers_planted_worst_language_gain_at_low_mean_cost():
    row = hyp.h4_budget(_budget_rows(0.82), n_resamples=300)
    assert row['improvement'] == pytest.approx(0.9 - 0.65 - (0.9 - 0.82), abs=0.03)
    assert row['improvement_low'] > 0
    assert row['best_baseline'] == 'uniform'


def test_h4_price_of_parity_compares_with_the_average_objective():
    row = hyp.h4_budget(_budget_rows(0.82), n_resamples=100)
    # Mean accuracy: average 0.74, minimax 0.83, so parity is negative-priced here.
    assert row['price_of_parity'] == pytest.approx(0.74 - 0.83, abs=0.03)
    assert row['price_low'] <= row['price_of_parity'] <= row['price_high']


def test_h4_verdict_needs_two_budgets_clear_of_zero():
    clear = {'improvement_low': .01, 'improvement_high': .1, 'mean_cost': .0}
    crossing = {'improvement_low': -.01, 'improvement_high': .1, 'mean_cost': .0}
    assert hyp.k3_verdict([clear, clear, crossing]) == 'POSITIVE'
    assert hyp.k3_verdict([clear, crossing, crossing]) == 'NEGATIVE_RESULT'


def test_h4_refuses_excess_mean_cost():
    costly = {'improvement_low': .01, 'improvement_high': .1, 'mean_cost': .02}
    assert hyp.k3_verdict([costly, costly, costly]) == 'NEGATIVE_RESULT'


def test_language_config_table_applies_bh_over_the_whole_family():
    fp16 = _cell({'en': .9, 'te': .9})
    a = _cell({'en': .9, 'te': .4}, weight='w4_gptq', calib='multi')
    b = _cell({'en': .9, 'te': .9}, weight='w8_rtn')
    table = hyp.language_config_table(pd.concat([fp16, a, b], ignore_index=True), n_resamples=50)
    assert len(table) == 4
    hit = table[(table.weight == 'w4_gptq') & (table.lang == 'te')].iloc[0]
    assert hit.delta == pytest.approx(-0.5, abs=0.06) and bool(hit.reject)
    assert hit.high < 0
    assert not table[table.weight == 'w8_rtn'].reject.any()


def test_attribution_needs_all_eight_cells():
    frames = []
    for w, weight in enumerate(('fp16', 'w4_gptq')):
        for k, kv in enumerate(('full', 'evict')):
            for s in (False, True):
                acc = 0.9 - 0.1 * w - 0.2 * k
                frame = _cell({'te': acc}, weight=weight, kv=kv)
                frames.append(frame.assign(spec=s))
    rows = pd.concat(frames, ignore_index=True)
    shapley = hyp.attribution(rows, 'te', 'w4_gptq', 'evict')
    assert shapley['W'] == pytest.approx(-0.1, abs=0.03)
    assert shapley['K'] == pytest.approx(-0.2, abs=0.03)
    assert shapley['S'] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError, match='cell'):
        hyp.attribution(rows[rows.kv == 'full'], 'te', 'w4_gptq', 'evict')


def test_relative_loss_is_the_share_of_above_chance_accuracy_lost():
    base = np.array([0.9, 0.35, 0.25])
    treat = np.array([0.8, 0.30, 0.25])
    loss = hyp.relative_loss(base, treat, chance=0.25)
    assert loss[0] == pytest.approx(0.1 / 0.65)
    assert loss[1] == pytest.approx(0.5)
    assert np.isnan(loss[2])  # no headroom above chance: undefined, not zero


def test_h1_takes_chance_from_the_task():
    fp16 = _cell({lang: 0.9 for lang in LANGS})
    # Same raw loss everywhere (0.1); relative loss above chance is also equal at equal base.
    treated = _cell({lang: 0.8 for lang in LANGS}, weight='w4_gptq')
    assert hyp.CHANCE['belebele'] == 0.25 and hyp.CHANCE['mgsm'] == 0.0
    result = hyp.h1(fp16, treated, n_resamples=50)
    assert result['chance'] == 0.25
    with pytest.raises(ValueError, match='chance'):
        hyp.h1(fp16.drop(columns='task'), treated.drop(columns='task'), n_resamples=10)


def _nll_parity(increase_by_method, n=400):
    rng = np.random.default_rng(3)
    base = rng.gamma(20, 2, size=n)
    frames = []
    for method, inc in increase_by_method.items():
        for lang, delta in inc.items():
            frames.append(pd.DataFrame({'item_id': [f'i{i:03d}' for i in range(n)], 'lang': lang,
                                        'nll_sum': base + delta + rng.normal(0, 0.5, n),
                                        'method': method}))
    return pd.concat(frames, ignore_index=True)


def test_h4_on_nll_treats_lower_as_better():
    rows = _nll_parity({'fp16': {'en': 0, 'te': 0}, 'minimax': {'en': 6, 'te': 8},
                        'average': {'en': 3, 'te': 20}, 'uniform': {'en': 5, 'te': 15}})
    row = hyp.h4_budget(rows, column='nll_sum', higher_is_better=False, n_resamples=200)
    assert row['best_baseline'] == 'uniform'
    assert row['improvement'] == pytest.approx(15 - 8, abs=0.3)
    assert row['improvement_low'] > 0
    assert row['worst_minimax'] == pytest.approx(8, abs=0.3)
