import itertools

import numpy as np
import pandas as pd
import pytest

from whopays.stats.bootstrap import benjamini_hochberg, paired_bootstrap, paired_frame_bootstrap
from whopays.stats.inequality import atkinson, gini, inequality_summary
from whopays.stats.mixed import fit_mixed_logistic
from whopays.stats.oaxaca import oaxaca
from whopays.stats.shapley import pair_interactions, shapley_values


def test_bootstrap_preserves_pairing_and_language_dependence():
    x = np.random.default_rng(5).normal(size=(80, 3))
    a = paired_bootstrap(x, x + [1, 2, 3], strata=np.arange(80) % 2)
    b = paired_bootstrap(x, x + [1, 2, 3], strata=np.arange(80) % 2)
    np.testing.assert_allclose(a.estimate, [1, 2, 3])
    np.testing.assert_allclose(a.low, a.high)
    np.testing.assert_array_equal(a.samples, b.samples)
    noisy = paired_bootstrap(np.zeros_like(x), np.tile(x[:, :1], (1, 3)))
    np.testing.assert_array_equal(noisy.samples[:, 0], noisy.samples[:, 2])


def test_frame_alignment_and_missing_pair_rejected():
    ref = pd.DataFrame({'item_id': [1, 2, 1, 2], 'language': ['en', 'en', 'te', 'te'],
                        'correct': [0, 1, 0, 1]})
    alt = ref.iloc[::-1].copy()
    assert np.all(paired_frame_bootstrap(ref, alt).estimate == 0)
    with pytest.raises(ValueError, match='parallel|matching'):
        paired_frame_bootstrap(ref, alt.iloc[1:])


def test_bh_and_invalid_inputs():
    np.testing.assert_allclose(benjamini_hochberg([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])
    with pytest.raises(ValueError):
        benjamini_hochberg([np.nan])
    with pytest.raises(ValueError):
        paired_bootstrap([1], [1, 2])


def test_shapley_recovers_additive_and_interactions():
    players = ['W', 'K', 'S']
    cells = {}
    for n in range(4):
        for subset in itertools.combinations(players, n):
            s = frozenset(subset)
            cells[s] = 10 + sum({'W': 1, 'K': 2, 'S': 3}[p] for p in s)
            cells[s] += 4 * ({'W', 'K'} <= s)
    phi = shapley_values(cells)
    assert phi == pytest.approx({'W': 3, 'K': 4, 'S': 3})
    assert pair_interactions(cells)[('K', 'W')] == 4
    del cells[frozenset()]
    with pytest.raises(ValueError):
        shapley_values(cells)


def test_oaxaca_pooled_planted_endowment_and_pairing():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(300, 2))
    shifted = x + [1, 0]
    a = oaxaca(x, x @ [2, 1], shifted, shifted @ [2, 1], n_resamples=100)
    assert a['gap']['estimate'] == pytest.approx(2)
    assert a['endowment']['low'] <= 2 + 1e-12
    assert a['endowment']['high'] >= 2 - 1e-12
    assert abs(a['treatment']['estimate']) < 1e-12
    assert a['endowment']['estimate'] + a['treatment']['estimate'] == pytest.approx(2)


def test_inequality_domain_and_hand_values():
    assert gini([0, 1]) == 0.5
    assert atkinson([0, 1]) == 1
    assert atkinson([1, 4]) == pytest.approx(0.2)
    assert gini([0, 0]) == atkinson([0, 0]) == 0
    with pytest.raises(ValueError):
        gini([-1, 1])
    assert inequality_summary([-0.1, 0.2])['worst_degradation'] == 0.1


def test_mixed_logistic_recovers_strong_planted_effect():
    rng = np.random.default_rng(2)
    rows = []
    for item in range(60):
        random_intercept = rng.normal(scale=0.5)
        for w, k, lang in itertools.product([0, 1], [0, 1], ['en', 'te']):
            logit = -1 + 3 * w + random_intercept
            rows.append((item, w, k, lang, rng.binomial(1, 1 / (1 + np.exp(-logit)))))
    data = pd.DataFrame(rows, columns=['item_id', 'W', 'K', 'language', 'correct'])
    result = fit_mixed_logistic(data)
    effect = result['coefficients'].set_index('term').loc['W']
    assert effect['low'] < 3 < effect['high']
    assert effect['low'] > 0
    assert 'variational' in result['inference_note']


def test_oaxaca_planted_treatment_gap():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(300, 2))
    noise = rng.normal(scale=0.1, size=300)
    result = oaxaca(x, x @ [2, 1] + noise, x, x @ [2, 1] + noise + 1,
                    n_resamples=100)
    assert abs(result['endowment']['estimate']) < 1e-12
    assert result['treatment']['estimate'] == pytest.approx(1)
    assert result['treatment']['low'] <= 1 + 1e-12
    assert result['treatment']['high'] >= 1 - 1e-12


def test_paired_binary_exact_pvalues():
    from whopays.stats.bootstrap import paired_binary_pvalues

    reference = np.ones((10, 2))
    treatment = reference.copy()
    treatment[:, 0] = 0
    np.testing.assert_allclose(paired_binary_pvalues(reference, treatment), [2 / 1024, 1])
