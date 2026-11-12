"""Power analysis on synthetic data with planted effects."""

import numpy as np
import pytest

from whopays.stats import power


def test_mde_shrinks_with_items_and_grows_with_discordance():
    assert power.paired_mde(900, 0.10) < power.paired_mde(250, 0.10)
    assert power.paired_mde(900, 0.20) > power.paired_mde(900, 0.10)


def test_mde_matches_textbook_value():
    # (1.96 + 0.8416) * sqrt(0.1 / 900) = 0.02954
    assert power.paired_mde(900, 0.10) == pytest.approx(0.02954, abs=2e-4)


def test_simulated_power_at_the_mde_is_close_to_target():
    n, disc = 900, 0.12
    delta = power.paired_mde(n, disc)
    hit = power.simulate_paired_power(n, disc, delta, n_sims=3000, seed=1)
    assert 0.75 < hit < 0.85


def test_simulated_power_without_effect_is_alpha():
    hit = power.simulate_paired_power(900, 0.12, 0.0, n_sims=4000, seed=2)
    assert 0.03 < hit < 0.07


PREMIUM = np.array([1.1, 1.2, 1.3, 1.5, 1.8, 2.2, 2.6, 3.0, 3.6, 4.2])
BASE = np.array([.85, .84, .82, .80, .75, .70, .62, .55, .50, .45])


def test_h1_power_high_for_strong_planted_slope():
    hit = power.simulate_h1_power(PREMIUM, BASE, slope=0.03, n_items=900, discordance=0.12,
                                  n_sims=60, n_resamples=200, seed=3)
    assert hit > 0.9


def test_h1_false_positive_rate_low_without_slope():
    hit = power.simulate_h1_power(PREMIUM, BASE, slope=0.0, n_items=900, discordance=0.12,
                                  n_sims=60, n_resamples=200, seed=4)
    assert hit < 0.25  # two correlations, either may exclude zero: about 2 x 5% plus noise
