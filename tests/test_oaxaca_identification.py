"""Identification checks for the descriptive pooled decomposition."""

import numpy as np
import pytest

from whopays.stats.oaxaca import oaxaca


def test_script_indicator_cannot_absorb_planted_treatment_gap():
    x = np.linspace(-1, 1, 40)
    reference = np.column_stack([x, np.zeros_like(x)])
    group = np.column_stack([x, np.ones_like(x)])
    with pytest.raises(ValueError, match='group.*script'):
        oaxaca(reference, 2 * x, group, 2 * x + 1, n_resamples=20)


def test_combination_of_varying_covariates_cannot_encode_group():
    x = np.linspace(-1, 1, 40)
    # Neither column alone is constant within a group; their difference is group.
    reference = np.column_stack([x, x])
    group = np.column_stack([x, x + 1])
    with pytest.raises(ValueError, match='group.*script'):
        oaxaca(reference, 2 * x, group, 2 * x + 1, n_resamples=20)


def test_shared_constant_reference_feature_is_allowed():
    x = np.linspace(-1, 1, 40)
    design = np.column_stack([x, np.ones_like(x), np.zeros_like(x)])
    result = oaxaca(design, 2 * x, design, 2 * x + 1, n_resamples=20)
    assert result['endowment']['estimate'] == pytest.approx(0)
    assert result['treatment']['estimate'] == pytest.approx(1)


def test_identifiable_shifted_design_preserves_descriptive_components():
    x = np.linspace(-1, 1, 40)[:, None]
    group = x + 0.25
    result = oaxaca(x, 2 * x[:, 0], group, 2 * group[:, 0], n_resamples=20)
    assert result['endowment']['estimate'] == pytest.approx(0.5)
    assert result['treatment']['estimate'] == pytest.approx(0, abs=1e-12)
