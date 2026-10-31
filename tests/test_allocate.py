import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from whopays.parityquant.allocate import AllocationError, allocate


@pytest.mark.parametrize('objective', ['minimax', 'average', 'softmax'])
def test_allocation_matches_exhaustive_optimum(objective):
    rng = np.random.default_rng(3)
    s = rng.random((4, 3, 3))
    counts = [1, 2, 1, 2]
    bits = [2, 3, 4]
    result = allocate(s, counts, 3, bits=bits, objective=objective, temperature=0.7)
    candidates = []
    for choice in itertools.product(range(3), repeat=4):
        if sum(counts[j] * bits[b] for j, b in enumerate(choice)) == 18:
            losses = sum(s[j, b] for j, b in enumerate(choice))
            candidates.append({'minimax': np.max(losses), 'average': np.mean(losses),
                               'softmax': 0.7 * logsumexp(losses / 0.7)}[objective])
    assert result.objective_value == pytest.approx(min(candidates), abs=1e-6)
    assert sum(p * b for p, b in zip(counts, result.bits, strict=True)) == 18


def test_uniform_and_infeasible_budget():
    s = np.zeros((2, 2, 2))
    assert allocate(s, [3, 5], 3, bits=[2, 3], objective='uniform').bits == (3, 3)
    with pytest.raises(AllocationError):
        allocate(s, [3, 5], 2.5, bits=[2, 3], objective='uniform')
    with pytest.raises(AllocationError):
        allocate(s, [3, 5], 2.1, bits=[2, 3])
    with pytest.raises(ValueError):
        allocate(s, [0, 1], 3, bits=[2, 3])


def test_36_layer_weighted_budget_and_negative_losses():
    s = -np.random.default_rng(13).uniform(size=(36, 4, 11))
    counts = np.full(36, 1_000_000)
    result = allocate(s, counts, 3.5, layer_names=[f'layer.{i}' for i in range(36)])
    assert sum(result.bits) == 126
    assert result.objective_value < 0
    assert result.bit_map['layer.0'] == result.bits[0]
