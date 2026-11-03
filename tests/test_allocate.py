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


def test_matched_uniform_ignores_sensitivity_and_breaks_ties_toward_early_layers():
    counts = [1, 1, 1, 1]
    bits = [3, 4]
    sensitive = np.zeros((4, 2, 2))
    sensitive[-1, 0, :] = 100
    first = allocate(sensitive, counts, 3.5, bits=bits, objective='matched_uniform')
    second = allocate(np.ones_like(sensitive), counts, 3.5, bits=bits, objective='matched_uniform')
    assert first.bits == second.bits == (4, 4, 3, 3)
    best = None
    for choice in itertools.product(range(2), repeat=4):
        selected = tuple(bits[i] for i in choice)
        if sum(counts[j] * selected[j] for j in range(4)) != 14:
            continue
        deviation = sum(p * (b - 3.5) ** 2 for p, b in zip(counts, selected, strict=True)) / 4
        tie = sum((j + 1) * b for j, b in enumerate(selected))
        candidate = (deviation, tie, selected)
        best = candidate if best is None or candidate < best else best
    assert first.bits == best[2]
    assert first.objective_value == pytest.approx(best[0])
    # Widths 4 and 2 beat 2 and 8 on squared error, and sensitivities are not an input.
    forced = allocate(np.zeros((2, 4, 1)), [3, 1], 3.5, bits=[2, 3, 4, 8],
                      objective='matched_uniform')
    assert forced.bits == (4, 2)
    palette = allocate(np.zeros((2, 3, 1)), [3, 5], 3, bits=[2, 3, 4], objective='matched_uniform')
    assert palette.bits == (3, 3)
    assert palette.objective_value == 0
    with pytest.raises(AllocationError):
        allocate(np.zeros((1, 2, 1)), [1], 3.5, bits=[3, 4], objective='matched_uniform')


def test_36_layer_weighted_budget_and_negative_losses():
    s = -np.random.default_rng(13).uniform(size=(36, 4, 11))
    counts = np.full(36, 1_000_000)
    result = allocate(s, counts, 3.5, layer_names=[f'layer.{i}' for i in range(36)])
    assert sum(result.bits) == 126
    assert result.objective_value < 0
    assert result.bit_map['layer.0'] == result.bits[0]


def test_lock_pins_one_solver_stack_on_every_python():
    # PuLP 4 removed PULP_CBC_CMD and the bundled CBC, so a Python-specific pin breaks Kaggle.
    from pathlib import Path

    lock = (Path(__file__).parents[1] / 'requirements.lock').read_text().splitlines()
    pins = [line for line in lock if line.startswith('pulp==')]
    assert len(pins) == 1 and pins[0].startswith('pulp==3.') and ';' not in pins[0]
