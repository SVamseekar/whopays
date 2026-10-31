"""Exact factorial attribution; incomplete coalitions are an error."""
from itertools import combinations
from math import factorial

import numpy as np


def _validate(values):
    cells = {frozenset(k): float(v) for k, v in values.items()}
    players = sorted(set().union(*cells)) if cells else []
    expected = {frozenset(s) for n in range(len(players) + 1)
                for s in combinations(players, n)}
    if cells.keys() != expected or not all(np.isfinite(v) for v in cells.values()):
        raise ValueError('All finite coalition values, including the empty coalition, are required')
    return cells, players


def shapley_values(values):
    cells, players = _validate(values)
    n = len(players)
    return {p: sum(factorial(len(s)) * factorial(n - len(s) - 1) / factorial(n)
                   * (cells[s | {p}] - value)
                   for s, value in cells.items() if p not in s) for p in players}


def pair_interactions(values):
    cells, players = _validate(values)
    return {(a, b): cells[frozenset((a, b))] - cells[frozenset((a,))]
            - cells[frozenset((b,))] + cells[frozenset()]
            for a, b in combinations(players, 2)}
