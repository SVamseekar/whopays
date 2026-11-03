"""Exact parameter-weighted bit budgets with CBC mixed-integer optimization.

``matched_uniform`` ignores sensitivities. It is the fractional-budget baseline:
the feasible assignment closest to the target average in parameter-weighted
squared width error, with higher widths placed on earlier layers when that
error ties.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd
from time import monotonic

import numpy as np
import pulp
from scipy.special import logsumexp, softmax


class AllocationError(RuntimeError):
    """Budget is infeasible or the solver did not certify the requested optimum."""


@dataclass(frozen=True)
class Allocation:
    bits: tuple[int, ...]
    bit_map: dict[str, int]
    average_bits: float
    language_losses: tuple[float, ...]
    objective_value: float
    objective: str
    optimality_gap: float = 0.0


def _matched_uniform_choice(counts, widths, budget, target, time_limit):
    """Indices of the sensitivity-free exact-budget assignment.

    Primary criterion: parameter-weighted squared deviation from ``budget``.
    Tie-break: smaller sum of (layer index + 1) × width, which puts higher
    widths on earlier layers. Both passes use integer coefficients.
    """
    denom, numer = budget.denominator, budget.numerator

    def solve(sse_cap, label):
        model = pulp.LpProblem('matched_uniform', pulp.LpMinimize)
        chosen = {(j, b): pulp.LpVariable(f'{label}_{j}_{b}', cat='Binary')
                  for j in range(len(counts)) for b in range(len(widths))}
        for j in range(len(counts)):
            model += pulp.lpSum(chosen[j, b] for b in range(len(widths))) == 1
        coefficients = {(j, b): counts[j] * widths[b] for j, b in chosen}
        divisor = reduce(gcd, [target, *coefficients.values()])
        model += (pulp.lpSum((v // divisor) * chosen[k] for k, v in coefficients.items())
                  == target // divisor)
        sse = pulp.lpSum(counts[j] * (widths[b] * denom - numer) ** 2 * chosen[j, b]
                         for j, b in chosen)
        if sse_cap is None:
            model += sse
        else:
            model += sse == sse_cap
            model += pulp.lpSum((j + 1) * widths[b] * chosen[j, b] for j, b in chosen)
        status = model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=1))
        if status != pulp.LpStatusOptimal or model.sol_status != pulp.LpSolutionOptimal:
            raise AllocationError(
                f'No certified matched-uniform allocation: {pulp.LpStatus[status]}')
        return [max(range(len(widths)), key=lambda b: chosen[j, b].value() or 0)
                for j in range(len(counts))]

    provisional = solve(None, 'u')
    cap = sum(counts[j] * (widths[provisional[j]] * denom - numer) ** 2
              for j in range(len(counts)))
    return solve(cap, 'v')


def allocate(sensitivities, parameter_counts, budget, *, bits=(2, 3, 4, 8),
             objective='minimax', layer_names=None, temperature=1.0,
             time_limit=60, tolerance=1e-6, max_iterations=500):
    """Allocate one width per layer; tensor axes are (layer, bit, language).

    ``parameter_counts`` count the weights actually quantized; excluded embeddings
    and heads are not in this average. Sensitivities are additive NLL increases
    and may be signed. Uniform requires the budget to equal an available width.
    ``matched_uniform`` meets a fractional budget without reading sensitivities.
    Softmax minimizes temperature * logsumexp(language_losses / temperature),
    via convex tangent cuts and MILP lower bounds until a certified absolute gap.
    No budget slack, rounding or fallback to a different objective is permitted.
    """
    s = np.asarray(sensitivities, dtype=float)
    raw_counts, raw_bits = list(parameter_counts), list(bits)
    if (not raw_counts or not raw_bits or any(int(p) != p or p <= 0 for p in raw_counts)
            or any(int(b) != b or b < 2 or b > 16 for b in raw_bits)):
        raise ValueError('Positive integer counts and bit widths in [2,16] required')
    counts, widths = list(map(int, raw_counts)), list(map(int, raw_bits))
    if (s.ndim != 3 or s.shape[:2] != (len(counts), len(widths)) or s.shape[2] == 0
            or not np.isfinite(s).all() or len(set(widths)) != len(widths)):
        raise ValueError('Invalid sensitivity tensor or duplicate widths')
    if objective not in {'minimax', 'average', 'uniform', 'softmax', 'matched_uniform'}:
        raise ValueError('Unknown allocation objective')
    if (not np.isfinite(temperature) or temperature <= 0 or time_limit <= 0
            or tolerance <= 0 or max_iterations < 1):
        raise ValueError('Invalid solver controls')
    names = list(layer_names) if layer_names is not None else [str(j) for j in range(len(counts))]
    if len(names) != len(counts) or len(set(names)) != len(names):
        raise ValueError('Layer names must be unique and match layers')
    target_average = Fraction(str(budget))
    target = target_average * sum(counts)
    if target.denominator != 1:
        raise AllocationError('Exact weighted bit budget is infeasible (nonintegral bit count)')
    target = int(target)

    def score(loss):
        if objective in {'minimax', 'uniform'}:
            return float(np.max(loss))
        if objective == 'average':
            return float(np.mean(loss))
        return float(temperature * logsumexp(loss / temperature))

    def result(choice, gap=0.0):
        selected = tuple(widths[b] for b in choice)
        actual = sum(p * b for p, b in zip(counts, selected, strict=True))
        if actual != target:
            raise AllocationError('Solver returned an allocation violating the exact budget')
        loss = s[np.arange(len(counts)), choice].sum(axis=0)
        if objective == 'matched_uniform':
            # Report the criterion that was optimized, not a sensitivity score.
            deviation = sum(p * (b - float(target_average)) ** 2
                            for p, b in zip(counts, selected, strict=True)) / sum(counts)
            value = float(deviation)
        else:
            value = score(loss)
        return Allocation(selected, dict(zip(names, selected, strict=True)), actual / sum(counts),
                          tuple(map(float, loss)), value, objective, gap)

    if objective == 'uniform':
        if target_average not in widths:
            raise AllocationError('Uniform allocation cannot meet this exact budget')
        return result([widths.index(int(budget))] * len(counts))
    if objective == 'matched_uniform':
        return result(_matched_uniform_choice(counts, widths, target_average, target, time_limit))
    model = pulp.LpProblem('parityquant', pulp.LpMinimize)
    x = {(j, b): pulp.LpVariable(f'x_{j}_{b}', cat='Binary')
         for j in range(len(counts)) for b in range(len(widths))}
    for j in range(len(counts)):
        model += pulp.lpSum(x[j, b] for b in range(len(widths))) == 1
    # Divide integer coefficients by their GCD to avoid billion-scale numerics.
    coefficients = {(j, b): counts[j] * widths[b] for j, b in x}
    divisor = reduce(gcd, [target, *coefficients.values()])
    model += pulp.lpSum((v // divisor) * x[k] for k, v in coefficients.items()) == target // divisor
    losses = [pulp.lpSum(float(s[j, b, lang]) * x[j, b] for j, b in x)
              for lang in range(s.shape[2])]
    t = pulp.LpVariable('objective', lowBound=None)
    model += t
    if objective in {'minimax', 'softmax'}:
        for loss in losses:
            model += t >= loss
    else:
        model += t == pulp.lpSum(losses) / len(losses)
    start = monotonic()
    best, upper = None, float('inf')
    for _iteration in range(max_iterations):
        remaining = time_limit - (monotonic() - start)
        if remaining <= 0:
            raise AllocationError('Solver time limit reached without certified optimum')
        status = model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=remaining, threads=1))
        if status != pulp.LpStatusOptimal or model.sol_status != pulp.LpSolutionOptimal:
            raise AllocationError(f'No certified optimal allocation: {pulp.LpStatus[status]}')
        choice = [max(range(len(widths)), key=lambda b: x[j, b].value())
                  for j in range(len(counts))]
        candidate = result(choice)
        if objective != 'softmax':
            return candidate
        if candidate.objective_value < upper:
            best, upper = choice, candidate.objective_value
        lower = float(pulp.value(t))
        gap = max(0.0, upper - lower)
        if gap <= tolerance:
            return result(best, gap)
        point = np.asarray(candidate.language_losses)
        gradient = softmax(point / temperature)
        model += t >= candidate.objective_value + pulp.lpSum(
            float(gradient[lang]) * (losses[lang] - float(point[lang]))
            for lang in range(len(losses)))
    raise AllocationError('Softmax outer approximation did not converge within max_iterations')
