"""Inequality requires nonnegative harms; do not silently shift signed changes."""
import numpy as np


def _values(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Expected a nonempty finite nonnegative vector')
    return a


def gini(values):
    a = np.sort(_values(values))
    if a.sum() == 0:
        return 0.0
    return float(2 * np.dot(np.arange(1, len(a) + 1), a) / (len(a) * a.sum())
                 - (len(a) + 1) / len(a))


def atkinson(values, epsilon=1.0):
    a = _values(values)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError('epsilon must be finite and nonnegative')
    if a.mean() == 0:
        return 0.0
    if epsilon >= 1 and (a == 0).any():
        return 1.0
    equivalent = (np.exp(np.log(a).mean()) if epsilon == 1 else
                  np.mean(a ** (1 - epsilon)) ** (1 / (1 - epsilon)))
    return float(1 - equivalent / a.mean())


def inequality_summary(quality_changes):
    """Signed worst/spread; Gini/Atkinson explicitly computed on max(0,-delta).

    Improvements are zero harm. This truncation is reported in the output, and
    never changes the signed worst degradation (negative if every language improves).
    """
    delta = np.asarray(quality_changes, dtype=float)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError('Expected a nonempty finite change vector')
    harms = np.maximum(0, -delta)
    return {'worst_degradation': float(-delta.min()), 'spread': float(np.ptp(delta)),
            'gini': gini(harms), 'atkinson': atkinson(harms),
            'inequality_domain': 'nonnegative harm max(0, -quality_change)'}
