"""Percentile bootstrap on parallel item clusters; never resample conditions separately."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    estimate: np.ndarray
    low: np.ndarray
    high: np.ndarray
    samples: np.ndarray


def paired_bootstrap(reference, treatment, *, n_resamples=2000, seed=0, strata=None,
                     confidence=0.95, statistic=None):
    """Treatment minus reference, rows=items, columns=parallel languages/conditions.

    A single row draw is shared by every column, preserving cross-language pairing
    and each language's sample size (language stratification). Optional ``strata``
    further stratifies item clusters. Inputs must already be aligned by item ID.
    The default statistic is the per-column mean; custom statistics receive the
    resampled difference array. CIs are percentile intervals, not hypothesis tests.
    """
    x, y = np.asarray(reference, dtype=float), np.asarray(treatment, dtype=float)
    if x.shape != y.shape or x.ndim not in (1, 2) or not len(x):
        raise ValueError('Expected matching nonempty item arrays')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Observations must be finite')
    if not isinstance(n_resamples, int) or n_resamples < 2 or not 0 < confidence < 1:
        raise ValueError('Invalid resampling count or confidence')
    labels = np.zeros(len(x), dtype=int) if strata is None else np.asarray(strata)
    if labels.shape != (len(x),):
        raise ValueError('strata must contain one label per item')
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    if sum(map(len, groups)) != len(x):
        raise ValueError('Missing stratum labels')
    rng = np.random.default_rng(seed)
    delta = y - x
    stat = statistic if statistic is not None else lambda a: a.mean(axis=0)
    samples = np.asarray([stat(delta[np.concatenate([
        rng.choice(g, len(g), replace=True) for g in groups])]) for _ in range(n_resamples)])
    alpha = (1 - confidence) / 2
    low, high = np.quantile(samples, [alpha, 1 - alpha], axis=0)
    return BootstrapResult(np.asarray(stat(delta)), low, high, samples)


def paired_frame_bootstrap(reference, treatment, *, value='correct', item='item_id',
                           language='language', **kwargs):
    """Strict rectangular item/language alignment; refuses missing/duplicate pairs.

    Output columns follow sorted language labels. Filter to a single model, task,
    calibration seed and config before calling; repeats are not independent items.
    """
    keys = [item, language]
    if reference.duplicated(keys).any() or treatment.duplicated(keys).any():
        raise ValueError('Duplicate item/language pairs')
    a = reference.pivot(index=item, columns=language, values=value).sort_index().sort_index(axis=1)
    b = treatment.pivot(index=item, columns=language, values=value).sort_index().sort_index(axis=1)
    if not a.index.equals(b.index) or not a.columns.equals(b.columns):
        raise ValueError('Expected matching parallel item/language sets')
    if a.isna().any().any() or b.isna().any().any():
        raise ValueError('Incomplete parallel item panel')
    return paired_bootstrap(a.to_numpy(), b.to_numpy(), **kwargs)


def benjamini_hochberg(pvalues):
    """BH adjusted p-values; supply the complete preregistered comparison family."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('p-values must be a finite vector in [0,1]')
    if not len(p):
        return p.copy()
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1)
    return result


def paired_binary_pvalues(reference, treatment):
    """Exact two-sided McNemar tests on matched binary outcomes, one per column.

    These are valid inputs to BH for per-language config comparisons, assuming
    independent item clusters; they are not tests of mixed-model interactions.
    Pass the complete language/config family to BH in one call.
    """
    from scipy.stats import binomtest

    a, b = np.asarray(reference), np.asarray(treatment)
    if (a.shape != b.shape or a.ndim not in (1, 2) or not len(a)
            or not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all()):
        raise ValueError('Expected matching nonempty binary item arrays')
    if a.ndim == 1:
        a, b = a[:, None], b[:, None]
    lost = ((a == 1) & (b == 0)).sum(axis=0)
    gained = ((a == 0) & (b == 1)).sum(axis=0)
    return np.asarray([binomtest(int(k), int(k + g)).pvalue if k + g else 1.0
                       for k, g in zip(lost, gained, strict=True)])
