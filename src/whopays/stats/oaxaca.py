"""Pooled (Neumark) twofold linear decomposition with item bootstrap intervals."""
import numpy as np


def oaxaca(x_reference, y_reference, x_group, y_group, *, paired=True,
           n_resamples=2000, seed=0, confidence=0.95):
    """Decompose group minus reference outcome, adding an intercept automatically.

    Covariates must be numerical with common coding and column order. Parallel
    rows are jointly resampled by default; callers must align IDs first. Pooled
    coefficients exclude a group dummy. Endowment and treatment are descriptive
    components; treatment is the residual group gap, not a causal estimate.
    Covariates must not encode group membership, including through combinations
    of columns (for example, different group-constant script indicators). Other
    collinear covariates use the least-squares pseudoinverse, so component
    attribution depends on coding.
    """
    xa, xb = np.asarray(x_reference, dtype=float), np.asarray(x_group, dtype=float)
    ya, yb = np.asarray(y_reference, dtype=float), np.asarray(y_group, dtype=float)
    if (xa.ndim != 2 or xb.ndim != 2 or xa.shape[1] != xb.shape[1]
            or ya.shape != (len(xa),) or yb.shape != (len(xb),)
            or min(len(xa), len(xb)) < 2 or (paired and len(xa) != len(xb))):
        raise ValueError('Invalid outcome/covariate shapes or unmatched paired items')
    if not all(np.isfinite(v).all() for v in [xa, xb, ya, yb]):
        raise ValueError('Inputs must be finite')
    if not isinstance(n_resamples, int) or n_resamples < 2 or not 0 < confidence < 1:
        raise ValueError('Invalid resampling count or confidence')
    xa, xb = [np.column_stack([np.ones(len(x)), x]) for x in [xa, xb]]
    design = np.concatenate([xa, xb])
    indicator = np.concatenate([np.zeros(len(xa)), np.ones(len(xb))])
    # Test the entire column span, including the intercept, rather than only
    # detecting individual group-constant columns. Normalize for mixed units.
    norms = np.linalg.norm(design, axis=0)
    scaled = design[:, norms > 0] / norms[norms > 0]
    fitted = scaled @ np.linalg.lstsq(scaled, indicator, rcond=None)[0]
    tolerance = 10 * np.finfo(float).eps * max(scaled.shape) * np.linalg.norm(indicator)
    if np.linalg.norm(indicator - fitted) <= tolerance:
        raise ValueError(
            'Pooled covariates encode the group indicator and mechanically force '
            'the treatment component to zero; remove group-identifying covariates '
            'such as script for pairwise decomposition'
        )

    def estimate(a, b):
        beta = np.linalg.lstsq(np.concatenate([xa[a], xb[b]]),
                               np.concatenate([ya[a], yb[b]]), rcond=None)[0]
        gap = yb[b].mean() - ya[a].mean()
        endowment = (xb[b].mean(axis=0) - xa[a].mean(axis=0)) @ beta
        return np.array([gap, endowment, gap - endowment])

    point = estimate(np.arange(len(xa)), np.arange(len(xb)))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        a = rng.integers(len(xa), size=len(xa))
        b = a if paired else rng.integers(len(xb), size=len(xb))
        draws.append(estimate(a, b))
    low, high = np.quantile(draws, [(1 - confidence) / 2, (1 + confidence) / 2], axis=0)
    return {name: {'estimate': float(point[i]), 'low': float(low[i]), 'high': float(high[i])}
            for i, name in enumerate(['gap', 'endowment', 'treatment'])}
