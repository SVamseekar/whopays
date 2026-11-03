"""Item-level logistic summaries for the preregistered factorial.

``fit_mixed_logistic`` is a descriptive conditional fit. Its variational
standard deviations are not sampling variances and are not inputs to
Benjamini–Hochberg. ``interaction_tests`` is the confirmatory H2 procedure:
population-averaged logit interactions with item-clustered p-values.
"""
import re

import numpy as np
import pandas as pd
import patsy
from scipy.stats import norm
from scipy.stats import t as student_t
from statsmodels.formula.api import gee
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.genmod.cov_struct import Independence
from statsmodels.genmod.families import Binomial

from whopays.stats.bootstrap import benjamini_hochberg

_COLUMN = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def fit_mixed_logistic(data, *, formula='correct ~ W * K * C(language)',
                       item='item_id', confidence=0.95):
    """Fit actual binomial mixed model with deterministic variational initialization.

    Intervals are approximate Bayesian credible intervals from mean-field VB,
    NOT frequentist Wald CIs. No p-values are manufactured for BH; preregistered
    frequentist interaction tests require a separately justified inference method.
    W/K should be numeric binary contrasts or explicitly categorical in formula.
    """
    if not 0 < confidence < 1 or item not in data or data[item].isna().any():
        raise ValueError('Invalid confidence or missing item IDs')
    if data.isna().any().any():
        raise ValueError('Missing observations must be resolved explicitly before model fitting')
    model = BinomialBayesMixedGLM.from_formula(formula, {'item': f'0 + C({item})'}, data,
                                              vcp_p=1, fe_p=2)
    if not np.isin(model.endog, [0, 1]).all() or len(np.unique(model.endog)) != 2:
        raise ValueError('Outcome must contain both binary classes')
    n = model.k_fep + model.k_vcp + model.k_vc
    fit = model.fit_vb(mean=np.zeros(n), sd=np.full(n, 0.1))
    if not fit.optim_retvals.success:
        raise RuntimeError(f'Mixed model did not converge: {fit.optim_retvals.message}')
    z = norm.ppf((1 + confidence) / 2)
    table = pd.DataFrame({'term': model.exog_names, 'estimate': fit.fe_mean,
                          'sd': fit.fe_sd, 'low': fit.fe_mean - z * fit.fe_sd,
                          'high': fit.fe_mean + z * fit.fe_sd})
    return {'coefficients': table, 'fit': fit,
            'inference_note': 'Bayesian mean-field variational credible intervals; '
                              'not frequentist p-values or BH-ready interaction tests.'}


def interaction_tests(data, *, item='item_id', language='language', weight='W',
                      cache='K', outcome='correct', q=0.05):
    """Language-specific weight × KV logit interactions and their BH decisions.

    The mean model is ``correct ~ W * K * language``. Dependence within an item
    is handled by a binomial GEE with an independence working correlation and
    the Mancl–DeRouen bias-reduced cluster covariance, grouped by item. Each
    language contributes one contrast, logit(W=1,K=1) − logit(W=1,K=0) −
    logit(W=0,K=1) + logit(W=0,K=0). Two-sided p-values use a t reference with
    degrees of freedom equal to the number of items minus one. Benjamini–Hochberg
    at ``q`` is applied only to this family.

    These are population-averaged log-odds interactions, not the conditional
    coefficients of a random-intercept GLMM. Pinned statsmodels has no
    frequentist likelihood for that GLMM, and its conditional logit does not
    return a covariance for this saturated within-item design. A cell with no
    outcome variation makes the logit interaction undefined; that is an error,
    not a language to drop.
    """
    if not 0 < q < 1:
        raise ValueError('q must be between 0 and 1')
    frame = _interaction_frame(data, item, language, weight, cache, outcome)
    languages = sorted(frame[language].unique())
    n_items = int(frame[item].nunique())
    if n_items < 2:
        raise ValueError('interaction tests need at least two items')
    # Patsy's first category is the reference. Contrasts below do not depend on it.
    frame[language] = pd.Categorical(frame[language], categories=languages)
    # GEE treats a repeated group id as a new cluster unless each item is one block.
    frame = frame.sort_values([item, language, weight, cache], kind='mergesort')
    frame = frame.reset_index(drop=True)
    formula = f'{outcome} ~ {weight} * {cache} * C({language})'
    fit = gee(formula, item, frame, family=Binomial(), cov_struct=Independence()).fit(
        cov_type='bias_reduced')
    if fit is None or not fit.converged:
        raise RuntimeError('clustered logit did not converge')
    design = fit.model.data.model_spec
    names = list(design.column_names)
    params = np.array([fit.params[name] for name in names], dtype=float)
    covariance = np.asarray(fit.cov_params().loc[names, names], dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    df = n_items - 1
    rows = [_language_interaction(design, params, covariance, lang, weight, cache, language, df)
            for lang in languages]
    adjusted = benjamini_hochberg([row['pvalue'] for row in rows])
    for row, p_bh in zip(rows, adjusted, strict=True):
        row['p_bh'] = float(p_bh)
        row['reject'] = bool(p_bh <= q)
    table = pd.DataFrame(rows)
    return {'tests': table, 'q': q, 'df': df, 'n_items': n_items, 'fit': fit,
            'family': 'language-specific weight × KV population-averaged logit interactions',
            'inference_note': 'Mancl–DeRouen item-clustered GEE Wald tests; '
                              'not variational GLMM posterior tails.'}


def _interaction_frame(data, item, language, weight, cache, outcome):
    columns = (item, language, weight, cache, outcome)
    if any(not _COLUMN.fullmatch(name) for name in columns):
        raise ValueError('analysis column names must be identifiers')
    if len(set(columns)) != len(columns):
        raise ValueError('item, language, factors and outcome need distinct columns')
    missing = [name for name in columns if name not in data.columns]
    if missing:
        raise ValueError(f'missing columns: {missing}')
    frame = data.loc[:, columns].copy()
    if frame.isna().any().any():
        raise ValueError('missing observations must be resolved before interaction tests')
    if set(np.unique(frame[weight])) != {0, 1} or set(np.unique(frame[cache])) != {0, 1}:
        raise ValueError('weight and cache factors must be coded 0/1')
    if not np.isin(frame[outcome], [0, 1]).all():
        raise ValueError('outcome must be binary 0/1')
    keys = [item, language, weight, cache]
    if frame.duplicated(keys).any():
        raise ValueError('duplicate item/language/factor cells')
    languages = sorted(frame[language].unique())
    n_items = int(frame[item].nunique())
    sizes = frame.groupby(item, observed=True).size()
    if (sizes != 4 * len(languages)).any():
        raise ValueError('incomplete W×K×language panel within items')
    for lang in languages:
        part = frame[frame[language] == lang]
        for level_w in (0, 1):
            for level_k in (0, 1):
                outcomes = part.loc[(part[weight] == level_w) & (part[cache] == level_k), outcome]
                if len(outcomes) != n_items or outcomes.nunique() < 2:
                    raise ValueError(
                        f'complete separation in {lang} at {weight}={level_w}, '
                        f'{cache}={level_k}; logit interaction is undefined')
    return frame


def _language_interaction(design, params, covariance, lang, weight, cache, language, df):
    grid = pd.DataFrame({weight: [0, 1, 0, 1], cache: [0, 0, 1, 1], language: [lang] * 4})
    exog = np.asarray(patsy.build_design_matrices([design], grid)[0], dtype=float)
    contrast = exog[3] - exog[1] - exog[2] + exog[0]
    estimate = float(contrast @ params)
    variance = float(contrast @ covariance @ contrast)
    scale = float(np.trace(np.abs(covariance))) + 1.0
    if variance < 0:
        if variance < -1e-8 * scale:
            raise RuntimeError('interaction covariance is not positive semidefinite')
        variance = 0.0
    if variance == 0:
        if abs(estimate) > 1e-8:
            raise RuntimeError('nonzero interaction with singular item-cluster covariance')
        se, statistic, pvalue = 0.0, 0.0, 1.0
    else:
        se = float(np.sqrt(variance))
        statistic = estimate / se
        pvalue = float(2 * student_t.sf(abs(statistic), df))
    return {'language': lang, 'estimate': estimate, 'se': se, 'statistic': statistic,
            'df': df, 'pvalue': pvalue}
