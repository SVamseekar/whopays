"""Item-random-intercept logistic regression using statsmodels Bayesian GLMM."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


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
