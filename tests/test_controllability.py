"""Controllability gate on synthetic rows with planted effects."""
import numpy as np
import pandas as pd

from whopays.parityquant import controllability as ct

M = "m"
LANGS = ("en", "te", "bn")


def _flores(effects):
    rows = []
    for lang in LANGS:
        for i in range(600):
            rows.append({"model": M, "task": "flores_nll", "lang": lang, "weight": "fp16",
                         "calib": "none", "seed": 0, "item_id": f"{lang}{i:04d}",
                         "nll_sum": 10.0})
    for (calib, seed), per_lang in effects.items():
        for lang in LANGS:
            for i in range(600):
                # Items before the held-out cut get a huge value: they must be ignored.
                bump = 1000.0 if i < ct.HELD_OUT_FROM else per_lang[lang]
                rows.append({"model": M, "task": "flores_nll", "lang": lang,
                             "weight": "w3_gptq", "calib": calib, "seed": seed,
                             "item_id": f"{lang}{i:04d}", "nll_sum": 10.0 + bump})
    return pd.DataFrame(rows)


def _effects(te_own):
    multi = {("multi", s): {"en": 20.0, "te": 85.0 + 3 * s, "bn": 70.0} for s in (0, 1, 2)}
    own = {("te", s): {"en": 60.0, "te": te_own + s, "bn": 90.0} for s in (0, 1)}
    return multi | own


def test_held_out_items_only_and_planted_increase():
    inc = ct.nll_increases(_flores(_effects(40.0)), M)
    te0 = inc[(inc.calib == "te") & (inc.seed == 0) & (inc.lang == "te")].iloc[0]
    assert te0.increase == 40.0 and te0.n_items == 100


def test_gate_passes_only_beyond_seed_spread():
    strong = ct.gate(ct.nll_increases(_flores(_effects(40.0)), M), targets=("te",)).iloc[0]
    assert strong.verdict == "CONTROLLABLE" and strong.spread == 6.0
    assert strong.worst11_own == 90.0  # bn rises; reported, not gated
    weak = ct.gate(ct.nll_increases(_flores(_effects(80.0)), M), targets=("te",)).iloc[0]
    assert weak.verdict == "NOT_CONTROLLABLE"  # margin 4 < spread 6
    missing = ct.gate(ct.nll_increases(_flores(_effects(40.0)), M), targets=("bn",)).iloc[0]
    assert missing.verdict == "INCOMPLETE"


def test_belebele_raw_and_calibrated():
    lp = [[0.0, -5, -5, -5]] * 3 + [[-5, 0.0, -5, -5]]
    rows = pd.DataFrame({"model": M, "task": "belebele", "lang": "te", "weight": "w3_gptq",
                         "calib": "te", "seed": 0, "gold": [0, 1, 2, 3],
                         "letter_logprobs": [np.array(x) for x in lp]})
    acc = ct.belebele_accuracy(rows, M).iloc[0]
    assert acc.raw == 25.0 and 0 <= acc.calibrated <= 100
