"""Local exploration helpers: bit-band maps, scoring, comparison, letter Fisher."""
import numpy as np
import pytest

from whopays.parityquant import explore as ex

ITEM = {"passage": "The river town kept its records in a stone archive by the bridge.",
        "question": "Where were the records kept?", "options": ["bridge", "archive", "river",
                                                                "town"], "gold": 1}


def test_band_bitmap():
    bm = ex.band_bitmap(4, 4, range(1, 3), 3)
    assert bm == {"model.layers.0": 4, "model.layers.1": 3, "model.layers.2": 3,
                  "model.layers.3": 4}


def test_compare_against_itself_is_zero_and_kl_positive_when_changed():
    lp = np.log(np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]]))
    s = {"te": {"nll": np.array([5.0, 6.0]), "letter_lp": lp, "gold": np.array([0, 1])}}
    same = ex.compare(s, s)["te"]
    assert same["nll_increase"] == 0 and same["letter_kl"] == pytest.approx(0, abs=1e-12)
    assert same["raw"] == 100.0
    other = {"te": {**s["te"], "letter_lp": lp[::-1].copy(), "nll": np.array([6.0, 8.0])}}
    diff = ex.compare(other, s)["te"]
    assert diff["nll_increase"] == 1.5 and diff["letter_kl"] > 0


def test_score_and_letter_fisher_on_tiny_model(tiny_model_dir):
    from whopays.config import RunConfig
    from whopays.models.loader import load
    from whopays.parityquant import langgptq as lg

    bundle = load(RunConfig(model=str(tiny_model_dir), task="belebele", lang="en"), "cpu")
    flores = {"en": [{"text": "Das Wetter ist schön."}, {"text": "A B C D"}]}
    scores = ex.score(bundle, flores, {"en": [ITEM, ITEM]})
    assert scores["en"]["nll"].shape == (2,) and scores["en"]["letter_lp"].shape == (2, 4)
    fisher = lg.output_fisher(bundle.model,
                              {"en": ex.belebele_batches(bundle.tokenizer, [ITEM])},
                              ex.letter_fisher_loss(bundle.tokenizer))
    assert all(f.sum() > 0 for f in fisher["en"].values())
