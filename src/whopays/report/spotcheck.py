"""Blinded rating sheets so native speakers can check MGSM generations by hand.

Automatic scoring only reads the final number. A native speaker can see what it misses: an
answer in the wrong language or script, broken grammar, or right number by wrong reasoning.
Each sheet row shows the FP16 and the treated output for one problem in random order
(A/B), so the rater cannot tell which is quantized; the key is kept in a separate file.
Half of each language's sample comes from items whose correctness flipped, where
quantization changed something, and half from the rest.

Workflow: `rating_sheet` -> send `sheet.csv` to a rater -> keep `key.csv` -> `score`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RATINGS = ("a_language_ok", "a_reasoning_ok", "b_language_ok", "b_reasoning_ok")


def rating_sheet(fp16: pd.DataFrame, treated: pd.DataFrame, *, per_lang: int = 20,
                 seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(sheet for raters, unblinding key). Rows need lang, item_id, gold, correct, output_text."""
    cols = ["lang", "item_id", "gold", "correct", "output_text"]
    pair = fp16[cols].merge(treated[cols], on=["lang", "item_id"], suffixes=("_f", "_t"),
                            validate="one_to_one")
    if len(pair) != len(fp16) or len(pair) != len(treated):
        raise ValueError("FP16 and treated rows must cover the same items")
    pair["flipped"] = pair.correct_f != pair.correct_t
    rng = np.random.default_rng(seed)
    picked = []
    for lang in sorted(pair.lang.unique()):
        g = pair[pair.lang == lang].sort_values("item_id")
        flips, rest = g[g.flipped], g[~g.flipped]
        n_flip = min(len(flips), per_lang // 2)
        n_rest = min(len(rest), per_lang - n_flip)
        n_flip = min(len(flips), per_lang - n_rest)
        picked.append(pd.concat([
            flips.iloc[rng.choice(len(flips), n_flip, replace=False)] if n_flip else flips[:0],
            rest.iloc[rng.choice(len(rest), n_rest, replace=False)] if n_rest else rest[:0]]))
    sample = pd.concat(picked).sample(frac=1, random_state=seed).reset_index(drop=True)
    fp16_is = np.where(rng.random(len(sample)) < 0.5, "a", "b")
    sample["sheet_id"] = [f"s{i:04d}" for i in range(len(sample))]
    sheet = pd.DataFrame({
        "sheet_id": sample.sheet_id, "lang": sample.lang, "item_id": sample.item_id,
        "gold_answer": sample.gold_f,
        "text_a": np.where(fp16_is == "a", sample.output_text_f, sample.output_text_t),
        "text_b": np.where(fp16_is == "a", sample.output_text_t, sample.output_text_f),
        **{col: "" for col in RATINGS}, "notes": ""})
    key = pd.DataFrame({"sheet_id": sample.sheet_id, "fp16_is": fp16_is,
                        "flipped": sample.flipped, "fp16_correct": sample.correct_f,
                        "treated_correct": sample.correct_t})
    return sheet, key


def score(sheet: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """Per language: share of FP16 and treated outputs rated language-OK and reasoning-OK."""
    df = sheet.merge(key, on="sheet_id", validate="one_to_one")
    out = {}
    for what in ("language_ok", "reasoning_ok"):
        a, b = df[f"a_{what}"].astype(bool), df[f"b_{what}"].astype(bool)
        df[f"fp16_{what}"] = np.where(df.fp16_is == "a", a, b)
        df[f"treated_{what}"] = np.where(df.fp16_is == "a", b, a)
        out[f"fp16_{what}"] = "mean"
        out[f"treated_{what}"] = "mean"
    table = df.groupby("lang").agg(**{k: (k, v) for k, v in out.items()},
                                   n=("sheet_id", "size"), flipped=("flipped", "sum"))
    return table.reset_index()
