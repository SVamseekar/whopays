"""Blinded native-speaker rating sheets for MGSM generations."""

import pandas as pd
import pytest

from whopays.report import spotcheck


def _rows(lang, n, flip_every):
    fp16 = pd.DataFrame({"lang": lang, "item_id": [f"{lang}{i}" for i in range(n)],
                         "gold": range(n), "correct": True,
                         "output_text": [f"fp16 {i}" for i in range(n)]})
    treated = fp16.copy()
    treated["correct"] = [i % flip_every != 0 for i in range(n)]
    treated["output_text"] = [f"w4 {i}" for i in range(n)]
    return fp16, treated


def test_sheet_is_blinded_balanced_and_keyed():
    f1, t1 = _rows("te", 60, 4)
    f2, t2 = _rows("bn", 60, 4)
    sheet, key = spotcheck.rating_sheet(pd.concat([f1, f2]), pd.concat([t1, t2]), per_lang=10)
    assert sheet.groupby("lang").size().to_dict() == {"bn": 10, "te": 10}
    assert {"text_a", "text_b", "a_language_ok", "b_reasoning_ok"} <= set(sheet.columns)
    assert not {"correct", "fp16_is", "output_text"} & set(sheet.columns)
    merged = sheet.merge(key, on="sheet_id")
    for _, r in merged.iterrows():
        assert r["text_a" if r.fp16_is == "a" else "text_b"].startswith("fp16")
    assert 0 < (merged.fp16_is == "a").mean() < 1
    assert merged.flipped.sum() == 10  # half of each language's sample, where available


def test_sheet_is_reproducible():
    f, t = _rows("te", 40, 3)
    a, _ = spotcheck.rating_sheet(f, t, per_lang=8, seed=3)
    b, _ = spotcheck.rating_sheet(f, t, per_lang=8, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_score_unblinds_ratings():
    f, t = _rows("te", 40, 2)
    sheet, key = spotcheck.rating_sheet(f, t, per_lang=6)
    for col in ("a_language_ok", "a_reasoning_ok"):
        sheet[col] = (key.set_index("sheet_id").loc[sheet.sheet_id, "fp16_is"] == "a").to_numpy()
    for col in ("b_language_ok", "b_reasoning_ok"):
        sheet[col] = (key.set_index("sheet_id").loc[sheet.sheet_id, "fp16_is"] == "b").to_numpy()
    out = spotcheck.score(sheet, key)
    row = out.set_index("lang").loc["te"]
    assert row.fp16_language_ok == 1.0 and row.treated_language_ok == 0.0
    assert row.n == 6


def test_mismatched_items_rejected():
    f, t = _rows("te", 10, 2)
    with pytest.raises(ValueError, match="same items"):
        spotcheck.rating_sheet(f, t.iloc[:5], per_lang=4)
