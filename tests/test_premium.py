import pandas as pd
import pytest

from whopays.tokens import premium


def test_premium_table_recovers_planted_ratios():
    en = {f"p{i}": 10 + i for i in range(20)}
    counts = {
        "en": en,
        "de": {k: 1.5 * v for k, v in en.items()},
        "te": {k: 7 * v for k, v in en.items()},
    }
    table = premium.premium_table(counts).set_index("lang")
    assert table.loc["en", "premium_total"] == 1
    assert table.loc["de", "premium_total"] == pytest.approx(1.5)
    assert table.loc["te", "premium_median"] == pytest.approx(7)
    assert table.loc["te", "n_passages"] == 20


def test_premium_table_rejects_non_parallel():
    with pytest.raises(ValueError, match="not parallel"):
        premium.premium_table({"en": {"a": 1, "b": 2}, "te": {"a": 3}})


def test_counts_are_cached(tmp_path, monkeypatch):
    calls = []

    def fake_counts(tok, lang):
        calls.append(lang)
        return {"p1": 10, "p2": 20} if lang == "en" else {"p1": 20, "p2": 40}

    monkeypatch.setattr(premium, "passage_counts", fake_counts)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda name: None)
    a = premium.belebele_premium("org/tok", tmp_path)
    b = premium.belebele_premium("org/tok", tmp_path)
    assert len(calls) == 11  # second call read the cache
    pd.testing.assert_frame_equal(a, b)
    assert (tmp_path / "org__tok.parquet").exists()
    assert set(a[a.lang != "en"].premium_total) == {2.0}


@pytest.mark.slow
def test_qwen_premium_table(tmp_path):
    table = premium.belebele_premium("Qwen/Qwen2.5-0.5B-Instruct", tmp_path)
    assert len(table) == 11 and table.n_passages.nunique() == 1
    assert table.set_index("lang").loc["en", "premium_total"] == 1
