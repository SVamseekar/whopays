import pandas as pd
import pytest

from whopays.languages import CODES
from whopays.report.langtable import MISSING, language_table


def _frame(langs=CODES):
    return pd.DataFrame([{"weight": w, "lang": code, "delta": i + (0.5 if w == "w3" else 0),
                          "low": i - 1, "high": i + 1}
                         for w in ("w4", "w3") for i, code in enumerate(langs)])


def test_all_languages_in_canonical_order():
    rows = language_table(_frame(), "delta", ["weight"]).splitlines()[2:]
    assert [r.split(" | ")[0].strip("| ") for r in rows] == list(CODES)
    assert "| lang | script | w4 | w3 |" in language_table(_frame(), "delta", ["weight"])


def test_missing_language_is_shown_not_dropped():
    table = language_table(_frame(CODES[:-1]), "delta", ["weight"])
    assert table.splitlines()[-1].startswith("| te | Telugu | " + MISSING)
    assert len(table.splitlines()) == 2 + len(CODES)


def test_interval_and_duplicates():
    table = language_table(_frame(), "delta", ["weight"], interval=("low", "high"), digits=1)
    assert "| en | Latin | 0.0 [-1.0, 1.0] | 0.5 [-1.0, 1.0] |" in table
    with pytest.raises(ValueError, match="duplicate"):
        language_table(_frame(), "delta")
    with pytest.raises(ValueError, match="unknown"):
        language_table(pd.DataFrame({"lang": ["xx"], "delta": [1.0]}), "delta")
