"""All-language markdown tables: every per-language result is shown for all 11 languages.

Rows are languages in the canonical order of `whopays.languages.LANGUAGES`, with script, so a
reader sees the whole distribution rather than the extremes. A language with no value is shown
as "missing", never dropped.
"""
from __future__ import annotations

import pandas as pd

from whopays.languages import CODES, LANGUAGES

MISSING = "missing"


def language_table(frame: pd.DataFrame, value: str, by: list[str] | None = None, *,
                   lang: str = "lang", digits: int = 2, interval: tuple[str, str] | None = None,
                   ) -> str:
    """Markdown table with one row per language and one column per combination of `by`.

    `interval` names (low, high) columns, shown as "value [low, high]". Duplicate cells raise,
    because a table cell must hold one logged value.
    """
    by = list(by or [])
    unknown = sorted(set(frame[lang]) - set(CODES))
    if unknown:
        raise ValueError(f"unknown language codes {unknown}")
    if frame.duplicated([*by, lang]).any():
        raise ValueError(f"duplicate rows per {[*by, lang]}; filter first")

    def cell(row) -> str:
        text = f"{row[value]:.{digits}f}"
        if interval:
            low, high = interval
            text += f" [{row[low]:.{digits}f}, {row[high]:.{digits}f}]"
        return text

    frame = frame.assign(_cell=frame.apply(cell, axis=1))
    if by:
        frame = frame.assign(_col=frame[by].astype(str).agg(" / ".join, axis=1))
        columns = list(dict.fromkeys(frame["_col"]))
    else:
        frame = frame.assign(_col=value)
        columns = [value]
    wide = frame.pivot(index=lang, columns="_col", values="_cell")

    header = ["lang", "script", *columns]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for code in CODES:
        values = [wide.at[code, c] if code in wide.index and pd.notna(wide.at[code, c])
                  else MISSING for c in columns]
        lines.append("| " + " | ".join([code, LANGUAGES[code].script, *values]) + " |")
    return "\n".join(lines)
