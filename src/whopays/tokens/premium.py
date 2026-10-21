"""Token premium (FR-9): tokens a language needs for parallel text, relative to English.

Measured on Belebele passages, deduplicated (several questions share one passage). Two
summaries per language:
  premium_total   sum of tokens in l / sum of tokens in English, over the same passages
  premium_median  median of per-passage ratios, robust to a few very long passages
Per-passage counts are kept too: they are the token-count covariate in the Oaxaca-Blinder
decomposition. Counts depend only on the tokenizer, never on hardware.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from whopays.languages import CODES


def passage_counts(tokenizer, lang: str) -> dict[str, int]:
    """Tokens per unique Belebele passage, keyed by passage ID (the item ID minus `-qN`)."""
    from whopays.datasets import belebele

    counts: dict[str, int] = {}
    for item in belebele(lang):
        pid = item["item_id"].rsplit("-q", 1)[0]
        if pid not in counts:
            counts[pid] = len(tokenizer(item["passage"], add_special_tokens=False)["input_ids"])
    return counts


def premium_table(counts: dict[str, dict[str, int]]) -> pd.DataFrame:
    """One row per language from {lang: {passage_id: tokens}}; English must be present."""
    en = pd.Series(counts["en"], dtype=float)
    rows = []
    for lang, c in counts.items():
        s = pd.Series(c, dtype=float)
        shared = s.index.intersection(en.index)
        if len(shared) != len(en):
            raise ValueError(f"{lang}: passages are not parallel with English")
        rows.append({
            "lang": lang,
            "n_passages": len(shared),
            "total_tokens": int(s[shared].sum()),
            "premium_total": s[shared].sum() / en[shared].sum(),
            "premium_median": float((s[shared] / en[shared]).median()),
        })
    return pd.DataFrame(rows)


def belebele_premium(tokenizer_name: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """Premium table for one tokenizer, cached as per-passage counts in `cache_dir`."""
    from transformers import AutoTokenizer

    from whopays.env import results_root

    cache_dir = Path(cache_dir) if cache_dir else results_root() / "token_premium"
    path = cache_dir / f"{tokenizer_name.replace('/', '__')}.parquet"
    if path.exists():
        long = pd.read_parquet(path)
    else:
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        long = pd.DataFrame(
            [{"lang": lang, "passage_id": pid, "tokens": n}
             for lang in CODES for pid, n in passage_counts(tok, lang).items()]
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        long.to_parquet(path, index=False)
    counts = {lang: dict(zip(g.passage_id, g.tokens, strict=True))
              for lang, g in long.groupby("lang", sort=False)}
    return premium_table(counts)
