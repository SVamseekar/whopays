"""GPTQ calibration sets.

Current source (fourth preregistration amendment, 2026-11-07): 128 windows of 2,048 tokens,
cut from native-language Wikipedia articles. The budget, 262,144 tokens, is the GPTQ paper's
"128 random 2048 token segments" (Frantar et al., 2023, arXiv 2210.17323) and the default of
Williams & Aletras (ACL 2024, arXiv 2311.09755). Native Wikipedia per language follows Chimoto
et al. (EACL 2026, arXiv 2601.18306). English and multilingual sets have the same token budget;
the multilingual set splits its windows equally over the 11 languages (their "multi10" design).

The token budget is what matters for GPTQ: each layer's Hessian is X^T X over calibration
tokens, and with fewer tokens than input features (8,960 for Qwen2.5-1.5B's down_proj) it is
rank-deficient. The legacy source, 128 single FLORES dev sentences (a few thousand tokens),
had that problem; it is kept only so runs made with it can be identified and reproduced.
"""
from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Iterable, Iterator

from whopays.languages import CODES

SOURCE = "wikimedia/wikipedia"
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"  # pinned commit, 2024-01-09
DUMP = "20231101"
# Parquet shards per language at REVISION.
SHARDS = {"en": 41, "de": 20, "fr": 17, "es": 13, "ru": 21, "zh": 6, "ja": 15, "th": 3,
          "bn": 2, "sw": 1, "te": 2}
N_WINDOWS = 128
WINDOW = 2048
# At most this many tokens from each article (its opening), so a language's windows draw on
# many articles rather than a few long ones. GPTQ's reference loader takes one segment per
# document for the same reason; Williams & Aletras find calibration diversity matters.
MAX_ARTICLE_TOKENS = 512
SEPARATOR = "\n\n"


def window_allocation(mode: str, seed: int, n_windows: int = N_WINDOWS) -> dict[str, int]:
    """Windows per language: all English, or equal over the 11 languages."""
    if mode == "en":
        return {"en": n_windows}
    if mode != "multi":
        raise ValueError("expected calibration mode 'en' or 'multi'")
    langs = list(CODES)
    random.Random(f"whopays-calibration-allocation:{seed}").shuffle(langs)
    base, extra = divmod(n_windows, len(langs))
    # The seed rotates which languages receive the remainder.
    return {lang: base + (i < extra) for i, lang in enumerate(langs)}


def shard_path(lang: str, shard: int) -> str:
    count = SHARDS[lang]
    if not 0 <= shard < count:
        raise ValueError(f"{lang} has {count} shards")
    return (f"datasets/{SOURCE}@{REVISION}/{DUMP}.{lang}/"
            f"train-{shard:05d}-of-{count:05d}.parquet")


def _open_shard(lang: str, shard: int):
    """Open a shard for range reads: only the footer and the row groups used are fetched."""
    from huggingface_hub import HfFileSystem

    return HfFileSystem().open(shard_path(lang, shard), "rb", block_size=8 * 2**20)


def wikipedia_articles(lang: str, seed: int) -> Iterator[dict]:
    """Articles of one language in a seeded order: a seeded shard, then row groups and the
    articles inside each row group shuffled. Reading stops when the caller stops."""
    import pyarrow.parquet as pq

    rng = random.Random(f"whopays-calibration:{lang}:{seed}")
    shard = rng.randrange(SHARDS[lang])
    with _open_shard(lang, shard) as handle:
        parquet = pq.ParquetFile(handle)
        groups = list(range(parquet.metadata.num_row_groups))
        rng.shuffle(groups)
        for group in groups:
            rows = parquet.read_row_group(group, columns=["id", "text"]).to_pylist()
            rng.shuffle(rows)
            for row in rows:
                if row["text"] and row["text"].strip():
                    yield {"id": f"{lang}:{row['id']}", "text": row["text"]}


def pack_windows(tokenizer, articles: Iterable[dict], n_windows: int, window: int,
                 max_article_tokens: int = MAX_ARTICLE_TOKENS
                 ) -> tuple[list[list[int]], list[str]]:
    """Concatenate article openings (separated by a blank line) and cut consecutive windows.

    Returns the windows and the IDs of the articles that contributed tokens.
    """
    need = n_windows * window
    separator = tokenizer(SEPARATOR, add_special_tokens=False)["input_ids"]
    stream: list[int] = []
    used: list[str] = []
    for article in articles:
        if stream:
            stream.extend(separator)
        ids = tokenizer(article["text"], add_special_tokens=False)["input_ids"]
        stream.extend(ids[:max_article_tokens])
        used.append(article["id"])
        if len(stream) >= need:
            break
    if len(stream) < need:
        raise ValueError(f"not enough text: {len(stream)} of {need} tokens")
    return [stream[i * window:(i + 1) * window] for i in range(n_windows)], used


def calibration_set(tokenizer, mode: str, seed: int, *, n_windows: int = N_WINDOWS,
                    window: int = WINDOW,
                    articles: Callable[[str, int], Iterable[dict]] = wikipedia_articles):
    """Batches of shape (1, window) for GPTQ, and the provenance to store with the run."""
    import torch

    allocation = window_allocation(mode, seed, n_windows)
    windows, used, per_lang = [], {}, {}
    for lang, count in allocation.items():
        if not count:
            continue
        cut, ids = pack_windows(tokenizer, articles(lang, seed), count, window)
        windows += cut
        used[lang] = ids
        per_lang[lang] = count
    random.Random(f"whopays-calibration-order:{mode}:{seed}").shuffle(windows)
    digest = hashlib.sha256()
    for w in windows:
        digest.update(b"".join(t.to_bytes(4, "little") for t in w))
    # No attention_mask: windows have no padding, and without one SDPA takes its causal path
    # instead of materialising a 2,048 × 2,048 mask per window during input capture.
    batches = [{"input_ids": torch.tensor([w], dtype=torch.long)} for w in windows]
    provenance = {"source": SOURCE, "revision": REVISION, "dump": DUMP, "mode": mode,
                  "seed": seed, "window": window, "tokens": len(windows) * window,
                  "max_article_tokens": MAX_ARTICLE_TOKENS,
                  "windows": per_lang, "articles": used, "sha256": digest.hexdigest()}
    return batches, provenance


# Legacy source: single FLORES dev sentences. Identifies and reproduces runs made before the
# fourth amendment; do not use for new runs.
def sample_calibration(mode='en', seed=0, n_samples=128):
    from whopays.datasets import flores
    if mode not in ('en', 'multi') or n_samples < 1:
        raise ValueError('expected en/multi and a positive sample count')
    langs = ['en'] if mode == 'en' else list(CODES)
    rng = random.Random(seed)
    # Rotate remainder allocation by seed, avoiding a systematic first-language preference.
    rng.shuffle(langs)
    rows = []
    for i, lang in enumerate(langs):
        count = n_samples // len(langs) + (i < n_samples % len(langs))
        items = sorted(flores(lang, 'dev'), key=lambda x: x['item_id'])
        if count > len(items):
            raise ValueError('not enough distinct calibration sentences')
        rows.extend({**r, 'lang': lang} for r in rng.sample(items, count))
    rng.shuffle(rows)
    return rows


def calibration_batches(tokenizer, rows, max_length=512):
    return [dict(tokenizer(r['text'], return_tensors='pt', truncation=True,
                           max_length=max_length, add_special_tokens=False)) for r in rows]


def batches_for(tokenizer, source: str, mode: str, seed: int):
    """Calibration batches and provenance for a RunConfig's calib_source."""
    if source == "wikipedia_128x2048":
        return calibration_set(tokenizer, mode, seed)
    if source == "flores_dev_sentences":
        rows = sample_calibration(mode, seed)
        return calibration_batches(tokenizer, rows), {"source": "flores_dev_sentences",
                                                      "mode": mode, "seed": seed,
                                                      "sentences": len(rows)}
    raise ValueError(f"unknown calibration source {source!r}")
