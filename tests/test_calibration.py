"""GPTQ calibration sets: 128 windows of 2,048 tokens from Wikipedia (fourth amendment)."""
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from whopays.languages import CODES
from whopays.quant import calibration as cal


class CharTokenizer:
    """One token per character, so window arithmetic is easy to check."""

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {'input_ids': [ord(c) for c in text]}


def fake_articles(lang, seed):
    for i in range(10_000):
        yield {'id': f'{lang}-{seed}-{i}', 'text': f'{lang}{seed}:{i:05d} ' * 3}


def test_allocation_english_is_all_english():
    assert cal.window_allocation('en', 0, 128) == {'en': 128}


def test_allocation_multi_is_balanced_and_seeded():
    a = cal.window_allocation('multi', 0, 128)
    assert set(a) == set(CODES) and sum(a.values()) == 128
    assert set(a.values()) == {11, 12}
    assert a == cal.window_allocation('multi', 0, 128)
    assert any(cal.window_allocation('multi', s, 128) != a for s in (1, 2))


def test_pack_windows_cuts_exact_windows_from_one_stream():
    windows, used = cal.pack_windows(CharTokenizer(), fake_articles('en', 0), n_windows=3,
                                     window=50)
    assert [len(w) for w in windows] == [50, 50, 50]
    stream = ''.join(chr(t) for w in windows for t in w)
    assert stream.startswith('en0:00000 en0:00000 en0:00000 \n\nen0:00001')
    assert used[0] == 'en-0-0'


def test_pack_windows_caps_tokens_per_article():
    long = ({'id': str(i), 'text': 'x' * 1000} for i in range(100))
    windows, used = cal.pack_windows(CharTokenizer(), long, n_windows=2, window=100,
                                     max_article_tokens=30)
    assert len(used) == 7  # 30 tokens + 2 separator tokens each; 200 tokens needed


def test_pack_windows_refuses_short_sources():
    with pytest.raises(ValueError, match='not enough'):
        cal.pack_windows(CharTokenizer(), iter([{'id': 'a', 'text': 'short'}]), 2, 50)


def test_calibration_set_matches_token_budget_and_is_reproducible():
    tok = CharTokenizer()
    batches, prov = cal.calibration_set(tok, 'multi', 1, n_windows=22, window=40,
                                        articles=fake_articles)
    assert len(batches) == 22
    assert all(b['input_ids'].shape == (1, 40) and b['input_ids'].dtype == torch.long
               for b in batches)
    assert all(set(b) == {'input_ids'} for b in batches)  # unpadded: SDPA's causal path
    assert prov['tokens'] == 22 * 40 and sum(prov['windows'].values()) == 22
    assert prov['source'] == cal.SOURCE and prov['revision'] == cal.REVISION
    again, prov2 = cal.calibration_set(tok, 'multi', 1, n_windows=22, window=40,
                                       articles=fake_articles)
    assert prov2['sha256'] == prov['sha256']
    _, other = cal.calibration_set(tok, 'multi', 2, n_windows=22, window=40,
                                   articles=fake_articles)
    assert other['sha256'] != prov['sha256']
    _, en = cal.calibration_set(tok, 'en', 1, n_windows=22, window=40, articles=fake_articles)
    assert en['tokens'] == prov['tokens'] and en['windows'] == {'en': 22}


def test_windows_never_mix_languages():
    import re

    batches, prov = cal.calibration_set(CharTokenizer(), 'multi', 0, n_windows=11, window=60,
                                        articles=fake_articles)
    found = Counter()
    for b in batches:
        text = ''.join(chr(t) for t in b['input_ids'][0].tolist())
        langs = set(re.findall(r'([a-z]{2})0:', text))
        assert len(langs) == 1
        found[langs.pop()] += 1
    assert found == Counter(prov['windows'])


def test_wikipedia_articles_reads_pinned_shards_in_seeded_order(tmp_path, monkeypatch):
    path = tmp_path / 'shard.parquet'
    table = pa.table({'id': [str(i) for i in range(40)], 'url': [''] * 40,
                      'title': [''] * 40, 'text': [f'text {i}' for i in range(40)]})
    pq.write_table(table, path, row_group_size=10)
    opened = []

    def open_shard(lang, shard):
        opened.append((lang, shard))
        return open(path, 'rb')

    monkeypatch.setattr(cal, '_open_shard', open_shard)
    first = [a['id'] for a in cal.wikipedia_articles('sw', 0)]
    assert sorted(first, key=lambda x: int(x.split(':')[1])) == [f'sw:{i}' for i in range(40)]
    assert first == [a['id'] for a in cal.wikipedia_articles('sw', 0)]
    assert first != [a['id'] for a in cal.wikipedia_articles('sw', 1)]
    assert {shard for lang, shard in opened} == {0}  # Swahili has one shard
    assert set(cal.SHARDS) == set(CODES)


def test_shard_path_is_pinned_to_the_revision():
    path = cal.shard_path('en', 3)
    assert path == (f'datasets/{cal.SOURCE}@{cal.REVISION}/'
                    '20231101.en/train-00003-of-00041.parquet')
    with pytest.raises(ValueError):
        cal.shard_path('en', 41)
