"""Seeded sentence sampling from FLORES dev only (never devtest)."""
import random

from whopays.languages import CODES


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
