"""Serving disparity cards (FR-16), with strict paired-item validation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FACTORS = ['model', 'task', 'weight', 'kv', 'spec', 'calib', 'seed', 'limit']


def disparity_cards(rows: pd.DataFrame) -> list[dict]:
    """One card per served configuration/seed, comparing identical FP16 items.

    Never pools calibration seeds as independent questions. Quality is a fraction,
    not percentage points. Speed metrics are kept separate from quality.
    """
    required = set(FACTORS + ['lang', 'item_id', 'correct', 'gpu', 'input_tokens'])
    if missing := required - set(rows):
        raise ValueError(f'missing result columns: {sorted(missing)}')
    if rows.empty:
        return []
    if rows.gpu.isna().any() or rows.gpu.astype(str).str.contains(
            r'cpu|mps|apple|unknown|none', case=False, regex=True).any():
        raise ValueError('research reports require logged GPU results')
    if rows.gpu.nunique() != 1:
        raise ValueError('separate GPU types before comparing speed')
    baseline = rows[(rows.weight == 'fp16') & (rows.kv == 'full') & ~rows.spec]
    keys = ['model', 'task', 'lang', 'item_id', 'limit']
    if baseline.duplicated(keys).any():
        raise ValueError('duplicate FP16 reference items')
    cards = []
    factors = FACTORS + [c for c in ['bitmap', 'budget', 'group_size', 'kv_budget',
                                    'draft_model', 'spec_k'] if c in rows]
    treatments = rows[~((rows.weight == 'fp16') & (rows.kv == 'full') & ~rows.spec)]
    for config, group in treatments.groupby(factors, dropna=False, sort=True):
        if group.duplicated(keys).any():
            raise ValueError('duplicate treatment items')
        cfg = {k: (None if pd.isna(v) else v.item() if hasattr(v, 'item') else v)
               for k, v in zip(factors, config, strict=True)}
        reference = baseline[(baseline.model == cfg['model']) & (baseline.task == cfg['task'])]
        reference = reference[reference['limit'].isna()] if cfg['limit'] is None else reference[
            reference['limit'] == cfg['limit']]
        paired = group.merge(reference, on=keys, suffixes=('', '_fp16'), how='outer',
                             indicator=True, validate='one_to_one')
        if not paired['_merge'].eq('both').all():
            raise ValueError('paired item coverage differs from FP16 reference')
        english = reference[reference.lang == 'en'].set_index('item_id').input_tokens
        from whopays.stats.bootstrap import paired_frame_bootstrap

        ci = paired_frame_bootstrap(reference, group, language='lang')
        ci_by_lang = {lang: (float(ci.low[i]), float(ci.high[i]))
                      for i, lang in enumerate(sorted(group.lang.unique()))}
        worst_samples = -ci.samples.min(axis=1)
        languages = {}
        for lang, frame in paired.groupby('lang', sort=True):
            eng = english.reindex(frame.item_id)
            if eng.isna().any() or (eng <= 0).any():
                raise ValueError('paired English items required for token premium')
            delta = frame.correct.astype(float).to_numpy() - frame.correct_fp16.astype(
                float).to_numpy()
            languages[lang] = {
                'n_items': len(frame), 'quality_delta': float(delta.mean()),
                'quality_delta_ci95': ci_by_lang[lang],
                'flip_rate': float((frame.correct_fp16.astype(bool) &
                                    ~frame.correct.astype(bool)).mean()),
                'token_premium': float(np.mean(frame.input_tokens.to_numpy() / eng.to_numpy())),
            }
            for metric in ['accepted_tokens', 'proposed_tokens', 'steps', 'output_tokens']:
                if metric in frame:
                    languages[lang][metric] = float(frame[metric].sum())
            if {'accepted_tokens', 'proposed_tokens'} <= set(frame):
                proposed = frame.proposed_tokens.sum()
                languages[lang]['acceptance_rate'] = (
                    float(frame.accepted_tokens.sum() / proposed) if proposed else None)
            if {'steps', 'output_tokens'} <= set(frame):
                steps = frame.steps.sum()
                languages[lang]['tokens_per_target_step'] = (
                    float(frame.output_tokens.sum() / steps) if steps else None)
        deltas = [x['quality_delta'] for x in languages.values()]
        cards.append({'configuration': cfg, 'gpu': str(rows.gpu.iloc[0]),
                      'languages': languages, 'worst_language_degradation': -min(deltas),
                      'disparity_spread': max(deltas) - min(deltas),
                      'worst_language_ci95': np.quantile(worst_samples, [.025, .975]).tolist(),
                      'memory_note': 'Fake-quant weight savings are theoretical bit counts.'})
    return cards


def write_cards(rows: pd.DataFrame, output: str | Path) -> list[Path]:
    from whopays.config import stable_hash

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for card in disparity_cards(rows):
        stem = stable_hash(card['configuration'])
        path = output / f'{stem}.json'
        path.write_text(json.dumps(card, indent=2, allow_nan=False) + '\n')
        lines = ['# Serving disparity card', '',
                 f"Configuration: `{json.dumps(card['configuration'], sort_keys=True)}`", '',
                 '| Language | Items | Accuracy change | Token premium |',
                 '| --- | ---: | ---: | ---: |']
        for lang, values in card['languages'].items():
            lines.append(f"| {lang} | {values['n_items']} | {values['quality_delta']:.6f} | "
                         f"{values['token_premium']:.4f} |")
        lines += ['', card['memory_note'], '']
        path.with_suffix('.md').write_text('\n'.join(lines))
        paths.append(path)
    return paths
