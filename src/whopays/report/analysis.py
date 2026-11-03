"""Load released Parquet and rebuild reports without model downloads or GPU calls."""
import json
import time
from pathlib import Path

import pandas as pd

from whopays.report import hypotheses as hyp
from whopays.report.card import disparity_cards, write_cards
from whopays.report.figures import gap_heatmap, price_of_parity
from whopays.stats.inequality import inequality_summary

TEN_MINUTES = 600.0
WEIGHTS = ("w8_rtn", "w4_gptq", "w3_gptq")
KV = ("quant4", "evict")


def load_parquet(root):
    paths = sorted(Path(root).rglob('items.parquet'))
    if not paths:
        raise ValueError('no released items.parquet files found')
    rows = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if rows.duplicated(['run_id', 'item_id']).any():
        raise ValueError('duplicate run/item rows in release')
    return rows


def research_rows(rows):
    """Keep rows that may enter results: logged on a GPU, excluding the GPTQ checkpoint."""
    local = rows.gpu.astype(str).str.contains(r'cpu|mps|apple|unknown|none', case=False)
    reference = rows.weight == 'prequant'
    excluded = rows[local | reference]
    if len(excluded):
        print(f'excluded {len(excluded)} rows from {excluded.run_id.nunique()} local or '
              'reference runs')
    return rows[~(local | reference)].copy()


def regenerate(root, output):
    """Rebuild cards and quality heatmaps; elapsed time is measured by the caller."""
    rows = research_rows(load_parquet(root))
    quality = rows[rows.task.isin(['belebele', 'mgsm'])].copy()
    output = Path(output)
    artifacts = write_cards(quality, output / 'cards')
    for task, frame in quality.groupby('task'):
        cards = disparity_cards(frame)
        if cards:
            artifacts.append(gap_heatmap(cards, output / f'{task}-gaps.png'))
    return artifacts


def _primary(rows):
    """One seed per configuration: seed repetitions are not independent questions."""
    return rows[rows.calib.isin(['none', 'multi']) & (rows.seed == 0)
                & ~rows.spec.astype(bool) & rows['limit'].isna()]


def _cell(rows, weight, kv):
    return rows[(rows.weight == weight) & (rows.kv == kv)]


def _treatments(rows):
    return [(w, k) for w, k in rows[['weight', 'kv']].drop_duplicates().itertuples(index=False)
            if (w, k) != ('fp16', 'full')]


def _model_tables(rows, n_resamples):
    h1_rows, h3_rows, inequality = [], [], []
    fp16 = _cell(rows, 'fp16', 'full')
    for weight, kv in _treatments(rows):
        treated = _cell(rows, weight, kv)
        if set(fp16.lang) != set(treated.lang):
            continue
        result = hyp.h1(fp16, treated, n_resamples=n_resamples)
        h1_rows.append({'weight': weight, 'kv': kv, 'supported': result['supported'],
                        **{f'{name}_{key}': result[name][key]
                           for name in ('token_premium', 'fp16_accuracy')
                           for key in ('rho', 'low', 'high')}})
        table = hyp.h3(fp16, treated, n_resamples=n_resamples)
        h3_rows.append(table.assign(weight=weight, kv=kv,
                                    h3_supported=hyp.h3_supported(table)))
        change = treated.groupby('lang').correct.mean() - fp16.groupby('lang').correct.mean()
        inequality.append({'weight': weight, 'kv': kv, **inequality_summary(change.to_numpy())})
    return h1_rows, h3_rows, inequality


def audit_report(root, output, *, n_resamples=2000, allocations=None):
    """Write every hypothesis table. A design with missing cells is listed as pending."""
    rows = research_rows(load_parquet(root))
    quality = rows[rows.task.isin(['belebele', 'mgsm'])]
    if quality.empty:
        raise ValueError('no research rows in this release')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hyp.language_config_table(quality[quality['limit'].isna()], n_resamples=n_resamples).to_csv(
        output / 'language_config.csv', index=False)
    summary = {'pending': {'h2': [], 'parity': []}, 'h2_supported': {}}
    h1_all, h3_all, inequality_all, h2_all = [], [], [], []
    for model, frame in _primary(rows[rows.task == 'belebele']).groupby('model'):
        h1_rows, h3_rows, inequality = _model_tables(frame, n_resamples)
        h1_all += [{'model': model, **r} for r in h1_rows]
        h3_all += [t.assign(model=model) for t in h3_rows]
        inequality_all += [{'model': model, **r} for r in inequality]
        for weight in WEIGHTS:
            for kv in KV:
                cells = [('fp16', 'full'), (weight, 'full'), ('fp16', kv), (weight, kv)]
                if any(_cell(frame, *cell).empty for cell in cells):
                    summary['pending']['h2'].append(f'{weight}/{kv}')
                    continue
                tests = hyp.h2(frame, weight, kv)['tests']
                h2_all.append(tests.assign(model=model, weight=weight, kv=kv))
    pd.DataFrame(h1_all).to_csv(output / 'h1.csv', index=False)
    pd.DataFrame(inequality_all).to_csv(output / 'inequality.csv', index=False)
    (pd.concat(h3_all) if h3_all else pd.DataFrame()).to_csv(output / 'h3.csv', index=False)
    h2_table = pd.concat(h2_all) if h2_all else pd.DataFrame()
    h2_table.to_csv(output / 'h2.csv', index=False)
    if not h2_table.empty:
        summary['h2_supported'] = bool(h2_table.reject.any())
    if allocations:
        summary['parity'] = _parity(rows, allocations, output, n_resamples)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def _parity(rows, allocations, output, n_resamples):
    from whopays.parityquant.plan import parity_report

    verdicts = {}
    for model, entries in allocations.items():
        table, verdict = parity_report(rows, entries, model, n_resamples=n_resamples)
        table.to_csv(output / f"parity_{model.split('/')[-1]}.csv", index=False)
        price_of_parity(table.rename(columns={'price_of_parity': 'price', 'price_low': 'low',
                                              'price_high': 'high'}),
                        output / f"price_of_parity_{model.split('/')[-1]}.png")
        verdicts[model] = verdict
    return verdicts


def timed_regenerate(root, output, *, n_resamples=2000, allocations=None):
    """The release requirement: every report rebuilt from Parquet in under 10 minutes."""
    start = time.monotonic()
    artifacts = regenerate(root, output)
    audit_report(root, output, n_resamples=n_resamples, allocations=allocations)
    seconds = time.monotonic() - start
    return {'seconds': seconds, 'within_limit': seconds < TEN_MINUTES,
            'artifacts': len(artifacts) + len(list(Path(output).glob('*.csv')))}
