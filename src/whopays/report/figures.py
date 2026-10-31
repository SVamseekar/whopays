"""Reproducible figures; caller supplies measured data, never embedded examples."""
from pathlib import Path

import numpy as np


def _save(fig, path):
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight', dpi=160)
    plt.close(fig)
    return path


def gap_heatmap(cards, path):
    import matplotlib.pyplot as plt

    if not cards:
        raise ValueError('no result cards')
    languages = sorted(cards[0]['languages'])
    if any(sorted(c['languages']) != languages for c in cards):
        raise ValueError('heatmap requires the same languages for every configuration')
    values = np.array([[c['languages'][lang]['quality_delta'] for lang in languages]
                       for c in cards])
    fig, ax = plt.subplots(figsize=(max(7, len(languages)), max(3, len(cards) * .4)))
    vmax = max(float(abs(values).max()), .001)
    mesh = ax.imshow(100 * values, cmap='RdBu', vmin=-100*vmax, vmax=100*vmax, aspect='auto')
    ax.set_xticks(range(len(languages)), languages)
    labels = [f"{c['configuration']['model'].split('/')[-1]} / "
              f"{c['configuration']['task']} / {c['configuration']['weight']} / "
              f"{c['configuration']['kv']} / {c['configuration']['calib']} / "
              f"seed {c['configuration']['seed']} / spec {c['configuration']['spec']}"
              for c in cards]
    ax.set_yticks(range(len(cards)), labels)
    fig.colorbar(mesh, ax=ax, label='Accuracy change vs FP16 (percentage points)')
    return _save(fig, path)


def shapley_bars(values, path):
    import matplotlib.pyplot as plt

    from whopays.stats.shapley import shapley_values

    attribution = shapley_values(values)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(attribution), list(attribution.values()))
    ax.axhline(0, color='black', linewidth=.6)
    ax.set_ylabel('Contribution to supplied disparity metric')
    return _save(fig, path)


def price_of_parity(table, path):
    """Table must hold measured budget, price, low and high (paired CI) columns."""
    import matplotlib.pyplot as plt

    required = ['budget', 'price', 'low', 'high']
    if not set(required) <= set(table) or table.empty:
        raise ValueError('measured budget/price/low/high required')
    table = table.sort_values('budget')
    if not np.isfinite(table[required].to_numpy()).all():
        raise ValueError('finite measurements required')
    if ((table.low > table.price) | (table.high < table.price)).any():
        raise ValueError('confidence bounds must bracket price')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(table.budget, table.price,
                yerr=[table.price-table.low, table.high-table.price], fmt='o-', capsize=4)
    ax.set(xlabel='Parameter-weighted average bits', ylabel='Mean accuracy cost of parity')
    return _save(fig, path)
