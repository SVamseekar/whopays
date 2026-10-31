"""Load released Parquet and rebuild reports without model downloads or GPU calls."""
from pathlib import Path

import pandas as pd

from whopays.report.card import disparity_cards, write_cards
from whopays.report.figures import gap_heatmap


def load_parquet(root):
    paths = sorted(Path(root).rglob('items.parquet'))
    if not paths:
        raise ValueError('no released items.parquet files found')
    rows = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if rows.duplicated(['run_id', 'item_id']).any():
        raise ValueError('duplicate run/item rows in release')
    return rows


def regenerate(root, output):
    """Rebuild cards and quality heatmaps; elapsed time is measured by the caller."""
    rows = load_parquet(root)
    quality = rows[rows.task.isin(['belebele', 'mgsm'])].copy()
    output = Path(output)
    artifacts = write_cards(quality, output / 'cards')
    for task, frame in quality.groupby('task'):
        cards = disparity_cards(frame)
        if cards:
            artifacts.append(gap_heatmap(cards, output / f'{task}-gaps.png'))
    return artifacts
