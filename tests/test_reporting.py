import pandas as pd
import pytest

from whopays.report.card import disparity_cards


def rows():
    out = []
    for lang in ['en', 'te']:
        for run, weight, values in [('base', 'fp16', [1, 1]), ('quant', 'w4_gptq', [1, 0])]:
            for i, value in enumerate(values):
                out.append(dict(model='tiny', task='belebele', lang=lang, run_id=run+lang,
                                item_id=str(i), correct=value, weight=weight, kv='full',
                                spec=False, calib='none' if weight == 'fp16' else 'multi',
                                seed=0, input_tokens=10 if lang == 'en' else 20,
                                gpu='Tesla T4', limit=None))
    return pd.DataFrame(out)


def test_cards_pair_by_item_and_include_all_languages():
    cards = disparity_cards(rows())
    assert len(cards) == 1
    card = cards[0]
    assert card['worst_language_degradation'] == .5
    assert card['languages']['te']['token_premium'] == 2
    assert card['languages']['en']['quality_delta'] == -.5


def test_cards_reject_missing_pairs_and_local_research_rows():
    data = rows()
    with pytest.raises(ValueError, match='paired'):
        disparity_cards(data.drop(index=0))
    with pytest.raises(ValueError, match='GPU'):
        disparity_cards(data.assign(gpu='CPU'))


def test_cards_reject_duplicate_reference():
    data = rows()
    with pytest.raises(ValueError, match='duplicate'):
        disparity_cards(pd.concat([data, data.iloc[[0]]]))
