from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

from whopays.config import RunConfig
from whopays.crosscheck import compare, k1_verdict
from whopays.languages import CODES


def test_partial_crosscheck_is_incomplete_and_duplicates_rejected():
    ours = pd.DataFrame([dict(task='belebele', lang='en', item_id='a', correct=True)])
    ref = ours.drop(columns='correct').assign(filter='none', metric='acc', value=1.)
    assert k1_verdict(compare(ours, ref)).set_index('task').loc['belebele', 'K1'] == 'INCOMPLETE'
    with pytest.raises(ValueError, match='duplicate'):
        compare(pd.concat([ours, ours]), ref)


def test_gate_uses_unrounded_threshold_and_requires_both_tasks():
    table = pd.DataFrame([dict(task='belebele', lang=lang, n_items=900,
                              ours=50., lmeval=48.999, diff_pts=1.001, agreement=.99,
                              complete=True) for lang in CODES])
    verdict = k1_verdict(table).set_index('task')
    assert verdict.loc['belebele', 'K1'] == 'FAIL'
    assert verdict.loc['mgsm', 'K1'] == 'INCOMPLETE'


@pytest.mark.parametrize('task', ['mgsm', 'flores_nll'])
def test_resume_preserves_original_batch_membership(task):
    import importlib
    mod = importlib.import_module('whopays.tasks.' + task)
    cfg = RunConfig(model='tiny', task=task, lang='en', batch_size=4)
    items = [dict(item_id=str(i), question=str(i), text=str(i), gold=i) for i in range(8)]
    seen = []
    def generation(bundle, cfg, prompts):
        seen.append(prompts)
        return [dict(text='0', input_tokens=1, output_tokens=1) for _ in prompts]
    def nll(bundle, texts):
        seen.append(texts)
        return [(1., 1) for _ in texts]
    name, fn = ('generate_batch', generation) if task == 'mgsm' else ('nll_batch', nll)
    with mock.patch.object(mod, name, side_effect=fn):
        rows = list(mod.score_items(cfg, SimpleNamespace(device='cpu'), items,
                                    {str(i) for i in range(5)}))
    assert len(seen) == 1 and len(seen[0]) == 4
    assert [r['item_id'] for r in rows] == ['5', '6', '7']


def test_full_crosscheck_passes():
    ours, refs = [], []
    for task, n, column, flt, metric in (
        ('belebele', 900, 'correct', 'none', 'acc'),
        ('mgsm', 250, 'correct_ascii', 'flexible-extract', 'exact_match'),
    ):
        for lang in CODES:
            base = dict(task=task, lang=lang, item_id=[f'i{i}' for i in range(n)])
            ours.append(pd.DataFrame({**base, column: True}))
            refs.append(pd.DataFrame({**base, 'filter': flt, 'metric': metric, 'value': 1.}))
    verdict = k1_verdict(compare(pd.concat(ours), pd.concat(refs)))
    assert verdict.K1.tolist() == ['PASS', 'PASS']
