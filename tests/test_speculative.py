import pytest
import torch

from whopays.config import RunConfig
from whopays.decoding.speculative import generate
from whopays.models.loader import load


@pytest.mark.parametrize('kv', ['full', 'evict'])
@pytest.mark.parametrize('same_draft', [True, False])
def test_exact_greedy(tiny_model_dir, kv, same_draft):
    cfg = RunConfig(model=str(tiny_model_dir), task='mgsm', lang='en', kv=kv,
                    kv_budget=6, spec=True, draft_model=str(tiny_model_dir))
    b = load(cfg, 'cpu')
    if not same_draft:
        with torch.no_grad():
            b.draft.lm_head.weight.mul_(-1)
    ids = [1, 2, 3, 4, 5, 6, 7]
    plain = generate(b, cfg.model_copy(update={'spec': False}), ids, max_new_tokens=19)
    spec = generate(b, cfg, ids, max_new_tokens=19)
    assert spec['token_ids'] == plain['token_ids']
    assert len(spec['token_ids']) == 19
    assert 0 <= spec['accepted_tokens'] <= spec['proposed_tokens']
    assert spec['steps'] > 0
    assert spec['cache_storage_bytes'] > 0


def test_full_matches_transformers_greedy(tiny_model_dir):
    cfg = RunConfig(model=str(tiny_model_dir), task='mgsm', lang='en')
    b = load(cfg, 'cpu')
    ids = [1, 2, 3, 4]
    ours = generate(b, cfg, ids, max_new_tokens=12)
    expected = b.model.generate(torch.tensor([ids]), max_new_tokens=12,
                                do_sample=False, repetition_penalty=1.0)
    assert ours['token_ids'] == expected[0, len(ids):].tolist()
