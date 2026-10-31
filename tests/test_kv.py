import pytest
import torch

from whopays.config import RunConfig
from whopays.kv.policies import after_decode, after_prefill, cache_bytes, new_cache, rollback
from whopays.models.loader import load


def test_evict_retains_sink_recent_and_budget(tiny_model_dir):
    cfg = RunConfig(model=str(tiny_model_dir), task='mgsm', lang='en', kv='evict', kv_budget=6)
    b = load(cfg, 'cpu')
    cache = new_cache(cfg, b.model)
    with torch.no_grad():
        b.model(torch.arange(1, 11)[None], past_key_values=cache, use_cache=True)
    original = cache.layers[0].keys.clone()
    assert after_prefill(cache, cfg) == 6
    assert torch.equal(cache.layers[0].keys, original[:, :, [0, 1, 2, 3, 8, 9]])
    assert cache_bytes(cache) > 0
    with torch.no_grad():
        b.model(torch.tensor([[11]]), position_ids=torch.tensor([[10]]),
                past_key_values=cache, use_cache=True)
    assert after_decode(cache, cfg) == 6


def test_full_rollback(tiny_model_dir):
    cfg = RunConfig(model=str(tiny_model_dir), task='mgsm', lang='en')
    b = load(cfg, 'cpu')
    cache = new_cache(cfg, b.model)
    with torch.no_grad():
        b.model(torch.tensor([[1, 2, 3]]), past_key_values=cache, use_cache=True)
    rollback(cache, 2)
    assert cache.get_seq_length() == 1
    with pytest.raises(ValueError):
        rollback(cache, 2)
