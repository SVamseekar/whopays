"""Cache invariants, including rollback at a compressed residual boundary."""

from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache, Qwen2Config
from transformers.cache_utils import Cache, QuantizedLayer

from whopays.config import RunConfig
from whopays.kv import policies


def config(**kwargs):
    return RunConfig(model="tiny", task="mgsm", lang="en", **kwargs)


def states(length, start=0):
    return torch.arange(start, start + length, dtype=torch.float32).reshape(1, 1, length, 1)


class CPUQuantizedLayer(QuantizedLayer):
    """Exercise the installed residual/flush code without optional HQQ kernels."""

    def _quantize(self, tensor, axis):
        return tensor.clone()

    def _dequantize(self, tensor):
        return tensor


def quantized_cache(residual_length=4, layers=1):
    return Cache(layers=[CPUQuantizedLayer(residual_length=residual_length) for _ in range(layers)])


def append(cache, length, start=0):
    for index in range(len(cache.layers)):
        cache.update(states(length, start), states(length, start) + 100, index)


@pytest.mark.parametrize("budget,expected", [(1, [0]), (3, [0, 1, 2]), (4, [0, 1, 2, 3]),
                                              (6, [0, 1, 2, 3, 8, 9]), (12, list(range(10)))])
def test_sink_recent_respects_even_tiny_budgets(budget, expected):
    cache = DynamicCache()
    cache.update(states(10), states(10) + 100, 0)
    cfg = config(kv="evict", kv_budget=budget)
    assert policies.after_prefill(cache, cfg) == len(expected)
    assert cache.layers[0].keys.flatten().tolist() == expected
    assert cache.layers[0].values.flatten().tolist() == [p + 100 for p in expected]
    cache.update(states(1, 10), states(1, 110), 0)
    assert policies.after_decode(cache, cfg) == min(budget, len(expected) + 1)
    if 4 < budget < 10:
        assert cache.layers[0].keys.flatten().tolist() == [0, 1, 2, 3, 9, 10]


def test_full_policy_preserves_cache_and_empty_cache_is_valid():
    cache = DynamicCache()
    cfg = config()
    assert policies.after_prefill(cache, cfg) == 0
    assert policies.after_decode(cache, cfg) == 0
    assert policies.cache_bytes(cache) == 0
    policies.rollback(cache, 0)
    cache.update(states(3), states(3), 0)
    original = cache.layers[0].keys
    assert policies.after_prefill(cache, cfg) == 3
    assert policies.after_decode(cache, cfg) == 3
    assert cache.layers[0].keys is original


def test_full_rollback_removes_count_and_validates_before_changing():
    cache = DynamicCache()
    cache.update(states(5), states(5), 0)
    policies.rollback(cache, 2)
    assert cache.layers[0].keys.flatten().tolist() == [0, 1, 2]
    for invalid in [-1, 4]:
        with pytest.raises(ValueError):
            policies.rollback(cache, invalid)
        assert cache.get_seq_length() == 3
    policies.rollback(cache, 3)
    assert cache.get_seq_length() == 0


def test_quantized_rollback_rewinds_counter_at_entire_residual_boundary():
    cache = quantized_cache()
    append(cache, 5)
    append(cache, 3, 5)
    policies.rollback(cache, 3)
    assert cache.get_seq_length() == 5
    assert cache.layers[0].keys.shape[-2] == 0
    append(cache, 1, 8)
    assert cache.get_seq_length() == 6
    assert cache.layers[0].keys.flatten().tolist() == [8]


def test_quantized_rollback_refuses_flushed_tokens_without_mutation():
    cache = quantized_cache()
    append(cache, 5)
    append(cache, 3, 5)
    append(cache, 1, 8)  # This update flushes the residual into the compressed store.
    original = cache.layers[0]._quantized_keys.clone()
    with pytest.raises(ValueError, match="residual"):
        policies.rollback(cache, 1)
    assert cache.get_seq_length() == 9
    assert torch.equal(cache.layers[0]._quantized_keys, original)


def test_quantized_rollback_preflights_every_layer():
    cache = quantized_cache(layers=2)
    append(cache, 5)
    cache.update(states(2, 5), states(2, 5), 0)
    cache.update(states(1, 5), states(1, 5), 1)
    with pytest.raises(ValueError, match="residual"):
        policies.rollback(cache, 2)
    assert cache.layers[0].get_seq_length() == 7
    assert cache.layers[0].keys.shape[-2] == 2


def test_cache_bytes_counts_compressed_metadata_and_shared_storage_once():
    cache = quantized_cache()
    append(cache, 5)
    layer = cache.layers[0]
    packed = torch.ones(8, dtype=torch.uint8)
    scale = torch.ones(2, dtype=torch.float16)
    layer._quantized_keys = (packed, {"scale": scale, "zero": scale[:1]})
    layer._quantized_values = (packed[2:], {"scale": scale})
    assert policies.cache_bytes(cache) == 12
    # A crop view still retains the entire underlying allocation.
    layer.keys = states(6)[..., :2, :]
    layer.values = layer.keys
    assert policies.cache_bytes(cache) == 36


def test_quant4_requires_optional_hqq_only_when_selected(monkeypatch):
    import transformers.cache_utils

    monkeypatch.setattr(transformers.cache_utils, "is_hqq_available", lambda: False)
    model = SimpleNamespace(config=Qwen2Config(num_hidden_layers=1))
    assert isinstance(policies.new_cache(config(), model), DynamicCache)
    assert isinstance(policies.new_cache(config(kv="evict"), model), DynamicCache)
    with pytest.raises(ImportError, match="HQQ|hqq"):
        policies.new_cache(config(kv="quant4"), model)


def test_eviction_matches_masked_full_cache_at_original_rope_position(tiny_model_dir):
    from whopays.models.loader import load

    cfg = RunConfig(
        model=str(tiny_model_dir), task="mgsm", lang="en", kv="evict", kv_budget=6
    )
    model = load(cfg, "cpu").model
    full, evicted = DynamicCache(config=model.config), policies.new_cache(cfg, model)
    prompt = torch.arange(1, 11)[None]
    with torch.no_grad():
        model(prompt, past_key_values=full, use_cache=True)
        model(prompt, past_key_values=evicted, use_cache=True)
        policies.after_prefill(evicted, cfg)
        # A full cache with precisely the omitted history masked out is a reference
        # for the physical eviction, provided the query keeps absolute RoPE position 10.
        mask = torch.zeros(1, 1, 1, 11)
        mask[..., 4:8] = torch.finfo(mask.dtype).min
        reference = model(torch.tensor([[11]]), position_ids=torch.tensor([[10]]),
                          attention_mask=mask, past_key_values=full, use_cache=True).logits
        actual = model(torch.tensor([[11]]), position_ids=torch.tensor([[10]]),
                       past_key_values=evicted, use_cache=True).logits
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)


def test_quantized_rollback_replay_has_identical_qwen_logits(tiny_model_dir):
    from whopays.models.loader import load

    cfg = RunConfig(model=str(tiny_model_dir), task="mgsm", lang="en")
    model = load(cfg, "cpu").model
    cache = quantized_cache(residual_length=128, layers=model.config.num_hidden_layers)
    query = torch.tensor([[6, 7]])
    with torch.no_grad():
        model(torch.tensor([[1, 2, 3, 4, 5]]), past_key_values=cache, use_cache=True)
        reference = model(query, past_key_values=cache, use_cache=True).logits
        policies.rollback(cache, 2)
        actual = model(query, past_key_values=cache, use_cache=True).logits
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
