"""KIVI 4-bit KV cache (Liu et al., ICML 2024): keys per channel, values per token."""
import pytest
import torch

from whopays.kv import policies
from whopays.kv.policies import KIVILayer, fake_quant


def test_per_channel_keys_survive_an_outlier_channel_per_token_does_not():
    torch.manual_seed(0)
    k = torch.randn(1, 2, 32, 128)
    k[..., 7] *= 80  # Qwen2.5 layer-0 keys: |k| about 320 in a few channels, median about 4
    per_channel = fake_quant(k, dim=-2, bits=4)  # min/max over the 32 tokens of each channel
    per_token = fake_quant(k.reshape(1, 2, 32, 2, 64), dim=-1, bits=4).reshape(k.shape)
    normal = [c for c in range(128) if c != 7]
    err_c = (per_channel - k)[..., normal].abs().mean()
    err_t = (per_token - k)[..., normal].abs().mean()
    assert err_c < 0.1 < err_t  # per-token groups lose the ordinary channels


def test_fake_quant_is_idempotent_and_uses_16_levels():
    x = torch.randn(3, 32, 8)
    q = fake_quant(x, dim=-2, bits=4)
    assert torch.equal(fake_quant(q, dim=-2, bits=4), q)
    assert all(len(torch.unique(q[i, :, j])) <= 16 for i in range(3) for j in range(8))


def _layer(group=4, residual=8):
    return KIVILayer(nbits=4, group=group, residual=residual)


def test_old_tokens_quantize_in_whole_groups_and_the_residual_stays_exact():
    torch.manual_seed(1)
    layer = _layer()
    k, v = torch.randn(1, 2, 19, 16), torch.randn(1, 2, 19, 16)
    out_k, out_v = layer.update(k, v)
    assert torch.equal(out_k, k) and torch.equal(out_v, v)  # this forward attends in full precision
    # 19 tokens, residual 8: 11 are old enough, 8 of them (two groups of 4) are quantized
    assert layer.n_quant.tolist() == [8]
    assert not torch.equal(layer.keys[..., :8, :], k[..., :8, :])
    assert torch.equal(layer.keys[..., 8:, :], k[..., 8:, :])
    assert torch.equal(layer.values[..., 8:, :], v[..., 8:, :])


def test_quantized_history_is_never_requantized():
    torch.manual_seed(2)
    layer = _layer()
    layer.update(torch.randn(1, 1, 16, 8), torch.randn(1, 1, 16, 8))
    frozen = layer.keys[..., :8, :].clone()
    for _ in range(9):
        layer.update(torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8))
    assert layer.n_quant.tolist() == [16]
    assert torch.equal(layer.keys[..., :8, :], frozen)


def test_left_padded_rows_quantize_like_the_same_sequence_alone():
    torch.manual_seed(3)
    seq_k, seq_v = torch.randn(1, 2, 13, 8), torch.randn(1, 2, 13, 8)
    alone = _layer()
    alone.update(seq_k, seq_v)
    pad = 5
    junk = torch.full((1, 2, pad, 8), 1e3)  # pad keys must not enter any group
    batched = _layer()
    batched.set_padding(torch.tensor([pad, 0]))
    other = torch.randn(1, 2, 13 + pad, 8)
    batched.update(torch.cat([torch.cat([junk, seq_k], 2), other]),
                   torch.cat([torch.cat([junk, seq_v], 2), other]))
    assert torch.equal(batched.keys[0, :, pad:], alone.keys[0])
    assert torch.equal(batched.values[0, :, pad:], alone.values[0])
    assert batched.n_quant.tolist() == [4, 8]


def test_rollback_only_within_the_residual():
    layer = _layer()
    layer.update(torch.randn(1, 1, 14, 8), torch.randn(1, 1, 14, 8))  # 4 quantized, 10 exact
    from transformers.cache_utils import Cache
    cache = Cache(layers=[layer])
    policies.rollback(cache, 10)
    assert layer.get_seq_length() == 4
    with pytest.raises(ValueError, match="residual"):
        policies.rollback(cache, 1)


def test_storage_bytes_count_bits_not_fake_quant_tensors():
    layer = _layer(group=4, residual=8)
    layer.update(torch.zeros(1, 2, 16, 8, dtype=torch.float16),
                 torch.zeros(1, 2, 16, 8, dtype=torch.float16))
    # 8 quantized tokens: keys 2 heads x 8 ch x 8 tok x 4 bits = 64 B, per-channel scale+zero
    # per group (2 groups x 2 heads x 8 ch x 2 x 2 B = 128 B); values 64 B + per token, one
    # group of 8 channels (8 tok x 2 heads x 2 x 2 B = 64 B); residual 8 tokens FP16 for both:
    # 2 x 2 heads x 8 tok x 8 ch x 2 B = 512 B.
    assert layer.storage_bytes() == 64 + 128 + 64 + 64 + 512


def test_recipe_change_moves_quant4_run_ids_only():
    from whopays import config
    from whopays.config import RunConfig

    assert config.KV_QUANT_RECIPE == 3
    base = dict(model="Qwen/Qwen2.5-1.5B-Instruct", task="belebele", lang="en")
    assert RunConfig(**base, kv="quant4").canonical()["kv_recipe"] == 3
    assert "kv_recipe" not in RunConfig(**base, kv="full").canonical()
