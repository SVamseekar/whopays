import pytest
import torch
from transformers import DynamicCache
from transformers.cache_utils import Cache, QuantizedLayer

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


class _RoundingLayer(QuantizedLayer):
    """CPU stand-in for HQQ: lossy, stateful and flushing exactly like the real layer."""

    scale = 4.0

    def _quantize(self, tensor, axis):
        return torch.round(tensor * self.scale) / self.scale

    def _dequantize(self, q_tensor):
        return q_tensor


def _lossy_cache(residual, scale=4.0):
    def build(cfg, model):
        if cfg.kv != 'quant4':  # the draft keeps a full cache, as in production
            return DynamicCache(config=model.config)
        layers = [_RoundingLayer(residual_length=residual)
                  for _ in range(model.config.num_hidden_layers)]
        for layer in layers:
            layer.scale = scale
        return Cache(layers=layers)
    return build


def _bundle(tiny_model_dir, kv, *, budget=6, same_draft=True):
    cfg = RunConfig(model=str(tiny_model_dir), task='mgsm', lang='en', kv=kv, kv_budget=budget,
                    spec=True, draft_model=str(tiny_model_dir))
    b = load(cfg, 'cpu')
    if not same_draft:
        with torch.no_grad():
            b.draft.lm_head.weight.mul_(-1)
    return b, cfg


@pytest.mark.parametrize('budget', [4, 5, 6, 9, 30, 256])
@pytest.mark.parametrize('same_draft', [True, False])
def test_eviction_block_verification_matches_greedy(tiny_model_dir, budget, same_draft):
    b, cfg = _bundle(tiny_model_dir, 'evict', budget=budget, same_draft=same_draft)
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plain = generate(b, cfg.model_copy(update={'spec': False}), ids, max_new_tokens=40)
    spec = generate(b, cfg, ids, max_new_tokens=40, compressed_block=True)
    assert spec['token_ids'] == plain['token_ids']
    assert spec['verification_mode'] == 'block'


def test_eviction_block_verification_saves_target_forwards(tiny_model_dir):
    # Budget above the sequence: a full-cache draft then agrees with the target. Tight budgets
    # make them disagree, so acceptance falls; the exactness tests above cover that case.
    b, cfg = _bundle(tiny_model_dir, 'evict', budget=256)
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plain = generate(b, cfg.model_copy(update={'spec': False}), ids, max_new_tokens=40)
    spec = generate(b, cfg, ids, max_new_tokens=40, compressed_block=True)
    assert spec['token_ids'] == plain['token_ids']
    assert spec['target_forwards'] < plain['target_forwards']
    assert spec['target_acceleration_enabled'] is True
    assert spec['block_rounds'] > 0


@pytest.mark.parametrize('residual', [3, 6, 11, 128])
@pytest.mark.parametrize('same_draft', [True, False])
def test_quantized_block_verification_matches_greedy_across_flushes(
        tiny_model_dir, monkeypatch, residual, same_draft):
    monkeypatch.setattr('whopays.decoding.speculative.new_cache', _lossy_cache(residual))
    b, cfg = _bundle(tiny_model_dir, 'quant4', same_draft=same_draft)
    ids = [1, 2, 3, 4, 5, 6, 7]
    plain = generate(b, cfg.model_copy(update={'spec': False}), ids, max_new_tokens=60)
    spec = generate(b, cfg, ids, max_new_tokens=60, compressed_block=True)
    assert spec['token_ids'] == plain['token_ids']


def test_quantized_block_verification_falls_back_near_a_flush(tiny_model_dir, monkeypatch):
    monkeypatch.setattr('whopays.decoding.speculative.new_cache', _lossy_cache(8, scale=1e4))
    b, cfg = _bundle(tiny_model_dir, 'quant4')
    plain_cfg = cfg.model_copy(update={'spec': False})
    plain = generate(b, plain_cfg, [1, 2, 3, 4], max_new_tokens=60)
    spec = generate(b, cfg, [1, 2, 3, 4], max_new_tokens=60, compressed_block=True)
    assert spec['block_rounds'] > 0 and spec['sequential_rounds'] > 0
    assert spec['target_forwards'] < plain['target_forwards']


def test_compressed_block_stays_off_until_the_gpu_gate_passes(tiny_model_dir):
    # DECISIONS 2026-10-31: the sequential control stays until a T4 exactness gate passes.
    b, cfg = _bundle(tiny_model_dir, 'evict')
    result = generate(b, cfg, [1, 2, 3, 4, 5, 6, 7], max_new_tokens=8)
    assert result['verification_mode'] == 'sequential'
    assert result['target_acceleration_enabled'] is False
