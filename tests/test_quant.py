import pytest
import torch

from whopays.quant.gptq import quantize_weight
from whopays.quant.rtn import quantize


@pytest.mark.parametrize('bits', [2, 3, 4, 8])
def test_rtn_error_bound(bits):
    torch.manual_seed(12)
    w = torch.randn(8, 33)
    q = quantize(w, bits, group_size=8)
    for start in range(0, 33, 8):
        x = w[:, start:start + 8]
        scale = (x.amax(1).clamp(min=0) - x.amin(1).clamp(max=0)) / (2**bits - 1)
        assert torch.all((q[:, start:start + 8] - x).abs() <= scale[:, None] / 2 + 1e-6)
    assert torch.equal(quantize(torch.zeros(3, 7), bits), torch.zeros(3, 7))


def test_gptq_reduces_calibration_output_error():
    torch.manual_seed(42)
    w = torch.randn(12, 32)
    x = torch.randn(512, 32)
    x[:, 1:] += x[:, :1] * 3
    h = x.T @ x / len(x)
    q = quantize_weight(w, h, bits=3, group_size=32)
    rtn = quantize(w, 3, group_size=32)
    assert ((x @ (q-w).T)**2).mean() < ((x @ (rtn-w).T)**2).mean()
    assert torch.isfinite(quantize_weight(w, torch.zeros(32, 32), 4)).all()


def test_calibration_balanced_and_seeded(monkeypatch):
    from whopays.quant.calibration import sample_calibration
    monkeypatch.setattr('whopays.datasets.flores', lambda lang, split: [
        {'item_id': str(i), 'text': f'{lang}-{i}-{split}'} for i in range(30)])
    a = sample_calibration('multi', seed=3, n_samples=22)
    assert a == sample_calibration('multi', seed=3, n_samples=22)
    assert a != sample_calibration('multi', seed=4, n_samples=22)
    from collections import Counter
    assert set(Counter(r['lang'] for r in a).values()) == {2}
    assert all(r['text'].endswith('-dev') for r in a)


def test_apply_bitmap_and_preserve_head(tiny_model_dir):
    from whopays.config import RunConfig
    from whopays.models.loader import load
    from whopays.quant.fakequant import apply
    b = load(RunConfig(model=str(tiny_model_dir), task='belebele', lang='en'), 'cpu')
    head = b.model.lm_head.weight.clone()
    before = b.model.model.layers[0].self_attn.q_proj.weight.clone()
    summary = apply(b.model, {'model.layers.0': 2, 'model.layers.1': 4})
    assert torch.equal(head, b.model.lm_head.weight)
    assert not torch.equal(before, b.model.model.layers[0].self_attn.q_proj.weight)
    assert summary['memory_kind'] == 'computed_weight_bits'
    with pytest.raises(ValueError, match='missing'):
        apply(b.model, {'model.layers.0': 4})


def test_profile_restores_weights_and_resumes(tiny_model_dir, tmp_path):
    from whopays.config import RunConfig
    from whopays.models.loader import load
    from whopays.parityquant.profile import profile
    b = load(RunConfig(model=str(tiny_model_dir), task='belebele', lang='en'), 'cpu')
    before = {k: v.clone() for k, v in b.model.state_dict().items()}
    path = tmp_path / 'profile.jsonl'
    rows = profile(b, {'en': ['the river'], 'de': ['Das Wetter']}, bits=(2,),
                   output_path=path, run_id='test')
    assert len(rows) == 4
    assert all(torch.equal(v, before[k]) for k, v in b.model.state_dict().items())
    assert rows == profile(b, {'en': ['the river'], 'de': ['Das Wetter']}, bits=(2,),
                           output_path=path, run_id='test')
    with pytest.raises(ValueError, match='manifest'):
        profile(b, {'en': ['changed']}, bits=(2,), output_path=path, run_id='test')
    import json

    meta = json.loads(path.with_suffix('.manifest.json').read_text())
    assert meta['env']['code_version']
    assert meta['model']['num_hidden_layers'] == b.model.config.num_hidden_layers
    from whopays.parityquant import profile as profiler
    original = profiler.env_info
    profiler.env_info = lambda: {**original(), 'code_version': 'not-this-checkout'}
    try:
        with pytest.raises(ValueError, match='manifest'):
            profile(b, {'en': ['the river'], 'de': ['Das Wetter']}, bits=(2,),
                    output_path=path, run_id='test')
    finally:
        profiler.env_info = original
    b.model.config.num_hidden_layers += 1
    with pytest.raises(ValueError, match='manifest'):
        profile(b, {'en': ['the river'], 'de': ['Das Wetter']}, bits=(2,),
                output_path=path, run_id='test')


def test_gptq_improves_tiny_qwen_projection(tiny_model_dir):
    from whopays.config import RunConfig
    from whopays.models.loader import load
    from whopays.quant.fakequant import activation_hessian
    b = load(RunConfig(model=str(tiny_model_dir), task='belebele', lang='en'), 'cpu')
    module = b.model.model.layers[0].self_attn.q_proj
    h = activation_hessian(b.model, module, [{'input_ids': torch.arange(1, 200)[None]}])
    weight = module.weight.detach()
    gptq = quantize_weight(weight, h, 3, 32)
    rtn = quantize(weight, 3, 32)
    def error(q):
        diff = q - weight
        return ((diff @ h) * diff).sum()
    assert error(gptq) < error(rtn)


def test_profile_restores_on_scoring_failure(tiny_model_dir, monkeypatch):
    from whopays.config import RunConfig
    from whopays.models.loader import load
    from whopays.parityquant import profile as profiler
    b = load(RunConfig(model=str(tiny_model_dir), task='belebele', lang='en'), 'cpu')
    before = {k: v.clone() for k, v in b.model.state_dict().items()}
    calls = 0
    def fail_after_baseline(*args):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError('interrupted')
        return 1.0
    monkeypatch.setattr(profiler, '_score', fail_after_baseline)
    with pytest.raises(RuntimeError, match='interrupted'):
        profiler.profile(b, {'en': ['test']}, bits=(2,))
    assert all(torch.equal(v, before[k]) for k, v in b.model.state_dict().items())


@torch.no_grad()
def _legacy_gptq(model, bits, batches, group_size):
    """The pre-2026-11-07 path: recapture each block's inputs from the partly quantized model
    and quantize one projection at a time. The faster path must reproduce it."""
    from whopays.quant import fakequant as fq
    layers = fq.linear_layers(model)
    modules = dict(model.named_modules())
    current, captured = None, None
    for name, module in layers.items():
        block_name = '.'.join(name.split('.')[:3])
        block = modules[block_name]
        if block_name != current:
            captured = fq.capture_block_inputs(model, block, batches)
            current = block_name
        h = fq.activation_hessian(model, module, batches, block=block, captured=captured)
        module.weight.copy_(quantize_weight(module.weight, h, bits, group_size))


def test_gptq_apply_matches_per_block_recapture(tiny_model_dir):
    from whopays.config import RunConfig
    from whopays.models.loader import load
    from whopays.quant.fakequant import apply
    torch.manual_seed(0)
    cfg = RunConfig(model=str(tiny_model_dir), task='belebele', lang='en')
    fast, slow = load(cfg, 'cpu').model, load(cfg, 'cpu').model
    batches = [{'input_ids': torch.randint(1, 200, (1, 48)),
                'attention_mask': torch.ones(1, 48, dtype=torch.long)} for _ in range(3)]
    apply(fast, 3, method='gptq', calibration=batches, group_size=16)
    _legacy_gptq(slow, 3, batches, 16)
    for (name, a), (_, b) in zip(fast.named_parameters(), slow.named_parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-5, rtol=0), name


def test_gptq_reports_hessian_weighted_error_per_layer(tiny_model_dir):
    from whopays.config import RunConfig
    from whopays.models.loader import load
    from whopays.quant.fakequant import apply
    torch.manual_seed(0)
    cfg = RunConfig(model=str(tiny_model_dir), task='belebele', lang='en')
    batches = [{'input_ids': torch.randint(1, 200, (1, 48)),
                'attention_mask': torch.ones(1, 48, dtype=torch.long)} for _ in range(3)]
    errors = {}
    for bits in (3, 4):
        model = load(cfg, 'cpu').model
        summary = apply(model, bits, method='gptq', calibration=batches, group_size=16)
        errors[bits] = summary['layer_error']
        assert set(errors[bits]) == set(summary['bitmap'])
        assert all(0 < e < 1 for e in errors[bits].values())
    assert sum(errors[3].values()) > sum(errors[4].values())


def test_rtn_reports_no_layer_error(tiny_model_dir):
    from whopays.config import RunConfig
    from whopays.models.loader import load
    from whopays.quant.fakequant import apply
    model = load(RunConfig(model=str(tiny_model_dir), task='belebele', lang='en'), 'cpu').model
    assert 'layer_error' not in apply(model, 4)
