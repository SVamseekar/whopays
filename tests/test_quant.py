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
