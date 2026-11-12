import pytest
import torch

from whopays.config import RunConfig
from whopays.models.loader import load


def test_fp16_config_loads_fp32_on_cpu(tiny_model_dir):
    b = load(RunConfig(model=str(tiny_model_dir), task="belebele", lang="en"), device="cpu")
    assert b.dtype == torch.float32 and b.device == "cpu"
    assert next(b.model.parameters()).dtype == torch.float32
    assert b.model.config._attn_implementation == "sdpa"
    assert not b.model.training and b.draft is None


def test_speculative_loads_draft(tiny_model_dir):
    cfg = RunConfig(model=str(tiny_model_dir), task="mgsm", lang="en", spec=True,
                    draft_model=str(tiny_model_dir))
    assert load(cfg, device="cpu").draft is not None


def test_rtn_weights_load(tiny_model_dir):
    cfg = RunConfig(model=str(tiny_model_dir), task="belebele", lang="en", weight="w8_rtn")
    b = load(cfg, device="cpu")
    assert b.quantization["method"] == "rtn"
    assert next(b.model.parameters()).dtype == torch.float32


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_mps_uses_fp16(tiny_model_dir):
    b = load(RunConfig(model=str(tiny_model_dir), task="belebele", lang="en"), device="mps")
    assert next(b.model.parameters()).dtype == torch.float16


def test_gptq_loads_tiny_model(tiny_model_dir):
    cfg = RunConfig(model=str(tiny_model_dir), task='belebele', lang='en',
                    weight='w4_gptq', calib='en', group_size=32)
    batches = [{'input_ids': torch.tensor([[1, 2, 3, 4]])}]
    b = load(cfg, device='cpu', calibration=batches)
    assert b.quantization['method'] == 'gptq'
    assert torch.isfinite(b.model(**batches[0]).logits).all()
    assert not any(m._forward_pre_hooks for m in b.model.modules())


def test_gptqmodel_checkpoints_load_in_the_run_dtype(monkeypatch):
    # GPTQModel 7.6 picks the config dtype (bf16 for Qwen2.5) unless told; T4 runs are FP16.
    import sys
    import types

    import pytest
    import torch

    from whopays.models import loader

    seen = {}

    def fake_load(path, backend, dtype):
        seen.update(path=path, backend=backend, dtype=dtype)
        weight = torch.zeros(2, 2, dtype=seen.pop('actual', dtype))
        model = types.SimpleNamespace(get_input_embeddings=lambda: types.SimpleNamespace(
            weight=weight), eval=lambda: model)
        return types.SimpleNamespace(model=model)

    fake = types.SimpleNamespace(BACKEND=types.SimpleNamespace(GPTQ_TORCH='torch'),
                                 GPTQModel=types.SimpleNamespace(load=fake_load))
    monkeypatch.setitem(sys.modules, 'gptqmodel', fake)
    loader._load_gptqmodel('ckpt', torch.float16)
    assert seen == {'path': 'ckpt', 'backend': 'torch', 'dtype': torch.float16}
    seen['actual'] = torch.bfloat16
    with pytest.raises(RuntimeError, match='bfloat16'):
        loader._load_gptqmodel('ckpt', torch.float16)


def test_quantized_weights_cache_reuses_identical_weights(tiny_model_dir, tmp_path, monkeypatch):
    from whopays.quant import calibration, fakequant

    batches = [{'input_ids': torch.randint(1, 200, (1, 32)),
                'attention_mask': torch.ones(1, 32, dtype=torch.long)} for _ in range(2)]
    monkeypatch.setattr(calibration, 'batches_for', lambda *a, **k: (batches, {'mode': 'en'}))
    monkeypatch.setenv('WHOPAYS_QUANT_CACHE', str(tmp_path))
    cfg = RunConfig(model=str(tiny_model_dir), task='belebele', lang='en',
                    weight='w4_gptq', calib='en', group_size=32)
    first = load(cfg, device='cpu')
    calls = []
    real = fakequant.apply
    monkeypatch.setattr(fakequant, 'apply', lambda *a, **k: calls.append(1) or real(*a, **k))
    second = load(cfg, device='cpu')
    assert not calls  # served from the cache, no second GPTQ pass
    for (n, a), (_, b) in zip(first.model.named_parameters(), second.model.named_parameters(),
                              strict=True):
        assert torch.equal(a, b), n
    assert second.quantization['weights_sha256'] == first.quantization['weights_sha256']
    assert second.quantization['calibration'] == {'mode': 'en'}
    assert second.quantization['layer_error'] == first.quantization['layer_error']


def test_corrupt_cache_is_rejected(tiny_model_dir, tmp_path, monkeypatch):
    from safetensors.torch import load_file, save_file

    from whopays.quant import calibration

    batches = [{'input_ids': torch.randint(1, 200, (1, 32))}]
    monkeypatch.setattr(calibration, 'batches_for', lambda *a, **k: (batches, {'mode': 'en'}))
    monkeypatch.setenv('WHOPAYS_QUANT_CACHE', str(tmp_path))
    cfg = RunConfig(model=str(tiny_model_dir), task='belebele', lang='en',
                    weight='w4_gptq', calib='en', group_size=32)
    load(cfg, device='cpu')
    (path,) = tmp_path.rglob('weights.safetensors')
    tensors = load_file(path)
    name = sorted(tensors)[0]
    tensors[name] = tensors[name] + 1
    save_file(tensors, path)
    with pytest.raises(RuntimeError, match='digest'):
        load(cfg, device='cpu')
