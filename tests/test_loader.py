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
