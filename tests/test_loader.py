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


def test_quantized_weights_not_yet_supported(tiny_model_dir):
    cfg = RunConfig(model=str(tiny_model_dir), task="belebele", lang="en", weight="w8_rtn")
    with pytest.raises(NotImplementedError, match="M1"):
        load(cfg, device="cpu")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_mps_uses_fp16(tiny_model_dir):
    b = load(RunConfig(model=str(tiny_model_dir), task="belebele", lang="en"), device="mps")
    assert next(b.model.parameters()).dtype == torch.float16
