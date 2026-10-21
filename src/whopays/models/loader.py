"""Model loading (FR-2). M0 supports FP16 weights; fake-quant, GPTQ and pre-quantized modes
arrive with the M1 quantization tasks.

Dtype follows golden rule 5: float16 on CUDA (T4 has no bf16) and MPS, float32 on CPU, where
float16 matmuls are slow and imprecise. Attention is always PyTorch SDPA (no FlashAttention-2
on sm75).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from whopays.config import RunConfig
from whopays.env import device as best_device

DTYPES = {"cuda": torch.float16, "mps": torch.float16, "cpu": torch.float32}


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    device: str
    dtype: torch.dtype
    draft: Any = None  # speculative-decoding draft model, same tokenizer


def _load_causal_lm(name: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa")
    return model.to(device).eval()


def load(cfg: RunConfig, device: str | None = None) -> ModelBundle:
    from transformers import AutoTokenizer

    if cfg.weight != "fp16":
        raise NotImplementedError(
            f"weight={cfg.weight!r} is not implemented yet (BUILD_PLAN M1: quant/, loader "
            "fake-quant mode)"
        )
    device = device or best_device()
    dtype = DTYPES[device]
    model = _load_causal_lm(cfg.model, device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    draft = _load_causal_lm(cfg.draft_model, device, dtype) if cfg.spec else None
    return ModelBundle(model=model, tokenizer=tokenizer, device=device, dtype=dtype, draft=draft)
