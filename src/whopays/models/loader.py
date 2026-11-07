"""FP16, portable RTN/GPTQ fake quantization and real GPTQ checkpoint loading (FR-2).

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
    quantization: dict | None = None


def _load_causal_lm(name: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    if device == "cuda" and "7B" in name:  # about 15 GB in FP16, more than one T4 holds
        if torch.cuda.device_count() < 2:
            raise ValueError(f"{name} needs two GPUs: use Kaggle's T4×2 accelerator")
        return AutoModelForCausalLM.from_pretrained(
            name, dtype=dtype, attn_implementation="sdpa", device_map="auto").eval()
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa")
    return model.to(device).eval()


def load(cfg: RunConfig, device: str | None = None, *, calibration=None) -> ModelBundle:
    from transformers import AutoTokenizer

    device = device or best_device()
    dtype = DTYPES[device]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if cfg.weight == "prequant":
        if device != "cuda":
            raise ValueError("real GPTQ checkpoints require CUDA; use fake quantization locally")
        try:
            from gptqmodel import BACKEND, GPTQModel
        except ImportError as exc:
            raise ImportError("install GPTQModel in its isolated reference environment") from exc
        model = GPTQModel.load(cfg.model, backend=BACKEND.GPTQ_TORCH).model.eval()
    elif cfg.weight == "w4_gptqmodel":
        if device != "cuda":
            raise ValueError("GPTQModel quantization requires CUDA")
        try:
            from gptqmodel import BACKEND, GPTQModel
        except ImportError as exc:
            raise ImportError("install GPTQModel in its isolated reference environment") from exc
        from whopays.replicate import quantize_with_gptqmodel
        path, provenance = quantize_with_gptqmodel(cfg, tokenizer)
        model = GPTQModel.load(str(path), backend=BACKEND.GPTQ_TORCH).model.eval()
        return ModelBundle(model=model, tokenizer=tokenizer, device=device, dtype=dtype,
                           quantization={"method": "gptqmodel", "calibration": provenance})
    else:
        model = _load_causal_lm(cfg.model, device, dtype)
    quantization = None
    if cfg.weight in {"w8_rtn", "w4_gptq", "w3_gptq", "bitmap"}:
        from whopays.quant.fakequant import apply
        method = "rtn" if cfg.weight == "w8_rtn" else "gptq"
        bits = cfg.bitmap if cfg.weight == "bitmap" else int(cfg.weight[1])
        provenance = None
        if method == "gptq" and calibration is None:
            from whopays.quant.calibration import batches_for
            calibration, provenance = batches_for(tokenizer, cfg.calib_source, cfg.calib,
                                                  cfg.seed)
        quantization = apply(model, bits, method=method, calibration=calibration,
                             group_size=cfg.group_size)
        if provenance is not None:
            quantization["calibration"] = provenance
    draft = _load_causal_lm(cfg.draft_model, device, dtype) if cfg.spec else None
    return ModelBundle(model=model, tokenizer=tokenizer, device=device, dtype=dtype, draft=draft,
                       quantization=quantization)
