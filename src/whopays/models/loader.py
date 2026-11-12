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


def _load_gptqmodel(path: str, dtype: torch.dtype):
    """Load a GPTQ checkpoint with GPTQModel's torch kernel in the run dtype.

    GPTQModel 7.6 otherwise takes the dtype from the model config (bfloat16 for Qwen2.5),
    which the T4 does not support natively and which differs from every other run.
    """
    try:
        from gptqmodel import BACKEND, GPTQModel
    except ImportError as exc:
        raise ImportError("install GPTQModel in its isolated reference environment") from exc
    model = GPTQModel.load(path, backend=BACKEND.GPTQ_TORCH, dtype=dtype).model.eval()
    got = model.get_input_embeddings().weight.dtype
    if got != dtype:
        raise RuntimeError(f"GPTQModel loaded {path} in {got}, expected {dtype}")
    return model


def load(cfg: RunConfig, device: str | None = None, *, calibration=None) -> ModelBundle:
    from transformers import AutoTokenizer

    device = device or best_device()
    dtype = DTYPES[device]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if cfg.weight == "prequant":
        if device != "cuda":
            raise ValueError("real GPTQ checkpoints require CUDA; use fake quantization locally")
        model = _load_gptqmodel(cfg.model, dtype)
    elif cfg.weight == "w4_gptqmodel":
        if device != "cuda":
            raise ValueError("GPTQModel quantization requires CUDA")
        from whopays.replicate import quantize_with_gptqmodel
        path, provenance = quantize_with_gptqmodel(cfg, tokenizer, dtype=dtype)
        model = _load_gptqmodel(str(path), dtype)
        return ModelBundle(model=model, tokenizer=tokenizer, device=device, dtype=dtype,
                           quantization={"method": "gptqmodel", "calibration": provenance})
    else:
        model = _load_causal_lm(cfg.model, device, dtype)
    quantization = None
    cache = _cache_dir(cfg) if calibration is None and cfg.weight != "w8_rtn" else None
    if cache is not None and (cache / "quantization.json").exists():
        quantization = _load_cached(model, cache)
    elif cfg.weight in {"w8_rtn", "w4_gptq", "w3_gptq", "bitmap"}:
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
        if cache is not None:
            _save_cached(model, quantization, cache)
    draft = _load_causal_lm(cfg.draft_model, device, dtype) if cfg.spec else None
    return ModelBundle(model=model, tokenizer=tokenizer, device=device, dtype=dtype, draft=draft,
                       quantization=quantization)


def _cache_dir(cfg: RunConfig):
    """Folder for this model key's quantized weights, or None when caching is off.

    Set WHOPAYS_QUANT_CACHE (Modal sets it to a Volume) to quantize each model key once and
    reuse the exact weights in every later process. The folder name also carries the torch and
    transformers versions, so a different software stack never reuses weights it did not make.
    """
    import os
    from pathlib import Path

    import transformers

    root = os.environ.get("WHOPAYS_QUANT_CACHE")
    if not root or cfg.weight not in {"w4_gptq", "w3_gptq", "bitmap"}:
        return None
    stack = f"torch-{torch.__version__}_transformers-{transformers.__version__}"
    return Path(root) / stack / cfg.model_key()


def _save_cached(model, quantization: dict, cache) -> None:
    import json
    import os
    import tempfile

    from safetensors.torch import save_file

    from whopays.quant.fakequant import linear_layers

    layers = linear_layers(model)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=cache.parent, prefix=".tmp-")
    save_file({n: layers[n].weight.detach().cpu().contiguous() for n in quantization["bitmap"]},
              os.path.join(tmp, "weights.safetensors"))
    with open(os.path.join(tmp, "quantization.json"), "w") as f:
        json.dump(quantization, f, sort_keys=True)
    try:
        os.rename(tmp, cache)  # atomic: a concurrent writer of the same key loses harmlessly
    except OSError:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


@torch.no_grad()
def _load_cached(model, cache) -> dict:
    import json

    from safetensors.torch import load_file

    from whopays.quant.fakequant import linear_layers, weights_digest

    quantization = json.loads((cache / "quantization.json").read_text())
    tensors = load_file(str(cache / "weights.safetensors"))
    layers = linear_layers(model)
    for name in quantization["bitmap"]:
        layers[name].weight.copy_(tensors[name].to(layers[name].weight.device))
    if weights_digest(layers, quantization["bitmap"]) != quantization.get("weights_sha256"):
        raise RuntimeError(f"cached weights in {cache} do not match their recorded digest")
    return quantization
