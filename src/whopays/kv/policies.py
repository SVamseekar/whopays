"""KV-cache policies for the two-pass context/query protocol (FR-4).

Prefill returns full-precision attention even with HQQ; quantization affects subsequent
queries. Eviction retains already rotated keys, so callers must pass the original absolute
``position_ids`` for every later forward instead of deriving them from the retained length.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache
from transformers.cache_utils import Cache, QuantizedLayer

from whopays.config import RunConfig


def new_cache(cfg: RunConfig, model) -> Cache:
    """Create a per-sequence cache; HQQ is needed only for the quant4 policy."""
    if cfg.kv == "quant4":
        from transformers import QuantizedCache

        try:
            return QuantizedCache(backend="hqq", config=model.config, nbits=4)
        except ImportError as exc:
            raise ImportError("kv='quant4' requires HQQ; install whopays[gpu]") from exc
    return DynamicCache(config=model.config)


def after_prefill(cache: Cache, cfg: RunConfig) -> int:
    """Apply the context policy and return the number of cached positions retained."""
    return _apply_policy(cache, cfg)


def after_decode(cache: Cache, cfg: RunConfig) -> int:
    """Maintain the policy after committed decode tokens, returning the retained length.

    Apply eviction only after accepting tokens: discarded history cannot be restored by
    rollback. Full and quantized caches already implement their storage policy on update.
    """
    return _apply_policy(cache, cfg)


def _apply_policy(cache: Cache, cfg: RunConfig) -> int:
    if cfg.kv == "evict":
        for layer in cache.layers:
            length = layer.get_seq_length()
            if length <= cfg.kv_budget:
                continue
            sink = min(4, cfg.kv_budget)
            recent = cfg.kv_budget - sink
            indices = torch.cat([
                torch.arange(sink, device=layer.keys.device),
                torch.arange(length - recent, length, device=layer.keys.device),
            ])
            layer.keys = layer.keys.index_select(-2, indices)
            layer.values = layer.values.index_select(-2, indices)
    return cache.get_seq_length()


def rollback(cache: Cache, n_remove: int) -> None:
    """Remove a suffix, refusing any rollback through quantized history.

    Transformers' inherited crop trims a quantized layer's residual but does not rewind its
    cumulative length. Validate every layer before mutating any, then repair that counter.
    Once rejected tokens have flushed into compressed storage, callers must rebuild instead.
    """
    if not isinstance(n_remove, int) or n_remove < 0:
        raise ValueError("n_remove must be a nonnegative integer")
    if n_remove == 0:
        return
    if not cache.layers:
        raise ValueError("cannot roll back more tokens than the cache contains")
    for layer in cache.layers:
        if n_remove > layer.get_seq_length():
            raise ValueError("cannot roll back more tokens than the cache contains")
        if not layer.is_croppable:
            raise ValueError("cache layer does not support rollback")
        if isinstance(layer, QuantizedLayer):
            residual = layer.keys.shape[-2] if layer.keys.ndim == 4 else 0
            if n_remove > residual:
                raise ValueError(
                    "rollback exceeds the quantized cache FP16 residual; rebuild cache"
                )
    for layer in cache.layers:
        layer.crop(-n_remove)
        if isinstance(layer, QuantizedLayer):
            layer.cumulative_length -= n_remove


def cache_bytes(cache: Cache) -> int:
    """Count unique tensor-storage bytes held by the cache, including HQQ metadata.

    This excludes weights, temporary attention tensors and allocator overhead. Views count
    their entire retained allocation; aliased storages are counted once. This is cache storage,
    not measured model memory or a claim about fake-quantized weight memory savings.
    """
    storages: dict[tuple[str, int], int] = {}

    def visit(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(vars(cache))
    for layer in cache.layers:
        visit(vars(layer))
    return sum(storages.values())
