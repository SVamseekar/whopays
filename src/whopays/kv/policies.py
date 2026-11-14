"""KV-cache policies for the two-pass context/query protocol (FR-4).

Prefill returns full-precision attention even with the 4-bit cache; quantization affects
subsequent queries. Eviction retains already rotated keys, so callers must pass the original
absolute ``position_ids`` for every later forward instead of deriving them from the retained length.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache
from transformers.cache_utils import Cache, DynamicLayer, QuantizedLayer

from whopays.config import RunConfig

# 4-bit cache: KIVI (Liu et al., ICML 2024, arXiv 2402.02750, section 4.1): keys quantized per
# channel and values per token, asymmetric round-to-nearest, "group size G ... 32 across all
# experiments" and "residual length R ... 128" tokens kept in full precision. The earlier HQQ
# recipe grouped keys per token (axis 1); Qwen2.5 keys carry outlier channels (layer 0: |k|
# about 320 against a median of 4), which flattened every other channel, and MGSM collapsed to
# 0/16 in English on a T4 and on CPU (RUN_LOG 17; DECISIONS 2026-11-14).
KIVI_BITS = 4
KIVI_GROUP = 32
KIVI_RESIDUAL = 128


def fake_quant(x: torch.Tensor, dim: int, bits: int = KIVI_BITS) -> torch.Tensor:
    """Asymmetric min-max round-to-nearest over `dim`, returned dequantized in x's dtype."""
    work = x.float()
    low = work.amin(dim=dim, keepdim=True)
    high = work.amax(dim=dim, keepdim=True)
    scale = ((high - low) / (2**bits - 1)).clamp(min=1e-8)
    q = ((work - low) / scale).round().clamp(0, 2**bits - 1)
    return (q * scale + low).to(x.dtype)


class KIVILayer(DynamicLayer):
    """Fake-quantized KIVI layer: values are stored dequantized; memory is counted from bits.

    Each row of a batch is handled as if alone: `set_padding` gives each row's left-padding
    count, and groups start at the row's first real token, so padding never enters a group.
    A token is quantized once, when a whole group of `group` tokens has aged past the
    `residual` most recent ones; the forward that writes tokens attends to them unquantized.
    """

    def __init__(self, nbits: int = KIVI_BITS, group: int = KIVI_GROUP,
                 residual: int = KIVI_RESIDUAL):
        super().__init__()
        self.nbits, self.group, self.residual = nbits, group, residual
        self.pad: torch.Tensor | None = None
        self.n_quant: torch.Tensor | None = None

    def set_padding(self, pad: torch.Tensor) -> None:
        self.pad = pad.detach().to("cpu", torch.long)

    def update(self, key_states, value_states, *args, **kwargs):
        keys, values = super().update(key_states, value_states, *args, **kwargs)
        batch = self.keys.shape[0]
        if self.pad is None:
            self.pad = torch.zeros(batch, dtype=torch.long)
        if self.n_quant is None:
            self.n_quant = torch.zeros(batch, dtype=torch.long)
        self._flush()
        return keys, values

    def _targets(self, extra: int = 0) -> torch.Tensor:
        real = self.keys.shape[-2] + extra - self.pad
        return ((real - self.residual).clamp(min=0) // self.group) * self.group

    def _flush(self) -> None:
        target = self._targets()
        if bool((target <= self.n_quant).all()):
            return
        keys, values = self.keys.clone(), self.values.clone()
        heads, dim = keys.shape[1], keys.shape[-1]
        vgroup = min(KIVI_GROUP, dim)
        for row in range(keys.shape[0]):
            start = int(self.pad[row] + self.n_quant[row])
            stop = int(self.pad[row] + target[row])
            if stop <= start:
                continue
            blocks = (stop - start) // self.group
            k = keys[row, :, start:stop].reshape(heads, blocks, self.group, dim)
            keys[row, :, start:stop] = fake_quant(k, dim=-2, bits=self.nbits).reshape(
                heads, stop - start, dim)
            v = values[row, :, start:stop].reshape(heads, stop - start, dim // vgroup, vgroup)
            values[row, :, start:stop] = fake_quant(v, dim=-1, bits=self.nbits).reshape(
                heads, stop - start, dim)
        self.keys, self.values = keys, values
        self.n_quant = torch.maximum(self.n_quant, target)

    def flush_free(self, size: int) -> bool:
        """True when appending `size` tokens would quantize no further group."""
        if self.n_quant is None:
            return True
        return bool((self._targets(size) <= self.n_quant).all())

    def residual_tokens(self) -> int:
        if self.n_quant is None:
            return self.get_seq_length()
        return int((self.keys.shape[-2] - self.pad - self.n_quant).min())

    def storage_bytes(self) -> int:
        """Cache storage implied by the recipe: 4-bit codes, FP16 scale and zero per group,
        FP16 residual (rule 7: memory from bit counts, not from fake-quantized tensors)."""
        if self.n_quant is None:
            return 0
        heads, dim = self.keys.shape[1], self.keys.shape[-1]
        vgroups = dim // min(KIVI_GROUP, dim)
        quant = int(self.n_quant.sum())
        exact = int((self.keys.shape[-2] - self.pad - self.n_quant).sum())
        codes = 2 * quant * heads * dim * self.nbits // 8
        meta = (quant // self.group) * heads * dim * 2 * 2 + quant * heads * vgroups * 2 * 2
        return codes + meta + 2 * exact * heads * dim * 2


def new_cache(cfg: RunConfig, model) -> Cache:
    """Create a cache; quant4 gets one KIVI layer per decoder layer. A quant4 cache serves a
    whole batch once `set_padding` has given each row's left padding."""
    if cfg.kv == "quant4":
        return Cache(layers=[KIVILayer() for _ in range(model.config.num_hidden_layers)])
    return DynamicCache(config=model.config)


def set_padding(cache: Cache, attention_mask: torch.Tensor) -> None:
    """Tell KIVI layers each row's left padding (no-op for other caches)."""
    pad = (attention_mask == 0).sum(dim=1)
    for layer in cache.layers:
        if isinstance(layer, KIVILayer):
            layer.set_padding(pad)


def after_prefill(cache: Cache, cfg: RunConfig, budget: int | None = None) -> int:
    """Apply the context policy and return the number of cached positions retained.

    `evict_share` needs `budget`, the item's own token budget; `evict` uses `cfg.kv_budget`.
    """
    return _apply_policy(cache, cfg, budget)


def after_decode(cache: Cache, cfg: RunConfig) -> int:
    """Maintain the policy after committed decode tokens, returning the retained length.

    Apply eviction only after accepting tokens: discarded history cannot be restored by
    rollback. Full and quantized caches already implement their storage policy on update.
    """
    return _apply_policy(cache, cfg)


def _apply_policy(cache: Cache, cfg: RunConfig, budget: int | None = None) -> int:
    if cfg.kv == "evict_share" and budget is None:
        raise ValueError("kv='evict_share' needs the item's budget")
    if cfg.kv in ("evict", "evict_share"):
        budget = cfg.kv_budget if cfg.kv == "evict" else budget
        for layer in cache.layers:
            length = layer.get_seq_length()
            if length <= budget:
                continue
            sink = min(4, budget)
            recent = budget - sink
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
        if isinstance(layer, KIVILayer) and n_remove > layer.residual_tokens():
            raise ValueError("rollback exceeds the quantized cache FP16 residual; rebuild cache")
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

    kivi = [layer for layer in cache.layers if isinstance(layer, KIVILayer)]
    if kivi:  # fake quantization: report the recipe's storage, not the FP16 tensors
        return sum(layer.storage_bytes() for layer in kivi)
    visit(vars(cache))
    for layer in cache.layers:
        visit(vars(layer))
    return sum(storages.values())
