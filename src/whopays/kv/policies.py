"""KV-cache policies behind one interface (FR-4).

Protocol (DECISIONS 2026-10-21): the context is prefilled with full-precision attention into
`new_cache(...)`, `after_prefill(...)` applies the policy to that cache, and the query is then
run against the result. This is the only way a single-pass score (Belebele) sees the policy.

M0 implements `full`. `quant4` and `evict` arrive with the M2 task `kv/policies.py`.
"""

from __future__ import annotations

from transformers import DynamicCache

from whopays.config import RunConfig


def new_cache(cfg: RunConfig, model) -> DynamicCache:
    if cfg.kv == "full":
        return DynamicCache(config=model.config)
    raise NotImplementedError(f"kv={cfg.kv!r} is not implemented yet (BUILD_PLAN M2)")


def after_prefill(cache: DynamicCache, cfg: RunConfig) -> int:
    """Apply the policy to a prefilled cache. Returns the number of context positions kept."""
    if cfg.kv == "full":
        return cache.get_seq_length()
    raise NotImplementedError(f"kv={cfg.kv!r} is not implemented yet (BUILD_PLAN M2)")
