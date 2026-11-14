"""Greedy draft/verify with block verification (FR-5).

The full cache verifies a draft block in one forward. Compressed caches are stateful, so a
plain block forward would change greedy semantics. Two exact replays exist behind
``compressed_block``: eviction gives each block token a 4D attention row equal to the cache it
would have seen sequentially, and the quantized cache takes a block only when no residual
flush can occur inside it. These stay off (labelled sequential, no acceleration) until
COMPRESSED_BLOCK_VERIFIED records a passed T4 exactness gate (DECISIONS 2026-10-31).
Counters include every model forward, including prefill and correction/bonus forwards.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from whopays.kv.policies import after_decode, after_prefill, cache_bytes, new_cache, rollback

COMPRESSED_BLOCK_VERIFIED = False  # set True only with a logged notebook 05 exactness PASS


def eviction_attention(cached: int, block: int, budget: int, device) -> torch.Tensor:
    """Boolean (1, 1, block, cached + block) mask replaying sequential sink+recent eviction."""
    sink = min(4, budget)
    recent = budget - sink
    mask = torch.zeros((1, 1, block, cached + block), dtype=torch.bool, device=device)
    kept = list(range(cached))
    for j in range(block):
        current = cached + j
        mask[0, 0, j, [*kept, current]] = True
        kept.append(current)
        if len(kept) > budget:
            kept = kept[:sink] + (kept[-recent:] if recent else [])
    return mask


class _State:
    def __init__(self, model, cfg, device):
        self.model, self.cfg, self.device = model, cfg, device
        self.cache = new_cache(cfg, model)
        self.forwards = 0
        self.position = 0
        self.peak_bytes = 0
        self.logits = None

    def forward(self, tokens, *, prefill=False, commit=True, mask=None):
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        positions = torch.arange(self.position, self.position + len(tokens), device=self.device)
        if mask is None:
            mask = torch.ones((1, self.cache.get_seq_length() + len(tokens)),
                              dtype=torch.long, device=self.device)
        out = self.model(input_ids=ids, attention_mask=mask, position_ids=positions[None],
                         past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.position += len(tokens)
        self.forwards += 1
        self.logits = out.logits[0, -1]
        if prefill:
            after_prefill(self.cache, self.cfg)
        if commit:
            self.commit()
        else:
            self.peak_bytes = max(self.peak_bytes, cache_bytes(self.cache))
        return out.logits[0]

    def commit(self):
        """Apply the decode policy; evicted history cannot be rolled back after this."""
        after_decode(self.cache, self.cfg)
        self.peak_bytes = max(self.peak_bytes, cache_bytes(self.cache))

    def block_is_exact(self, size):
        """True when one forward of `size` tokens equals `size` sequential forwards."""
        if self.cfg.kv != "quant4":
            return True
        from whopays.kv.policies import KIVILayer

        return all(layer.flush_free(size) if isinstance(layer, KIVILayer)
                   else (layer.keys.shape[-2] if layer.keys.ndim == 4 else 0) + size
                   < layer.residual_length for layer in self.cache.layers)

    def block_mask(self, size):
        if self.cfg.kv != "evict":
            return None
        return eviction_attention(self.cache.get_seq_length(), size, self.cfg.kv_budget,
                                  self.device)

    def rewind(self, n):
        if n:
            rollback(self.cache, n)
            self.position -= n

    def next(self):
        return int(self.logits.argmax())


@torch.no_grad()
def generate(bundle, cfg, input_ids: Sequence[int] | torch.Tensor, max_new_tokens: int = 256,
             *, stop_strings: Sequence[str] | str | None = None,
             eos_token_id: int | Sequence[int] | None = None,
             compressed_block: bool | None = None) -> dict:
    """Generate one unpadded prompt; return new token IDs and honest inference accounting.

    EOS and a completed stop string are included in token_ids, matching Transformers.
    steps/target_forwards/target_forward_passes count actual target calls including prefill.
    cache_storage_bytes is the peak target cache storage after policy application.
    """
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim == 2 and input_ids.shape[0] == 1:
            input_ids = input_ids[0]
        if input_ids.ndim != 1:
            raise ValueError("generate requires one unpadded prompt")
        input_ids = input_ids.tolist()
    ids = list(input_ids)
    if not ids:
        raise ValueError("input_ids must contain at least one token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if cfg.spec and bundle.draft is None:
        raise ValueError("speculative generation requires a loaded draft model")
    if eos_token_id is None:
        generation_config = getattr(bundle.model, "generation_config", None)
        eos_token_id = getattr(generation_config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(bundle.tokenizer, "eos_token_id", None)
    eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple, set))
              else [eos_token_id]) - {None}
    stops = [stop_strings] if isinstance(stop_strings, str) else list(stop_strings or [])
    if any(not s for s in stops):
        raise ValueError("stop_strings must be nonempty strings")
    if compressed_block is None:
        compressed_block = COMPRESSED_BLOCK_VERIFIED
    mode = "block" if cfg.spec and (cfg.kv == "full" or compressed_block) else "sequential"
    result = {"token_ids": [], "accepted_tokens": 0, "proposed_tokens": 0,
              "steps": 0, "target_forwards": 0, "target_forward_passes": 0,
              "target_prefill_forwards": 0, "target_decode_forwards": 0,
              "draft_forwards": 0, "draft_prefill_forwards": 0,
              "cache_storage_bytes": 0, "verification_mode": mode,
              "target_acceleration_enabled": mode == "block",
              "block_rounds": 0, "sequential_rounds": 0,
              "stopped_at_eos": False, "stopped_at_stop_string": False}
    if max_new_tokens == 0:
        return result
    target = _State(bundle.model, cfg, bundle.device)
    target.forward(ids, prefill=True)
    draft = None
    if cfg.spec:
        # A full draft cache needs no irreversible compression or policy rollback.
        draft_cfg = cfg.model_copy(update={"kv": "full"})
        draft = _State(bundle.draft, draft_cfg, bundle.device)
        draft.forward(ids, prefill=True)
    generated = result["token_ids"]

    def emit(token):
        generated.append(token)
        result["stopped_at_eos"] = token in eos
        if stops:
            text = bundle.tokenizer.decode(generated, skip_special_tokens=True)
            result["stopped_at_stop_string"] = any(s in text for s in stops)
        return (len(generated) == max_new_tokens or result["stopped_at_eos"]
                or result["stopped_at_stop_string"])

    while len(generated) < max_new_tokens:
        if draft is None:
            token = target.next()
            if emit(token):
                break
            target.forward([token])
            continue

        proposals = []
        k = min(cfg.spec_k, max_new_tokens - len(generated))
        for _ in range(k):
            token = draft.next()
            proposals.append(token)
            draft.forward([token])
            if token in eos:
                break
        result["proposed_tokens"] += len(proposals)
        if mode == "sequential" or not target.block_is_exact(len(proposals)):
            result["sequential_rounds"] += 1
            # Verify the same k-proposal prefix, but commit target state one token at
            # a time. Compressed history is never rolled back or replayed.
            accepted = 0
            ended = False
            for proposed in proposals:
                if proposed != target.next():
                    break
                accepted += 1
                result["accepted_tokens"] += 1
                if emit(proposed):
                    ended = True
                    break
                target.forward([proposed])
            draft.rewind(len(proposals) - accepted)
            if ended:
                break
            correction = target.next()  # rejection correction, or all-accepted bonus
            if emit(correction):
                break
            target.forward([correction])
            draft.forward([correction])
            continue

        result["block_rounds"] += 1
        first = target.next()
        verified = target.forward(proposals, commit=False, mask=target.block_mask(len(proposals)))
        predictions = [first] + verified[:-1].argmax(dim=-1).tolist()
        accepted = 0
        ended = False
        for proposed, predicted in zip(proposals, predictions, strict=True):
            if proposed != predicted:
                break
            accepted += 1
            result["accepted_tokens"] += 1
            if emit(proposed):
                ended = True
                break
        # Keep only the verified, emitted prefix. Draft and target roll back together.
        target.rewind(len(proposals) - accepted)
        draft.rewind(len(proposals) - accepted)
        target.commit()
        if ended:
            break
        correction = (predictions[accepted] if accepted < len(proposals)
                      else int(verified[-1].argmax()))
        if emit(correction):
            break
        # Materialise the correction/bonus in both caches before drafting the next block.
        target.forward([correction])
        draft.forward([correction])

    result.update(steps=target.forwards, target_forwards=target.forwards,
                  target_forward_passes=target.forwards, target_prefill_forwards=1,
                  target_decode_forwards=target.forwards - 1,
                  draft_forwards=draft.forwards if draft else 0,
                  draft_prefill_forwards=int(draft is not None),
                  cache_storage_bytes=target.peak_bytes)
    return result
