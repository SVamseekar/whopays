"""Per-sentence negative log-likelihood on FLORES-200 devtest (FR-6, feeds FR-12 profiling).

Each sentence is scored after one prefix token (the model's BOS, which for Qwen is
<|endoftext|>), so every sentence token, including the first, is predicted. Rows keep the NLL
*sum* with token, character and UTF-8 byte counts, so analysis can report per-token NLL (the
quantity profiled) or bits per byte (comparable across languages whose tokenizers differ).
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence

import torch

from whopays.config import RunConfig
from whopays.tasks import register


def prefix_token(bundle) -> int:
    bos = bundle.model.config.bos_token_id
    return bos if bos is not None else bundle.tokenizer.eos_token_id


@torch.no_grad()
def nll_batch(bundle, texts: Sequence[str]) -> list[tuple[float, int]]:
    """(sum of token NLL in nats, number of scored tokens) per text, right-padded batch."""
    tok, model, dev = bundle.tokenizer, bundle.model, bundle.device
    prefix = prefix_token(bundle)
    seqs = [[prefix, *tok(t, add_special_tokens=False)["input_ids"]] for t in texts]
    width = max(map(len, seqs))
    pad = tok.pad_token_id if tok.pad_token_id is not None else prefix
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    ids, mask = ids.to(dev), mask.to(dev)
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    targets = ids[:, 1:]
    nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    nll = nll * mask[:, 1:]
    return [(float(nll[i].sum()), len(s) - 1) for i, s in enumerate(seqs)]


@torch.no_grad()
def nll_batch_low_memory(bundle, texts: Sequence[str]) -> list[tuple[float, int]]:
    """Same quantity as `nll_batch`, without the (batch, length, vocab) float32 logits.

    `nll_batch` at batch 32 asks for about 4.7 GB for the float32 logits alone (Qwen2.5's
    151,936-token vocabulary), which ran a T4 out of memory in the K2 probe (2026-11-12). Here
    the decoder runs once on the batch and the vocabulary projection plus cross-entropy run
    one sequence at a time. Used by the sensitivity probe only; FLORES task runs keep
    `nll_batch`, so logged results are unchanged.
    """
    tok, model, dev = bundle.tokenizer, bundle.model, bundle.device
    prefix = prefix_token(bundle)
    seqs = [[prefix, *tok(t, add_special_tokens=False)["input_ids"]] for t in texts]
    width = max(map(len, seqs))
    pad = tok.pad_token_id if tok.pad_token_id is not None else prefix
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    ids, mask = ids.to(dev), mask.to(dev)
    hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    out = []
    for i, s in enumerate(seqs):
        n = len(s)
        logits = model.lm_head(hidden[i, : n - 1]).float()
        nll = torch.nn.functional.cross_entropy(logits, ids[i, 1:n], reduction="sum")
        out.append((float(nll), n - 1))
    return out


def score_items(cfg: RunConfig, bundle, items: Sequence[dict], skip: set[str]) -> Iterator[dict]:
    if cfg.kv != "full" or cfg.spec:
        raise ValueError("flores_nll measures weights only; use kv='full' and spec=False")
    # Keep original padding and batch composition when resuming a partial batch.
    for start in range(0, len(items), cfg.batch_size):
        batch = items[start : start + cfg.batch_size]
        if all(it["item_id"] in skip for it in batch):
            continue
        if bundle.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        scores = nll_batch(bundle, [it["text"] for it in batch])
        if bundle.device == "cuda":
            torch.cuda.synchronize()
        per_item = (time.perf_counter() - t0) / len(batch)
        for item, (nll_sum, n_tokens) in zip(batch, scores, strict=True):
            if item["item_id"] in skip:
                continue
            yield {
                "item_id": item["item_id"],
                "nll_sum": round(nll_sum, 5),
                "n_tokens": n_tokens,
                "nll_per_token": round(nll_sum / n_tokens, 6),
                "n_chars": len(item["text"]),
                "n_bytes": len(item["text"].encode("utf-8")),
                "batch_size": cfg.batch_size,
                "wall_time_s": round(per_item, 4),
            }


@register("flores_nll")
def run(cfg: RunConfig, bundle, skip: set[str]) -> Iterator[dict]:
    from whopays.datasets import flores

    items = flores(cfg.lang, "devtest")
    yield from score_items(cfg, bundle, items[: cfg.limit] if cfg.limit else items, skip)
