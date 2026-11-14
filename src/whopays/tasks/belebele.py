"""Belebele reading comprehension: answer-letter scoring (FR-6).

The prompt is lm-evaluation-harness's zero-shot template, so FP16 numbers are comparable at
the K1 gate. lm-eval scores the log-likelihood of " A".." D" after "Answer:", which for
single-token letters equals the next-token log-probabilities used here.

Two-pass scoring (DECISIONS 2026-10-21): the context "P: {passage}\\n" is prefilled into a
cache, the KV policy is applied, and the query "Q: ...\\nAnswer:" runs against that cache.
Both segments come from one tokenization of the full prompt, so the token IDs are exactly the
single-pass ones.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Iterator, Sequence

import numpy as np
import torch

from whopays.config import RunConfig
from whopays.kv.policies import after_prefill, new_cache
from whopays.tasks import register

LETTERS = ("A", "B", "C", "D")


def build_prompt(item: dict) -> tuple[str, str]:
    a, b, c, d = item["options"]
    context = f"P: {item['passage']}\n"
    query = f"Q: {item['question'].strip()}\nA: {a}\nB: {b}\nC: {c}\nD: {d}\nAnswer:"
    return context, query


def split_tokens(tokenizer, context: str, query: str) -> tuple[list[int], list[int]]:
    """Tokenize context+query once and split at the character boundary between them."""
    enc = tokenizer(context + query, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    cut = len(context)
    n_ctx = next((i for i, (s, e) in enumerate(offsets) if e > s and s >= cut), len(ids))
    if any(e > cut for _, e in offsets[:n_ctx]):
        raise ValueError("a token straddles the context/query boundary")
    return ids[:n_ctx], ids[n_ctx:]


def letter_token_ids(tokenizer) -> list[int]:
    ids = [tokenizer(" " + letter, add_special_tokens=False)["input_ids"] for letter in LETTERS]
    if any(len(i) != 1 for i in ids):
        raise ValueError(f"answer letters are not single tokens: {ids}")
    return [i[0] for i in ids]


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# Questions that share a passage share its prefilled cache. The context forward pass and the
# policy depend only on the context tokens, so reusing a copy gives exactly the cache a fresh
# prefill would (tests check bit identity); it saves 31-33% of Belebele tokens (900 questions
# over 488 passages per language, measured with the Qwen2.5 tokenizer, DECISIONS 2026-11-12).
# This is automatic prefix caching (vLLM docs, design/prefix_caching). The 4-bit KIVI cache is
# plain fake-quantized tensors with deterministic grouping, so it is reused too.
REUSABLE_KV = {"full", "evict", "evict_share", "quant4"}

# Exploratory retention-matched eviction (seventh amendment): each item keeps the share of its
# own context that `kv_budget` tokens is of the parallel English context, the "+RATIO" rule of
# "Equal Tokens, Unequal Losses" (anonymous ACL submission 2026, OpenReview 57mHZnnuEe).
MIN_SHARE_BUDGET = 8  # 4 sink tokens plus at least 4 recent ones


def share_budget(kv_budget: int, *, n_lang: int, n_en: int) -> int:
    """Tokens to keep for one item: kv_budget × (its context tokens / English's)."""
    return max(MIN_SHARE_BUDGET, round(kv_budget * n_lang / n_en))


def english_share_budgets(tokenizer, kv_budget: int, items: Sequence[dict]) -> dict[str, int]:
    """Per-item budgets for `items`, from the parallel English items with the same IDs."""
    from whopays.datasets import belebele

    english = {it["item_id"]: len(split_tokens(tokenizer, *build_prompt(it))[0])
               for it in belebele("en")}
    return {it["item_id"]: share_budget(
        kv_budget, n_lang=len(split_tokens(tokenizer, *build_prompt(it))[0]),
        n_en=english[it["item_id"]]) for it in items}


@torch.no_grad()
def prefill(bundle, cfg: RunConfig, ctx_ids: list[int], budget: int | None = None):
    """Context pass plus KV policy; returns (cache, context positions kept)."""
    cache = new_cache(cfg, bundle.model)
    bundle.model(torch.tensor([ctx_ids], device=bundle.device), past_key_values=cache,
                 use_cache=True)
    return cache, after_prefill(cache, cfg, budget)


@torch.no_grad()
def query(bundle, cache, n_ctx: int, q_ids: list[int], letters: list[int]) -> list[float]:
    dev = bundle.device
    # Original positions continue after the full context even if the policy evicted some.
    pos = torch.arange(n_ctx, n_ctx + len(q_ids), device=dev)[None]
    out = bundle.model(torch.tensor([q_ids], device=dev), past_key_values=cache,
                       position_ids=pos, use_cache=True)
    return out.logits[0, -1].float().log_softmax(-1)[letters].tolist()


@torch.no_grad()
def score_item(bundle, cfg: RunConfig, ctx_ids: list[int], q_ids: list[int],
               letters: list[int]) -> tuple[list[float], int]:
    """Letter log-probabilities (over the full vocabulary) and context positions kept."""
    cache, kept = prefill(bundle, cfg, ctx_ids)
    return query(bundle, cache, len(ctx_ids), q_ids, letters), kept


def score_items(cfg: RunConfig, bundle, items: Sequence[dict], skip: set[str], *,
                budgets: dict[str, int] | None = None) -> Iterator[dict]:
    if cfg.kv == "evict_share" and budgets is None:
        raise ValueError("kv='evict_share' needs per-item budgets")
    letters = letter_token_ids(bundle.tokenizer)
    shared_ctx, shared_cache, shared_kept = None, None, 0
    for item in items:
        if item["item_id"] in skip:
            continue
        ctx_ids, q_ids = split_tokens(bundle.tokenizer, *build_prompt(item))
        _sync(bundle.device)
        start = time.perf_counter()
        budget = budgets[item["item_id"]] if budgets is not None else None
        reused = cfg.kv in REUSABLE_KV and ctx_ids == shared_ctx
        if not reused:
            cache, kept = prefill(bundle, cfg, ctx_ids, budget)
            if cfg.kv in REUSABLE_KV:
                shared_ctx, shared_cache, shared_kept = ctx_ids, copy.deepcopy(cache), kept
        else:
            cache, kept = copy.deepcopy(shared_cache), shared_kept
        lp = query(bundle, cache, len(ctx_ids), q_ids, letters)
        _sync(bundle.device)
        pred = int(np.argmax(lp))
        yield {
            "item_id": item["item_id"],
            "gold": item["gold"],
            "pred": pred,
            "correct": pred == item["gold"],
            "gold_logprob": round(lp[item["gold"]], 6),
            "letter_logprobs": [round(x, 6) for x in lp],
            "context_tokens": len(ctx_ids),
            "query_tokens": len(q_ids),
            "input_tokens": len(ctx_ids) + len(q_ids),
            "kept_context_tokens": kept,
            **({"kv_item_budget": budget} if budget is not None else {}),
            "prefill_reused": reused,
            "wall_time_s": round(time.perf_counter() - start, 4),
        }


@register("belebele")
def run(cfg: RunConfig, bundle, skip: set[str]) -> Iterator[dict]:
    from whopays.datasets import belebele

    items = belebele(cfg.lang)
    items = items[: cfg.limit] if cfg.limit else items
    budgets = (english_share_budgets(bundle.tokenizer, cfg.kv_budget, items)
               if cfg.kv == "evict_share" else None)
    yield from score_items(cfg, bundle, items, skip, budgets=budgets)


def calibrated_accuracy(letter_logprobs: Sequence[Sequence[float]], gold: Sequence[int]) -> float:
    """Accuracy after removing the run's average preference for each letter.

    Subtracting each letter's mean log-probability over the run's items cancels a constant
    A/B/C/D prior (the PRD's letter-bias risk) without extra forward passes.
    """
    lp = np.asarray(letter_logprobs, dtype=float)
    pred = (lp - lp.mean(axis=0, keepdims=True)).argmax(axis=1)
    return float((pred == np.asarray(gold)).mean())
