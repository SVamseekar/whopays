"""Small-model evaluation helpers for local ParityQuant v2 exploration (never results tables).

Used by scripts/m1_langgptq.py on Qwen2.5-0.5B: score a model in memory on held-out FLORES
devtest NLL and a Belebele subset, compare with an FP reference, and build the label-free
answer-letter loss used for a task-aware output Fisher.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from whopays.tasks.belebele import build_prompt, letter_token_ids, split_tokens
from whopays.tasks.flores_nll import nll_batch


def band_bitmap(n_blocks: int, base: int, band: Sequence[int], band_bits: int) -> dict:
    """Every block at `base` bits except the blocks in `band`."""
    return {f"model.layers.{i}": (band_bits if i in band else base) for i in range(n_blocks)}


def belebele_batches(tokenizer, items: Sequence[dict]) -> list[dict]:
    """One batch per item: the full prompt ids, for the answer-letter Fisher."""
    out = []
    for item in items:
        ctx, q = split_tokens(tokenizer, *build_prompt(item))
        out.append({"input_ids": torch.tensor([ctx + q])})
    return out


def letter_fisher_loss(tokenizer, seed: int = 0):
    """Label-free loss whose squared gradients estimate the Fisher of the answer-letter choice.

    The letter is sampled from the model's own distribution over A–D (a one-sample estimate of
    the true Fisher), so no gold labels are used.
    """
    letters = letter_token_ids(tokenizer)
    gen = torch.Generator().manual_seed(seed)

    def loss(model, batch):
        ids = batch["input_ids"].to(next(model.parameters()).device)
        logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
        logp = logits[letters].log_softmax(-1)
        k = int(torch.multinomial(logp.detach().exp().cpu(), 1, generator=gen))
        return -logp[k]

    return loss


@torch.no_grad()
def score(bundle, flores_items: dict, belebele_items: dict, batch_size: int = 16) -> dict:
    """Per language: FLORES NLL sums per sentence and Belebele letter log-probabilities."""
    letters = letter_token_ids(bundle.tokenizer)
    out = {}
    for lang in flores_items:
        texts = [it["text"] for it in flores_items[lang]]
        nll = []
        for i in range(0, len(texts), batch_size):
            nll += [s for s, _ in nll_batch(bundle, texts[i:i + batch_size])]
        lps, gold = [], []
        for item in belebele_items[lang]:
            ctx, q = split_tokens(bundle.tokenizer, *build_prompt(item))
            ids = torch.tensor([ctx + q], device=bundle.device)
            logits = bundle.model(input_ids=ids, use_cache=False).logits[0, -1].float()
            lps.append(logits.log_softmax(-1)[letters].cpu().numpy())
            gold.append(item["gold"])
        out[lang] = {"nll": np.array(nll), "letter_lp": np.stack(lps), "gold": np.array(gold)}
    return out


def compare(scores: dict, reference: dict) -> dict:
    """Per language: NLL increase per sentence, raw and calibrated accuracy, letter KL."""
    from whopays.tasks.belebele import calibrated_accuracy

    out = {}
    for lang, s in scores.items():
        ref = reference[lang]
        lp4 = s["letter_lp"] - np.logaddexp.reduce(s["letter_lp"], axis=1, keepdims=True)
        ref4 = ref["letter_lp"] - np.logaddexp.reduce(ref["letter_lp"], axis=1, keepdims=True)
        out[lang] = {
            "nll_increase": float((s["nll"] - ref["nll"]).mean()),
            "raw": 100 * float((s["letter_lp"].argmax(1) == s["gold"]).mean()),
            "calibrated": 100 * calibrated_accuracy(s["letter_lp"], s["gold"]),
            "letter_kl": float((np.exp(ref4) * (ref4 - lp4)).sum(1).mean()),
        }
    return out
