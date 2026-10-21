"""MGSM: greedy generation with final-number extraction (FR-6).

The prompt is lm-evaluation-harness's zero-shot `mgsm_native_cot`: each language is asked in
its own language ("ప్రశ్న: {q}\\nదశలవారీగా సమాధానం:"), generation stops at the next question
marker or EOS, and at most 256 tokens are generated.

Scoring takes the *last* number in the output, like lm-eval's flexible-extract, but also
reads native-script digits (Bengali, Telugu, Thai, ...) and thousands separators. ASCII-only
extraction would turn a model answering "১৮" into a fake Bengali quality gap (PRD risk table).
`correct_ascii` keeps the ASCII-only reading for the K1 comparison with lm-eval, and
`extraction_failed` flags outputs with no number at all, reported per language.

Decoding is pure greedy: Qwen's generation_config sets repetition_penalty=1.1, which is
overridden to 1.0 (speculative exactness needs plain argmax). An lm-eval cross-check must pass
`--gen_kwargs repetition_penalty=1.0` to match.

Batches use left padding. In FP16, padding and batch composition can move near-tied greedy
choices, so `batch_size` is logged per row.
"""

from __future__ import annotations

import re
import string
import time
import unicodedata
from collections.abc import Iterator, Sequence

import torch

from whopays.config import RunConfig
from whopays.tasks import register

MAX_NEW_TOKENS = 256

# lm-eval mgsm utils.py: question marker and "step-by-step answer" cue per language.
PROMPTS = {
    "bn": ("প্রশ্ন:", "ধাপে ধাপে উত্তর:"),
    "de": ("Frage:", "Schritt-für-Schritt-Antwort:"),
    "en": ("Question:", "Step-by-Step Answer:"),
    "es": ("Pregunta:", "Respuesta paso a paso:"),
    "fr": ("Question :", "Réponse étape par étape :"),
    "ru": ("Задача:", "Пошаговоерешение:"),
    "sw": ("Swali:", "Jibu la Hatua kwa Hatua:"),
    "te": ("ప్రశ్న:", "దశలవారీగా సమాధానం:"),
    "th": ("โจทย์:", "คำตอบทีละขั้นตอน:"),
    "ja": ("問題：", "ステップごとの答え:"),
    "zh": ("问题：", "逐步解答:"),
}

# Any Unicode decimal digit, optionally signed, with , or . groups: "1,234.5", "১৮", "-3".
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*")
_ASCII_NUMBER = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")  # lm-eval flexible-extract


def build_prompt(lang: str, question: str) -> str:
    q_marker, answer_cue = PROMPTS[lang]
    return f"{q_marker} {question}\n{answer_cue}"


def _to_number(token: str) -> float | None:
    s = "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in token).rstrip(".,")
    if "," in s and "." in s:  # the later separator is the decimal mark: 1,234.5 / 1.234,5
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        s = s.replace("." if dec == "," else ",", "").replace(dec, ".")
    elif "," in s or "." in s:
        sep = "," if "," in s else "."
        groups = s.split(sep)
        lead = groups[0].lstrip("-")
        thousands = (len(groups) > 2 or len(groups[-1]) == 3) and lead not in ("", "0")
        s = "".join(groups) if thousands else s.replace(sep, ".")
    try:
        return float(s)
    except ValueError:
        return None


def extract_number(text: str) -> float | None:
    """Last number in `text`, any script; None when there is none."""
    for match in reversed(_NUMBER.findall(text)):
        value = _to_number(match)
        if value is not None:
            return value
    return None


def extract_ascii_lmeval(text: str) -> str | None:
    """lm-eval flexible-extract plus its ignore_punctuation: last ASCII match, punctuation off."""
    matches = _ASCII_NUMBER.findall(text)
    if not matches:
        return None
    last = matches[-1][0] or matches[-1][1]
    return last.translate(str.maketrans("", "", string.punctuation)) or None


def truncate_at_stop(text: str, lang: str) -> tuple[str, bool]:
    marker = PROMPTS[lang][0]
    idx = text.find(marker)
    return (text[:idx], True) if idx >= 0 else (text, False)


@torch.no_grad()
def generate_batch(bundle, cfg: RunConfig, prompts: Sequence[str]) -> list[dict]:
    if cfg.spec:
        raise NotImplementedError("speculative decoding arrives with BUILD_PLAN M2")
    if cfg.kv != "full":
        raise NotImplementedError(f"kv={cfg.kv!r} for generation arrives with BUILD_PLAN M2")
    tok, model = bundle.tokenizer, bundle.model
    tok.padding_side = "left"
    enc = tok(list(prompts), return_tensors="pt", padding=True).to(bundle.device)
    out = model.generate(
        **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
        stop_strings=[PROMPTS[cfg.lang][0]], tokenizer=tok,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        # Pure greedy: override Qwen's generation_config (repetition_penalty=1.1, sampling
        # params), which would otherwise apply even with do_sample=False.
        repetition_penalty=1.0, top_p=None, top_k=None, temperature=None,
    )
    results = []
    width = enc["input_ids"].shape[1]
    for row, mask in zip(out, enc["attention_mask"], strict=True):
        gen = row[width:].tolist()
        eos = {tok.eos_token_id, tok.pad_token_id} - {None}
        n_out = next((i + 1 for i, t in enumerate(gen) if t in eos), len(gen))
        results.append({"input_tokens": int(mask.sum()), "output_tokens": n_out,
                        "text": tok.decode(gen[:n_out], skip_special_tokens=True)})
    return results


def score_items(cfg: RunConfig, bundle, items: Sequence[dict], skip: set[str]) -> Iterator[dict]:
    todo = [it for it in items if it["item_id"] not in skip]
    for start in range(0, len(todo), cfg.batch_size):
        batch = todo[start : start + cfg.batch_size]
        if bundle.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        gens = generate_batch(bundle, cfg, [build_prompt(cfg.lang, it["question"]) for it in batch])
        if bundle.device == "cuda":
            torch.cuda.synchronize()
        per_item = (time.perf_counter() - t0) / len(batch)
        for item, gen in zip(batch, gens, strict=True):
            text, stopped = truncate_at_stop(gen["text"], cfg.lang)
            pred = extract_number(text)
            ascii_pred = extract_ascii_lmeval(text)
            yield {
                "item_id": item["item_id"],
                "gold": item["gold"],
                "pred_number": pred,
                "correct": pred is not None and abs(pred - item["gold"]) < 1e-6,
                "correct_ascii": ascii_pred == str(item["gold"]),
                "extraction_failed": pred is None,
                "output_text": text,
                "stopped_at_question": stopped,
                "hit_max_tokens": gen["output_tokens"] >= MAX_NEW_TOKENS,
                "input_tokens": gen["input_tokens"],
                "output_tokens": gen["output_tokens"],
                "batch_size": cfg.batch_size,
                "wall_time_s": round(per_item, 4),  # batch time / batch size
            }


@register("mgsm")
def run(cfg: RunConfig, bundle, skip: set[str]) -> Iterator[dict]:
    from whopays.datasets import mgsm

    items = mgsm(cfg.lang)
    yield from score_items(cfg, bundle, items[: cfg.limit] if cfg.limit else items, skip)
