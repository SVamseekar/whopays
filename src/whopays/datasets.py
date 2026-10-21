"""Benchmark files at pinned Hub revisions. Every task reads its data through here.

Revisions are commit SHAs recorded on 2026-10-21. Changing one changes every number derived
from that dataset, so bump SCHEMA_VERSION in config.py and log it in DECISIONS.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from whopays.env import get_secret
from whopays.languages import LANGUAGES

REVISIONS = {
    "facebook/belebele": "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
    "juletxara/mgsm": "b2f13d426afe3be8d69a7e739b36724db8b66bbc",
    "facebook/flores": "71abf77d8b7beb5cfef59898d6b24d92ab7654fc",  # gated: accept terms on HF
}


def hub_file(repo: str, path: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, path, repo_type="dataset", revision=REVISIONS[repo],
                                token=get_secret("HF_TOKEN")))


def belebele_item_id(link: str, question_number: int) -> str:
    """Language-independent ID: the same question has the same ID in all 11 languages."""
    return f"{hashlib.sha1(link.encode()).hexdigest()[:10]}-q{question_number}"


def belebele(lang: str) -> list[dict]:
    """900 items as {item_id, passage, question, options, gold}, sorted by item_id.

    The per-language files list the same items in different orders; sorting makes `limit`
    select the same parallel items in every language.
    """
    path = hub_file("facebook/belebele", f"data/{LANGUAGES[lang].flores}.jsonl")
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            items.append({
                "item_id": belebele_item_id(r["link"], r["question_number"]),
                "passage": r["flores_passage"],
                "question": r["question"],
                "options": [r[f"mc_answer{i}"] for i in range(1, 5)],
                "gold": int(r["correct_answer_num"]) - 1,
            })
    return sorted(items, key=lambda it: it["item_id"])


def mgsm(lang: str) -> list[dict]:
    """250 problems as {item_id, question, gold}; MGSM rows are parallel by position."""
    import pandas as pd

    df = pd.read_parquet(hub_file("juletxara/mgsm", f"{lang}/test-00000-of-00001.parquet"))
    return [{"item_id": f"mgsm-{i:03d}", "question": q, "gold": int(a)}
            for i, (q, a) in enumerate(zip(df["question"], df["answer_number"], strict=True))]


def flores(lang: str, split: str) -> list[dict]:
    """FLORES-200 sentences as {item_id, text}; rows are parallel by position.

    `dev` is calibration-only and `devtest` is scoring-only (PRD §Datasets); callers pick.
    facebook/flores is frozen upstream (successor: openlanguagedata/flores_plus). It is kept
    because Belebele passages come from FLORES-200, so both benchmarks share one text version.
    """
    import pandas as pd

    if split not in ("dev", "devtest"):
        raise ValueError(f"split must be dev or devtest, not {split!r}")
    code = LANGUAGES[lang].flores
    df = pd.read_parquet(hub_file("facebook/flores",
                                  f"data/language/{code}/{split}-00000-of-00001.parquet"))
    column = next((c for c in ("sentence", "text") if c in df.columns), None)
    if column is None:
        raise KeyError(f"no sentence column in FLORES {code}/{split}: {list(df.columns)}")
    return [{"item_id": f"flores-{split}-{i:04d}", "text": t} for i, t in enumerate(df[column])]
