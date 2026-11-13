"""Run the sensitivity profile, read it back, decide K2 and solve bit allocations.

Texts are split into two disjoint halves per language (`en#a`, `en#b`). The merged vector
feeds K2 and the allocator. The halves give a noise ceiling: a low correlation between
languages means little if one language's own two halves disagree as much.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Sequence
from itertools import combinations
from pathlib import Path
from statistics import fmean

import pandas as pd
from scipy.stats import spearmanr

from whopays.config import RunConfig, stable_hash
from whopays.env import results_root
from whopays.io import hub_sync
from whopays.languages import CODES
from whopays.parityquant.allocate import allocate
from whopays.parityquant.profile import profile

PRIMARY_BITS = 4  # K2 reads one width: the uniform baseline's, where budgets 3-4 operate
K2_THRESHOLD = 0.9
HALF = "#"


def probe_texts(limit: int = 100, languages: Iterable[str] = CODES) -> dict[str, list[str]]:
    """First `limit` FLORES devtest sentences by item ID, as two alternating halves."""
    from whopays.datasets import flores

    texts = {}
    for lang in languages:
        rows = sorted(flores(lang, "devtest"), key=lambda r: r["item_id"])[:limit]
        texts[f"{lang}{HALF}a"] = [r["text"] for r in rows[0::2]]
        texts[f"{lang}{HALF}b"] = [r["text"] for r in rows[1::2]]
    return texts


CALIB_SOURCE = "wikipedia_128x2048"  # the quantizer's calibration source (fourth amendment)


def _calibration(tokenizer, calib: str, seed: int):
    from whopays.quant.calibration import batches_for

    return batches_for(tokenizer, CALIB_SOURCE, calib, seed)[0]


def _folder(model: str, bits, method: str, limit: int, calib: str, seed: int) -> str:
    return "probe-" + stable_hash({"model": model, "bits": list(bits), "method": method,
                                   "limit": limit, "calib": calib, "seed": seed,
                                   "calib_source": CALIB_SOURCE})


def load_profile(root: Path, model: str, *, bits=(2, 3, 4, 8), method: str = "gptq",
                 limit: int = 100, calib: str = "multi", seed: int = 0) -> list[dict]:
    path = Path(root) / _folder(model, bits, method, limit, calib, seed) / "profile.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def run_probe(model: str, *, bits=(2, 3, 4, 8), method: str = "gptq", limit: int = 100,
              calib: str = "multi", seed: int = 0, batch_size: int = 32,
              root: Path | None = None, sync: bool = True, loader=None) -> list[dict]:
    """Profile one model resumably, pushing to the results dataset and printing a projection."""
    from whopays.models.loader import load

    root = Path(root) if root is not None else results_root()
    name = _folder(model, bits, method, limit, calib, seed)
    run_dir = root / name
    if sync and not run_dir.exists():
        hub_sync.pull_runs(root, [name])
    bundle = (loader or load)(RunConfig(model=model, task="flores_nll", lang="en"))
    texts = probe_texts(limit=limit)
    calibration = _calibration(bundle.tokenizer, calib, seed) if method == "gptq" else None
    push = hub_sync.ThrottledPush(run_dir, background=True) if sync else None
    start, count = time.monotonic(), 0
    per_block = len(bits) * len(texts)
    n_blocks = getattr(bundle.model.config, "num_hidden_layers", 0)

    def on_row(_row):
        nonlocal count
        count += 1
        if count % per_block == 0:
            elapsed = time.monotonic() - start
            print(f"blocks done this session: {count // per_block}/{n_blocks}; full-model "
                  f"projection {elapsed / count * per_block * n_blocks / 3600:.2f} h")
        if push:
            push()

    rows = profile(bundle, texts, bits, method=method, calibration=calibration,
                   batch_size=batch_size, output_path=run_dir / "profile.jsonl", run_id=name,
                   on_row=on_row)
    if push:
        push(force=True)
    return rows


def merge_halves(rows: Iterable[dict]) -> list[dict]:
    """One row per layer × bits × language: the mean sensitivity of its text halves."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["layer"], row["bits"], row["lang"].split(HALF)[0]), []).append(row)
    return [{**members[0], "lang": key[2],
             "sensitivity": fmean(m["sensitivity"] for m in members)}
            for key, members in groups.items()]


def _layer_order(name: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name))


def sensitivity_tensor(rows: Iterable[dict]):
    """(layers, bits, languages, tensor[layer, bit, language], parameter counts)."""
    frame = pd.DataFrame(list(rows))
    layers = sorted(frame.layer.unique(), key=_layer_order)
    bits, languages = sorted(frame.bits.unique().tolist()), sorted(frame.lang.unique())
    if frame.duplicated(["layer", "bits", "lang"]).any():
        raise ValueError("duplicate profile rows")
    if len(frame) != len(layers) * len(bits) * len(languages):
        raise ValueError("incomplete profile: every layer × bits × language is required")
    tensor = (frame.set_index(["layer", "bits", "lang"]).sensitivity
              .unstack("lang")[languages].unstack("bits").reindex(layers))
    values = tensor.to_numpy().reshape(len(layers), len(languages), len(bits)).transpose(0, 2, 1)
    params = frame.groupby("layer").parameters.agg(["min", "max"]).reindex(layers)
    if (params["min"] != params["max"]).any():
        raise ValueError("inconsistent layer parameter counts")
    return layers, bits, languages, values, params["min"].astype(int).tolist()


def _vectors(rows: Iterable[dict]) -> dict[str, pd.Series]:
    frame = pd.DataFrame([r for r in rows if r["bits"] == PRIMARY_BITS])
    if frame.empty:
        raise ValueError(f"K2 requires {PRIMARY_BITS}-bit sensitivities")
    wide = frame.pivot(index="layer", columns="lang", values="sensitivity")
    if wide.isna().any().any():
        raise ValueError("incomplete profile for K2")
    return {lang: wide[lang] for lang in wide}


def _rho(a: pd.Series, b: pd.Series) -> float:
    value = float(spearmanr(a, b).statistic)
    if value != value:
        raise ValueError("constant sensitivity vector; Spearman correlation is undefined")
    return value


def k2_verdict(rows: Sequence[dict]) -> pd.Series:
    """PRD K2 at PRIMARY_BITS: the minimum pairwise Spearman must be at most 0.9.

    `K2` is the literal PRD threshold. `noise_limited` is a separate warning: the
    Spearman-Brown-corrected split-half reliability is itself at or below the threshold, so
    a PASS cannot be attributed to real differences between languages.
    """
    rows = list(rows)
    merged = _vectors(merge_halves(rows))
    if len(merged) < 2:
        raise ValueError("K2 needs at least two languages")
    pairs = {pair: _rho(merged[pair[0]], merged[pair[1]])
             for pair in combinations(sorted(merged), 2)}
    min_pair = min(pairs, key=pairs.get)
    halves = _vectors([r for r in rows if HALF in r["lang"]]) if any(
        HALF in r["lang"] for r in rows) else {}
    within = [_rho(halves[f"{lang}{HALF}a"], halves[f"{lang}{HALF}b"]) for lang in merged
              if f"{lang}{HALF}a" in halves and f"{lang}{HALF}b" in halves]
    split_half = min(within) if within else float("nan")
    corrected = 2 * split_half / (1 + split_half) if split_half > 0 else split_half
    return pd.Series({"bits": PRIMARY_BITS, "languages": len(merged),
                      "min_spearman": pairs[min_pair], "min_pair": min_pair,
                      "split_half_min": split_half,
                      "noise_limited": bool(within) and corrected <= K2_THRESHOLD,
                      "K2": "PASS" if pairs[min_pair] <= K2_THRESHOLD else "FAIL"})


def solve(rows: Iterable[dict], budget: float, objective: str = "minimax", **kwargs):
    """Solve one allocation from a stored profile; layer names are decoder blocks."""
    layers, bits, _languages, tensor, params = sensitivity_tensor(merge_halves(rows))
    return allocate(tensor, params, budget, bits=bits, objective=objective, layer_names=layers,
                    **kwargs)
