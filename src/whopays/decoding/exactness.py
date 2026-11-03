"""FR-5 gate: speculative output must equal plain greedy, token for token.

Generation runs past EOS so every prompt reaches `max_new_tokens`, which forces eviction and
quantized residual flushes. A divergence is a harness bug, never a finding.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from whopays.config import RunConfig
from whopays.decoding.speculative import generate
from whopays.languages import CODES

POLICIES = ("full", "quant4", "evict")


def report(bundle, model: str, prompts: dict[str, list[list[int]]], *, max_new_tokens: int = 256,
           policies: Sequence[str] = POLICIES, kv_budget: int = 256,
           compressed_block: bool = True) -> pd.DataFrame:
    """Per policy × language: prompts compared, exact outputs, first divergence, target calls."""
    rows = []
    for kv in policies:
        for lang, group in prompts.items():
            cfg = RunConfig(model=model, task="mgsm", lang=lang, kv=kv, kv_budget=kv_budget,
                            spec=True)
            plain_cfg = cfg.model_copy(update={"spec": False})
            exact, first, plain_calls, spec_calls = 0, None, 0, 0
            for ids in group:
                plain = generate(bundle, plain_cfg, ids, max_new_tokens=max_new_tokens,
                                 eos_token_id=[])
                spec = generate(bundle, cfg, ids, max_new_tokens=max_new_tokens, eos_token_id=[],
                                compressed_block=compressed_block)
                plain_calls += plain["target_forwards"]
                spec_calls += spec["target_forwards"]
                if plain["token_ids"] == spec["token_ids"]:
                    exact += 1
                elif first is None:
                    pairs = zip(plain["token_ids"], spec["token_ids"], strict=False)
                    first = next((i for i, (a, b) in enumerate(pairs) if a != b),
                                 min(len(plain["token_ids"]), len(spec["token_ids"])))
            rows.append({"kv": kv, "lang": lang, "n_prompts": len(group), "exact": exact,
                         "first_divergence": first, "plain_target_forwards": plain_calls,
                         "spec_target_forwards": spec_calls})
    return pd.DataFrame(rows)


def verdict(table: pd.DataFrame, *, min_prompts: int = 100,
            languages: Sequence[str] | None = None) -> str:
    """FAIL on any divergence; INCOMPLETE below `min_prompts` per policy or a missing language."""
    if (table.exact < table.n_prompts).any():
        return "FAIL"
    covered = all(set(g.lang) >= set(languages or []) and g.n_prompts.sum() >= min_prompts
                  for _, g in table.groupby("kv"))
    return "PASS" if covered else "INCOMPLETE"


def mgsm_prompts(tokenizer, per_language: int = 10, languages: Sequence[str] = CODES):
    """Fixed first MGSM prompts per language, as token IDs."""
    from whopays.datasets import mgsm
    from whopays.tasks.mgsm import build_prompt

    return {lang: [tokenizer.encode(build_prompt(lang, item["question"]), add_special_tokens=True)
                   for item in mgsm(lang)[:per_language]] for lang in languages}


def run_gate(model: str, *, per_language: int = 10, root=None, sync: bool = True, loader=None):
    """Run the T4 gate over 11 languages and store its table in the results dataset."""
    from whopays.env import env_info, results_root
    from whopays.io import hub_sync
    from whopays.models.loader import load

    root = Path(root) if root is not None else results_root()
    folder = root / f"exactness-{model.split('/')[-1]}"
    if sync and not folder.exists():
        hub_sync.pull_runs(root, [folder.name])
    stored = folder / "verdict.json"
    if stored.exists():
        record = json.loads(stored.read_text())
        if record["verdict"] == "PASS" and record["code_version"] == env_info()["code_version"]:
            return pd.read_json(folder / "table.jsonl", lines=True), "PASS"
    cfg = RunConfig(model=model, task="mgsm", lang="en", spec=True)
    bundle = (loader or load)(cfg)
    table = report(bundle, model, mgsm_prompts(bundle.tokenizer, per_language))
    state = verdict(table, min_prompts=100, languages=CODES)
    folder.mkdir(parents=True, exist_ok=True)
    table.to_json(folder / "table.jsonl", orient="records", lines=True)
    (folder / "verdict.json").write_text(json.dumps({"verdict": state, **env_info()}) + "\n")
    if sync:
        hub_sync.push_run(folder)
    return table, state
