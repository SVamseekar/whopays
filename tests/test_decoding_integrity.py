"""Offline decoder checks with planted token schedules and real Qwen cache updates."""

import torch

from whopays.config import RunConfig
from whopays.decoding.speculative import generate
from whopays.models.loader import load
from whopays.tasks import mgsm


def scheduled(model, fn, monkeypatch):
    original = model.forward
    calls = []

    def forward(*args, **kwargs):
        calls.append(kwargs["input_ids"].shape[-1])
        out = original(*args, **kwargs)
        logits = torch.full_like(out.logits, -100)
        for i, position in enumerate(kwargs["position_ids"][0].tolist()):
            logits[:, i, fn(position)] = 100
        out.logits = logits
        return out

    monkeypatch.setattr(model, "forward", forward)
    return calls


def config(path, **kwargs):
    return RunConfig(model=str(path), task="mgsm", lang="en", spec=True,
                     draft_model=str(path), **kwargs)


def test_block_rejections_match_greedy_and_count_actual_forwards(tiny_model_dir, monkeypatch):
    cfg = config(tiny_model_dir)
    bundle = load(cfg, "cpu")
    target_calls = scheduled(bundle.model, lambda p: 3 + p % 5, monkeypatch)
    draft_calls = scheduled(bundle.draft, lambda p: 3 + (p + (p % 4 > 1)) % 5, monkeypatch)
    plain = generate(bundle, cfg.model_copy(update={"spec": False}), [1, 2, 3, 4], 17)
    target_calls.clear()
    actual = generate(bundle, cfg, [1, 2, 3, 4], 17)
    assert actual["token_ids"] == plain["token_ids"]
    assert 0 < actual["accepted_tokens"] < actual["proposed_tokens"]
    assert actual["steps"] == actual["target_forwards"] == len(target_calls)
    assert actual["draft_forwards"] == len(draft_calls)
    assert actual["target_prefill_forwards"] == 1
    assert actual["verification_mode"] == "block"
    assert max(target_calls[1:]) > 1


def test_eos_and_limit_truncate_accepted_block(tiny_model_dir, monkeypatch):
    cfg = config(tiny_model_dir)
    bundle = load(cfg, "cpu")
    for model in (bundle.model, bundle.draft):
        scheduled(model, lambda p: 0 if p == 5 else 3, monkeypatch)
    result = generate(bundle, cfg, [1, 2, 3, 4], 10)
    assert result["token_ids"] == [3, 3, 0]
    assert result["stopped_at_eos"]
    limited = generate(bundle, cfg, [1, 2, 3, 4], 2)
    assert limited["token_ids"] == [3, 3]
    assert not limited["stopped_at_eos"]
    empty = generate(bundle, cfg, [1, 2, 3, 4], 0)
    assert empty["token_ids"] == []
    assert empty["steps"] == 0


def test_stop_string_checked_after_each_accepted_token(tiny_model_dir, monkeypatch):
    cfg = config(tiny_model_dir)
    bundle = load(cfg, "cpu")
    token = bundle.tokenizer.encode("A", add_special_tokens=False)[0]
    for model in (bundle.model, bundle.draft):
        scheduled(model, lambda p: token, monkeypatch)
    text = bundle.tokenizer.decode([token], skip_special_tokens=True)
    result = generate(bundle, cfg, [1, 2], 10, stop_strings=[text])
    assert result["token_ids"] == [token]
    assert result["stopped_at_stop_string"]


def test_evict_uses_absolute_positions_and_sequential_verification(tiny_model_dir, monkeypatch):
    cfg = config(tiny_model_dir, kv="evict", kv_budget=5)
    bundle = load(cfg, "cpu")
    calls = scheduled(bundle.model, lambda p: 3 + p % 5, monkeypatch)
    scheduled(bundle.draft, lambda p: 9, monkeypatch)
    plain = generate(bundle, cfg.model_copy(update={"spec": False}), list(range(1, 9)), 12)
    calls.clear()
    actual = generate(bundle, cfg, list(range(1, 9)), 12)
    assert actual["token_ids"] == plain["token_ids"] == [3 + p % 5 for p in range(7, 19)]
    assert actual["verification_mode"] == "sequential"
    assert actual["target_forwards"] == len(calls) == 12
    assert all(n == 1 for n in calls[1:])


def test_mgsm_preserves_baseline_dispatch_and_logs_spec_counters(tiny_model_dir, monkeypatch):
    cfg = config(tiny_model_dir, batch_size=2)
    bundle = load(cfg, "cpu")
    monkeypatch.setattr(mgsm, "MAX_NEW_TOKENS", 3)
    original = bundle.model.generate
    calls = []

    def baseline(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(bundle.model, "generate", baseline)
    mgsm.generate_batch(bundle, cfg.model_copy(update={"spec": False}), ["A", "B"])
    assert len(calls) == 1
    assert calls[0]["repetition_penalty"] == 1.0
    assert calls[0]["stop_strings"] == ["Question:"]
    rows = list(mgsm.score_items(cfg, bundle, [
        {"item_id": "a", "question": "A", "gold": 1},
        {"item_id": "b", "question": "B", "gold": 2},
    ], set()))
    assert len(calls) == 1
    for row in rows:
        assert row["batch_size"] == 1
        assert row["requested_batch_size"] == 2
        assert row["effective_batch_size"] == 1
        assert row["steps"] == row["target_forwards"] > 0
        assert row["cache_storage_bytes"] > 0
        assert row["verification_mode"] == "block"


def test_identical_draft_reduces_target_calls_without_exceeding_limit(tiny_model_dir, monkeypatch):
    cfg = config(tiny_model_dir)
    bundle = load(cfg, "cpu")
    for model in (bundle.model, bundle.draft):
        scheduled(model, lambda p: 3, monkeypatch)
    plain = generate(bundle, cfg.model_copy(update={"spec": False}), [1, 2], 19)
    actual = generate(bundle, cfg, [1, 2], 19)
    assert actual["token_ids"] == plain["token_ids"] == [3] * 19
    assert actual["target_forwards"] < plain["target_forwards"]
    assert actual["accepted_tokens"] == actual["proposed_tokens"]


def test_eos_correction_on_first_rejected_proposal(tiny_model_dir, monkeypatch):
    cfg = config(tiny_model_dir)
    bundle = load(cfg, "cpu")
    scheduled(bundle.model, lambda p: 0, monkeypatch)
    scheduled(bundle.draft, lambda p: 3, monkeypatch)
    actual = generate(bundle, cfg, [1, 2], 9)
    assert actual["token_ids"] == [0]
    assert actual["accepted_tokens"] == 0
    assert actual["proposed_tokens"] == cfg.spec_k
    assert actual["stopped_at_eos"]


def test_quantized_flush_uses_exact_sequential_reference(tiny_model_dir, monkeypatch):
    from transformers.cache_utils import Cache, QuantizedLayer

    from whopays.decoding import speculative

    class CPUQuantizedLayer(QuantizedLayer):
        def _quantize(self, tensor, axis):
            return tensor.clone()

        def _dequantize(self, tensor):
            return tensor

    original = speculative.new_cache
    compressed = []

    def factory(cfg, model):
        if cfg.kv != "quant4":
            return original(cfg, model)
        cache = Cache(layers=[CPUQuantizedLayer(residual_length=4)
                              for _ in range(model.config.num_hidden_layers)])
        compressed.append(cache)
        return cache

    monkeypatch.setattr(speculative, "new_cache", factory)
    cfg = config(tiny_model_dir, kv="quant4")
    bundle = load(cfg, "cpu")
    with torch.no_grad():
        bundle.draft.lm_head.weight.mul_(-1)
    plain = generate(bundle, cfg.model_copy(update={"spec": False}), [1, 2, 3, 4, 5], 19)
    actual = generate(bundle, cfg, [1, 2, 3, 4, 5], 19)
    assert actual["token_ids"] == plain["token_ids"]
    assert actual["target_forwards"] == plain["target_forwards"] == 19
    assert actual["verification_mode"] == "sequential"
    assert not actual["target_acceleration_enabled"]
    assert compressed[-1].layers[0]._quantized_keys.shape[-2] > 5


def test_sequential_verification_counts_rejected_draft_block_suffix(tiny_model_dir, monkeypatch):
    cfg = config(tiny_model_dir, kv="evict", kv_budget=5)
    bundle = load(cfg, "cpu")
    scheduled(bundle.model, lambda p: 3, monkeypatch)
    scheduled(bundle.draft, lambda p: 3 if p % 3 == 1 else 4, monkeypatch)
    actual = generate(bundle, cfg, [1, 2], 7)
    assert actual["token_ids"] == [3] * 7
    # Five blocks accept 1,0,1,0,1; the discarded suffix still counts as proposed.
    assert actual["proposed_tokens"] == 4 + 4 + 4 + 2 + 1
    assert actual["accepted_tokens"] == 3
    assert actual["target_forwards"] == 7
    assert not actual["target_acceleration_enabled"]


def test_run_retains_effective_batch_size(tiny_model_dir, tmp_path, monkeypatch):
    import json

    from whopays import datasets, run

    cfg = config(tiny_model_dir, batch_size=2)
    bundle = load(cfg, "cpu")
    monkeypatch.setattr(mgsm, "MAX_NEW_TOKENS", 3)
    monkeypatch.setattr(datasets, "mgsm", lambda lang: [
        {"item_id": "a", "question": "A", "gold": 1},
        {"item_id": "b", "question": "B", "gold": 2},
    ])
    [result] = run([cfg], root=tmp_path, loader=lambda c: bundle, sync=False,
                   stop_on_error=True)
    assert result.status == "done"
    rows = [json.loads(line) for line in (tmp_path / cfg.run_id / "items.jsonl")
            .read_text().splitlines()]
    for row in rows:
        assert row["batch_size"] == row["requested_batch_size"] == 2
        assert row["effective_batch_size"] == 1
        assert row["steps"] == row["target_forwards"]
