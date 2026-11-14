import numpy as np
import pytest
import torch

from whopays import datasets, run
from whopays.config import RunConfig
from whopays.models.loader import load
from whopays.tasks import belebele as bb

ITEM = {
    "item_id": "x-q1",
    "passage": "The river town kept its records in a stone archive by the bridge.",
    "question": " Where were the records kept? ",
    "options": ["bridge", "archive", "river", "town"],
    "gold": 1,
}


def items(n=6):
    return [{**ITEM, "item_id": f"x-q{i}", "gold": i % 4} for i in range(n)]


@pytest.fixture(scope="module")
def bundle(tiny_model_dir):
    cfg = RunConfig(model=str(tiny_model_dir), task="belebele", lang="en")
    return load(cfg, device="cpu")


def cfg_for(tiny_model_dir, **kw):
    return RunConfig(model=str(tiny_model_dir), task="belebele", lang="te", **kw)


def test_prompt_matches_lm_eval_template():
    context, query = bb.build_prompt(ITEM)
    # lm-eval: "P: {{flores_passage}}\nQ: {{question.strip()}}\nA: ...\nD: ...\nAnswer:"
    assert context + query == (
        "P: The river town kept its records in a stone archive by the bridge.\n"
        "Q: Where were the records kept?\nA: bridge\nB: archive\nC: river\nD: town\nAnswer:"
    )


def test_split_tokens_reproduces_single_tokenization(bundle):
    context, query = bb.build_prompt(ITEM)
    ctx_ids, q_ids = bb.split_tokens(bundle.tokenizer, context, query)
    assert ctx_ids + q_ids == bundle.tokenizer(context + query)["input_ids"]
    assert ctx_ids and q_ids


def test_split_tokens_rejects_straddling_token(bundle):
    with pytest.raises(ValueError, match="straddles"):
        bb.split_tokens(bundle.tokenizer, "P: x\nAns", "wer:")


def test_two_pass_equals_single_pass(bundle, tiny_model_dir):
    """The KV protocol must not change scores when the policy is `full` (fp32, CPU)."""
    ctx_ids, q_ids = bb.split_tokens(bundle.tokenizer, *bb.build_prompt(ITEM))
    letters = bb.letter_token_ids(bundle.tokenizer)
    two_pass, kept = bb.score_item(bundle, cfg_for(tiny_model_dir), ctx_ids, q_ids, letters)
    with torch.no_grad():
        logits = bundle.model(torch.tensor([ctx_ids + q_ids])).logits[0, -1]
    single = logits.float().log_softmax(-1)[letters].tolist()
    assert kept == len(ctx_ids)
    np.testing.assert_allclose(two_pass, single, atol=1e-5)


def test_rows_and_skip(bundle, tiny_model_dir):
    rows = list(bb.score_items(cfg_for(tiny_model_dir), bundle, items(), skip={"x-q2"}))
    assert [r["item_id"] for r in rows] == ["x-q0", "x-q1", "x-q3", "x-q4", "x-q5"]
    r = rows[0]
    assert r["correct"] == (r["pred"] == r["gold"])
    assert r["gold_logprob"] == r["letter_logprobs"][r["gold"]]
    assert r["input_tokens"] == r["context_tokens"] + r["query_tokens"]
    assert r["kept_context_tokens"] == r["context_tokens"]
    assert all(lp <= 0 for lp in r["letter_logprobs"])


@pytest.mark.parametrize("budget", [6, 256])
def test_eviction_policy_retains_context_budget(bundle, tiny_model_dir, budget):
    [row] = bb.score_items(cfg_for(tiny_model_dir, kv="evict", kv_budget=budget),
                           bundle, items(1), set())
    assert row["kept_context_tokens"] == min(budget, row["context_tokens"])
    assert np.isfinite(row["letter_logprobs"]).all()
    if budget >= row["context_tokens"]:
        [reference] = bb.score_items(cfg_for(tiny_model_dir), bundle, items(1), set())
        np.testing.assert_allclose(row["letter_logprobs"], reference["letter_logprobs"])


def test_calibrated_accuracy_removes_letter_prior():
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 4, 2000)
    signal = np.eye(4)[gold] * 1.0 + rng.normal(0, 0.6, (2000, 4))
    biased = signal + np.array([1.5, 0, 0, 0])  # planted preference for "A"
    raw = float((biased.argmax(1) == gold).mean())
    assert bb.calibrated_accuracy(biased, gold) > raw + 0.1
    assert abs(bb.calibrated_accuracy(biased, gold) - bb.calibrated_accuracy(signal, gold)) < 0.02


def test_end_to_end_through_run(tmp_path, tiny_model_dir, monkeypatch):
    monkeypatch.setattr(datasets, "belebele", lambda lang: items(9))
    jobs = [{"model": str(tiny_model_dir), "task": "belebele", "lang": "te", "limit": 7}]
    [res] = run(jobs, root=tmp_path, loader=lambda c: load(c, device="cpu"), sync=False)
    assert (res.status, res.n_items) == ("done", 7)


@pytest.mark.slow
def test_belebele_download_is_parallel_and_sorted():
    en, te = datasets.belebele("en"), datasets.belebele("te")
    assert len(en) == len(te) == 900
    assert [i["item_id"] for i in en] == [i["item_id"] for i in te]
    assert [i["gold"] for i in en] == [i["gold"] for i in te]


@pytest.mark.slow
def test_qwen_tokenizer_splits_every_item_cleanly():
    from transformers import AutoTokenizer

    from whopays.languages import CODES

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    assert len(bb.letter_token_ids(tok)) == 4
    for lang in CODES:
        for item in datasets.belebele(lang):
            context, query = bb.build_prompt(item)
            ctx_ids, q_ids = bb.split_tokens(tok, context, query)
            assert ctx_ids + q_ids == tok(context + query)["input_ids"]


def shared_passage_items():
    out = []
    for p, passage in enumerate(["The archive sits by the bridge in the old river town.",
                                 "Farmers moved the market to the hill after the flood."]):
        for q, question in enumerate(["Where is it?", "What moved?", "Why did it change?"]):
            out.append({**ITEM, "item_id": f"p{p}-q{q}", "passage": passage,
                        "question": question, "gold": q % 4})
    return out


@pytest.mark.parametrize("kv", ["full", "evict", "quant4"])
def test_prefill_reuse_is_bit_identical_to_fresh_prefill(bundle, tiny_model_dir, kv,
                                                         monkeypatch):
    from whopays.kv import policies
    monkeypatch.setattr(policies.KIVILayer.__init__, "__defaults__", (4, 4, 4))  # quantize
    cfg = cfg_for(tiny_model_dir, kv=kv, kv_budget=8)
    letters = bb.letter_token_ids(bundle.tokenizer)
    rows = list(bb.score_items(cfg, bundle, shared_passage_items(), set()))
    assert [r["prefill_reused"] for r in rows] == [False, True, True] * 2
    for item, row in zip(shared_passage_items(), rows, strict=True):
        ctx, q = bb.split_tokens(bundle.tokenizer, *bb.build_prompt(item))
        fresh, kept = bb.score_item(bundle, cfg, ctx, q, letters)
        assert [round(x, 6) for x in fresh] == row["letter_logprobs"]
        assert kept == row["kept_context_tokens"]


def test_prefill_reuse_survives_skipped_items(bundle, tiny_model_dir):
    cfg = cfg_for(tiny_model_dir)
    full = {r["item_id"]: r for r in bb.score_items(cfg, bundle, shared_passage_items(), set())}
    part = list(bb.score_items(cfg, bundle, shared_passage_items(), {"p0-q0", "p1-q1"}))
    for row in part:
        assert row["letter_logprobs"] == full[row["item_id"]]["letter_logprobs"]


def test_share_budget_scales_with_the_language_token_ratio():
    assert bb.share_budget(256, n_lang=600, n_en=300) == 512
    assert bb.share_budget(256, n_lang=300, n_en=300) == 256
    assert bb.share_budget(256, n_lang=10, n_en=3000) == bb.MIN_SHARE_BUDGET


def test_evict_share_keeps_the_english_share_of_each_item(bundle, tiny_model_dir):
    budgets = {"x-q0": 7}
    [row] = bb.score_items(cfg_for(tiny_model_dir, kv="evict_share", kv_budget=256), bundle,
                           items(1), set(), budgets=budgets)
    assert row["kv_item_budget"] == 7
    assert row["kept_context_tokens"] == min(7, row["context_tokens"])
    [same] = bb.score_items(cfg_for(tiny_model_dir, kv="evict", kv_budget=7), bundle, items(1),
                            set())
    np.testing.assert_allclose(row["letter_logprobs"], same["letter_logprobs"])


def test_evict_share_needs_budgets(bundle, tiny_model_dir):
    with pytest.raises(ValueError, match="per-item budgets"):
        list(bb.score_items(cfg_for(tiny_model_dir, kv="evict_share"), bundle, items(1), set()))


def test_evict_share_budgets_come_from_the_english_items(bundle, tiny_model_dir, monkeypatch):
    long = {**ITEM, "passage": ITEM["passage"] * 3}
    monkeypatch.setattr(datasets, "belebele",
                        lambda lang: [ITEM] if lang == "en" else [long])
    budgets = bb.english_share_budgets(bundle.tokenizer, 20, [long])
    en_ctx = len(bb.split_tokens(bundle.tokenizer, *bb.build_prompt(ITEM))[0])
    te_ctx = len(bb.split_tokens(bundle.tokenizer, *bb.build_prompt(long))[0])
    assert budgets == {"x-q1": bb.share_budget(20, n_lang=te_ctx, n_en=en_ctx)}
