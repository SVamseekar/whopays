import pytest
import torch

from whopays import datasets, run
from whopays.config import RunConfig
from whopays.languages import CODES
from whopays.models.loader import load
from whopays.tasks import mgsm


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("16 - 3 - 4 = 9 eggs. 9 * 2 = 18. The answer is 18.", 18),
        ("She pays $1,234 in total.", 1234),
        ("Total: 1,234.50 dollars", 1234.5),
        ("Insgesamt 1.234,5 Euro", 1234.5),
        ("Es sind 3,5 Liter", 3.5),
        ("It costs 0.125 each", 0.125),
        ("সমাধান: ৯ × ২ = ১৮", 18),  # Bengali digits
        ("సమాధానం ౧౮", 18),  # Telugu digits
        ("คำตอบคือ ๑๘", 18),  # Thai digits
        ("答えは18です。", 18),
        ("The temperature fell to -3 degrees", -3),
        ("No number here.", None),
    ],
)
def test_extract_number(text, expected):
    assert mgsm.extract_number(text) == expected


def test_ascii_extraction_mirrors_lm_eval():
    assert mgsm.extract_ascii_lmeval("so 9 * 2 = $18.") == "18"
    assert mgsm.extract_ascii_lmeval("সমাধান ১৮") is None  # lm-eval misses native digits
    assert mgsm.extract_ascii_lmeval("1,234") == "1234"


def test_prompts_cover_all_languages_and_match_lm_eval_shape():
    assert set(mgsm.PROMPTS) == set(CODES)
    assert mgsm.build_prompt("te", "Q?") == "ప్రశ్న: Q?\nదశలవారీగా సమాధానం:"
    assert mgsm.build_prompt("en", "Q?") == "Question: Q?\nStep-by-Step Answer:"


def test_truncate_at_next_question():
    text, stopped = mgsm.truncate_at_stop("9*2=18.\nQuestion: next", "en")
    assert (text, stopped) == ("9*2=18.\n", True)


@pytest.fixture(scope="module")
def bundle(tiny_model_dir):
    return load(RunConfig(model=str(tiny_model_dir), task="mgsm", lang="en"), device="cpu")


def items(n):
    return [{"item_id": f"mgsm-{i:03d}", "question": f"What is {i} + {i}?", "gold": 2 * i}
            for i in range(n)]


def test_generation_rows(bundle, tiny_model_dir, monkeypatch):
    monkeypatch.setattr(mgsm, "MAX_NEW_TOKENS", 8)
    cfg = RunConfig(model=str(tiny_model_dir), task="mgsm", lang="en", batch_size=2)
    rows = list(mgsm.score_items(cfg, bundle, items(5), skip={"mgsm-001"}))
    assert [r["item_id"] for r in rows] == ["mgsm-000", "mgsm-002", "mgsm-003", "mgsm-004"]
    for r in rows:
        assert 1 <= r["output_tokens"] <= 8
        assert r["extraction_failed"] == (r["pred_number"] is None)
        assert r["batch_size"] == 2


def test_greedy_is_batch_invariant_in_fp32(bundle, tiny_model_dir, monkeypatch):
    """Left padding must not change outputs (fp32 CPU); in FP16 near-ties may still move."""
    monkeypatch.setattr(mgsm, "MAX_NEW_TOKENS", 8)
    one = RunConfig(model=str(tiny_model_dir), task="mgsm", lang="en", batch_size=1)
    three = RunConfig(model=str(tiny_model_dir), task="mgsm", lang="en", batch_size=3)
    a = [r["output_text"] for r in mgsm.score_items(one, bundle, items(3), set())]
    b = [r["output_text"] for r in mgsm.score_items(three, bundle, items(3), set())]
    assert a == b


def test_speculative_not_yet_supported(bundle, tiny_model_dir):
    cfg = RunConfig(model=str(tiny_model_dir), task="mgsm", lang="en", spec=True,
                    draft_model=str(tiny_model_dir))
    with pytest.raises(NotImplementedError, match="M2"):
        list(mgsm.score_items(cfg, bundle, items(1), set()))


def test_end_to_end_through_run(tmp_path, tiny_model_dir, monkeypatch):
    monkeypatch.setattr(mgsm, "MAX_NEW_TOKENS", 4)
    monkeypatch.setattr(datasets, "mgsm", lambda lang: items(5))
    jobs = [{"model": str(tiny_model_dir), "task": "mgsm", "lang": "sw", "limit": 3}]
    [res] = run(jobs, root=tmp_path, loader=lambda c: load(c, device="cpu"), sync=False)
    assert (res.status, res.n_items) == ("done", 3)


@pytest.mark.slow
def test_mgsm_download_is_parallel():
    en, te = datasets.mgsm("en"), datasets.mgsm("te")
    assert len(en) == len(te) == 250
    assert [i["gold"] for i in en] == [i["gold"] for i in te]


def test_quantized_cache_batches_per_sequence(tiny_hd128_dir, monkeypatch):
    """Batched quant4 generation equals batch-1 generation row by row (fp32 CPU), with groups
    quantized during decoding, because KIVI groups start at each row's first real token."""
    from whopays.kv import policies
    monkeypatch.setattr(mgsm, "MAX_NEW_TOKENS", 20)
    monkeypatch.setattr(policies, "KIVI_RESIDUAL", 4)
    monkeypatch.setattr(policies, "KIVI_GROUP", 4)
    monkeypatch.setattr(policies.KIVILayer.__init__, "__defaults__", (4, 4, 4))
    b = load(RunConfig(model=str(tiny_hd128_dir), task="mgsm", lang="en"), device="cpu")
    one = RunConfig(model=str(tiny_hd128_dir), task="mgsm", lang="en", kv="quant4", batch_size=1)
    four = RunConfig(model=str(tiny_hd128_dir), task="mgsm", lang="en", kv="quant4", batch_size=4)
    full = RunConfig(model=str(tiny_hd128_dir), task="mgsm", lang="en", batch_size=1)
    qs = [{"item_id": f"m{i}", "question": "What is " + "x " * (3 * i) + f"{i}?", "gold": i}
          for i in range(4)]
    single = [r["output_text"] for r in mgsm.score_items(one, b, qs, set())]
    batched = list(mgsm.score_items(four, b, qs, set()))
    assert [r["output_text"] for r in batched] == single
    assert all(r["batch_size"] == 4 for r in batched)
    exact = [r["output_text"] for r in mgsm.score_items(full, b, qs, set())]
    assert single != exact  # the cache did quantize something


@pytest.mark.parametrize("budget", [6, 40])
def test_eviction_batches_per_sequence(bundle, tiny_model_dir, monkeypatch, budget):
    """Batched sink+recent eviction equals the sequential batch-1 loop row by row (fp32 CPU):
    prompts shorter and longer than the budget, eviction at prefill and during decoding."""
    monkeypatch.setattr(mgsm, "MAX_NEW_TOKENS", 20)
    kw = dict(model=str(tiny_model_dir), task="mgsm", lang="en", kv="evict", kv_budget=budget)
    one, four = RunConfig(**kw, batch_size=1), RunConfig(**kw, batch_size=4)
    qs = [{"item_id": f"m{i}", "question": "What is " + "x " * (6 * i) + f"{i}?", "gold": i}
          for i in range(5)]
    single = list(mgsm.score_items(one, bundle, qs, set()))
    batched = list(mgsm.score_items(four, bundle, qs, set()))
    assert [r["output_text"] for r in batched] == [r["output_text"] for r in single]
    assert [r["output_tokens"] for r in batched] == [r["output_tokens"] for r in single]
    assert [r["input_tokens"] for r in batched] == [r["input_tokens"] for r in single]
    assert all(r["batch_size"] == 4 for r in batched)  # as the full cache logs it
    lengths = [r["input_tokens"] for r in single]
    assert max(lengths) > budget  # eviction at prefill
    if budget == 40:
        assert min(lengths) < budget  # and rows evicted only while decoding


def test_eviction_batch_passes_absolute_positions(bundle, tiny_model_dir, monkeypatch):
    """After columns are evicted the mask no longer encodes positions; each decode step must
    pass every row's own absolute position (prompt length + tokens generated so far)."""
    import types

    monkeypatch.setattr(mgsm, "MAX_NEW_TOKENS", 6)
    seen = []
    real = bundle.model

    class Spy:
        def __getattr__(self, name):
            return getattr(real, name)

        def __call__(self, **kw):
            seen.append(kw["position_ids"].clone())
            return real(**kw)

    spied = types.SimpleNamespace(**{**vars(bundle), "model": Spy()})
    cfg = RunConfig(model=str(tiny_model_dir), task="mgsm", lang="en", kv="evict",
                    kv_budget=8, batch_size=3)
    qs = [{"item_id": f"m{i}", "question": "What is " + "x " * (6 * i) + "?", "gold": i}
          for i in range(3)]
    rows = list(mgsm.score_items(cfg, spied, qs, set()))
    lengths = torch.tensor([r["input_tokens"] for r in rows])
    prefill, *decode = seen
    assert prefill.shape[1] == int(lengths.max())
    assert (prefill[:, -1] == lengths - 1).all()
    for step, pos in enumerate(decode):
        assert (pos[:, 0] == lengths + step).all()


def test_batch_gate_reports_agreement_and_passes_in_fp32(bundle, tiny_model_dir, monkeypatch):
    from whopays import gates

    monkeypatch.setattr(mgsm, "MAX_NEW_TOKENS", 6)
    table, verdict = gates.batch_gate(str(tiny_model_dir), langs=("en",), n=3, batch_size=2,
                                      kv_policies=("full", "evict"), loader=lambda c: bundle,
                                      items=lambda lang: items(3))
    assert verdict == "PASS"
    assert list(table.kv) == ["full", "evict"]
    assert (table.same_text == 3).all() and (table.flips == 0).all()
