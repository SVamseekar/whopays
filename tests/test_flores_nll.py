import json
import math

import pytest
import torch

from whopays import datasets, run
from whopays.config import RunConfig
from whopays.models.loader import load
from whopays.tasks import flores_nll

TEXTS = ["Das Wetter ist schön.", "ఈ రోజు వాతావరణం బాగుంది.", "A B C D", "今天天气很好。"]


@pytest.fixture(scope="module")
def bundle(tiny_model_dir):
    return load(RunConfig(model=str(tiny_model_dir), task="flores_nll", lang="en"), device="cpu")


def reference_nll(bundle, text):
    """Unbatched, unpadded per-sentence NLL computed independently."""
    ids = [flores_nll.prefix_token(bundle), *bundle.tokenizer(text, add_special_tokens=False)[
        "input_ids"]]
    with torch.no_grad():
        logp = bundle.model(torch.tensor([ids])).logits[0, :-1].float().log_softmax(-1)
    return -sum(logp[i, t].item() for i, t in enumerate(ids[1:])), len(ids) - 1


def test_batched_padded_nll_matches_reference(bundle):
    got = flores_nll.nll_batch(bundle, TEXTS)
    for text, (nll, n) in zip(TEXTS, got, strict=True):
        ref_nll, ref_n = reference_nll(bundle, text)
        assert n == ref_n
        assert math.isclose(nll, ref_nll, rel_tol=1e-4)


def test_random_model_nll_is_near_log_vocab(bundle):
    nll, n = flores_nll.nll_batch(bundle, ["Das Wetter ist schön."])[0]
    assert abs(nll / n - math.log(len(bundle.tokenizer))) < 1.0  # untrained model ~ uniform


def test_rows(bundle, tiny_model_dir):
    cfg = RunConfig(model=str(tiny_model_dir), task="flores_nll", lang="te", batch_size=3)
    items = [{"item_id": f"flores-devtest-{i:04d}", "text": t} for i, t in enumerate(TEXTS)]
    rows = list(flores_nll.score_items(cfg, bundle, items, skip={"flores-devtest-0000"}))
    assert len(rows) == 3
    r = rows[0]
    assert r["n_bytes"] == len(TEXTS[1].encode()) > r["n_chars"]
    assert math.isclose(r["nll_per_token"], r["nll_sum"] / r["n_tokens"], rel_tol=1e-4)


def test_kv_and_spec_rejected(bundle, tiny_model_dir):
    cfg = RunConfig(model=str(tiny_model_dir), task="flores_nll", lang="te", kv="quant4")
    with pytest.raises(ValueError, match="weights only"):
        list(flores_nll.score_items(cfg, bundle, [{"item_id": "a", "text": "x"}], set()))


def test_end_to_end_through_run(tmp_path, tiny_model_dir, monkeypatch):
    fake = [{"item_id": f"flores-devtest-{i:04d}", "text": t} for i, t in enumerate(TEXTS)]
    monkeypatch.setattr(datasets, "flores", lambda lang, split: fake)
    jobs = [{"model": str(tiny_model_dir), "task": "flores_nll", "lang": "zh"}]
    [res] = run(jobs, root=tmp_path, loader=lambda c: load(c, device="cpu"), sync=False)
    assert (res.status, res.n_items) == ("done", 4)


def test_flores_split_guard():
    with pytest.raises(ValueError, match="dev or devtest"):
        datasets.flores("en", "test")


def test_low_memory_nll_matches_batched_nll(tiny_model_dir):
    from whopays.config import RunConfig
    from whopays.models.loader import load
    from whopays.tasks.flores_nll import nll_batch, nll_batch_low_memory

    bundle = load(RunConfig(model=str(tiny_model_dir), task="flores_nll", lang="en"), "cpu")
    texts = ["a short one", "a considerably longer sentence with several more tokens in it", "x"]
    for (a, n), (b, m) in zip(nll_batch(bundle, texts), nll_batch_low_memory(bundle, texts),
                              strict=True):
        assert n == m
        assert abs(a - b) <= 1e-4 * max(1.0, abs(a))


def test_offset_scores_items_after_the_probe_sentences(tmp_path, tiny_model_dir, monkeypatch):
    fake = [{"item_id": f"flores-devtest-{i:04d}", "text": TEXTS[i % 4]}
            for i in reversed(range(6))]
    monkeypatch.setattr(datasets, "flores", lambda lang, split: fake)
    jobs = [{"model": str(tiny_model_dir), "task": "flores_nll", "lang": "zh", "offset": 4}]
    [res] = run(jobs, root=tmp_path, loader=lambda c: load(c, device="cpu"), sync=False)
    assert res.n_items == 2
    items = (tmp_path / res.run_id / "items.jsonl").read_text().splitlines()
    ids = {json.loads(line)["item_id"] for line in items}
    assert ids == {"flores-devtest-0004", "flores-devtest-0005"}
