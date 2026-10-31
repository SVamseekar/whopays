import json

import pandas as pd
import pytest

from whopays import crosscheck
from whopays.datasets import belebele_item_id
from whopays.languages import CODES


def test_task_names_round_trip():
    for lang in CODES:
        for task in ("belebele", "mgsm"):
            assert crosscheck.parse_lmeval_task(crosscheck.lmeval_task(task, lang)) == (task, lang)
    assert crosscheck.lmeval_task("belebele", "te") == "belebele_tel_Telu"
    assert crosscheck.lmeval_task("mgsm", "te") == "mgsm_native_cot_te"
    with pytest.raises(ValueError):
        crosscheck.lmeval_task("flores_nll", "te")


def test_args_pin_greedy_fp16_and_resume(tmp_path):
    args = crosscheck.lmeval_args("Qwen/Qwen2.5-1.5B-Instruct", ["belebele_tel_Telu"], tmp_path,
                                  limit=5)
    joined = " ".join(args)
    assert args[0] == "run"
    assert "pretrained=Qwen/Qwen2.5-1.5B-Instruct,dtype=float16" in joined
    assert "--gen_kwargs repetition_penalty=1.0" in joined
    assert f"--use_cache {tmp_path / 'cache'}" in joined
    assert "--log_samples" in args and args[-2:] == ["--limit", "5"]


def _write_samples(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _belebele_rec(doc_id, link, qnum, acc):
    return {"doc_id": doc_id, "doc": {"link": link, "question_number": qnum}, "filter": "none",
            "metrics": ["acc", "acc_norm"], "acc": acc, "acc_norm": acc}


def _mgsm_recs(doc_id, flexible, strict):
    return [{"doc_id": doc_id, "doc": {}, "filter": "strict-match", "metrics": ["exact_match"],
             "exact_match": strict},
            {"doc_id": doc_id, "doc": {}, "filter": "flexible-extract",
             "metrics": ["exact_match"], "exact_match": flexible}]


def test_read_samples_maps_ids_and_newest_file_wins(tmp_path):
    model_dir = tmp_path / "Qwen__Qwen2.5-1.5B-Instruct"
    _write_samples(model_dir / "samples_belebele_tel_Telu_2026-10-21T10-00-00.000001.jsonl",
                   [_belebele_rec(0, "https://x/a", 1, 0.0)])
    _write_samples(model_dir / "samples_belebele_tel_Telu_2026-10-21T11-00-00.000001.jsonl",
                   [_belebele_rec(0, "https://x/a", 1, 1.0)])
    _write_samples(model_dir / "samples_mgsm_native_cot_sw_2026-10-21T11-00-00.000001.jsonl",
                   _mgsm_recs(7, 1.0, 0.0))
    df = crosscheck.read_samples(tmp_path)
    bel = df[(df.task == "belebele") & (df.metric == "acc")]
    assert bel.to_dict("records") == [{"task": "belebele", "lang": "te",
                                       "item_id": belebele_item_id("https://x/a", 1),
                                       "filter": "none", "metric": "acc", "value": 1.0}]
    flex = df[(df.task == "mgsm") & (df["filter"] == "flexible-extract")]
    assert (flex.lang.item(), flex.item_id.item(), flex.value.item()) == ("sw", "mgsm-007", 1.0)


def _theirs(task, lang, flt, metric, values):
    return pd.DataFrame({"task": task, "lang": lang, "item_id": [f"i{k}" for k in
                                                                 range(len(values))],
                         "filter": flt, "metric": metric, "value": values})


def test_compare_gap_agreement_and_verdict():
    n = 200
    ref = [1.0] * 100 + [0.0] * 100
    ours_te = ref.copy()
    ours_te[0], ours_te[150] = 0.0, 1.0           # two swaps: same accuracy, agreement 0.99
    ours_sw = ref.copy()
    ours_sw[100:106] = [1.0] * 6                  # +6 items = +3 points
    ours = pd.concat([
        pd.DataFrame({"task": "belebele", "lang": "te", "item_id": [f"i{k}" for k in range(n)],
                      "correct": ours_te}),
        pd.DataFrame({"task": "belebele", "lang": "sw", "item_id": [f"i{k}" for k in range(n)],
                      "correct": ours_sw}),
        pd.DataFrame({"task": "mgsm", "lang": "te", "item_id": [f"i{k}" for k in range(n)],
                      "correct_ascii": ref, "correct": ref}),
    ])
    theirs = pd.concat([
        _theirs("belebele", "te", "none", "acc", ref),
        _theirs("belebele", "sw", "none", "acc", ref),
        _theirs("mgsm", "te", "flexible-extract", "exact_match", ref),
        _theirs("mgsm", "te", "strict-match", "exact_match", [0.0] * n),  # must be ignored
    ])
    table = crosscheck.compare(ours, theirs).set_index(["task", "lang"])
    assert table.loc[("belebele", "te"), "diff_pts"] == 0.0
    assert table.loc[("belebele", "te"), "agreement"] == 0.99
    assert table.loc[("belebele", "sw"), "diff_pts"] == pytest.approx(3.0)
    assert bool(table.loc[("belebele", "sw"), "flag"])
    assert table.loc[("mgsm", "te"), "diff_pts"] == 0.0
    verdict = crosscheck.k1_verdict(table.reset_index()).set_index("task")
    assert verdict.loc["belebele", "mean_abs_diff_pts"] == pytest.approx(1.5)
    assert verdict.loc["belebele", "K1"] == "INCOMPLETE"
    assert verdict.loc["mgsm", "K1"] == "INCOMPLETE"


def test_compare_uses_only_common_items():
    ours = pd.DataFrame({"task": "belebele", "lang": "en", "item_id": ["a", "b", "c"],
                         "correct": [1.0, 1.0, 0.0]})
    theirs = _theirs("belebele", "en", "none", "acc", [1.0, 1.0]).assign(item_id=["a", "b"])
    row = crosscheck.compare(ours, theirs).iloc[0]
    assert (row.n_items, row.ours, row.lmeval) == (2, 100.0, 100.0)
