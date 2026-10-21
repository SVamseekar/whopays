import pandas as pd

from whopays import throughput
from whopays.grids import K1_MODEL, M0_MODELS, m0_fp16


def test_model_size():
    assert throughput.model_size("Qwen/Qwen2.5-1.5B-Instruct") == "1.5B"
    assert throughput.model_size("Qwen/Qwen2.5-3B-Instruct") == "3B"
    assert throughput.model_size("/tmp/tiny") == "/tmp/tiny"


def test_summary_projects_to_full_benchmark_and_flags_replan():
    # 990 Belebele items at 0.2 s each: 9,900 items would take 33 min vs 15 assumed (2.2x).
    bel = pd.DataFrame({"model": K1_MODEL, "task": "belebele", "item_id": range(990),
                        "wall_time_s": 0.2})
    # 275 MGSM items at 0.5 s and 100 output tokens each: 2,750 -> 22.9 min vs 30 assumed.
    mg = pd.DataFrame({"model": K1_MODEL, "task": "mgsm", "item_id": range(275),
                       "wall_time_s": 0.5, "output_tokens": 100})
    table = throughput.summary(pd.concat([bel, mg])).set_index("task")
    assert table.loc["belebele", "projected_full_min"] == 33.0
    assert table.loc["belebele", "ratio_vs_assumed"] == 2.2
    assert bool(table.loc["belebele", "replan"])
    assert table.loc["mgsm", "projected_full_min"] == 22.9
    assert not bool(table.loc["mgsm", "replan"])
    assert table.loc["mgsm", "out_tokens_per_s"] == 200.0


def test_summary_without_assumption():
    rows = pd.DataFrame({"model": "Qwen/Qwen2.5-0.5B-Instruct", "task": "flores_nll",
                         "item_id": range(10), "wall_time_s": 0.1})
    row = throughput.summary(rows).iloc[0]
    assert row.assumed_min is None and row.ratio_vs_assumed is None


def test_m0_grid():
    jobs = m0_fp16()
    assert jobs[0]["model"] == K1_MODEL  # K1 model runs first
    assert len(jobs) == len(M0_MODELS) * 3
    assert all(j["weight"] == "fp16" and j["kv"] == "full" and j["lang"] == "all" for j in jobs)
