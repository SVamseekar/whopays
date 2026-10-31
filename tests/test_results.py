import os
import signal
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from whopays.io.results import (
    ITEMS,
    ResultsWriter,
    is_done,
    mark_done,
    read_rows,
    repair,
    to_parquet,
)

N = 237


def fake_items(skip=frozenset(), crash_at=None):
    """Deterministic task: yields rows in a fixed order, skipping finished ones."""
    for i in range(N):
        item_id = f"item-{i:04d}"
        if item_id in skip:
            continue
        if i == crash_at:
            raise RuntimeError("session died")
        yield {"item_id": item_id, "correct": i % 3 == 0, "logprob": -i / 7, "text": "తెలుగు"}


def run_to_end(path, crash_at=None):
    with ResultsWriter(path) as w:
        for row in fake_items(w.done_ids(), crash_at):
            w.write(row)


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "ref" / "items.jsonl"
    run_to_end(path)
    return path.read_bytes()


def test_crash_then_resume_is_byte_identical(tmp_path, reference):
    path = tmp_path / "run" / "items.jsonl"
    with pytest.raises(RuntimeError):
        run_to_end(path, crash_at=120)
    assert len(read_rows(path)) == 120  # exception path flushes finished rows
    run_to_end(path)
    assert path.read_bytes() == reference


def test_partial_line_is_repaired_on_resume(tmp_path, reference):
    path = tmp_path / "run" / "items.jsonl"
    with pytest.raises(RuntimeError):
        run_to_end(path, crash_at=100)
    with path.open("ab") as f:
        f.write(b'{"item_id":"item-0100","corr')  # killed mid-write
    run_to_end(path)
    assert path.read_bytes() == reference


def test_sigkill_mid_run_then_resume_is_byte_identical(tmp_path, reference):
    path = tmp_path / "run" / "items.jsonl"
    # A real SIGKILL: no exception handlers or context managers run, buffered rows are lost.
    script = textwrap.dedent(f"""
        import os, signal, sys
        sys.path.insert(0, {os.path.dirname(__file__)!r})
        from test_results import fake_items
        from whopays.io.results import ResultsWriter
        w = ResultsWriter({str(path)!r})
        for n, row in enumerate(fake_items(w.done_ids())):
            if n == 173:
                os.kill(os.getpid(), signal.SIGKILL)
            w.write(row)
    """)
    proc = subprocess.run([sys.executable, "-c", script])
    assert proc.returncode == -signal.SIGKILL
    assert len(read_rows(path)) == 150  # three flushes of 50; 23 buffered rows lost
    run_to_end(path)
    assert path.read_bytes() == reference


def test_flushes_every_50(tmp_path):
    flushed = []
    path = tmp_path / "items.jsonl"
    w = ResultsWriter(path, on_flush=lambda p: flushed.append(len(read_rows(p))))
    for row in list(fake_items())[:120]:
        w.write(row)
    assert flushed == [50, 100]
    w.close()
    assert flushed == [50, 100, 120]


def test_duplicate_item_is_an_error(tmp_path):
    w = ResultsWriter(tmp_path / "items.jsonl")
    w.write({"item_id": "a"})
    with pytest.raises(ValueError, match="written twice"):
        w.write({"item_id": "a"})


def test_nan_is_rejected(tmp_path):
    w = ResultsWriter(tmp_path / "items.jsonl")
    with pytest.raises(ValueError):
        w.write({"item_id": "a", "logprob": float("nan")})


def test_repair_without_any_complete_line(tmp_path):
    path = tmp_path / "items.jsonl"
    path.write_bytes(b'{"item_id":')
    assert repair(path) == 11
    assert path.read_bytes() == b""


def test_parquet_and_done_marker(tmp_path):
    path = tmp_path / "items.jsonl"
    run_to_end(path)
    df = pd.read_parquet(to_parquet(path))
    assert len(df) == N and df["text"].iloc[0] == "తెలుగు"
    assert not is_done(tmp_path)
    (tmp_path / "config.yaml").write_text("{}")
    (tmp_path / "meta.json").write_text("{}")
    mark_done(tmp_path, len(df))
    assert is_done(tmp_path)


def test_load_runs_tags_model_and_reads_partial_runs(tmp_path):
    from whopays.config import RunConfig
    from whopays.io.results import load_runs

    done = RunConfig(model="m/a", task="belebele", lang="te")
    partial = RunConfig(model="m/b", task="belebele", lang="te")
    missing = RunConfig(model="m/c", task="belebele", lang="te")
    for cfg, n in ((done, 3), (partial, 2)):
        with ResultsWriter(tmp_path / cfg.run_id / ITEMS) as w:
            for i in range(n):
                w.write({"item_id": f"i{i}", "run_id": cfg.run_id})
    to_parquet(tmp_path / done.run_id / ITEMS)
    df = load_runs(tmp_path, [done, partial, missing])
    assert df.groupby("model").size().to_dict() == {"m/a": 3, "m/b": 2}
    assert set(df["weight"]) == {"fp16"} and set(df["kv"]) == {"full"}
    assert load_runs(tmp_path, [missing]).empty


def test_m0_grid_expands_to_every_language():
    from whopays.config import load_jobs
    from whopays.grids import m0_fp16

    configs = load_jobs(m0_fp16())
    assert len(configs) == 3 * 3 * 11
    assert len({c.run_id for c in configs}) == len(configs)
