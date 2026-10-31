"""K1 gate: compare our FP16 scores with lm-evaluation-harness on the same items (FR-6).

lm-eval runs in its own virtualenv. The venv sees the system torch and transformers through
--system-site-packages, and pip is constrained to the installed versions. Its own
dependencies (evaluate, sacrebleu, scikit-learn, ...) stay out of the main environment and
out of requirements.lock.

What is compared, like with like:
- Belebele: our `correct` vs lm-eval `acc`. Both take the argmax over the log-probabilities
  of " A".." D" after the same zero-shot prompt.
- MGSM: our `correct_ascii` vs lm-eval `exact_match` under its `flexible-extract` filter.
  Both take the last ASCII number. Our native-digit `correct` is reported alongside, not
  gated. lm-eval gets `repetition_penalty=1.0` so both decode pure greedy.

The gate (DECISIONS 2026-10-21) is per task: the mean over languages of |ours − lm-eval|
must be within 1 point. Per-language gaps over 1 point are flagged for inspection. Item-level
agreement shows whether a gap is a few FP16 near-ties or a systematic difference.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from contextlib import closing
from pathlib import Path

import pandas as pd

from whopays.languages import CODES, LANGUAGES

LMEVAL_VERSION = "0.4.13"
TOLERANCE_PTS = 1.0
_BY_FLORES = {lang.flores: code for code, lang in LANGUAGES.items()}
_SAMPLES = re.compile(r"samples_(?P<task>.+)_\d{4}-\d{2}-\d{2}T[\d.\-]+\.jsonl$")
# (lm-eval filter, lm-eval metric, our column) per task
GATED = {"belebele": ("none", "acc", "correct"),
         "mgsm": ("flexible-extract", "exact_match", "correct_ascii")}
log = logging.getLogger(__name__)


def lmeval_task(task: str, lang: str) -> str:
    if task == "belebele":
        return f"belebele_{LANGUAGES[lang].flores}"
    if task == "mgsm":
        return f"mgsm_native_cot_{lang}"
    raise ValueError(f"no lm-eval counterpart for task {task!r}")


def parse_lmeval_task(name: str) -> tuple[str, str]:
    """Inverse of lmeval_task: 'belebele_tel_Telu' -> ('belebele', 'te')."""
    if name.startswith("belebele_"):
        return "belebele", _BY_FLORES[name.removeprefix("belebele_")]
    if name.startswith("mgsm_native_cot_"):
        return "mgsm", name.removeprefix("mgsm_native_cot_")
    raise ValueError(f"unknown lm-eval task {name!r}")


def lmeval_args(model: str, task_names: Sequence[str], out_dir: Path, *, batch_size: int = 8,
                limit: int | None = None) -> list[str]:
    """Arguments after `python -m lm_eval`. `--use_cache` makes a restarted run resume."""
    args = ["run", "--model", "hf",
            "--model_args", f"pretrained={model},dtype=float16",
            "--tasks", ",".join(task_names),
            "--batch_size", str(batch_size),
            "--gen_kwargs", "repetition_penalty=1.0",
            "--output_path", str(out_dir),
            "--use_cache", str(out_dir / "cache"),
            "--log_samples"]
    if limit is not None:
        args += ["--limit", str(limit)]
    return args


def install(venv: Path = Path("/tmp/venv-lmeval")) -> Path:
    """Create the lm-eval venv (idempotent); return its python.

    Created without pip because Colab's Python has no ensurepip; the system pip, visible
    through --system-site-packages, installs into it.
    """
    import importlib.metadata as md

    python = venv / "bin" / "python"
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages",
                        str(venv)], check=True)
    constraints = venv / "constraints.txt"
    constraints.write_text("".join(f"{p}=={md.version(p)}\n"
                                   for p in ("torch", "transformers", "accelerate")))
    subprocess.run([str(python), "-m", "pip", "install", "-q", "-c", str(constraints),
                    f"lm-eval=={LMEVAL_VERSION}"], check=True)
    return python


def sync_lmeval(out_dir: Path) -> bool:
    """Mirror a consistent checkpoint, including committed SQLite/WAL requests.

    Copying a live SQLite file can corrupt the remote cache or omit its WAL. Use
    SQLite's online backup API, and upload a temporary snapshot under the same
    folder name. Hub failures leave the local reference run running.
    """
    from whopays.env import get_secret
    from whopays.io import hub_sync

    token = get_secret("HF_TOKEN")
    if not token:
        return False
    out_dir = Path(out_dir)
    try:
        with tempfile.TemporaryDirectory(prefix="whopays-lmeval-") as temp:
            snapshot = Path(temp) / out_dir.name
            snapshot.mkdir()
            for source in out_dir.rglob("*"):
                if not source.is_file() or source.name.endswith(("-wal", "-shm", "-journal")):
                    continue
                dest = snapshot / source.relative_to(out_dir)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with source.open("rb") as stream:
                    is_sqlite = stream.read(16) == b"SQLite format 3\x00"
                if is_sqlite:
                    uri = source.resolve().as_uri() + "?mode=ro"
                    with (closing(sqlite3.connect(uri, uri=True)) as reader,
                          closing(sqlite3.connect(dest)) as writer):
                        reader.backup(writer)
                        writer.execute("PRAGMA journal_mode=DELETE")
                else:
                    shutil.copy2(source, dest)
            return hub_sync.push_run(snapshot, token=token)
    except Exception as exc:
        log.warning("lm-eval checkpoint failed; local results retained: %s", exc)
        return False


def run_lmeval(model: str, tasks: Iterable[str], langs: Iterable[str], out_dir: Path, *,
               batch_size: int = 8, limit: int | None = None,
               venv: Path = Path("/tmp/venv-lmeval"), sync: bool = True,
               sync_interval: float = 600.0) -> Path:
    """Run reference tasks, checkpointing requests during evaluation and on exit.

    A dead GPU runtime loses at most the time since the last successful remote
    checkpoint. Without an HF token, progress remains local only.
    """
    if not math.isfinite(sync_interval) or sync_interval <= 0:
        raise ValueError("sync_interval must be positive finite seconds")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    python = install(venv)
    names = [lmeval_task(t, lang) for t in tasks for lang in langs]
    log = out_dir / "lmeval.log"
    with log.open("a") as f, subprocess.Popen([str(python), "-m", "lm_eval",
                            *lmeval_args(model, names, out_dir, batch_size=batch_size,
                                         limit=limit)],
                           stdout=f, stderr=subprocess.STDOUT,
                           env={**os.environ, "USE_TF": "0"}) as proc:
        try:
            while True:
                try:
                    returncode = proc.wait(timeout=sync_interval)
                    break
                except subprocess.TimeoutExpired:
                    if sync:
                        sync_lmeval(out_dir)
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
        finally:
            if sync:
                sync_lmeval(out_dir)
    if returncode != 0:
        raise RuntimeError(f"lm-eval failed (exit {returncode}); tail of {log}:\n"
                           + "".join(log.read_text().splitlines(keepends=True)[-40:]))
    return out_dir


def _item_id(task: str, record: dict) -> str:
    if task == "belebele":
        from whopays.datasets import belebele_item_id

        doc = record["doc"]
        return belebele_item_id(doc["link"], int(doc["question_number"]))
    return f"mgsm-{int(record['doc_id']):03d}"


def read_samples(out_dir: Path) -> pd.DataFrame:
    """lm-eval per-item results: columns task, lang, item_id, filter, metric, value.

    lm-eval writes samples_<task>_<timestamp>.jsonl per run; when a task was run more than
    once, the newest file wins.
    """
    newest: dict[str, Path] = {}
    for path in sorted(Path(out_dir).rglob("samples_*.jsonl")):
        if m := _SAMPLES.search(path.name):
            newest[m["task"]] = path  # sorted: later timestamps overwrite earlier ones
    rows = []
    for name, path in newest.items():
        task, lang = parse_lmeval_task(name)
        with path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                for metric in rec["metrics"]:
                    rows.append({"task": task, "lang": lang, "item_id": _item_id(task, rec),
                                 "filter": rec["filter"], "metric": metric,
                                 "value": float(rec[metric])})
    return pd.DataFrame(rows, columns=["task", "lang", "item_id", "filter", "metric", "value"])


def compare(ours: pd.DataFrame, theirs: pd.DataFrame) -> pd.DataFrame:
    """Per task × language on the items both sides scored: accuracies, gap, agreement.

    `ours` needs columns task, lang, item_id and the gated column per task (GATED).
    """
    out = []
    for task, (flt, metric, col) in GATED.items():
        mine = ours[ours["task"] == task]
        ref = theirs[(theirs["task"] == task) & (theirs["filter"] == flt)
                     & (theirs["metric"] == metric)]
        if mine.empty or ref.empty:
            continue
        for frame in (mine, ref):
            if frame.duplicated(["lang", "item_id"]).any():
                raise ValueError(f"duplicate {task} language/item keys in crosscheck")
        both = mine[["lang", "item_id", col]].merge(
            ref[["lang", "item_id", "value"]], on=["lang", "item_id"], how="inner")
        for lang, g in both.groupby("lang", sort=True):
            a, b = g[col].astype(float), g["value"].astype(float)
            out.append({"task": task, "lang": lang, "n_items": len(g),
                        "ours": 100 * a.mean(), "lmeval": 100 * b.mean(),
                        "diff_pts": 100 * (a - b).mean(),
                        "complete": (len(g) == {"belebele": 900, "mgsm": 250}[task]
                                     and len(mine[mine.lang == lang]) == len(g)
                                     and len(ref[ref.lang == lang]) == len(g)),
                        "agreement": round(float((a == b).mean()), 4)})
    table = pd.DataFrame(out, columns=["task", "lang", "n_items", "ours", "lmeval", "diff_pts",
                                       "agreement", "complete"])
    table["flag"] = table["diff_pts"].abs() > TOLERANCE_PTS
    return table


def k1_verdict(table: pd.DataFrame) -> pd.DataFrame:
    """Gate full benchmark coverage; partial comparisons never establish harness validity."""
    rows = []
    for task, expected in (("belebele", 900), ("mgsm", 250)):
        g = table[table.task == task]
        complete = (len(g) == len(CODES) and set(g.lang) == set(CODES)
                    and g.n_items.eq(expected).all()
                    and "complete" in g and g["complete"].all())
        gap = float(g.diff_pts.abs().mean()) if len(g) else float("nan")
        verdict = "INCOMPLETE" if not complete else "PASS" if gap <= TOLERANCE_PTS else "FAIL"
        rows.append({"task": task, "languages": len(g), "mean_abs_diff_pts": gap,
                     "max_abs_diff_pts": float(g.diff_pts.abs().max()),
                     "mean_agreement": float(g.agreement.mean()), "K1": verdict})
    return pd.DataFrame(rows)
