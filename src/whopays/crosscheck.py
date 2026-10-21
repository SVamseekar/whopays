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
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from whopays.languages import LANGUAGES

LMEVAL_VERSION = "0.4.13"
TOLERANCE_PTS = 1.0
_BY_FLORES = {lang.flores: code for code, lang in LANGUAGES.items()}
_SAMPLES = re.compile(r"samples_(?P<task>.+)_\d{4}-\d{2}-\d{2}T[\d.\-]+\.jsonl$")
# (lm-eval filter, lm-eval metric, our column) per task
GATED = {"belebele": ("none", "acc", "correct"),
         "mgsm": ("flexible-extract", "exact_match", "correct_ascii")}


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


def run_lmeval(model: str, tasks: Iterable[str], langs: Iterable[str], out_dir: Path, *,
               batch_size: int = 8, limit: int | None = None,
               venv: Path = Path("/tmp/venv-lmeval")) -> Path:
    """Run lm-eval on every task × language; output and a log go to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    python = install(venv)
    names = [lmeval_task(t, lang) for t in tasks for lang in langs]
    log = out_dir / "lmeval.log"
    with log.open("a") as f:
        proc = subprocess.run([str(python), "-m", "lm_eval",
                               *lmeval_args(model, names, out_dir, batch_size=batch_size,
                                            limit=limit)],
                              stdout=f, stderr=subprocess.STDOUT,
                              env={**os.environ, "USE_TF": "0"})
    if proc.returncode != 0:
        raise RuntimeError(f"lm-eval failed (exit {proc.returncode}); tail of {log}:\n"
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
        both = mine[["lang", "item_id", col]].merge(
            ref[["lang", "item_id", "value"]], on=["lang", "item_id"], how="inner")
        for lang, g in both.groupby("lang", sort=True):
            a, b = g[col].astype(float), g["value"].astype(float)
            out.append({"task": task, "lang": lang, "n_items": len(g),
                        "ours": round(100 * a.mean(), 2), "lmeval": round(100 * b.mean(), 2),
                        "diff_pts": round(100 * (a.mean() - b.mean()), 2),
                        "agreement": round(float((a == b).mean()), 4)})
    table = pd.DataFrame(out, columns=["task", "lang", "n_items", "ours", "lmeval", "diff_pts",
                                       "agreement"])
    table["flag"] = table["diff_pts"].abs() > TOLERANCE_PTS
    return table


def k1_verdict(table: pd.DataFrame) -> pd.DataFrame:
    """Per task: mean |gap| over languages, and PASS when it is within TOLERANCE_PTS."""
    g = table.groupby("task")
    verdict = pd.DataFrame({
        "languages": g.size(),
        "mean_abs_diff_pts": g["diff_pts"].apply(lambda d: round(float(d.abs().mean()), 2)),
        "max_abs_diff_pts": g["diff_pts"].apply(lambda d: round(float(d.abs().max()), 2)),
        "mean_agreement": g["agreement"].mean().round(4),
    }).reset_index()
    verdict["K1"] = verdict["mean_abs_diff_pts"].le(TOLERANCE_PTS).map({True: "PASS",
                                                                        False: "FAIL"})
    return verdict
