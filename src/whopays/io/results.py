"""Per-item results: append-only JSONL, periodic flushes, resume, Parquet (FR-7, FR-8).

Layout of one run folder, results/<run_id>/:
    config.yaml   the RunConfig
    meta.json     env_info() at first start, plus every resume
    items.jsonl   one JSON object per item, keys sorted
    items.parquet written when the run completes
    DONE          written last; its presence means the run is complete

A session can die at any moment. Rows are buffered and appended in whole-line batches, so the
file only ever holds complete rows, except that a kill during a write can leave a partial last
line. Opening a writer truncates that partial line. Resuming then reproduces the uninterrupted
file byte for byte, provided the task yields items in a fixed order.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd

ITEMS = "items.jsonl"
PARQUET = "items.parquet"
DONE = "DONE"


def dumps_row(row: dict[str, Any]) -> str:
    # allow_nan=False: NaN is not valid JSON; tasks must emit None for missing values.
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


def repair(path: Path) -> int:
    """Truncate a partial trailing line left by a killed write. Returns bytes removed."""
    if not path.exists() or path.stat().st_size == 0:
        return 0
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return 0
    keep = data.rfind(b"\n") + 1  # 0 when no complete line exists
    with path.open("r+b") as f:
        f.truncate(keep)
    return len(data) - keep


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class ResultsWriter:
    """Append rows to items.jsonl, flushing every `flush_every` rows and on close.

    `on_flush` runs after each flush with the file path (used for throttled hub sync).
    """

    def __init__(
        self,
        path: Path,
        flush_every: int = 50,
        on_flush: Callable[[Path], None] | None = None,
    ):
        self.path = Path(path)
        self.flush_every = flush_every
        self.on_flush = on_flush
        self.path.parent.mkdir(parents=True, exist_ok=True)
        repair(self.path)
        self._done = {row["item_id"] for row in read_rows(self.path)}
        self._buffer: list[str] = []

    def done_ids(self) -> set[str]:
        """Item IDs already on disk or buffered; tasks skip these on resume."""
        return set(self._done)

    def write(self, row: dict[str, Any]) -> None:
        item_id = row["item_id"]
        if item_id in self._done:
            raise ValueError(f"item {item_id!r} written twice; the task ignored done_ids")
        self._buffer.append(dumps_row(row) + "\n")
        self._done.add(item_id)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write("".join(self._buffer))
            f.flush()
            os.fsync(f.fileno())
        self._buffer.clear()
        if self.on_flush:
            self.on_flush(self.path)

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> ResultsWriter:
        return self

    def __exit__(self, *exc) -> None:
        # Flush on errors too: buffered rows are finished items, and keeping them saves GPU time.
        self.close()


def to_parquet(jsonl: Path, parquet: Path | None = None) -> Path:
    jsonl = Path(jsonl)
    parquet = parquet or jsonl.with_name(PARQUET)
    pd.DataFrame.from_records(read_rows(jsonl)).to_parquet(parquet, index=False)
    return parquet


def load_runs(root: Path, configs: Iterable[Any]) -> pd.DataFrame:
    """Rows of the given runs, finished or partial, tagged with model, weight and kv.

    Rows carry run_id, task and lang but not the model, so it comes from each RunConfig. Runs
    with no folder yet are skipped.
    """
    frames = []
    for cfg in configs:
        run_dir = Path(root) / cfg.run_id
        if (run_dir / PARQUET).exists():
            df = pd.read_parquet(run_dir / PARQUET)
        elif (run_dir / ITEMS).exists():
            df = pd.DataFrame.from_records(read_rows(run_dir / ITEMS))
        else:
            continue
        from whopays.config import canonical_json

        fields = cfg.model_dump(mode="json", exclude={"tag"})
        fields["bitmap"] = canonical_json(cfg.bitmap) if cfg.bitmap is not None else None
        fields["config_json"] = canonical_json(cfg.canonical())
        # Existing logged provenance wins over caller execution preferences.
        frames.append(df.assign(**{k: v for k, v in fields.items() if k not in df}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def is_done(run_dir: Path) -> bool:
    """Validate completion artifacts before accepting a DONE marker; fail closed on damage."""
    run_dir = Path(run_dir)
    if not (run_dir / DONE).exists():
        return False
    required = (ITEMS, PARQUET, "config.yaml", "meta.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"incomplete DONE run {run_dir.name}: missing {missing}")
    n = json.loads((run_dir / DONE).read_text())["n_items"]
    rows = read_rows(run_dir / ITEMS)
    frame = pd.read_parquet(run_dir / PARQUET)
    ids = [r["item_id"] for r in rows]
    if (n <= 0 or len(ids) != n or len(set(ids)) != n or len(frame) != n
            or "item_id" not in frame or frame.item_id.tolist() != ids):
        raise ValueError(f"incomplete or inconsistent DONE run {run_dir.name}")
    return True


def mark_done(run_dir: Path, n_items: int) -> None:
    (Path(run_dir) / DONE).write_text(json.dumps({"n_items": n_items}) + "\n")
