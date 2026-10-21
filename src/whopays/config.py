"""Run configuration and its stable hash (FR-1).

A run is one (model setup, task, language) cell. Its ID is a hash of the canonical JSON of the
fields that can change results. Fields that cannot affect a given cell (the seed of an FP16 run,
the eviction budget of a full cache) are dropped before hashing, so equivalent configs share
one ID and one results folder.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from whopays.languages import CODES

# Bump when canonicalisation or field semantics change; this deliberately orphans old run IDs.
SCHEMA_VERSION = 1
DEFAULT_DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
ALLOWED_BITS = {2, 3, 4, 8}

Weight = Literal["fp16", "w8_rtn", "w4_gptq", "w3_gptq", "bitmap", "prequant"]
Calib = Literal["none", "en", "multi"]
KVPolicy = Literal["full", "quant4", "evict"]
Task = Literal["belebele", "mgsm", "flores_nll"]

CALIBRATED: set[str] = {"w4_gptq", "w3_gptq", "bitmap"}
GROUPED: set[str] = {"w8_rtn", "w4_gptq", "w3_gptq", "bitmap"}
# Execution details: never part of the hash, so a smaller batch on a smaller GPU resumes the run.
EXEC_FIELDS = {"batch_size", "tag"}


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    task: Task
    lang: str

    weight: Weight = "fp16"
    group_size: int = Field(128, gt=0)
    calib: Calib = "none"
    seed: int = Field(0, ge=0)
    bitmap: dict[str, int] | None = None  # layer name -> bits, for weight="bitmap"
    budget: float | None = None  # average bits the bitmap was solved for (provenance)

    kv: KVPolicy = "full"
    kv_budget: int = Field(256, gt=0)

    spec: bool = False
    draft_model: str | None = None
    spec_k: int = Field(4, gt=0)

    limit: int | None = Field(None, ge=1)  # first N items only (smoke tests)

    batch_size: int = Field(8, gt=0)
    tag: str = ""

    @field_validator("lang")
    @classmethod
    def _known_lang(cls, v: str) -> str:
        if v not in CODES:
            raise ValueError(f"unknown language {v!r}; expected one of {CODES}")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> RunConfig:
        if self.weight in CALIBRATED and self.calib == "none":
            raise ValueError(f"weight={self.weight} needs calib='en' or 'multi'")
        if self.weight not in CALIBRATED and self.calib != "none":
            raise ValueError(f"weight={self.weight} takes no calibration set")
        if (self.weight == "bitmap") != (self.bitmap is not None):
            raise ValueError("bitmap is required for weight='bitmap' and only allowed there")
        if self.bitmap and not set(self.bitmap.values()) <= ALLOWED_BITS:
            raise ValueError(f"bitmap bits must be in {sorted(ALLOWED_BITS)}")
        if self.spec and self.draft_model is None:
            object.__setattr__(self, "draft_model", DEFAULT_DRAFT)
        return self

    def canonical(self) -> dict[str, Any]:
        """Fields that can change this cell's results, with irrelevant ones removed."""
        d = self.model_dump(mode="json", exclude=EXEC_FIELDS)
        d["schema_version"] = SCHEMA_VERSION
        if self.weight not in GROUPED:
            del d["group_size"]
        if self.weight not in CALIBRATED:
            del d["seed"]
        if self.weight != "bitmap":
            del d["bitmap"], d["budget"]
        if self.kv != "evict":
            del d["kv_budget"]
        if not self.spec:
            del d["draft_model"], d["spec_k"]
        return d

    @property
    def run_id(self) -> str:
        return stable_hash(self.canonical())

    def model_key(self) -> str:
        """Identity of the loaded weights (plus draft), shared across tasks, languages, KV."""
        d = self.canonical()
        keep = {"schema_version", "model", "weight", "group_size", "calib", "seed", "bitmap",
                "draft_model"}
        return stable_hash({k: v for k, v in d.items() if k in keep})

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True, allow_unicode=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        return cls(**yaml.safe_load(Path(path).read_text()))


def _normalise(obj: Any) -> Any:
    """Make numbers compare by value: 4.0 -> 4, 0.30000000000000004 -> 0.3."""
    if isinstance(obj, dict):
        return {str(k): _normalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalise(v) for v in obj]
    if isinstance(obj, float):
        obj = float(f"{obj:.12g}")
        return int(obj) if obj.is_integer() else obj
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(_normalise(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()[:12]


def load_jobs(source: str | Path | Iterable[RunConfig | dict]) -> list[RunConfig]:
    """Load a job list from YAML (`defaults:` + `jobs:`) or from configs / dicts.

    In YAML, `lang` may be a list or `all`, which expands to one job per language.
    """
    if isinstance(source, (str, Path)):
        doc = yaml.safe_load(Path(source).read_text())
        defaults, entries = doc.get("defaults", {}), doc["jobs"]
    else:
        defaults, entries = {}, list(source)
    jobs: list[RunConfig] = []
    for entry in entries:
        if isinstance(entry, RunConfig):
            jobs.append(entry)
            continue
        merged = {**defaults, **entry}
        langs = merged.pop("lang")
        langs = list(CODES) if langs == "all" else langs if isinstance(langs, list) else [langs]
        jobs.extend(RunConfig(**merged, lang=lang) for lang in langs)
    return jobs
