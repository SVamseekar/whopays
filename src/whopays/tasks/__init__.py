"""Task registry.

A task is `fn(cfg, bundle, skip) -> Iterator[dict]`. `bundle` is whatever the model loader
returned for cfg; `skip` holds item IDs already written, which the task must not recompute.
Each yielded dict is one item's result and must contain a unique `item_id`. Items must come
in a fixed order so a resumed run reproduces the uninterrupted file (FR-8).

Task modules register themselves on import; `get_task("mgsm")` imports `whopays.tasks.mgsm`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from typing import Any

TaskFn = Callable[[Any, Any, set[str]], Iterator[dict[str, Any]]]

_REGISTRY: dict[str, TaskFn] = {}


def register(name: str) -> Callable[[TaskFn], TaskFn]:
    def deco(fn: TaskFn) -> TaskFn:
        _REGISTRY[name] = fn
        return fn

    return deco


def get_task(name: str) -> TaskFn:
    if name not in _REGISTRY:
        try:
            importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            raise KeyError(f"task {name!r} is not implemented (no module {exc.name})") from exc
    return _REGISTRY[name]
