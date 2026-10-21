"""Job lists for each experiment, built in code so notebooks stay one-liners.

Each function returns plain dicts for `whopays.run()`. `lang: "all"` expands to the 11
languages there.
"""

from __future__ import annotations

from collections.abc import Sequence

QWEN = {size: f"Qwen/Qwen2.5-{size}-Instruct" for size in ("0.5B", "1.5B", "3B", "7B")}
K1_MODEL = QWEN["1.5B"]
M0_MODELS = (QWEN["1.5B"], QWEN["0.5B"], QWEN["3B"])  # K1 model first: it gates everything
M0_TASKS = ("belebele", "mgsm", "flores_nll")


def m0_fp16(models: Sequence[str] = M0_MODELS, tasks: Sequence[str] = M0_TASKS,
            limit: int | None = None) -> list[dict]:
    """FP16 baselines (BUILD_PLAN M0): every model × task × language, full cache, no draft."""
    return [{"model": m, "task": t, "lang": "all", "weight": "fp16", "kv": "full",
             "limit": limit} for m in models for t in tasks]
