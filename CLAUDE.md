# CLAUDE.md — WhoPays + ParityQuant

Read this first, every session. Then read `docs/BUILD_PLAN.md` to find the next unfinished task.

## What this project is

This is a research codebase with two parts.

1. **WhoPays.** An audit of how LLM inference optimizations change quality and speed differently across 11 languages, with each gap attributed to a layer. The optimizations are weight quantization, KV-cache compression and speculative decoding.
2. **ParityQuant.** A mixed-precision quantizer that minimizes the *worst* language's degradation at a fixed average bit-width.

The full spec is `docs/PRD.md`. Research context and novelty boundaries are in `docs/RESEARCH_CONTEXT.md`. Decisions already made are in `docs/DECISIONS.md`; don't relitigate them without saying so.

Owner: Soura Vamseekar Marti (GitHub: SVamseekar; Hugging Face: Heavy-Dervish1205). He prefers brief, direct communication and expects you to make technical calls and flag gaps proactively.

## Where code runs

| Machine | What it does | What it must never do |
| --- | --- | --- |
| MacBook Air M1, 16 GB (where you, Claude Code, run) | Write code, run pytest on CPU, smoke-test Qwen2.5-0.5B on MPS, run all statistics, ParityQuant solver, figures | Produce any number that goes into a results table |
| Google Colab free, T4 16 GB | Short runs, debugging on real GPU | Run jobs longer than ~3 h unattended |
| Kaggle, T4 / T4×2 | The main experiment grid (background "Save & Run All", 12 h max) | — |

You cannot execute notebooks on a GPU. The user uploads or opens them in Colab or Kaggle and runs them there. Results come back through the Hugging Face dataset (and optionally Google Drive). You then analyse them locally.

## Golden rules

1. **All logic lives in `src/whopays/`.** Notebooks are thin runners: install the package, set config, call `whopays.run(...)`, show a summary. If you're writing more than ~15 lines in a notebook cell, it belongs in `src/`.
2. **Generate notebooks from code, never by hand-editing JSON.** Use `scripts/build_notebooks.py` (jupytext or nbformat) so `.ipynb` files are reproducible. Regenerate after any change.
3. **Every run is a YAML config with a stable hash** (FR-1). The results path is `results/<run_hash>/items.jsonl`.
4. **Everything is resumable** (FR-8). Assume the GPU session dies at any moment.
5. **T4 constraints:** FP16 only (no bf16), PyTorch SDPA (no FlashAttention-2), 15 GB usable VRAM, compute capability 7.5. Never assume an A100.
6. **M1 constraints:** use `device="mps"` only for the 0.5B smoke test. bitsandbytes, GPTQ kernels and CUDA extensions don't work here. Guard their imports and skip their tests unless CUDA is present.
7. **Fake quantization is the default.** Quantize then dequantize to FP16. Memory savings are computed from bit counts, never claimed as measured.
8. **Pin versions.** Keep `requirements.lock` in sync. Log `torch`, `transformers` and the GPU name in every result row.
9. **No secrets in code or notebooks.** Read `HF_TOKEN` from Colab `userdata`, Kaggle `UserSecretsClient` or an env var, through `whopays.env.get_secret()`.
10. **No numbers in docs or the README that weren't produced by a logged run.** Never invent results.
11. **Tests first for stats and quantization code.** Synthetic data with planted effects for stats; a tiny random Qwen2-architecture model for model code. CI runs on CPU.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11
pip install -e ".[dev]"                               # dev = pytest, ruff, jupytext, nbformat
pytest -q                                             # CPU tests, must pass before any commit
ruff check src tests
python scripts/build_notebooks.py                     # regenerate notebooks/*.ipynb
python -m whopays.cli smoke --device mps              # 0.5B, 20 Belebele items, local
```

## How a working session goes

1. Open `docs/BUILD_PLAN.md`, take the first unchecked task, and implement it with tests.
2. Run `pytest -q`. Then regenerate any notebook the task touches.
3. Tick the task. Append anything non-obvious you decided to `docs/DECISIONS.md` (dated, one paragraph).
4. If a task needs a GPU to verify, write the notebook, then say exactly what the user should run and what output to paste back.
5. Commit with a clear message. Push only when the user asks.

## When the user pastes notebook output back

- Compare it against the "Done when" check for that FR in `docs/PRD.md`.
- If it fails a kill criterion (K1–K4 in the PRD), say so plainly and propose the pivot the PRD defines.
- Update `docs/BUILD_PLAN.md` status and `docs/DECISIONS.md`.
