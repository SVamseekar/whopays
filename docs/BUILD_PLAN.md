# Build plan

This is an ordered task list for Claude Code. Take the first unchecked task, finish it with tests, and tick it. Each task cites the PRD requirement it implements. "GPU check" means the user has to run a notebook and paste back the output.

## M0 — Harness (weeks 1–2, Oct 26 – Nov 8)

### Project skeleton
- [x] `pyproject.toml` (package `whopays`, Python 3.11), with extras `dev`, `gpu`, `stats`, `space`. Add `requirements.lock`, `.gitignore`, Apache-2.0 `LICENSE` and `README.md` (short, no results).
- [x] `src/whopays/env.py`: `platform()` returning colab/kaggle/local, `device()` returning cuda/mps/cpu, `get_secret(name)`, `results_root()`. Tests mock each platform.
- [x] GitHub Actions workflow running ruff + pytest on CPU.
- [x] `scripts/build_notebooks.py`: builds `notebooks/*.ipynb` from `notebooks/src/*.py` (jupytext percent format) and prepends the shared setup cell (see `docs/NOTEBOOKS.md`). Add a `--bundle` flag that also builds a wheel.

### Config and results (FR-1, FR-7, FR-8)
- [x] `config.py`: a pydantic (or dataclass) `RunConfig` and `stable_hash()` over canonical JSON. Test that hashes are identical across key order and float formatting.
- [x] `io/results.py`: JSONL writer that flushes every 50 items, `done_ids()` for resume, and `to_parquet()`. Test: kill mid-run, resume, and check the final file is byte-identical to an uninterrupted run.
- [x] `io/hub_sync.py`: push and pull a run folder to a private HF dataset `Heavy-Dervish1205/whopays-results`, with a no-op when there is no token.
- [x] `run.py`: `run(job_list)` loads configs, skips finished runs, dispatches to tasks and syncs results.

### Models and tasks (FR-2, FR-6, FR-9)
- [x] `models/loader.py`: FP16 loading only (other modes come in M1). Use `torch_dtype=float16` on CUDA, float32 on CPU and float16 on MPS.
- [x] `tasks/belebele.py`: prompt template, scoring with A/B/C/D letter logits, and a gold-answer logprob. Pin the dataset revision. Include a calibrated-accuracy option.
- [x] `tasks/mgsm.py`: greedy generation, final-number extraction, and an extraction-failure flag per item.
- [x] `tasks/flores_nll.py`: per-token NLL on devtest.
- [x] `tokens/premium.py`: token premium table for 11 languages on Belebele passages.
- [x] `cli.py smoke`: 0.5B on MPS, 20 Belebele items × 2 languages, under 3 minutes on the M1.
- [ ] Notebook `00_env_check` (built 2026-10-21; awaiting the Colab run). **GPU check:** the user pastes the pass/fail table; record the answers in `DECISIONS.md`.
- [ ] Notebook `01_fp16_baselines`. **GPU check, K1 gate:** FP16 within 1 point of lm-evaluation-harness, plus measured throughput. Re-plan the compute budget if throughput is more than 2× slower than assumed.

## M1 — Replicate + probe (weeks 3–4)
- [ ] `quant/rtn.py` and `quant/fakequant.py`: group-wise asymmetric RTN with a per-layer bit map. Unit-test the reconstruction error bounds.
- [ ] `quant/gptq.py`: minimal GPTQ (Hessian from calibration activations, damping 0.01, group 128). Test it on a tiny model so that GPTQ error is below RTN error.
- [ ] Calibration sampler: English-only and multilingual FLORES dev, with a seed.
- [ ] Loader: a fake-quant mode driven by a bit map.
- [ ] Notebook `02_replicate_uniform`. **GPU check:** FR-3 is within 2% of a GPTQModel W4 checkpoint, and the per-language gap goes in the published direction.
- [ ] `parityquant/profile.py` and notebook `03_sensitivity_probe`. **GPU check, K2 gate:** minimum pairwise Spearman correlation of layer-sensitivity vectors across languages ≤ 0.9.

## M2 — Audit grid (weeks 5–8)
- [ ] Write and commit `PREREGISTRATION.md` (dated) from PRD §Research questions and §Statistical protocol. **Do this before any M2 run.**
- [ ] `kv/policies.py`: full, 4-bit quantized cache, fixed 256-token eviction, behind one interface (FR-4).
- [ ] `decoding/speculative.py`: greedy draft-verify, k = 4, logging. The exactness test against plain greedy is mandatory (FR-5).
- [ ] `configs/` generator for the 12-cell × 2-model grid, plus job-list slices sized at ≤ 10 h each.
- [ ] Notebooks `04_audit_grid` and `05_speed_grid`.

## M3 — Decomposition (weeks 9–10, M1 only)
- [ ] `stats/bootstrap.py`, `stats/mixed.py`, `stats/shapley.py`, `stats/oaxaca.py`, `stats/inequality.py`, each tested on synthetic planted effects (FR-10, FR-11).
- [ ] `analysis.ipynb` and `report/figures.py`.
- [ ] Draft the audit preprint.

## M4 — ParityQuant (weeks 11–14)
- [ ] `parityquant/allocate.py`: the minimax MILP (PuLP + CBC), the average objective, uniform and soft-max variants. Test that the budget is met exactly.
- [ ] Notebook `06_parityquant_eval`. **GPU check, K3 gate.**
- [ ] Optional: `parityquant/refine.py` (FR-14).

## M5 — Scale + release (weeks 15–16)
- [ ] Notebook `07_scale_7b`.
- [ ] `report/card.py` (FR-16) and `space/app.py` (FR-15).
- [ ] Release checklist: every figure regenerates from released Parquet in under 10 minutes on the M1.
