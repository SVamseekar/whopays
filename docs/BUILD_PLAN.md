# Build plan

This is an ordered task list for Claude Code. Take the first unchecked task, finish it with tests, and tick it. Each task cites the PRD requirement it implements. "GPU check" means the user has to run a notebook and paste back the output.

**31 Oct 2026 reconciliation:** software existed beyond the checked milestones in the
starting working tree. See [the repository/research review](REVIEW_2026-10-31.md).
CPU implementation checks below do not complete GPU acceptance gates. K1 passed on
2026-11-06; the compute re-plan is now the first unfinished research task. No downstream
experiment is authorized by a passing unit test alone.

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
- [x] Notebook `00_env_check` (built 2026-10-21; 4 Colab runs; all 12 checks pass on the fourth, 2026-10-21. Answers in `DECISIONS.md`).
- [x] Notebook `01_fp16_baselines` (built 2026-10-21; **K1 PASS on Kaggle T4, 2026-11-06**, commit `200cea5`; see DECISIONS 2026-11-06). **GPU check, K1 gate:** FP16 within 1 point of lm-evaluation-harness, plus measured throughput. Re-plan the compute budget if throughput is more than 2× slower than assumed.
  - [x] CPU reliability follow-up: stop on baseline errors, checkpoint consistent live
    lm-eval SQLite caches, compare the unrounded throughput ratio; regenerate notebook 01.
  - Latest Colab T4 attempt stopped in setup because the wheel form value contained
    literal quotes; tolerate pasted quotes and clarify the form hint. No K1 scores were
    produced, so the gate remains untested. `HF_TOKEN` was also absent in that attempt.
  - Subsequent attempt reached model loading, then failed on `torchvision::nms` in an
    optional vision import. No baseline items were scored. Add a text-runtime import
    probe/removal for broken torchvision; retry K1 after restarting the Colab session.
  - After removing torchvision, model import reached torchaudio and failed on a CUDA
    build mismatch (torch 13.2 versus torchaudio 13.0). Extend the probe to both optional
    companions together. K1 score agreement and throughput remain untested.
  - 31 Oct T4 run, read from the results dataset on 3 Nov: all 33 FP16 baselines for 1.5B
    finished (Belebele 900, MGSM 250, FLORES 1,012 items per language), then lm-eval died
    with CUDA OOM at 32 of 39,600 requests. K1 is untested, not failed.
  - [x] CPU fix: lm-eval per task (Belebele batch 2, MGSM batch 8); throughput prints before
    lm-eval; a rejected HF token raises. **Next GPU check:** re-run notebook 01. Finished
    runs skip, so only throughput and lm-eval execute.
  - Start-stamp estimates (not the gate figure): MGSM about 76 min against 30 assumed;
    Belebele about 30 min against 15. The gate uses summed `wall_time_s`; if it exceeds 2×,
    re-plan the compute budget before M2.
  - 2026-11-06 Kaggle T4 run: K1 PASS. Belebele mean |ours − lm-eval| 0.11 points (max 0.33),
    MGSM 0.80 (max 1.60, German). Logged throughput: Belebele 18.7 GPU-min (1.25× assumed),
    MGSM 68.1 (2.27× assumed, so `replan` is true), FLORES 7.2. The re-plan is the next task.

## M1 — Replicate + probe (weeks 3–4)
- [x] `quant/rtn.py` and `quant/fakequant.py`: group-wise asymmetric RTN with a per-layer bit map. Unit-test the reconstruction error bounds. CPU verified 2026-10-31.
- [x] `quant/gptq.py`: minimal GPTQ (Hessian from calibration activations, damping 0.01, group 128). Test it on a tiny model so that GPTQ error is below RTN error. CPU verified 2026-10-31; FR-3 GPU reference comparison remains open.
- [x] Calibration sampler: English-only and multilingual, with a seed. CPU verified 2026-10-31 on FLORES dev; replaced 2026-11-07 by 128 × 2,048-token Wikipedia windows (fourth amendment).
- [x] Loader: a fake-quant mode driven by a bit map. CPU verified 2026-10-31.
- [ ] Notebook `02_replicate_uniform`. **GPU check:** FR-3 is within 2% of a GPTQModel W4 checkpoint, and the per-language gap goes in the published direction.
  - [x] CPU: notebook built; `replicate.py` runs the checkpoint in an isolated venv and computes both checks; `gates.require_k1` blocks it until K1 passes. GPU evidence is absent.
  - 2026-11-06 Kaggle run (all 308 runs, 7.4 h): FR-3 FAIL against Qwen's official GPTQ-Int4,
    mean 7.5% (Russian 25.8%). Cause: 128 single FLORES sentences gave a few thousand calibration
    tokens, fewer than the widest layer's 8,960 inputs. Gap direction mixed (DECISIONS 2026-11-07).
  - [x] CPU fix (fourth amendment): 128 × 2,048-token Wikipedia calibration, GPTQModel on the same
    windows as the primary FR-3 reference, layer-by-layer GPTQ pass. All GPTQ runs re-run; FP16
    and W8 runs keep their IDs. **Next GPU check:** re-run notebook 02; FR-3 now runs first and
    stops the notebook on failure.
- [ ] `parityquant/profile.py` and notebook `03_sensitivity_probe`. **GPU check, K2 gate:** minimum pairwise Spearman correlation of layer-sensitivity vectors across languages ≤ 0.9.
  - [x] CPU: profile resume rejects a different code version, GPU, torch, transformers, CUDA build or model identity. K2 diversity and timing remain open.
  - [x] CPU: notebook built; `parityquant/probe.py` runs a resumable profile, decides K2 at 4 bits and reports split-half reliability. A full profile would take about 33 h, so the default is 100 sentences per language (DECISIONS 2026-11-03).

## M2 — Audit grid (weeks 5–8)
- [x] Write and commit `PREREGISTRATION.md` (dated) from PRD §Research questions and §Statistical protocol. Prospective draft amended and committed 2026-10-31, before any M2 run. Remaining H2 inference and fractional-budget comparator specifications are explicitly open in that file.
- [ ] `kv/policies.py`: full, 4-bit quantized cache, fixed 256-token eviction, behind one interface (FR-4).
  - [x] Implement interface, eviction, residual-only rollback and cache storage accounting;
    CPU tests including absolute RoPE positions. Actual HQQ/T4 acceptance remains open.
- [ ] `decoding/speculative.py`: greedy draft-verify, k = 4, logging. The exactness test against plain greedy is mandatory (FR-5).
  - [x] CPU exactness, rejection/EOS/stop/boundary tests and MGSM integration. Full cache
    uses block verification; compressed caches use a labelled sequential correctness control.
  - [x] CPU: exact block verification for eviction and quantized caches, off behind `COMPRESSED_BLOCK_VERIFIED`; gate code in `decoding/exactness.py`.
  - [ ] Multilingual T4 exactness gate (notebook 05), then set the flag.
- [ ] `configs/` generator for the 12-cell × 2-model grid, plus job-list slices sized at ≤ 10 h each.
  - [x] CPU: `whopays.cli grid` writes the job lists; `whopays.cli plan` projects GPU-hours and writes slices from logged baselines.
  - [x] Audit cut and slices (2026-11-06): calibration and seed replication on 1.5B only (third preregistration amendment); 858 runs, about 61 projected GPU-hours; `configs/audit_slice_00`–`06`. GPTQ load time is unmeasured (DECISIONS 2026-11-06).
- [x] Notebooks `04_audit_grid` and `05_speed_grid` built. CPU only; each needs a K1 pass to run.

## M3 — Decomposition (weeks 9–10, M1 only)
- [ ] `stats/bootstrap.py`, `stats/mixed.py`, `stats/shapley.py`, `stats/oaxaca.py`, `stats/inequality.py`, each tested on synthetic planted effects (FR-10, FR-11).
  - [x] CPU estimator tests and Oaxaca group-identification guard. Complete H2 interaction
    inference before claiming FR-10: variational credible intervals do not supply BH p-values.
  - [x] H2 p-values: language-specific weight × KV Wald tests from an item-clustered binomial GEE, with BH limited to that family. The variational fit stays descriptive. GPU evidence remains absent.
- [x] `analysis.ipynb` and `report/figures.py`. CPU-tested on synthetic rows; H1-H4 tables come from `report/hypotheses.py`.
- [ ] Draft the audit preprint.
  - [x] Methods draft without results: `docs/preprint/AUDIT_DRAFT.md`. Results sections wait for M2.

## M4 — ParityQuant (weeks 11–14)
- [ ] `parityquant/allocate.py`: the minimax MILP (PuLP + CBC), the average objective, uniform and soft-max variants. Test that the budget is met exactly.
  - [x] CPU: `matched_uniform`, the sensitivity-free exact-budget comparator used when the target is not an allowed width. K3 remains open.
  - [x] CPU: `parityquant/plan.py` and `whopays.cli parityquant` solve 4 methods × 3 budgets, write job lists and compute H4; a K2 failure stops them.
- [ ] Notebook `06_parityquant_eval`. **GPU check, K3 gate.**
  - [x] CPU: notebook built. Needs K1 and K2 passes, then bit maps solved locally.
- [ ] Optional: `parityquant/refine.py` (FR-14). Not built; only useful after a positive K3 (DECISIONS 2026-11-03).

## M5 — Scale + release (weeks 15–16)
- [ ] Notebook `07_scale_7b`.
  - [x] CPU: notebook built; 7B loads across T4×2. GPU run is exploratory and needs a K1 pass.
- [ ] `report/card.py` (FR-16) and `space/app.py` (FR-15).
  - [x] CPU: both exist and card tests pass. FR-15's 10-second load needs the released Parquet.
- [ ] Release checklist: every figure regenerates from released Parquet in under 10 minutes on the M1.
  - [x] CPU: `scripts/release_check.py` measures it. Needs the released Parquet to pass.
