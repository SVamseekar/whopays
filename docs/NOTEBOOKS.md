# Notebooks and the Colab / Kaggle workflow

## The workflow

```
M1 (Claude Code)                 GitHub                    Colab / Kaggle (T4)
  write src/ + tests  ── push ──►  SVamseekar/whopays  ──►  notebook: pip install git+...@<commit>
  build_notebooks.py                                         run jobs, resumable
       ▲                                                          │
       └──── pull Parquet ◄──── HF dataset (private) ◄── push ────┘
```

Why this works better than uploading `.ipynb` files with code inside them:

- **One source of truth.** Code is in the repo, not copied into notebook cells that drift.
- **No re-uploads.** Open a notebook straight from GitHub at `https://colab.research.google.com/github/SVamseekar/whopays/blob/main/notebooks/<name>.ipynb`. A fix only needs a push and a refresh.
- **Pinned results.** The notebook installs a pinned commit, so every result row records exactly which code produced it.

**Fallback before the repo exists, or while it's private:**

- **Private repo:** use a fine-grained GitHub token stored in Colab Secrets as `GH_TOKEN`. The setup cell handles it.
- **No GitHub at all:** run `python scripts/build_notebooks.py --bundle`, upload the `.ipynb` plus `dist/whopays-*.whl` to Colab, and `pip install` the wheel.

## Every notebook starts with the same setup cell (generated, don't hand-edit)

1. Detect the platform: Colab, Kaggle or local.
2. Check for a GPU and print its name. Stop with a clear message if no CUDA device is attached ("Runtime → Change runtime type → T4 GPU").
3. Install the pinned repo commit plus `requirements.lock`.
4. Read `HF_TOKEN` (and optionally `GH_TOKEN`) from secrets.
   - **Colab:** the key icon in the left sidebar → add the secret → toggle notebook access.
   - **Kaggle:** Add-ons → Secrets.
5. On Colab, optionally mount Google Drive as a second results mirror.
6. Print the versions of torch, transformers and CUDA, and the GPU name.

## Notebook set

| Notebook | Runs on | Purpose | Milestone |
| --- | --- | --- | --- |
| `00_env_check.ipynb` | Colab T4 | Answers the week-1 open questions: GPTQModel, QuantizedCache, kvpress and the Qwen draft/target vocab check. Prints a pass/fail table | M0 |
| `01_fp16_baselines.ipynb` | Colab or Kaggle T4 | FP16 Belebele + MGSM + FLORES NLL for 0.5B/1.5B/3B; lm-eval cross-check (K1 gate); measures throughput | M0 |
| `02_replicate_uniform.ipynb` | Kaggle | Uniform W8/W4/W3 × calibration language × 3 seeds on 1.5B | M1 |
| `03_sensitivity_probe.ipynb` | Kaggle | Per-layer × bits × language NLL profile on 1.5B (K2 gate) | M1 |
| `04_audit_grid.ipynb` | Kaggle (background) | Takes a job list (a slice of the 12-cell grid) and runs it resumably | M2 |
| `05_speed_grid.ipynb` | Kaggle | Speculative on/off over the 12 cells, MGSM subset, batch size 1 | M2 |
| `06_parityquant_eval.ipynb` | Kaggle | Evaluates bit maps produced locally by `parityquant/allocate.py` | M4 |
| `07_scale_7b.ipynb` | Kaggle T4×2 | 7B confirmation | M5 |
| `analysis.ipynb` | M1, CPU | Pull Parquet, stats, figures. Never needs a GPU | M3+ |

## Colab rules

- **Runtime:** T4 GPU. Never TPU, because the code is PyTorch + CUDA.
- **Session length:** free Colab disconnects after idle time and caps sessions around 12 h. Keep Colab jobs under about 3 h and put long grids on Kaggle.
- **Losing a session:** if it dies, re-run all cells. The resumable writer skips finished items.
- **Notebook size:** keep outputs small. Print summaries, not per-item logs.

## Kaggle rules

- **GPU access:** phone verification is required once.
- **Accelerator:** "GPU T4 x2" (more memory headroom) or "GPU P100".
- **Internet:** Settings → Internet on, so pip and HF downloads work.
- **Long jobs:** "Save Version → Save & Run All (Commit)". It runs in the background for up to 12 h and doesn't need the browser open.
- **Quota:** check the weekly counter before launching a grid slice.
