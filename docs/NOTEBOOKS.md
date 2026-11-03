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
- **No GitHub at all (last resort; a commit install logs better provenance):** run `python scripts/build_notebooks.py --bundle`, upload the `.ipynb` plus `dist/whopays-*.whl` to Colab, and set `WHOPAYS_WHEEL` in the setup cell to the exact wheel path. Wheels are never picked up automatically.
  Current wheels embed `requirements.lock`; setup applies its constraints while retaining
  the platform scientific base, and logs a SHA-256 code identity. Rebuild an older wheel if
  setup reports that its embedded lock is missing.
  Restart the runtime before switching code builds; cached Python imports otherwise retain
  the old implementation. Setup rejects a changed build in an already imported kernel.

## Step by step: running a notebook on Colab

### A. One-time setup (do once; every later notebook reuses it)

**1. Hugging Face token (Write).** Results are saved to the private dataset `Heavy-Dervish1205/whopays-results`, which needs write access.
1. Sign in to https://huggingface.co as Heavy-Dervish1205.
2. Open https://huggingface.co/settings/tokens. If an existing token shows a **WRITE** badge, you can reuse it.
3. Otherwise, click **+ Create new token**, choose the **Write** tab, name it `colab-whopays` and click **Create token**.
4. Copy the value (`hf_...`). It is shown only once.
5. FLORES is gated. Open https://huggingface.co/datasets/facebook/flores and click **Agree and access repository**. Skip this if you already did it for `00_env_check`.

**2. GitHub token: not needed while the repo stays public (it is public as of 2026-11-03).** Skip this step. If the repo goes private again, create a read-only token as follows.
1. Open https://github.com/settings/personal-access-tokens/new. This is the "Fine-grained token" page.
2. **Token name:** `colab-whopays`. **Expiration:** 90 days, or whatever you prefer.
3. **Repository access:** select **Only select repositories**, then pick `SVamseekar/whopays`.
4. **Permissions → Repository permissions → Contents:** set it to **Read-only**. Metadata becomes read-only automatically. Leave everything else at "No access".
5. Click **Generate token** and copy the value (`github_pat_...`).

**3. Store both tokens in Colab.** Colab secrets belong to your Google account, so this is also a one-time step.
1. Open any notebook at https://colab.research.google.com.
2. Click the **key icon (Secrets)** in the left sidebar.
3. Click **+ Add new secret**, set the name to `HF_TOKEN` and paste the HF token as the value.
4. Do the same again with the name `GH_TOKEN` and the GitHub token.
5. Turn on the **Notebook access** toggle for both.

Never paste a token into a notebook cell.

### B. Each run

1. **Open the notebook from GitHub.** Use `https://colab.research.google.com/github/SVamseekar/whopays/blob/main/notebooks/<name>.ipynb`.
2. **Select the GPU.** Go to **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save**.
3. **Run it.** Go to **Runtime → Run all**. If Colab asks "Grant access to HF_TOKEN / GH_TOKEN?", click **Grant access**.
4. **Check the first cell's output.** It should print `platform: colab | GPU: Tesla T4` and then a version table. If it says "No CUDA GPU attached", redo step 2.
5. **Keep the tab open and the laptop awake.** Free Colab can end a session whose tab is closed. Long cells print a line per finished job; some steps, such as lm-eval, stay quiet for a long time while they work.
6. **If the session dies:** reopen it and do **Runtime → Run all** again. Finished jobs are skipped and resumed from the results dataset.
7. **When it finishes:** go to **File → Download → Download .ipynb**. The file lands in `~/Downloads`. Tell Claude Code; it reads the outputs and errors from there.

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

If model import fails with `operator torchvision::nms does not exist` or a torchaudio CUDA
version mismatch, an optional media package is broken in that runtime. For an existing
uploaded notebook, run `%pip uninstall -y torchvision torchaudio` in a code cell, restart
the session (not disconnect/delete), and Run all again with the same wheel. Current setup
probes both imports in subprocesses and removes failed companions together before
Transformers is loaded. Healthy companions and the platform CUDA torch are preserved.
The `No files matched allow_patterns`
message by itself only means that the requested run has no remote checkpoint yet.

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
