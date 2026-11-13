# Run log

This is the timeline of every GPU run and of every check whose numbers someone might cite. It runs oldest first: append new entries at the bottom. The **why** behind each result lives in `DECISIONS.md`, and task status lives in `BUILD_PLAN.md`. This file only records **what ran, where, at which commit, what it wrote to the results dataset, and what came out**.

**Rules**
- Write one entry per session, including sessions that failed, were stopped or were superseded. If a run produced no result rows, say so.
- Give times in UTC. Session times come from the run folders' `meta.json` start stamps, Kaggle's `lastRunTime` or the log's elapsed seconds. If an entry uses another source, name it.
- Copy numbers only from a logged run or from the results dataset. Never write numbers from the M1 here except as plumbing checks, and label them as such.
- To refresh the dataset snapshot at the end of this file, run `python scripts/run_inventory.py`.
- Superseded runs stay in the dataset with their own run IDs. Mark them superseded here, never delete them.

## Summary

| # | Date (UTC) | Platform | Notebook / job | Commit | Outcome | Rows written |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-10-21 | Colab T4 | 00 env check, 4 runs | `10f9bee` → `187dd94` | 12/12 checks pass on run 4 | none |
| 2 | 2026-10-28 | M1 (local) | `cli smoke` | `a9e2ae5` | plumbing check only | 2 runs (0.5B, 20 items), not results |
| 3 | ≤ 2026-10-31 | Colab T4 | 01 FP16 baselines, 3 failed setups | various | quoted wheel path; `torchvision::nms`; torchaudio CUDA mismatch | none |
| 4 | 2026-10-31 | Colab T4 | 01 FP16 baselines | wheel (unmapped) | 33 FP16 baselines done; lm-eval OOM | 33 runs |
| 5 | 2026-11-06 | Kaggle T4 | 01, first session | before `200cea5` | `platform: colab`, no `HF_TOKEN` | none |
| 6 | 2026-11-06 | Kaggle T4×2 | 01 K1 rerun | `200cea5` | **K1 PASS** | lm-eval outputs |
| 7 | 2026-11-06 | Kaggle T4×2 | 02 replication | `dd7d538` | **FR-3 FAIL** (vs official checkpoint); superseded | 297 runs |
| 8 | 2026-11-07 | Kaggle T4×2 | 02 replication | `ee25c07` | **FR-3 FAIL** (vs GPTQModel, recipe 1); reference superseded | 22 runs |
| 9 | 2026-11-12 | Modal T4 | FR-3 check | `6e01035` | **FR-3 PASS** | 11 runs |
| 10 | 2026-11-12 | Modal T4 | Belebele prefill reuse; MGSM batch-1 timings | `b820e74` | reuse exact; timings for planning | none |
| 11 | 2026-11-12 | Modal T4 | 02 replication grid | `b820e74` | **stopped after 1.4 h**; 112 done, 6 partial | 118 runs |
| 12 | 2026-11-12 | Kaggle T4×2 | 03 sensitivity probe | `b820e74` | CUDA OOM in first baseline pass | none |
| 13 | 2026-11-12 | Kaggle T4×2 | 03 sensitivity probe | `974c16a` | **K2 PASS, noise-limited** | probe profile, 2,464 rows |

## Entries

### 1. 2026-10-21, Colab T4: env check (notebook 00), runs 1–4
- Environment: Python 3.13.15, torch 2.11.0+cu130, CUDA 13.0, transformers 5.19.0, T4 (sm75, 14.6 GB).
- Run 1 (fixed in `eab7eff`): two failures. The quantized-cache crop does not rewind `cumulative_length`, and venvs could not be created because there is no ensurepip.
- Run 2 (`24052ec`): HQQ cache works and the patched crop is exact. Quanto's CUDA extension does not build. kvpress fails on an old `fire`, and GPTQModel's Marlin kernel fails on sm75.
- Run 3 (`3492f80`): GPTQModel 7.6.0 with `BACKEND.GPTQ_TORCH` loads the GPTQ-Int4 checkpoint in 31 s. kvpress fails at model load.
- Run 4 (`187dd94`): **all 12 checks pass**. Assisted generation matches greedy on 10/10 prompts. Two-pass scoring max logit difference is 0.027.
- Rows written: none. Details: DECISIONS 2026-10-21, "Env check, first Colab T4 run".

### 2. 2026-10-28 11:31, M1: `cli smoke`
- 0.5B, Belebele en and te, 20 items each, FP16, run IDs `88856f7d7524` and `1b5c168dbc55` (tag `smoke`), Python 3.11.15, torch 2.14.1.
- This is a plumbing check, not a result (CLAUDE.md: the M1 never produces table numbers).

### 3. Before 2026-10-31, Colab T4: notebook 01, three failed setups
- Attempt A: the wheel path form value held literal quotes. Fixed in `b617c50`.
- Attempt B: an optional import hit `torchvision::nms`. Fixed by a text-runtime import probe.
- Attempt C: a torchaudio CUDA build mismatch (13.2 vs 13.0). The probe now checks both companions.
- No items were scored. Exact dates were not logged. Details: BUILD_PLAN notebook 01 notes; DECISIONS 2026-10-31.

### 4. 2026-10-31 09:34–11:47, Colab T4: notebook 01
- 33 FP16 baselines for 1.5B (Belebele 900, MGSM 250, FLORES 1,012 items per language × 11 languages). Python 3.13.16, torch 2.15.0+cu132.
- `code_version` is recorded as the wheel content identity, not a commit (DECISIONS 2026-11-03, baseline provenance).
- lm-eval then hit CUDA OOM at 32 of 39,600 requests, so K1 was untested.
- These 33 runs are the FP16 reference reused by every later notebook.

### 5. 2026-11-06, Kaggle T4×2: notebook 01, first session
- The session printed `platform: colab` and then `secret HF_TOKEN not available (TimeoutException)`. Kaggle sets `COLAB_RELEASE_TAG`.
- Fixed in `200cea5`, which checks for Kaggle first. Rows written: none.

### 6. 2026-11-06, Kaggle T4×2 (one GPU used): notebook 01, K1 rerun at `200cea5`
- The 33 baselines were reused. lm-eval 0.4.13 ran on all items in about 74 min. Outputs are in `lmeval-Qwen2.5-1.5B-Instruct/` (results stamped 08:46 and 09:44 UTC).
- **K1 PASS.**
  - Belebele: mean |ours − lm-eval| 0.11 points, max 0.33 (ja), agreement 99.8%.
  - MGSM: mean 0.80 points, max 1.60 (de), agreement 98.4%.
- Throughput (GPU-minutes per 1.5B sweep): Belebele 18.7, MGSM 68.1, FLORES 7.2. MGSM is 2.27× the assumption, so a re-plan was triggered.
- Details: DECISIONS 2026-11-06.

### 7. 2026-11-06 12:40–20:01, Kaggle T4×2: notebook 02 at `dd7d538`
- 297 new runs: W3/W4 GPTQ over calibration en/multi × 3 seeds on Belebele and FLORES, W8 RTN, and Qwen's official GPTQ-Int4 (`prequant`) on FLORES.
- Session length 7.4 h.
- **FR-3 FAIL** against the official checkpoint: mean 7.5%, Russian 25.8%.
- Cause: calibration was 128 single FLORES sentences, too few tokens.
- **Superseded.** These GPTQ runs use the legacy calibration and have their own run IDs. The W8 RTN and `prequant` runs keep their IDs and are still valid.
- Details: DECISIONS 2026-11-07.

### 8. 2026-11-07 05:16–06:11, Kaggle T4×2: notebook 02 at `ee25c07`
- 22 runs: our W4 GPTQ on 128 × 2,048 Wikipedia windows, plus the GPTQModel reference (recipe 1), both on FLORES.
- Session length 1.13 h.
- **FR-3 FAIL**: mean 4.32%, max 10.96% (es). Our W4 beat the reference in 7 of 11 languages.
- Cause: the reference ran in bfloat16 and saved in v1 `gptq` format with asymmetric zeros.
- **Reference superseded.** Recipe-1 runs such as `8e8544b892fa` stay as a record. Our 11 W4 runs are reused.
- Details: DECISIONS 2026-11-07.

### 9. 2026-11-12 08:32–08:47, Modal T4: FR-3 check at `6e01035`
- 11 new GPTQModel reference runs (FP16, `gptq_v2`, recipe 2). Our W4 runs from entry 8 were reused.
- Environment: Python 3.13.18, torch 2.15.0+cu132, transformers 5.19.0.
- **FR-3 PASS**: mean 1.72% (limit 2%).
  - Above 2%: es 4.75%, fr 4.53%, sw 2.50%.
  - Ours is lower than the reference in 10 of 11 languages (bn is the exception).
- Details: DECISIONS 2026-11-12.

### 10. 2026-11-12, Modal T4: Belebele prefill reuse and MGSM batch-1 timings at `b820e74`
- Prefill reuse:
  - 300 en and 300 te items gave identical letter log-probabilities with and without reuse.
  - The same items were identical to the logged FP16 rows.
  - Wall time fell 12% in en and 31% in te.
- MGSM timings (8 problems per cell) gave planning numbers only.
  - Speculative decoding was 0.73–0.81× plain greedy.
  - The quantized cache hit the 256-token cap on all 8 te problems at 1.5B.
- Rows written: none. Details: DECISIONS 2026-11-12.

### 11. 2026-11-12 08:50–10:15, Modal T4: notebook 02 replication grid at `b820e74`
- 118 runs started, 112 finished: Belebele W4 ×66, Belebele W3 ×38, FLORES W4 ×14, all Wikipedia calibration.
- The last start stamp is 10:15:45. No later activity is in the dataset, and no Modal app is running (checked 2026-11-13).
- **The grid stopped. It did not finish.**
  - The stop cause was not logged.
  - The grid was launched attached, before `274bd85` (09:31 UTC) made Modal apps detached. A dropped connection from the M1 would end it.
- Six runs are partial, with no `DONE`. Resume finishes them: `0ca4b77238e4`, `1bf8eb84ac13`, `2fdde0dd55ed`, `64b2087b9e06`, `76044670ebc8`, `bee9e4d7c59b`.
- `configs/replication.yaml` status: 167 of 308 done, 6 partial, 135 not started.
  - Not started: all FLORES W3 (66), most FLORES W4 seeds, and the Belebele W3 multi seeds 1–2.
- BUILD_PLAN said "replication grid running on Modal" until 2026-11-13. That note was stale.

### 12. 2026-11-12, Kaggle T4×2: notebook 03 probe at `b820e74`
- Failed in the first baseline pass. `nll_batch` at batch 32 converted the full logits to float32 (4.66 GiB), which ran out of memory.
- Fixed in `974c16a` with `nll_batch_low_memory`. Rows written: none. Details: DECISIONS 2026-11-12.

### 13. 2026-11-12 09:00 – about 13:00, Kaggle T4×2: notebook 03 probe at `974c16a`
- Kernel `souramarti/whopays-runner`, status `COMPLETE`. Environment: torch 2.15.0+cu132, CUDA 13.2, transformers 5.19.0, Python 3.13.16, Tesla T4.
- Profile `probe-e537cbad92f7`: 28 blocks × bits {2, 3, 4, 8} × 11 languages × 2 halves = 2,464 rows. Settings: GPTQ, group 128, batch 32, 100 FLORES devtest sentences per language.
- **K2: PASS, noise-limited.**
  - `min_spearman` 0.348 (te, zh), limit ≤ 0.9.
  - `split_half_min` 0.548. Spearman-Brown correction gives about 0.71, which is ≤ 0.9, so `noise_limited = True`.
  - The literal gate passes. But at 100 sentences per language, the measured differences between languages cannot be separated from sampling noise.
  - I recomputed `k2_verdict` on the M1 from the dataset rows. It gives the same values.
- Time:
  - The session took about 4.0 h. Setup ended at 365 s.
  - Two HF commit rate limits (HTTP 429) each paused the job for about 1 h: one at 487 s, one at 4,691 s.
  - Blocks took about 220–300 s each, so the 28 blocks took about 1.9 h.
  - The final printed projection was 3.87 h, which includes the waits.
  - The first 429 came while the Modal grid (entry 11) was pushing to the same dataset. Both jobs counted against one HF account's commit limit.
- Log: `runs/kaggle/latest/20261113T073657Z/whopays-runner.log` (local, gitignored).

## Results dataset snapshot (2026-11-13)

`Heavy-Dervish1205/whopays-results`: 483 run folders, 477 with `DONE`, plus `lmeval-Qwen2.5-1.5B-Instruct/` and `probe-e537cbad92f7/`. Generated by `python scripts/run_inventory.py`.

| First start (UTC) | Last start (UTC) | Platform | Commit | Runs | Done | Cells |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-28T11:31 | 11:32 | local | `a9e2ae5` | 2 | 2 | 0.5B belebele fp16 ×2 |
| 2026-10-31T09:34 | 11:47 | colab | `wheel:s` | 33 | 33 | 1.5B belebele fp16 ×11; 1.5B flores_nll fp16 ×11; 1.5B mgsm fp16 ×11 |
| 2026-11-06T12:40 | 20:01 | kaggle | `dd7d538` | 297 | 297 | 1.5B belebele w3_gptq ×66; 1.5B belebele w4_gptq ×66; 1.5B belebele w8_rtn ×11; 1.5B flores_nll prequant ×11; 1.5B flores_nll w3_gptq ×66; 1.5B flores_nll w4_gptq ×66; 1.5B flores_nll w8_rtn ×11 |
| 2026-11-07T05:16 | 06:11 | kaggle | `ee25c07` | 22 | 22 | 1.5B flores_nll w4_gptq wikipedia_128x2048 ×11; 1.5B flores_nll w4_gptqmodel wikipedia_128x2048 ×11 |
| 2026-11-12T08:32 | 08:47 | modal | `6e01035` | 11 | 11 | 1.5B flores_nll w4_gptqmodel wikipedia_128x2048 ×11 |
| 2026-11-12T08:50 | 10:15 | modal | `b820e74` | 118 | 112 | 1.5B belebele w3_gptq wikipedia_128x2048 ×38; 1.5B belebele w4_gptq wikipedia_128x2048 ×66; 1.5B flores_nll w4_gptq wikipedia_128x2048 ×14 |

| Job list | Runs | Done | Partial | Not started |
| --- | --- | --- | --- | --- |
| `configs/replication.yaml` | 308 | 167 | 6 | 135 |
| `configs/audit.yaml` | 858 | 132 | 5 | 721 |
| `configs/speed.yaml` | 308 | 0 | 0 | 308 |

The audit's 132 done runs are Belebele full-cache cells shared with the replication grid.

## Open items (2026-11-13)
- Replication grid: resume the 141 unfinished runs (6 partial, 135 not started). After that, check the gap direction for FR-3.
- K2: re-profile at 500 sentences per language, to resolve `noise_limited` (DECISIONS 2026-11-13).
- HF uploads: fixed 2026-11-13. Modal grids make one commit at the end, and interval pushes run in a background thread (DECISIONS 2026-11-13).
