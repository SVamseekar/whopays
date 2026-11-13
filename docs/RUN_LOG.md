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
| 14 | 2026-11-13 | Modal T4 ×≤10 | 02 replication grid, resumed | `4ac29ab` | **grid complete, 308/308**; gap direction not reproduced in raw points (floor effect) | 135 new runs; 286 runs pushed in one commit |
| 15 | 2026-11-13 | Kaggle T4×2 | 03 sensitivity probe, 500 sentences per language | `4ac29ab` | running | `probe-6f52db8b4007` (in progress) |

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

### 14. 2026-11-13 08:15–08:56 UTC (first pass) and about 11:20 UTC (rerun), Modal T4 ×≤10: notebook 02 replication grid resumed at `4ac29ab`
- Job list: the 141 unfinished runs of `configs/replication.yaml`. These were 11 chunks of whole model groups, and the quantized weights for 11 model groups were already cached on the Volume.
- First pass:
  - 131 runs done. 6 were skipped: the 6 runs that looked partial on HF had actually finished on the Volume on 2026-11-12, but were never pushed.
  - 4 runs were refused by the resume guard: `bb2d3522d696`, `6d47d388601a`, `946da8c66189`, `1eb45efe5c96`. Each had a partial folder on the Volume started at `b820e74`, so resuming at `4ac29ab` would have mixed commits within a run.
  - Those 4 folders had no `DONE`, were not on HF, and were deleted from the Volume. The rerun finished them in 53–98 s each.
- Per-run wall time: 1.82 h summed over the first pass's 131 runs, plus 0.08 h for the 4 reruns. **Modal cost: $1.61** (`modal billing report`: $1.41 + $0.19 + $0.01).
- Driver hang:
  - In both passes, the M1 driver stayed in `run_jobs.map(...)` after every chunk had returned and Modal showed 0 tasks. `push_results` therefore never started. Cause unknown.
  - Workaround: kill the driver, then call `modal_app.push_results.remote(ids)` directly. One commit uploaded 286 finished run folders from the Volume.
- Commits mixed across runs: `b820e74` (2026-11-12) and `4ac29ab`. Between them, `src/` changed only in hub sync, the probe and a probe-only scorer (`nll_batch_low_memory`). FLORES and Belebele task scoring is identical, and each run sits under a single commit.
- **Gap direction** (computed on the M1 from these logged rows):
  - Method: paired bootstrap over parallel Belebele items, 2,000 resamples, accuracy change versus FP16 minus English's change, W4/W3 averaged over 3 seeds per item. Table: `docs/figures/replication_gap_direction.csv`.
  - W8 RTN: English −0.11 points; no language differs from English.
  - W4 GPTQ, English calibration: English −3.70 points. Russian (−3.96, CI −6.67 to −1.52) and Thai (−3.48, CI −6.33 to −0.48) lose more than English. The other 8 CIs include 0.
  - W4 GPTQ, multilingual calibration: English −6.26 points. No language loses more than English; Telugu (+4.44) and Chinese (+2.56) lose less.
  - W3 GPTQ: English −44.96 (English calibration) and −47.41 (multilingual calibration) points. Bengali, Swahili and Telugu lose only 6–17 points, because their FP16 accuracy (29.9–39.8%) already sits near the 25% chance level of 4-way choice.
  - **The published direction (non-Latin-script and lower-resource languages hurt more, Marchisio et al. 2024) is not reproduced in raw accuracy points.** The low-resource languages hit the chance floor, so a raw-points gap cannot show the published direction. Any metric that corrects for headroom would be a post-hoc choice and needs a preregistration amendment (DECISIONS 2026-11-13).

### 15. 2026-11-13, Kaggle T4×2: notebook 03 probe at 500 sentences per language, `4ac29ab`
- Kernel `souramarti/whopays-runner` version 5, status `RUNNING`. Profile `probe-6f52db8b4007` is pushing to HF about every 10 min from the background upload thread.
- Expected about 7 h. Result pending.

## Results dataset snapshot (2026-11-13)

`Heavy-Dervish1205/whopays-results` (refreshed after entry 14): 618 run folders, 618 with `DONE`, plus `lmeval-Qwen2.5-1.5B-Instruct/` and `probe-e537cbad92f7/`. Generated by `python scripts/run_inventory.py`.

| First start (UTC) | Last start (UTC) | Platform | Commit | Runs | Done | Cells |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-28T11:31 | 11:32 | local | `a9e2ae5` | 2 | 2 | 0.5B belebele fp16 ×2 |
| 2026-10-31T09:34 | 11:47 | colab | `wheel:s` | 33 | 33 | 1.5B belebele fp16 ×11; 1.5B flores_nll fp16 ×11; 1.5B mgsm fp16 ×11 |
| 2026-11-06T12:40 | 20:01 | kaggle | `dd7d538` | 297 | 297 | 1.5B belebele w3_gptq ×66; 1.5B belebele w4_gptq ×66; 1.5B belebele w8_rtn ×11; 1.5B flores_nll prequant ×11; 1.5B flores_nll w3_gptq ×66; 1.5B flores_nll w4_gptq ×66; 1.5B flores_nll w8_rtn ×11 |
| 2026-11-07T05:16 | 06:11 | kaggle | `ee25c07` | 22 | 22 | 1.5B flores_nll w4_gptq wikipedia_128x2048 ×11; 1.5B flores_nll w4_gptqmodel wikipedia_128x2048 ×11 |
| 2026-11-12T08:32 | 08:47 | modal | `6e01035` | 11 | 11 | 1.5B flores_nll w4_gptqmodel wikipedia_128x2048 ×11 |
| 2026-11-12T08:50 | 10:15 | modal | `b820e74` | 118 | 118 | 1.5B belebele w3_gptq wikipedia_128x2048 ×38; 1.5B belebele w4_gptq wikipedia_128x2048 ×66; 1.5B flores_nll w4_gptq wikipedia_128x2048 ×14 |
| 2026-11-13T08:15 | 08:56 | modal | `4ac29ab` | 135 | 135 | 1.5B belebele w3_gptq wikipedia_128x2048 ×28; 1.5B flores_nll w3_gptq wikipedia_128x2048 ×66; 1.5B flores_nll w4_gptq wikipedia_128x2048 ×41 | 

| Job list | Runs | Done | Partial | Not started |
| --- | --- | --- | --- | --- |
| `configs/replication.yaml` | 308 | 308 | 0 | 0 |
| `configs/audit.yaml` | 858 | 165 | 0 | 693 |
| `configs/speed.yaml` | 308 | 0 | 0 | 308 |

The audit's 165 done runs are Belebele full-cache cells shared with the replication grid.

## Open items (2026-11-13)
- Replication gap direction: the user must choose between raw points as preregistered (not reproduced) and a headroom-corrected metric, which needs an amendment (DECISIONS 2026-11-13).
- Modal driver hangs after `run_jobs.map` returns. Until that is fixed, call `push_results` directly.
- K2: re-profile at 500 sentences per language, to resolve `noise_limited` (DECISIONS 2026-11-13).
- HF uploads: fixed 2026-11-13. Modal grids make one commit at the end, and interval pushes run in a background thread (DECISIONS 2026-11-13).
