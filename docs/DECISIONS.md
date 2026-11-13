# Decisions log

Newest first. One paragraph per decision: what was decided, why, and what would reverse it.

## 2026-11-13 — Gap direction investigated: reproduced in FLORES NLL and in Belebele above-chance retention, not in raw points
**Symptom.** On Belebele raw points (previous entry), the published direction was not reproduced. Under W4 with English calibration, only ru and th were worse than en. Under W3, the low-resource languages lost fewer points than en.

**Investigation (M1 analysis of logged GPU rows; no new GPU runs).**
(1) *Letter-bias collapse.* At FP16, the share of answers going to the most chosen letter is 0.26–0.33 for most languages, but 0.45 for sw and 0.41 for te. Under W4 with English calibration it rises to 0.57 (bn), 0.58 (te) and 0.53 (sw). Under W3 it reaches 0.83 (ja) with English calibration and 0.82 (en) with multilingual calibration. Belebele letter scoring increasingly measures an A/B/C/D prior, not reading. The PRD risk register names this risk and its mitigation: "Also report calibrated accuracy". `tasks.belebele.calibrated_accuracy` subtracts each letter's mean log-probability per run. Calibrated accuracy barely changes the raw-points picture (W4 English calibration: ru −7.1 and th −7.5 vs en −3.6 are the only significant gaps), because the floor remains.
(2) *Floor.* With calibrated accuracy at FP16, te is 31.7%, sw 36.4% and bn 41.3%, against 25% chance. That leaves 6.7–16.3 points of headroom, against 53.7 for en, so a raw-points drop is capped. The share of above-chance accuracy retained, (acc_q − 25)/(acc_fp16 − 25), runs the published way. Under W4 with English calibration: en 0.93; bn 0.68, sw 0.67, te 0.80, th 0.80, ru 0.84, ja 0.89. Under W3 with English calibration: en 0.23; bn 0.04, sw 0.03, te −0.18 (below chance), th 0.02, ja 0.11, ru 0.11. zh (0.24) is the only language at or above en. No intervals have been computed for this ratio yet. It is unstable where headroom is under 10 points (te).
(3) *FLORES NLL has no floor.* The relative increase in total NLL over the 1,012 parallel devtest sentences, with a paired sentence bootstrap of 2,000 resamples, is in `docs/figures/replication_flores_nll_gap.csv`. Under W4 with English calibration, en rises 7.4%, and 9 of 10 languages rise significantly more: te +26%, bn +23%, ru and th +19%, ja +14%, zh +13%, es and fr +12%, de +11%. sw rises less (+5%). Under W3 with English calibration, en rises 22%, against bn +86%, te +77% and th +70%. Multilingual calibration roughly halves the low-resource excess (W4: bn +13%, te +14%). sw is the consistent exception. Its FP16 perplexity is 119 against 30 for en, so the model barely models Swahili, and a relative increase from that baseline is small.
(4) *Code path.* `replicate.gap_direction` computes raw paired points minus en, as notebook 02 prints. Nothing in it is wrong. It just cannot show a direction once accuracy is near chance.

**What the field does.** [Marchisio et al. (2024)](https://arxiv.org/abs/2407.03211) studied four multilingual models of 8B–103B and report "non-Latin script languages impacted worst", using automatic tasks, LLM-as-judge and human evaluation. [Uneven impact of PTQ in MT (2025)](https://arxiv.org/abs/2508.20893) covers 55 languages and 1.7B–70B models. It finds 4-bit holds for high-resource languages, low-resource and Indic languages degrade noticeably, models under 10B degrade more across the board, and 2-bit can remove low-resource translation entirely. That matches our W3 retention near zero for bn, sw, te and th. [Zheng et al. (ICLR 2024)](https://arxiv.org/abs/2309.03882) show multiple-choice letter scoring carries an option-ID token prior and propose prior debiasing (PriDe). Our calibrated accuracy is the cheap, run-level version of that. Chance-adjusted accuracy, (acc − chance)/(1 − chance), is common in multiple-choice evaluation ([None of the Others, 2025](https://arxiv.org/abs/2502.12896)), but no single standard exists. A limiting point: the published claims come from models of 8B and up, at accuracies well above chance, so a raw-points test at 1.5B is not a like-for-like replication in any case.

**Recommendation (owner decides; it changes the protocol after seeing results).** A fifth amendment, disclosed as following the raw-points result: replication direction is judged on FLORES relative NLL increase (primary: no floor, parallel sentences) and on Belebele above-chance retention of calibrated accuracy, with bootstrap intervals and raw points reported alongside. Rejected: raw points only, because it measures headroom, not sensitivity, at 1.5B. Also rejected: PriDe with permuted options, because it needs new forward passes and the run-level prior subtraction already in the PRD covers the constant bias. GPU cost of the recommendation: 0 h, since everything is computed from stored rows. Open consequence: H1 ("gaps grow with lower FP16 accuracy") is biased toward falsification by the same floor if gaps are raw points. The 165 audit runs already finished are Belebele full-cache cells seen here, so any H1 metric change is also post-hoc and must be disclosed.

## 2026-11-13 — Replication grid complete; published gap direction not reproduced in raw points; metric decision open
**Result.** All 308 runs of `configs/replication.yaml` are finished. The last 141 ran on Modal at `4ac29ab` for $1.61 (RUN_LOG entry 14). The PRD's M1 gate asks for the "published direction of the per-language quantization gap reproduced, with 95% CIs". The published finding is that quantization hurts non-Latin-script and lower-resource languages more (Marchisio et al. 2024). This was measured on Belebele as each language's paired accuracy change from FP16, minus English's change, with 95% item-bootstrap CIs (`docs/figures/replication_gap_direction.csv`). In raw points it is **not reproduced**. Under W4 with English calibration, only Russian and Thai lose significantly more than English (−3.96 and −3.48 points). Under W4 with multilingual calibration, no language does, and Telugu and Chinese lose less. Under W3, every model collapses toward chance: English loses 45–47 points, while Bengali, Swahili and Telugu lose only 6–17. Their FP16 accuracies (29.9–39.8%) leave 5–15 points above the 25% chance floor, so they cannot lose more points than English.

**Why this is not a verdict yet.** A raw-points gap confounds sensitivity with headroom. Low-resource languages at 1.5B start near chance, so their measured drop is capped. Marchisio et al. report relative degradation on much larger models (for example 103B), where low-resource accuracy sits well above chance. A headroom-corrected measure, for example the change in above-chance accuracy, (acc_q − 0.25)/(acc_fp16 − 0.25), or the change in FLORES NLL, which has no floor, would test the published claim more fairly. But choosing it now, after seeing the raw result, is a post-hoc metric change. It would have to be disclosed as a fifth preregistration amendment, with the raw-points result reported alongside. The multilingual-calibration result also shows calibration language moving the gap by several points: under W4, Telugu goes from +1.00 to +4.44 points relative to English. That is the calibration finding the PRD lists as a sanity anchor.

**Decision needed (owner).** (a) Report the replication as not reproduced in raw points and move on, since this M1 gate is not a K kill criterion. (b) Add a fifth amendment with a headroom-corrected Belebele metric and FLORES NLL change as the replication measures, reporting raw points alongside. The recommendation is (b), because the floor effect is mechanical rather than a finding. Either way, the audit's H1 (gap grows with lower base accuracy) faces the same floor problem and should be settled in the same amendment, before any audit result.

## 2026-11-13 — Fewer hub commits; probe re-profiled at 500 sentences per language
**Hub commits.** On 2026-11-12 the Kaggle probe lost about 2 h to two HF commit rate limits (HTTP 429). huggingface_hub then sleeps up to an hour inside `upload_folder`. The commits came from both jobs on one account: the Modal grid pushed every run at each flush and again when it finished, and `run()` also re-pushed every finished run it skipped, including runs it had just pulled from the hub. Three changes. (1) `run(push=False)` keeps the pulls and skips all pushes. `scripts/modal_run.py jobs` uses it, because results already sit on the durable Modal Volume, and then uploads every finished run in one commit with `hub_sync.push_runs` from a CPU-only container (`--push-each` restores the old behaviour). (2) A run restored from the hub is no longer pushed back. (3) `ThrottledPush(background=True)`, used by `run()` flushes and the probe, uploads in a daemon thread and skips a push while one is still running, so a 429 never stalls the GPU. The forced final push stays synchronous, so a Kaggle session cannot end before its last state is uploaded. A partial last line uploaded mid-write is harmless, because both `ResultsWriter` and `profile` trim it on resume.

**Probe limit.** K2 at 100 sentences per language was noise-limited (split-half 0.548, corrected about 0.71). Spearman-Brown projects the corrected reliability to pass 0.9 at about 3.7× the sentences. 300 sentences would project to about 0.88, which is still noise-limited, so the limit is now 500 (projected about 0.92). Scoring dominates the probe's cost, so the run should take about 7 h on one Kaggle T4, inside the 12 h session limit. A full 1,012-sentence profile would take about 14 h and is not worth it. The 500-sentence profile has a new run ID; the 100-sentence profile stays in the dataset as a record. Reverse if the 500-sentence run is still noise-limited. In that case, K2 is reported as passing only at the literal threshold, and M4 proceeds with that caveat stated.

## 2026-11-13 — K2 passes but is noise-limited; replication grid found stopped; one run timeline
**K2.** Notebook 03 ran on Kaggle T4×2 at `974c16a` (torch 2.15.0+cu132, CUDA 13.2, transformers 5.19.0, Python 3.13.16, Tesla T4) in about 4.0 h. It wrote profile `probe-e537cbad92f7` (28 blocks × 4 widths × 11 languages × 2 halves, 2,464 rows, 100 FLORES devtest sentences per language). At 4 bits the minimum pairwise Spearman is 0.348 (te, zh) against the 0.9 limit, so **K2 PASS** by the literal PRD rule, and M4 is not killed. `noise_limited` is true: the lowest split-half correlation is 0.548, about 0.71 after Spearman-Brown correction, which is also below 0.9. So the sample cannot show that the languages differ in which layers are sensitive. Under the 2026-11-03 rule, the profile is re-measured with more sentences per language before M4 allocation reads it. The K2 task stays unticked until then. About 2 h of the session was spent in two HF commit rate-limit waits (HTTP 429, about 1 h each). The first wait overlapped the Modal grid, which was pushing to the same dataset, so both jobs counted against one account's commit limit.

**Replication grid.** BUILD_PLAN said the notebook-02 grid was "running on Modal". The results dataset shows it ran from 08:50 to 10:15 UTC on 2026-11-12 at `b820e74` and then stopped: 112 runs finished, 6 are partial, and no Modal app is running. Against `configs/replication.yaml` that is 167 of 308 done. The cause was not logged. The grid was launched attached, before `274bd85` made Modal apps detached. Resume skips finished runs, so nothing is lost.

**Run log.** Run records were spread across BUILD_PLAN, DECISIONS and the dataset, and the stale grid note went unnoticed for a day. `docs/RUN_LOG.md` is now the single timeline: every GPU session, oldest first, including failed and superseded ones, with commit, times, rows written and outcome. `scripts/run_inventory.py` rebuilds its dataset snapshot and grid completeness from the dataset itself. Entries 1–12 were backfilled from DECISIONS, BUILD_PLAN, git history and the run folders' `meta.json` stamps.

## 2026-11-12 — FR-3 passes on Modal; Belebele prefill reuse verified; MGSM KV cells measured at batch 1
**FR-3.** `scripts/modal_run.py fr3` at `6e01035` on a Modal T4 (Python 3.13.18, torch 2.15.0+cu132, CUDA 13.2, transformers 5.19.0; Kaggle logs 3.13.16, the resume guard does not pin Python). Our 11 W4 FLORES runs were reused from 2026-11-07; the GPTQModel 7.6.0 reference (FP16, `gptq_v2`, same 128 × 2,048 Wikipedia windows) ran fresh. Mean absolute perplexity gap 1.72% (limit 2%), so **FR-3 PASS** under the implemented mean-over-languages rule. Flagged above 2%: Spanish 4.75%, French 4.53%, Swahili 2.50%; a strict per-language reading would fail those three. Our perplexity is lower than the reference in 10 of 11 languages (Bengali the exception), so the gap is ours being slightly better, not worse. Qwen's official GPTQ-Int4 (secondary, descriptive) differs by 2.5–25.8%, as expected from its undisclosed calibration and symmetric quantization.

**Belebele prefill reuse.** 900 questions per language share 488 passages; context is 66–71% of tokens, and reusing the prefilled cache removes 30.6–33.0% of tokens (en, te, zh, bn; Qwen2.5 tokenizer). On a Modal T4 at `b820e74`, 300 English and 300 Telugu items gave letter log-probabilities identical with and without reuse (300/300 each) and identical to the logged FP16 rows (300/300 each). Wall time fell 12% in English (16.9 s vs 19.2 s) and 31% in Telugu (39.4 s vs 56.7 s): at batch 1 per-call overhead dominates short contexts. This is automatic prefix caching ([vLLM](https://docs.vllm.ai/en/latest/design/prefix_caching.html)). The quantized cache is excluded because its HQQ state is not plain tensors.

**MGSM KV cells run at batch 1.** `tasks/mgsm.generate_batch` sends every `kv != "full"` cell through the sequential `decoding.speculative.generate` loop, one problem at a time; the full-KV cells use batched `generate` at batch 8. The 2026-11-06 audit projection priced all MGSM cells at the batch-8 rate, so it was too low. Measured on a Modal T4 (8 problems, English and Telugu mean, seconds per problem; planning numbers, not results): 1.5B full batch 8 1.63, full batch 1 6.80, quantized cache 11.95, eviction 6.45, speculative 9.17; 3B full batch 8 2.24, full batch 1 11.92, quantized 14.41, eviction 11.23, speculative 14.32. Projected MGSM audit cost: quantized-cache cells 36.5 h (1.5B) and 44.0 h (3B); eviction cells 19.7 h and 34.3 h; full cells 5.0 h and 6.8 h; speed grid 25.2 h. Speculative decoding was slower than plain greedy at both sizes on these 8 problems (0.73–0.81×); the speed grid measures it properly. The quantized cache hit the 256-token cap on all 8 Telugu problems at 1.5B and needs its outputs inspected before the audit. Batching the quantized-cache cells naively is not valid: transformers 5.19 hands HQQ the whole `[batch, heads, tokens, dim]` tensor, and hqq 0.2.8 with `axis=0` groups 64 values strided across the flattened tensor (`W.reshape([group_size, -1])`), so other sequences and padding would enter each sequence's quantization groups. The batch-8 full cells against batch-1 KV cells also mix batch shape into the KV contrast ([Thinking Machines on batch invariance](https://simonwillison.net/b/9008)); both points are addressed in the next decision.

**K2 probe out of memory.** The first Kaggle probe run (`b820e74`) failed in its first baseline pass: `nll_batch` at batch 32 converted the full logits to float32 (4.66 GiB). The probe now uses `nll_batch_low_memory` (decoder once per batch, vocabulary projection and cross-entropy per sequence; within 1e-4 of `nll_batch` on CPU). FLORES task runs keep `nll_batch`. Kaggle kernels pushed through the API also lose Secrets bindings; the private `whopays-secrets` dataset fallback worked on the rerun.

## 2026-11-12 — Both Kaggle T4s used; GPU runs launched from the M1; power, layer error and rating sheets added
**Problem.** Each GPU check needed the owner to upload a notebook and paste output back, and notebook 02 projects to about 9.5 T4-hours. From the logged notebook-01 throughput (Belebele 18.7, FLORES 7.2 GPU-minutes per 1.5B model) and the 2026-11-07 run (1.13 h for FR-3, about 20 min per GPTQ quantization with 128 × 2,048 windows): 11 GPTQ models to quantize (≈ 3.7 h), 12 Belebele model sweeps (≈ 3.7 h), 11 FLORES sweeps (≈ 1.3 h) and the FR-3 reference (≈ 0.8 h). Kaggle sessions are "GPU T4 x2", but `run()` uses one device, so half of every session was idle. The rest of the program (audit ≈ 57 h after the 154 Belebele runs shared with notebook 02, probe ≈ 3 h, ParityQuant ≈ 27 h, 7B ≈ 12 h, speed grid) projects to roughly 115 T4-hours in total, before re-runs. These are projections from logged rates, not results.

**Decision.** (1) `whopays.parallel.run_parallel` splits a job list into shards of whole model groups (each quantized once), balanced by a rough cost model, and runs one `whopays.cli run` worker per GPU pinned with `CUDA_VISIBLE_DEVICES`. Outputs and run IDs are unchanged, because each worker runs the same code and batch sizes on one T4 and each run folder has one writer. Notebooks 02, 04 and 06 call it, then call `whopays.run` in-process, which skips finished runs and gives the status table. It is not used for 7B (split over both GPUs) or the speed grid (workers share the host). (2) `scripts/kaggle_run.py` pushes a notebook pinned to the pushed HEAD to one private kernel, `whopays-runner`, starts it on T4 x2, polls it, and downloads the log and executed notebook, so the owner no longer handles notebooks. The Kaggle API cannot attach secrets: `HF_TOKEN` is attached once to that kernel in the web UI. (3) From the review of what practitioners would add: `stats/power.py` (paired minimum detectable effect, Connor 1987, and a simulation of the preregistered H1 rule through the exact `h1` code); GPTQ now writes each layer's relative Hessian-weighted output error to `layer_error.json` (diagnostic only, weights unchanged, tested); `report/spotcheck.py` builds blinded A/B rating sheets of MGSM generations for native speakers, half from items whose correctness flipped.

**Rejected.** Bigger GPUs (A100, H100) on paid clouds: faster per run, but the logged FP16 and W8 runs are T4 results, and the resume guard rightly refuses mixed hardware inside a comparison. Batched single-pass Belebele: it would change FP16 numerics against the logged baselines and break the two-pass path the KV cells share (2026-10-21). Colab: it has no supported way to start a run without a browser. Paid Indian GPU clouds (E2E, Jarvislabs) and marketplaces (Vast.ai, RunPod): they cost money and give no parallelism on T4 over what Kaggle's second GPU and Modal already give. Modal ($30 a month of free credit, up to 10 GPUs at once, T4 at $0.59/h, image pinnable to Kaggle's torch 2.15.0+cu132) is the burst option and waits for the owner's sign-up.

**Limits and reversal.** The Kaggle quota is assumed to count session time, not GPU count; check the quota page after the first T4 x2 run. The cost model only orders shards; a bad estimate costs balance, not correctness. Revert to one worker if two concurrent GPTQ passes ever produce different weights from a single-process run of the same model key.

## 2026-11-07 — FR-3 reference fixed: GPTQModel runs in FP16 and saves gptq_v2
**Result.** Notebook 02 ran on Kaggle T4×2 at commit `ee25c07` (torch 2.15.0+cu132, transformers 5.19.0, Python 3.13.16, GPTQModel 7.6.0 in its own venv) for 1.13 h. FR-3 against GPTQModel on our exact calibration windows: mean absolute perplexity gap 4.32%, maximum 10.96% (Spanish), limit 2%, **FAIL**. The notebook stopped before the grid as designed. Our W4 had *lower* perplexity than the reference in English (38.023 vs 42.476), Spanish, German, French, Swahili, Chinese and Thai, and was within 1.2% in Bengali, Japanese, Russian, Telugu and Thai. A reference that is worse than a hand-written quantizer with the same recipe points at the reference, not at ours.

**Cause, from the log and GPTQModel 7.6.0 source.** (1) GPTQModel quantized and evaluated in bfloat16: the log shows `Loader: Auto dtype (native bfloat16)` and `dtype: bf16` on every module. `GPTQModel.load` without `dtype` takes the model config's dtype (`utils/model.py`, `auto_dtype`), which is bfloat16 for Qwen2.5. The T4 (compute capability 7.5) has no native bfloat16, and every other run in this project is FP16 (CLAUDE.md rule 5). The official checkpoint's config is FP16, so its runs were FP16 (log of 2026-11-07 morning: `Auto dtype (native float16)`). (2) The checkpoint was saved in the v1 `gptq` format with `sym=False`. GPTQModel's own source says v1 stores zero points minus one and that "v1 checkpoint format with sym=False ... will overflow ~<=13% based on testing" (`utils/model.py`, `convert_gptq_v1_to_v2_format`); the `FORMAT.GPTQ_V2` enum is commented "v2 format fixed sym = False quantization" (`quantization/config.py`). Our quantizer is asymmetric, so the reference was likely corrupted in some groups. bfloat16 alone is unlikely to explain gaps of 7 to 11%: BF16-versus-FP16 perplexity differences are usually far below quantization effects ([llama.cpp PR 7150](https://github.com/ggerganov/llama.cpp/pull/7150)). The zero-point overflow is the stronger candidate; the re-run separates neither, but both were defects. T4 bfloat16 limits: [PyTorch forum](https://discuss.pytorch.org/t/current-cuda-device-does-not-support-bfloat16-please-switch-dtype-to-float16/201564), [OpenXLA issue 12429](https://www.github.com/openxla/xla/issues/12429).

**Decision.** `replicate.GPTQMODEL_CONFIG` adds `format="gptq_v2"`. Quantization and every GPTQModel load (`models/loader._load_gptqmodel`, also used for the official checkpoint) pass `dtype=float16` and fail if the loaded embeddings are not FP16. `config.GPTQMODEL_RECIPE = 2` enters the canonical form of `w4_gptqmodel` runs only, so the reference gets new run IDs and cache keys while the recipe-1 runs (for example `8e8544b892fa`) stay in the dataset as a record. Our W4 runs and the official-checkpoint runs keep their IDs and are reused. The 2% rule is unchanged. This is a bug fix to the reference, made after seeing the result, and disclosed as such in the fourth amendment.

**Reverse if** FR-3 still fails against the FP16, v2 reference. Then compare per-layer outputs of our fake-quantized weights against GPTQModel's dequantized weights on one block before any other change.

## 2026-11-07 — GPTQ calibration: 128 × 2,048 Wikipedia tokens; FR-3 against GPTQModel on the same data
**Problem.** FR-3 failed on 2026-11-06: our W4 was 7.5% worse in mean FLORES devtest perplexity than Qwen's official GPTQ-Int4 (Russian 25.8%, Chinese 15.8%, Japanese 11.3%, Thai 9.2%). Bengali, Swahili and Telugu were better than the reference, and English, German, Spanish and French were within 2%. Our calibration was 128 single FLORES dev sentences, a few thousand tokens. GPTQ's Hessian for a layer is X^T X over calibration tokens. Qwen2.5-1.5B's down_proj has 8,960 inputs, so with fewer tokens its Hessian is rank-deficient and the update fits the calibration sentences. That explains being better on same-domain low-resource FLORES languages and worse on languages Qwen's own calibration covers. The GPTQ arithmetic itself matched GPTQModel 7.6.0 line by line: min/max including zero, `zero = round(-min/scale)`, damping relative to the mean diagonal, group parameters from the updated weights, and dead-column handling.

**What the field does.** The GPTQ paper calibrates on "128 random 2048 token segments" of C4 ([Frantar et al., 2023](https://arxiv.org/abs/2210.17323)). [Williams & Aletras (ACL 2024)](https://arxiv.org/abs/2311.09755) use the same 262,144 tokens. They find diminishing returns with more examples (LLaMA-7B GPTQ C4 perplexity 6.13 with one example, 5.90 with 128). They ask authors to release calibration data and to check several calibration sets. [llm-compressor](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a16/) starts at 512 × 2,048. [GPTQModel](https://github.com/ModelCloud/GPTQModel)'s examples use 1,024 C4 documents. [Chimoto et al. (EACL 2026)](https://arxiv.org/abs/2601.18306) study multilingual calibration with GPTQModel on Qwen2.5-7B. They use native-language Wikipedia and C4 and an equal-per-language multilingual mix, and find GPTQ much more calibration-sensitive than AWQ. [Paglieri et al. (2024)](https://arxiv.org/abs/2405.20835) report smaller calibration effects for modern models. That finding concerns which data to use, not token counts below the layer width.

**Decision.** 128 windows × 2,048 tokens from native Wikipedia (dump 20231101, pinned commit `b04c8d1c`), at most 512 tokens per article, equal budgets for English and multilingual, equal windows per language in the multilingual set. FLORES dev was rejected because English has only about 30,000 tokens, too few for 262,144-token sets with distinct seeds. It also shares a domain with the FLORES devtest evaluation, which flatters low-resource NLL. 512 × 2,048 was rejected: it costs 4× the quantization time on a T4 across at least 14 quantizations, for gains the literature calls small. Known limitation: Chinese Wikipedia mixes Traditional and Simplified script, while FLORES and Belebele use Simplified. The run directory stores `calibration.json` (article IDs, shard, revision, SHA-256 of the token IDs), and every result row carries `calib_sha256`.

**FR-3 reference.** The official checkpoint was a confounded reference: undisclosed calibration, symmetric quantization, and made with AutoGPTQ, not GPTQModel as the PRD names. The primary reference is now GPTQModel 7.6.0 on our exact windows. Its defaults differ from classic GPTQ (group-aware reordering on, damping 0.05, symmetric), so `replicate.GPTQMODEL_CONFIG` pins asymmetric, group 128, `desc_act=False`, `act_group_aware=False`, damping 0.01, true-sequential, no MSE search. A gap then measures implementation, not data. The official checkpoint is printed as a secondary, descriptive comparison. Notebook 02 now runs FR-3 first and stops before the grid on failure.

**Engineering.** `fakequant.apply` now captures the first block's inputs once and feeds each quantized block's outputs forward. Previously it recaptured each block's inputs with a forward pass from the embeddings. It collects Hessians for projections that share inputs (q/k/v, gate/up) in one pass. A test shows identical weights to the old path. `RunConfig.calib_source` changes the run IDs of calibrated cells only. FP16 and W8 runs keep theirs, and the legacy source keeps the old IDs, so superseded runs stay identifiable. All GPTQ cells in the replication grid and the audit re-run; `configs/` was regenerated.

**Reverse if** FR-3 still fails against GPTQModel on identical windows, which would point to an implementation difference, or if a later study shows a larger calibration budget changes per-language results.

## 2026-11-06 — Audit cut: calibration and seed replication on 1.5B only
The user delegated the cut. Options were: full grid (80.1 h); replication on 1.5B only (61.4 h); that plus dropping 3B MGSM eviction (52.3 h); 1.5B only (26.7 h). Chosen: replication on 1.5B only, recorded as the third preregistration amendment before any quantized result. It keeps every factorial cell that H1 to H3 and Shapley need on both models. Dropping 3B MGSM eviction would break the weight × eviction factorial that H2 names first. Calibration language is not a confirmatory hypothesis, and the PRD's own M1 row already scopes the calibration × seed replication to 1.5B with a 3B spot-check. Chimoto et al. (EACL 2026, arXiv 2601.18306) report that GPTQ is notably sensitive to calibration language at 7 to 8B, so 1.5B keeps the full calibration × seed design, which is enough to measure that effect with its seed spread. A single-seed 3B calibration contrast could not separate calibration from seed noise. `audit_grid(replicate=...)` implements this; `configs/audit.yaml` is now 858 runs. Notebook 02's 154 Belebele full-cache runs share run IDs with the audit, so whichever runs first saves the other about 4 h. Slices `audit_slice_00` to `06` (≤ 10 projected hours each, seven sessions) were written with `grids.write_slices` from the rates in the logged notebook-01 throughput table, because no working `HF_TOKEN` was available locally for `cli plan`. GPTQ calibration and loading time is not yet measured and is covered only by the 2 h margin under Kaggle's 12 h limit. Reverse if the 1.5B calibration effect turns out large enough that its scale dependence becomes a main question.

## 2026-11-06 — K1 passes; full audit projects to 80 GPU-hours against 60
Notebook 01 ran on Kaggle T4×2 (one GPU used) at commit `200cea5`, torch 2.15.0+cu132, transformers 5.19.0, Python 3.13. The 33 FP16 baselines from the 31 Oct Colab run were reused unchanged; lm-eval 0.4.13 ran on all items of both tasks in about 74 minutes. Belebele: mean absolute gap 0.11 points over 11 languages (maximum 0.33, Japanese), item agreement 99.8%. MGSM: mean absolute gap 0.80 points (maximum 1.60, German; French and Russian 1.20), agreement 98.4%. Both pass the implemented K1 rule (`crosscheck.k1_verdict`, written 2026-10-21 and shown in notebook 01: per task, the mean absolute gap over languages is at most 1 point). The PRD only says "within 1 point" and does not say per task or per language. Under a strict per-language reading MGSM fails in German, French and Russian, by 0.2 to 0.6 points beyond the limit. On 250 items, 1 point is 2.5 answers, and these gaps are 3 to 4 answers with 96 to 98% item agreement. Differences of that size are expected from generation numerics, not from harness logic. The per-task reading is kept; reverse it if the user wants the strict reading, which would require explaining the three MGSM gaps before M1. The baselines and the reference ran on different machines and torch builds, which can only add disagreement. Logged throughput for 1.5B: Belebele 18.7 GPU-minutes (1.25× the assumed 15), MGSM 68.1 (2.27× the assumed 30), FLORES NLL 7.2. MGSM exceeds the 2× re-plan threshold. Projected from these rates with 3B at the PRD's 2× of 1.5B, the 1,188-run audit needs 80.1 GPU-hours (1.5B: Belebele 13.1, MGSM 13.6; 3B: Belebele 26.2, MGSM 27.2), excluding GPTQ loading and calibration, against 60 budgeted. This replaces the 108-hour start-stamp estimate of 2026-11-03. No cut is chosen yet; M2 waits for the user's decision.

## 2026-11-06 — Detect Kaggle before Colab
A Kaggle T4×2 session printed `platform: colab` and then `secret HF_TOKEN not available (TimeoutException)`. Kaggle's notebook image is built on Colab's and also sets `COLAB_RELEASE_TAG`, so the setup cell asked Colab's `userdata` for the token, which timed out. Both `whopays.env.platform()` and the generated setup cell now check `KAGGLE_KERNEL_RUN_TYPE` first. Without this, Kaggle runs get no `HF_TOKEN`, cannot pull finished runs and cannot push results. Reverse only if Colab itself starts setting Kaggle's variables.

## 2026-11-03 — Pin PuLP below 4 on every Python
The lock resolved PuLP 3.3.2 on Python 3.11 and 4.0.1 on 3.12 and later. PuLP 4 removed `PULP_CBC_CMD` and the bundled CBC binary and changed the `LpVariable` constructor, so the allocator failed on Python 3.12 in CI from `2ad1821` onward, and would have failed on Kaggle. Local tests passed because the M1 runs 3.11, and the branch had no CI until it was pushed. Cap `pulp<4` so every platform solves with the same PuLP and CBC, which also keeps allocation reproducible. Reverse by moving to `pulp[cbc]` with `COIN_CMD` and `add_variable` on all Pythons, once there is a reason to.

## 2026-11-03 — M3 to M5 software is complete on CPU; closed-loop refinement is not built
`report/hypotheses.py` computes H1 to H4 with planted-effect tests, and `report/analysis.py` rebuilds every table from Parquet, listing any factorial with missing cells as pending. H4 reselects the best baseline inside each bootstrap resample, so its identity is not treated as known. When two methods solve to the same bit map at the same calibration they are one run with one set of rows, which is correct and leaves the comparison tied. `whopays.cli parityquant` refuses a K2 failure. 7B loads across both GPUs and refuses one GPU with a clear message. FR-14 (`refine.py`) is P2 and optional, and is not built: it needs a profile measured against an already quantized background, which only makes sense after K3. Build it only if K3 is positive and the budget remains.

## 2026-11-03 — Compute plan exceeds the budget on logged timings; re-plan at the M0 gate
Run start stamps from the 31 Oct T4 run give about 30 min for Belebele and 76 min for MGSM (1.5B, 11 languages), against 15 and 30 assumed. These are wall-clock estimates, not the gate figure, which notebook 01 now prints from summed `wall_time_s`. Assuming 3B costs twice 1.5B, the audit needs about 108 GPU-hours: 42 Belebele cells at 30 and 60 min (63 h) plus 12 MGSM cells at 76 and 152 min (45 h). The PRD budgets 60 h. The PRD's own first lever, dropping MGSM seeds, is already applied. Do not start M2 until `python -m whopays.cli plan` runs on the logged baselines and the user chooses a cut (for example fewer Belebele seeds on 3B, or 1.5B-only replication). Reverse nothing here; this entry records the constraint.

## 2026-11-03 — Probe uses 100 sentences per language; a full profile would take about 33 h
FR-12 asks for a layers × bits × languages tensor in at most 3 GPU-hours on FLORES devtest. The logged run scored 10 languages of FLORES (10,120 sentences) in about 16 min at batch 8. A full 1.5B profile is 28 blocks × 4 widths = 112 passes over 11,132 sentences, about 33 h before quantization time. So the probe defaults to the first 100 sentences per language by item ID at batch 32, in two disjoint halves. Each finished block prints a projected total, and K2 reports split-half reliability, because a smaller sample can pass K2 through noise alone. K2 reads 4-bit vectors only: 8-bit sensitivities are mostly rounding noise and would make the gate pass trivially. Reverse the limit if `noise_limited` is true and the projection leaves room, or for the M4 profile, which feeds allocation directly.

## 2026-11-03 — Exact block verification for compressed caches, off until its T4 gate passes
A plain block forward breaks greedy exactness under compression. Eviction now gives each block token a 4D attention row equal to the cache it would have seen sequentially, and commits eviction only after rollback. The quantized cache takes a block only when no residual flush can occur inside it (`residual + k < residual_length`), and otherwise uses the sequential round. CPU tests with a lossy stand-in for HQQ match greedy across flushes. As the 31 Oct decision requires, `COMPRESSED_BLOCK_VERIFIED` stays False, so runs keep the labelled sequential control until notebook 05's gate passes on a T4. A full-cache draft disagrees with a target under tight eviction, so acceptance can fall there; that is a measured effect, not a bug. Reverse by setting the flag False if any token diverges.

## 2026-11-03 — Downstream notebooks enforce K1 in code
`gates.require_k1` rebuilds the verdict from stored baselines and lm-eval samples, and raises unless both tasks PASS. The PRD says to stop all experiments until K1 passes, and a notebook is otherwise one click from breaking that. An override needs a written reason, is logged to `gate_overrides.jsonl`, and never counts as a pass.

## 2026-11-03 — lm-eval runs per task at its own batch size; rejected tokens fail loudly
The 31 Oct T4 attempt finished all 33 FP16 baselines, then lm-eval died at 32 of 39,600 Belebele requests with `torch.OutOfMemoryError ... Tried to allocate 4.12 GiB`, so K1 is untested, not failed. Our scorer keeps only last-position logits; lm-eval takes `log_softmax` over vocabulary logits for every position, so long contexts at batch 8 exceed a T4. Belebele now runs at batch 2 and MGSM stays at batch 8, in separate processes so generation is not slowed and a finished task keeps its samples. `expandable_segments` reduces fragmentation. Throughput now prints before lm-eval, so a reference failure cannot hide the re-plan gate. Batch composition can shift FP16 near-ties; the K1 item-agreement check already covers that. Separately, the hub reports an invalid token as `RepositoryNotFoundError` with status 401, which `pull_runs` read as "no results yet", so a fresh session could restart finished runs. A 401 now raises `PermissionError`. Reverse batch 2 only if Belebele still runs out of memory (lower it) or the measured reference time forces a larger value that still fits.

## 2026-11-03 — Baseline provenance: run wheel not mapped to a commit
All 33 FP16 baselines log `wheel:sha256:1ffb8c58...`. That wheel is not retained locally and no record maps it to a commit; `dist/` holds a later rebuild with a different hash. `git diff 5b0ca6c..HEAD` is empty for tasks, models, config, results, run and crosscheck, so scoring code has not moved since the first commit that could have produced an embedded-lock wheel, but that does not prove which source built it. Accept these baselines for K1 and disclose the gap. The repository is now public, so later runs install by commit and log the SHA instead of uploading wheels. Reverse by re-running the baselines at a pinned commit if K1 item agreement looks systematic.

## 2026-11-03 — H2 p-values come from item-clustered GEE, not variational tails
Confirmatory H2 tests are language-specific weight × KV interactions on the population-averaged logit, with Mancl–DeRouen bias-reduced covariance clustered by item and t(items − 1) p-values, then Benjamini–Hochberg over that family only. Variational random-intercept standard deviations are not sampling variances, and turning them into p-values would manufacture the BH inputs the preregistration forbids. Pinned statsmodels has no frequentist binomial GLMM likelihood, and its conditional logit failed to return a covariance on the saturated within-item design, so the operational estimand is the marginal logit interaction rather than the written `(1|item)` coefficient. A cell with no outcome variation aborts the analysis instead of dropping a language. Reverse this if a pinned, frequentist item-intercept likelihood is added by a further dated amendment before any treatment results are inspected.

## 2026-11-03 — Fractional budgets use a sensitivity-free matched-uniform baseline
`matched_uniform` is the H4 comparator when the average bit budget is not an allowed width. It minimises parameter-weighted squared deviation of layer widths from the target, subject to the same exact bit total as the other objectives, and breaks ties by placing higher widths on earlier layers. It does not read the sensitivity tensor, so a 3.5-bit uniform baseline is no longer undefined and the comparison is not secretly another sensitivity policy. Uniform stays in place when the target itself is an allowed width. Reverse it only for another sensitivity-free rule that is also unique at every feasible budget and is declared before allocation results are inspected.

## 2026-11-03 — Profile resume uses the item-run environment pin
A sensitivity profile may resume only when the manifest matches the texts, settings, model identity, code version, GPU, torch, transformers and CUDA build. Item runs already refuse to mix those fields; profile resume previously checked texts and settings only, so a later environment could append rows to an earlier file. Rows now record the same pinned versions. An old manifest without these fields does not match and must start a new file. Reverse the pin only if a signed artifact manifest replaces code and library identity.

## 2026-10-31 — Dated technical map with a readable interview reference
Create an explanatory poster and companion interview guide from the specification, current
code and latest user-shared Colab log. Separate CPU implementation checks, partial GPU job
completion, pending gates and proposed research contributions; do not draw invented result
charts. The guide carries exact equations and limitations, including the compressed-cache
sequential control, descriptive Oaxaca, incomplete H2 inference and allocation-budget
semantics. The poster is a snapshot, not live monitoring or an experiment figure. Update
its evidence labels only after reviewing new logs; keep the generation and correction
briefs with the artifact so the visual can be regenerated.

## 2026-10-31 — Check both PyTorch media companions in one text-runtime repair
After torchvision removal, the next Colab attempt reached torchaudio and failed because
its CUDA build was 13.0 while PyTorch reported 13.2. Both are optional for the text tasks.
Extend the existing subprocess preflight to check torchvision and torchaudio before any
removal, uninstall only failed companions together, and preserve healthy companions and
the platform torch. This avoids serial vision/audio failures and repeated restarts. The
existing uploaded notebook can use `%pip uninstall -y torchvision torchaudio`, then a
session restart and Run all with the same wheel. No scores were produced; K1 remains
untested. Revisit if a task starts using audio or vision, requiring compatible builds.

## 2026-10-31 — Repair broken optional torchvision before text model imports
The subsequent Colab T4 baseline attempt failed before scoring: Qwen2's import reached
torchvision, whose registration raised `operator torchvision::nms does not exist`. The
missing-run Hugging Face warning preceding it was a normal first-run resume lookup.
Probe torchvision in a subprocess during notebook setup; preserve healthy installations,
and uninstall only a failed optional torchvision while keeping the platform torch build.
If Transformers was already imported, require a session restart to clear cached package
availability. This text-only repair lives in `env.py` and is called before version logging.
K1 remains untested, rather than failed on score agreement. Revisit if a future task
actually requires vision, in which case a matching torch/torchvision stack is necessary.

## 2026-10-31 — Accept quoted wheel paths pasted into Colab forms
The next Colab T4 setup attempt received literal double quotes around the wheel path and
stopped before package installation or evaluation; this does not test or fail K1. Strip
outer whitespace and one matching pair of path quotes before validation, and show an
unquoted form example. Regression tests exercise quoted form values through the actual
setup code. The existing uploaded wheel remains usable after correcting the form; this
change is in notebook bootstrap only. The absent HF token must be supplied through secrets
for gated FLORES access and remote result recovery. Keep strict file/extension validation;
reverse quote normalization only if literal surrounding quotes become meaningful filenames.

## 2026-10-31 — Offline wheels carry the lock and use content identity
Include `requirements.lock` in built wheels, enforce those constraints in the upload setup
while preserving the platform scientific base as in the Git install, and log the wheel's
SHA-256 as code identity. Filename-only identity allowed rebuilt code to resume the same
results folder, and the old wheel path skipped dependency pins. Reject older wheels without
the embedded lock. Force reinstall our wheel without reinstalling dependencies, and require
a fresh kernel when changing builds so cached imports cannot be labelled with new code
identity. Revisit only if a signed artifact manifest supplies equivalent immutable
identity and pinned installation. The local build used setuptools' bundled wheel builder
with `--no-isolation --skip-dependency-check` because standalone `wheel` is absent here.

## 2026-10-31 — Reconcile implementation, keep K1 first, and checkpoint the reference run
Preserve and verify the existing uncommitted CPU research modules while restoring the
missing KV/decode APIs that prevented test collection. This is software progress, not
evidence that M1–M5 gates passed. K1 remains the next research run. Baseline failure now
stops notebook 01 before reference evaluation; lm-eval checkpoints consistent SQLite
snapshots during execution and on exit so a dead runtime can recover remote requests.
The throughput gate uses the unrounded ratio. Reverse the execution order only after
reviewing the complete logged T4 K1 outputs and measured compute projection.

## 2026-10-31 — Correct the pairwise Oaxaca specification before observing results
Amend the PRD/prospective preregistration to omit script from English-versus-language
decompositions and reject any covariate span that contains the group indicator. Differing
group-constant scripts make pooled OLS fit both group means and force the residual treatment
gap to zero, predetermining H3. Keep the split descriptive and the H3 threshold unchanged.
A separately specified across-language exploratory model could use script; reversing this
restriction would require a design that does not mechanically identify pairwise group.

## 2026-10-31 — Compressed-cache speculation is currently a correctness control
Use real k-token draft/block verification with rollback for the full cache. For eviction and
quant4, draft k proposals but verify the accepted prefix sequentially so every target token
sees the same eviction/quantization state as plain greedy. Roll back only the full draft
cache, count all actual forwards and wasted proposals, and log no target acceleration.
Do not interpret this fallback's timing as an accelerated-stack result. Replace it only
when a compressed-state block verifier passes the multilingual token-exactness gate.

## 2026-10-31 — Narrow novelty claims after primary-source search (K4 review)
FairQuant already covers group-aware fairness/bit budgets in medical imaging; Colla-Q
covers minimax balancing across MoE experts. Retain only the specific worst-language
degradation objective for dense multilingual LLMs as a provisional ParityQuant distinction.
The indexed Dynamic KV-Cache Compression paper also contains multilingual analyses, so
withdraw the earlier blanket absence claim pending full-text review. Cite these and the
recent dialect-quantization study; neither a universal low-resource disadvantage nor
priority over all multilingual KV work is assumed. Keep the composed-stack audit as the
main direction unless closer prior art covers that actual comparison. Sources and proposed
low-compute extensions are recorded in `REVIEW_2026-10-31.md`.

## 2026-10-21 — K1 gate: how ours vs lm-evaluation-harness is compared
**Setup.** lm-eval 0.4.13 runs the K1 model (Qwen2.5-1.5B-Instruct) on `belebele_<flores>` and `mgsm_native_cot_<lang>` for all 11 languages, in FP16 with `--gen_kwargs repetition_penalty=1.0`. It is installed in its own venv, constrained to the system torch, transformers and accelerate, so its dependencies stay out of the lock. `--log_samples` gives per-item results. These are matched to our items: Belebele by link plus question number, MGSM by row position.

**What is compared.**
- Belebele: our `correct` against lm-eval `acc`.
- MGSM: our `correct_ascii` against lm-eval `exact_match` under the `flexible-extract` filter.

Our native-digit `correct` is reported but not gated, because lm-eval has no equivalent.

**Gate.** For each task, the mean over languages of |ours − lm-eval| must be ≤ 1 point. Any single language over 1 point is flagged and inspected through item-level agreement. A gap made of a few FP16 near-ties, at agreement of 0.98 or above, is benign. A systematic gap is a harness bug.

**Constraints.**
- Belebele is compared on full sets only. lm-eval's `--limit` takes items in file order, while ours takes them in sorted ID order, so limited runs score different items.
- lm-eval's request cache (`--use_cache`) is pushed to the results dataset after the run, and pulled at the start of the next session to resume.

**Local plumbing check (M1, CPU, FP32; not a result).** lm-eval ran with exactly these arguments on 3 Belebele te and 3 MGSM te items with the 0.5B. `read_samples` mapped every item to our IDs. Our MGSM generations were text-identical to lm-eval's, and Belebele correctness matched item for item.

## 2026-10-21 — Env check, first Colab T4 run: what it settled and what it broke
The run used Colab's T4 (sm75, 14.6 GB) with Python 3.13.15, torch 2.11.0+cu130 and CUDA 13.0. The pinned transformers 5.19.0 installed over Colab's 5.20.1 without moving torch.

**Settled:**
- FP16 with SDPA generates correctly.
- Two-pass scoring (prefill the context, then score the query) matches one forward pass: max logit difference 0.027, same argmax.
- Manual sink-plus-recent eviction is exact at no eviction.
- `DynamicCache.crop(-n)` rolls back exactly.
- The 0.5B, 1.5B, 3B and 7B tokenizers give identical IDs on all 20 multilingual probe sentences, so drafting needs no token mapping.
- On 10 prompts, FP16 assisted generation with a 0.5B draft and k=4 matched plain greedy token for token on the 1.5B target.
- All three dataset SHAs match the pins. FLORES parquet has a `sentence` column, which the loader already reads.

**Broke:**
- **Quantized-cache rollback.** `QuantizedLayer` inherits `DynamicLayer.crop`, which trims the FP16 residual buffer but leaves `cumulative_length` unchanged. `get_seq_length()` reports that counter, so after a crop the next forward builds a mask too long by the cropped count (`expanded size 436 must match 442`). Speculative decoding combined with `quant4` KV therefore needs our own rollback, which trims the residual buffer and rewinds `cumulative_length`. That rollback works only while the rejected tokens are still in the residual buffer. The env check now tests both the stock crop and this patched crop.
- **Isolated venvs (kvpress, GPTQModel).** Colab's Python 3.13 has no ensurepip, so `python -m venv` failed. Venvs are now created with `--without-pip`, and the system pip, visible through `--system-site-packages`, installs into them.

**Second run (same day):**
- **HQQ works.** The 4-bit cache perturbs the prefill-then-query logits (max diff 4.36, same argmax). The stock crop fails as predicted, and the patched crop is exact (diff 0.0).
- **Quanto does not work.** optimum-quanto first failed to import because Colab's preinstalled diffusers breaks against its torchao; uninstalling diffusers got past that. Its CUDA extension then failed to JIT-build on the T4 (`quanto_cuda.so: cannot open shared object file`, after 259 s).
- **kvpress** failed: Colab preinstalls an old `fire`, which imports `pipes`, and Python 3.13 removed `pipes`. The fix pins `fire>=0.7` in the venv.
- **GPTQModel** failed: its AUTO backend picked Marlin, whose CUDA kernels fail to compile for sm75. The fix loads with `BACKEND.GPTQ_TORCH`, which needs no build.

**Third run (same day):**
- **GPTQModel passes.** GPTQModel 7.6.0 with `BACKEND.GPTQ_TORCH` loads `Qwen2.5-1.5B-Instruct-GPTQ-Int4` in 31 s and generates correct text on the T4, so it is usable as the FR-3 reference.
- **kvpress still fails**, now at model load: the transformers 4.x that kvpress pulls in takes `torch_dtype`, not `dtype`. The script now uses `torch_dtype`, and runs with `USE_TF=0` so Colab's TensorFlow log noise no longer buries the traceback.

**Fourth run (same day, fresh session): all 12 checks pass.**
- kvpress 0.2.10 runs in its own venv on transformers 4.53.3. `KnormPress(0.5)` keeps 215 of 430 prompt tokens, and the next token is " bridge".
- GPTQModel loads in 22 s with the `TorchLinear` kernel.
- FP16 assisted generation again matches plain greedy on 10 of 10 prompts.

kvpress stays a reference only (see the kvpress conflict entry); the grid uses hand-written eviction.

## 2026-10-21 — KV quantization backend: HQQ, not quanto
`kv=quant4` uses `QuantizedCache(backend="hqq", nbits=4)`, rolled back with our patched crop. Quanto is removed from the `gpu` extra and from the lock. Reverse only if HQQ breaks under a future transformers pin and quanto ships a prebuilt sm75 kernel.

## 2026-10-21 — Task protocols: lm-eval templates, pure greedy, native-digit extraction, parallel item IDs
**Prompts.** All three tasks use raw prompts, not chat templates, matching lm-evaluation-harness defaults so the K1 gate compares like with like. Belebele uses lm-eval's zero-shot template. MGSM uses zero-shot `mgsm_native_cot`: each language gets its own question marker and "step-by-step answer" cue, generation stops at the next question marker or EOS, and output is capped at 256 tokens.

**Greedy decoding.** MGSM decoding is pure greedy. Qwen2.5's `generation_config` sets `repetition_penalty=1.1` plus sampling parameters, and these would otherwise apply even with `do_sample=False`. Pure argmax is also required for the speculative-decoding exactness test. An lm-eval MGSM cross-check therefore needs `--gen_kwargs repetition_penalty=1.0`.

**Answer extraction.** MGSM takes the last number in the output and reads any Unicode decimal digits. ASCII-only extraction would score a correct "১৮" as wrong and create a fake Bengali/Telugu/Thai gap. The lm-eval-style ASCII reading is stored as `correct_ascii`, and missing numbers as `extraction_failed`.

**Item IDs.** Item IDs are parallel across languages, so cross-language pairing is by ID. Belebele uses a hash of the source link plus the question number; its per-language files order items differently, so items are sorted by ID and `limit` takes the same items in every language. MGSM and FLORES use row position, since they are parallel by position.

**FLORES scoring.** Each sentence is scored after the model's BOS token (`<|endoftext|>` for Qwen), so its first token is predicted too. Rows store the NLL sum with token, character and byte counts, so per-token NLL or bits per byte can be derived.

**FLORES source.** The source stays `facebook/flores`, pinned. It is frozen upstream in favour of `openlanguagedata/flores_plus`, but Belebele is built from FLORES-200, so both benchmarks share one text version. Reverse if a FLORES-200 error affects one of the 11 languages.

**Resume caveat.** Batched generation and NLL in FP16 can shift near-tied greedy choices with padding and batch composition, so `batch_size` is logged per row. A run resumed with a different batch size is therefore not guaranteed byte-identical; with the same batch size it is.

## 2026-10-21 — KV policies act on the context; the question is scored against the compressed cache
Belebele uses the lm-evaluation-harness template. The prompt is split at `\nQ:`. The context (`P: {passage}`) is prefilled with full-precision attention, the KV policy is then applied to its cache, and the query (`Q: … A: … D: …\nAnswer:`) is forwarded against that cache. The score is the A/B/C/D letter logits at the last position. The `full` policy uses the same two-pass path, so cells differ only by policy. Why this design:
- In transformers 5.19, `QuantizedLayer.update` returns unquantized K/V on its first call, so quantization only affects tokens processed later (checked in the source). In a single forward pass, `quant4` would therefore be a no-op.
- kvpress compresses only context tokens, and its docs warn that compressing the question too "artificially favors" query-aware methods.
- The long-context KV benchmark (arXiv 2407.01527) likewise never uses the quantized cache during prefill.

Compressing without seeing the question (query-agnostic) is also the realistic case for document and prefix caching. The eviction rule is query-agnostic sink + recent: keep the first 4 tokens plus the most recent ones up to the budget, as in StreamingLLM, with the original RoPE positions kept. SDPA exposes no attention weights, and attention-scored rules (H2O, SnapKV) would need eager attention on the T4 and are query-aware. For MGSM, the policy is applied after the prompt prefill, and the budget is maintained during decoding.

Consequence to keep in mind: on Belebele, a fixed 256-token budget leaves English and the Latin-script European languages almost untouched, but evicts from nearly every Telugu and Bengali passage. The `evict` cell is therefore largely a token-premium treatment, which is the H2 mechanism. **Proposed, not adopted (it changes the PRD grid):** add a ratio-matched eviction level (keep 50% of the context) on Belebele only. That would separate "more tokens are evicted" from "the language is more sensitive to eviction". Reverse the main protocol only if the env check shows the two-pass `full` path drifts from single-pass scoring by more than FP16 noise.

## 2026-10-21 — Rolling back a quantized cache only works inside the FP16 residual window
`QuantizedLayer` inherits `DynamicLayer.crop`, which trims only the unquantized residual buffer (default 128 tokens). Tokens already folded into the quantized store cannot be rolled back. The speculative loop under `quant4` (FR-4 × FR-5) must therefore verify and crop before a residual flush, or rebuild the store. `crop` takes a negative count (tokens to remove) in 5.x; positive arguments are deprecated.

## 2026-10-21 — Dependency pins: transformers 5.19.0 pinned, torch and the scientific base not forced
`requirements.lock` (from `scripts/lock.sh`, one-week `--exclude-newer` buffer) pins everything except torch and the CUDA stack, which Colab and Kaggle preinstall. The notebook setup cell also leaves the platform's numpy, pandas, pyarrow, scipy and matplotlib alone, because replacing them breaks preinstalled packages. torch, transformers, CUDA, GPU and commit are logged in every result row. kvpress 0.5.5 requires `transformers<5.3`, but GPTQModel 7.6 requires `>=5.14`. So kvpress is not a dependency: fixed-budget eviction will be hand-written (the PRD already allows this), and `00_env_check` probes kvpress only in an isolated venv. Reverse if kvpress supports transformers ≥ 5.14 and offers a press we need.

## 2026-10-21 — Run identity: one run = weights × KV × decoding × task × language
`RunConfig.run_id` is the first 12 hex characters of the SHA-256 of canonical JSON (sorted keys, numbers normalised to 12 significant digits, integral floats written as ints). Fields that cannot affect a cell are dropped before hashing: the seed for FP16/RTN, the group size for FP16, the KV budget unless eviction is on, and the draft/k unless speculative decoding is on. Execution details (`batch_size`, `tag`) are never hashed, so a smaller batch on a different GPU still resumes the same run. `model_key()` identifies the loaded weights, so jobs that differ only in task, language or KV policy reuse one quantized model. `SCHEMA_VERSION` must be bumped, and logged here, whenever canonicalisation changes; a test pins one hash to catch accidental changes.

## 2026-10-21 — Resume refuses to mix environments; hub mirrors run folders at the repo root
`meta.json` records the environment of every (re)start. Resuming under a different code commit, GPU, torch or transformers fails, unless `allow_env_change=True` is passed, because a mixed file would confound the measurements. Run folders are uploaded to the root of the HF dataset (`<run_id>/…`), the same layout as the local results root. Pushes are throttled to one every 10 minutes plus a final push, to stay under HF commit-rate limits. Before running, a missing local folder is pulled from the hub, so a fresh Colab/Kaggle session resumes or skips finished runs. Hub failures are logged, never fatal: the local copy is authoritative. The repo defaults to `Heavy-Dervish1205/whopays-results`, the owner's HF account (confirmed 2026-10-21; it differs from the GitHub username). Override it with `WHOPAYS_HF_REPO`.

## 2026-10-21 — Notebook installs: git fetch at a ref, default `main`
The setup cell fetches the repo at `WHOPAYS_REF` (a branch, tag or SHA), installs it under the lock constraints, and exports the commit as `WHOPAYS_CODE_VERSION`. Committed notebooks default to `main` so they don't churn on every commit. Grid launches should be built with `--ref <sha>`, and the resume guard above stops a grid from silently mixing commits. For a private repo, `GH_TOKEN` is passed through `GIT_CONFIG_*` environment variables, so it never appears in a URL or the log. Offline fallback: a `whopays-*.whl` uploaded to `/content` or a Kaggle input is installed instead.

## 2026-10-21 — Python ≥ 3.11
Local development runs on 3.11 (CLAUDE.md), but Colab and Kaggle ship 3.12. `requires-python = ">=3.11,<3.14"`, and CI tests both 3.11 and 3.12.

## 2026-10-21 — Notebooks are thin runners generated from code
Notebooks install the repo at a pinned commit and call `whopays.run()`. They are generated by `scripts/build_notebooks.py`, never hand-edited. Why: Claude Code writes them on the M1 but cannot run them, so a single source of truth keeps Colab, Kaggle and local in sync. Reverse if Colab install time becomes a bottleneck (more than 3 minutes per session).

## 2026-10-21 — Colab for debugging, Kaggle for the grid
Free Colab gives one T4 with dynamic limits. Kaggle gives about 30 h a week of T4 with background commits. The M2 grid (~60 GPU-h) goes on Kaggle. Reverse if the Kaggle quota drops below 15 h a week.

## 2026-10-21 — Fake quantization and our own GPTQ / speculative loop
Library kernels for T4 (sm75) are unreliable, and per-layer mixed bit-widths aren't supported by off-the-shelf kernels. Quality is measured with quantize-dequantize in FP16, and memory is computed from bit counts. Reverse only for an optional "real kernel" spot-check.

## 2026-10-21 — Qwen2.5 Instruct family (0.5B draft; 1.5B and 3B targets; 7B confirm)
It has a shared tokenizer for speculative drafting, strong multilingual coverage and use in prior work. Llama-3.2 is the optional second family.

## 2026-10-21 — Project choice: WhoPays + ParityQuant
Chosen over AegisServe and the on-device compression pipeline. See `RESEARCH_CONTEXT.md`.
