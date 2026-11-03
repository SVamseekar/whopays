# Decisions log

Newest first. One paragraph per decision: what was decided, why, and what would reverse it.

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
