# Decisions log

Newest first. One paragraph per decision: what was decided, why, and what would reverse it.

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
