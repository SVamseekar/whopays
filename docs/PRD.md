# WhoPays + ParityQuant — Product Requirements Document

Version 1.0 · 21 Oct 2026 · Owner: Soura Vamseekar Marti
Live, editable copy: https://claude.ai/artifact/K2am6Qk83YT8GmRFwWWAjj
This file is the source of truth for Claude Code. If the live doc changes, re-export it here.

## Summary and compute verdict

**The whole project fits on free cloud GPUs.** Kaggle runs the heavy experiments, Colab handles quick tests, and the M1 Air handles code, statistics and writing. Total budget is about 145 T4 GPU-hours over 16 weeks: roughly 9 hours a week on average, about 15 in the busiest month. That is well under Kaggle's free weekly quota.

Two design choices make it fit:

- **Models stay at 0.5–3B.** 7B appears only in a final confirmation run.
- **Mixed-precision quality is measured with simulated ("fake") quantization.** Weights are quantized and then dequantized in FP16, so no special GPU kernels are needed. Memory savings are computed from bit counts, not measured.

**What the project is:**

- **WhoPays** measures how quantization, KV-cache compression and speculative decoding change quality and speed differently across 11 languages, then attributes each gap to the individual layer that causes it.
- **ParityQuant** is a new mixed-precision quantizer that minimizes the worst language's degradation at a fixed memory budget.

| Platform | GPU | Limits | Role in this project |
| --- | --- | --- | --- |
| Kaggle Notebooks | 2× T4 16 GB, or P100 | About 30 GPU-h/week (the quota floats); 12 h per session; about 20 GB disk | Main experiment runner, using background "Save & Run All" |
| Google Colab (free) | 1× T4 16 GB, not guaranteed | Dynamic limits; sessions up to about 12 h; idle disconnects | Debugging and short runs |
| Colab Pro (optional) | L4 or A100 when available | $9.99 for 100 compute units, which run out | Only if 7B runs are too slow. Linking it to Kaggle reportedly adds 15 h/week there |
| MacBook Air M1, 16 GB | Apple GPU (MPS), unified memory | CUDA-only libraries do not run | Code, unit tests on the 0.5B model, all statistics, plots, writing |

**What will not work, and the workaround:**

- **7B in FP16 on one T4.** The weights alone are about 15 GB, plus the KV cache. Use Kaggle T4×2 or official pre-quantized 4-bit checkpoints.
- **bf16 and FlashAttention-2 on T4.** T4 lacks native support. Use FP16 with PyTorch SDPA.
- **Wall-clock latency compared across platforms.** Report hardware-independent speed metrics instead: acceptance rate, accepted tokens per step, tokens per request. Measure milliseconds only on one pinned GPU type.
- **Long unattended runs on free Colab.** Sessions get killed, so every job must be resumable (FR-8).

## Problem, goals and success criteria

Inference optimizations are validated on average, mostly on English benchmarks, but their costs fall unevenly across languages. Nobody has measured the combined stack's quality and speed gaps per language, or built a quantizer whose objective is the worst-off language.

**What is already known:**

- **Quantization:** hurts non-Latin-script and lower-resource languages more ([Marchisio et al., 2024](https://arxiv.org/pdf/2407.03211); [MT study](https://arxiv.org/pdf/2508.20893)).
- **Calibration language:** multilingual calibration data improves quantized multilingual models ([Chimoto et al., EACL 2026](https://arxiv.org/abs/2601.18306)).
- **Speculative decoding:** acceptance rates fall for lower-resource languages ([Sandler et al.](https://arxiv.org/abs/2510.02128); [Paudel et al.](https://arxiv.org/pdf/2605.30580)).
- **Composition:** a factorial design over weight, KV and speculative-decoding changes has been proposed for safety, not for languages or speed ([Waheed et al., Oct 2026](https://www.researchsquare.com/article/rs-10796407/v1)).

**What is missing:**

1. A per-language audit of the *combined* stack that measures quality gaps and speed gaps separately.
2. An attribution of each gap to the layer that causes it, separating what the language brings (token premium, base-model weakness) from what the stack does.
3. A quantizer that directly minimizes the worst language's degradation.

### Goals

| ID | Goal | Output |
| --- | --- | --- |
| G1 | Audit per-language quality and speed under weight quantization × KV policy × speculative decoding, on 2 model sizes and 11 languages | Per-item results dataset; gap tables |
| G2 | Decompose each gap by layer (Shapley) and into endowment vs treatment (Oaxaca–Blinder) | Attribution figures |
| G3 | Build ParityQuant: mixed-precision bit allocation that minimizes worst-language degradation at a fixed average bit-width | Method, code, price-of-parity curve |
| G4 | Release everything reproducibly | GitHub repo, Kaggle/Colab notebooks, HF dataset, HF Space dashboard, arXiv preprint |

### Non-goals

- **No training or fine-tuning of models.** Calibration only.
- **No production serving at scale.** No vLLM clusters, no new CUDA kernels.
- **No routing or semantic caching in v1.** Both are stretch goals.
- **No demographic fairness beyond language.** BBQ is an optional control.
- **Not trying to beat state-of-the-art average perplexity.** The claim is about worst-language behaviour at equal bits.

### Success criteria

| Criterion | Target | Checked by |
| --- | --- | --- |
| Harness validity | FP16 scores within 1 point of lm-evaluation-harness on the same model and task | M0 gate |
| Replication | Published direction of the per-language quantization gap reproduced, with 95% CIs | M1 gate |
| Audit coverage | 2 sizes × 11 languages × 12 quality cells, plus a speed subset; all per-item outputs released | M2 gate |
| ParityQuant result | At equal average bits, worst-language degradation lower than the best baseline with a CI excluding 0, at 2 of 3 budgets; mean accuracy cost ≤ 1 point | M4 gate. A null result is still published |
| Reproducibility | Every figure regenerates from released files in under 10 minutes on the M1, CPU only | Release checklist |
| Write-up | arXiv preprint plus one workshop submission | Week 16 |

## Research questions and hypotheses

There are four research questions. Each has a pre-registered hypothesis and a falsification test, so a null result is still a finding.

| RQ | Question | Hypothesis | Falsified if |
| --- | --- | --- | --- |
| RQ1 | How large are per-language quality and speed gaps for each optimization alone? | H1: Gaps grow with token premium and with lower base accuracy. Quality gaps come from weight and KV compression. Speed gaps come from speculative decoding, whose output is unchanged under greedy verification | Gap size is uncorrelated with both token premium and FP16 accuracy (the Spearman ρ CI includes 0) |
| RQ2 | Are the effects additive when layers are combined? | H2: At least one pairwise interaction is non-zero. Leading candidate: weight precision × fixed-budget KV eviction, through token premium | All interaction terms are indistinguishable from 0 after Benjamini–Hochberg correction |
| RQ3 | How much of each language's gap is endowment versus treatment? | H3: Endowment explains less than half the gap for at least 3 non-Latin-script languages | Endowment share is above 50% for every language |
| RQ4 | Can bit allocation shrink the worst-language gap at equal memory? | H4: ParityQuant reduces worst-language degradation versus the best baseline at 3.0, 3.5 and 4.0 average bits, at ≤ 1 point mean cost | The improvement CI includes 0 at 2 or more of the 3 budgets |

Two further rules:

- **Exactness control for speculative decoding.** Under greedy verification, outputs must equal non-speculative greedy outputs token for token. Any difference is a harness bug, not a finding.
- **Keep claims tied to the RQs.** Results outside these four questions are labelled exploratory.

## Scope: models, languages, datasets, factors

Primary scope: one model family (Qwen2.5 Instruct, 0.5B–3B) and 11 languages that appear in both MGSM and Belebele.

### Models

| Model | Role | FP16 weights | Where it runs |
| --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | Draft model for speculative decoding; dev and unit tests | ~1 GB | Anywhere, including the M1 (MPS) |
| Qwen2.5-1.5B-Instruct | Primary target | ~3 GB | T4 |
| Qwen2.5-3B-Instruct | Second target | ~6 GB | T4 |
| Qwen2.5-7B-Instruct | Confirmation only (M5) | ~15 GB | Kaggle T4×2, or official 4-bit checkpoints on one T4 |
| Llama-3.2-1B / 3B-Instruct | Optional second family | ~2.5 / 6.5 GB | T4 (gated: request access in week 1) |

### Languages

| Language | MGSM code | Belebele / FLORES code | Script |
| --- | --- | --- | --- |
| English (reference) | en | eng_Latn | Latin |
| German | de | deu_Latn | Latin |
| French | fr | fra_Latn | Latin |
| Spanish | es | spa_Latn | Latin |
| Russian | ru | rus_Cyrl | Cyrillic |
| Chinese (Simplified) | zh | zho_Hans | Han |
| Japanese | ja | jpn_Jpan | Kana / Kanji |
| Thai | th | tha_Thai | Thai |
| Bengali | bn | ben_Beng | Bengali |
| Swahili | sw | swh_Latn | Latin |
| Telugu | te | tel_Telu | Telugu |

### Datasets

| Dataset | Use | Size per language | Scoring |
| --- | --- | --- | --- |
| Belebele | Reading comprehension; main quality metric (items are parallel across languages) | 900 questions | Single forward pass; argmax over A/B/C/D answer-letter logits |
| MGSM | Generative maths reasoning; speed and decoding measurements | 250 problems | Greedy decoding; numeric answer extraction |
| FLORES-200 dev | Calibration data for quantization | 997 sentences | Not scored |
| FLORES-200 devtest | Held-out per-language perplexity; sensitivity profiling | 1,012 sentences | Token-level NLL |

FLORES dev is never used for scoring. FLORES devtest is never used for calibration.

### Factors and levels

| Factor | Levels | Notes |
| --- | --- | --- |
| Weight precision | FP16 · W8 RTN · W4 GPTQ (group 128) · W3 GPTQ (group 128) | AWQ W4 is an alternative if it runs on T4 (open question) |
| Calibration set | English-only · multilingual FLORES mix | Replicates the calibration-language finding as a sanity anchor |
| KV policy | Full FP16 · 4-bit quantized cache · eviction to a fixed 256-token budget | A fixed budget evicts proportionally more of high-token-premium languages (the H2 mechanism) |
| Speculative decoding | Off · on (0.5B draft, greedy, 4 draft tokens) | Speed only; outputs must match greedy exactly |
| ParityQuant budget | 3.0 · 3.5 · 4.0 average bits; per-layer bits from {2, 3, 4, 8} | M4 only |

The cells in each grid:

- **Quality grid:** 4 weight levels × 3 KV policies = 12 cells per model.
- **Speed grid:** the same 12 cells × speculative decoding on/off, on a 30-problem MGSM subset per language.
- **Calibration language:** crossed with W4 and W3 only.
- **Seeds:** every quantized cell is repeated with 3 calibration-subsample seeds.

## Functional requirements

| Priority | Meaning |
| --- | --- |
| P0 | Must exist before M2 starts |
| P1 | Needed before release |
| P2 | Stretch goal |

Each requirement has a "Done when" check.

### A. Experiment harness

| ID | Requirement | Priority | Done when |
| --- | --- | --- | --- |
| FR-1 | Every run is defined by a YAML config (model, weight level, calibration set, KV policy, speculative flag, dataset, language, seed). A stable hash of the config is the run ID | P0 | Same config gives the same hash on Kaggle, Colab and the M1 |
| FR-2 | Model loader supports FP16, fake-quant (any per-layer bit map), real GPTQ checkpoints and official pre-quantized checkpoints | P0 | Loads all four on a T4; the 0.5B model also loads on M1 MPS |
| FR-3 | Our own minimal GPTQ (group 128, act-order off) with fake quantization and per-layer bit-widths, plus RTN | P0 | Uniform W4 fake-quant perplexity on FLORES devtest is within 2% of a GPTQModel W4 checkpoint |
| FR-4 | KV policies (full, 4-bit quantized cache, fixed-budget eviction) behind one interface | P0 | Each policy runs Belebele for all 11 languages without errors; memory per policy is logged |
| FR-5 | Our own greedy speculative-decoding loop (draft k = 4, target verifies) that logs accepted tokens per step | P0 | Output equals plain greedy output token for token on 100 prompts in all 11 languages |

### B. Evaluation

| ID | Requirement | Priority | Done when |
| --- | --- | --- | --- |
| FR-6 | Task runners: Belebele letter scoring, MGSM generation with numeric answer extraction, FLORES per-token NLL | P0 | FP16 within 1 point of lm-evaluation-harness (M0 gate); answer extraction hand-checked on 50 items per language |
| FR-7 | Per-item results written as JSONL (run ID, language, item ID, correct, gold-answer logprob, input tokens, output tokens, accepted tokens, steps, wall time, GPU type), compacted to Parquet and synced to a private Hugging Face dataset | P0 | Any figure can be rebuilt from Parquet alone |
| FR-8 | Resumable runs: finished items are skipped, results flush every 50 items, an interrupted session continues from the last flush | P0 | Killing a run mid-way and restarting gives an identical final file |
| FR-9 | Token premium: tokens per parallel item divided by the English count, per language and tokenizer | P0 | A cached table of 11 premiums |

### C. Analysis

| ID | Requirement | Priority | Done when |
| --- | --- | --- | --- |
| FR-10 | Statistics module: paired bootstrap CIs over parallel items, mixed-effects logistic regression, Benjamini–Hochberg correction, Shapley attribution over factorial cells, Oaxaca–Blinder decomposition | P0 | Unit tests on synthetic data with planted effects recover those effects within the CI |
| FR-11 | Inequality summaries: worst-language degradation, max–min gap, and Gini and Atkinson indices over per-language degradation | P1 | Matches hand-computed values on a toy example |

### D. ParityQuant

| ID | Requirement | Priority | Done when |
| --- | --- | --- | --- |
| FR-12 | Sensitivity profiler: for each layer and bit-width in {2, 3, 4, 8}, the per-language NLL increase on FLORES devtest, with the other layers at reference precision | P0 | Produces a layers × bits × languages tensor for 1.5B in ≤ 3 GPU-hours |
| FR-13 | Allocator: minimax bit allocation as a mixed-integer program (PuLP + CBC, CPU). Baselines: uniform, average-sensitivity allocation, and multilingual-calibration uniform | P0 | Solves in under 60 s on the M1 for a 36-layer model; bit budget met exactly |
| FR-14 | Closed-loop refinement that re-profiles after the first allocation, to correct for interactions between layers | P2 | Reported as an ablation |

### E. Reporting and release

| ID | Requirement | Priority | Done when |
| --- | --- | --- | --- |
| FR-15 | Figure scripts (per-language gap heat-map, Shapley bars, price-of-parity curve) and a Hugging Face Space dashboard (Gradio, free CPU) reading the released Parquet | P1 | Dashboard loads in under 10 s; filters by model, language and config |
| FR-16 | "Serving disparity card": a one-page markdown/JSON summary per served configuration (per-language quality and speed deltas, token premium, worst-language degradation), for assurance use | P1 | Generated automatically for every audited cell |

## Metrics and statistics

Every headline number is a per-language change relative to FP16 on the same items, with a paired bootstrap 95% CI. Belebele items are parallel across languages, so cross-language comparisons are paired too.

| Metric | Definition | Channel |
| --- | --- | --- |
| Quality change, Δacc(l, c) | Accuracy of config c in language l, minus FP16 accuracy in l | Quality |
| Flip rate | Share of items correct in FP16 and wrong under c | Quality |
| NLL change | Mean per-token NLL change on FLORES devtest | Quality (profiling) |
| Acceptance rate α | Accepted draft tokens / proposed draft tokens | Speed |
| Tokens per step τ | Mean tokens committed per target forward pass | Speed |
| Token premium TP(l) | Tokens for a parallel item in l / tokens for the English version | Endowment |
| Effective cost per answer | Target forward passes per completed answer (combines token premium and acceptance) | Speed |
| Worst-language degradation | max over l of −Δacc(l, c) | Disparity |
| Disparity spread | max − min of Δacc across languages; Gini and Atkinson (ε = 1) over degradations | Disparity |
| Price of parity | Mean-accuracy loss of ParityQuant minus that of the average-objective allocation, at equal bits | ParityQuant |

### Shapley attribution

The three layers (weights W, KV policy K, speculative decoding S) are the players. v(T) is a disparity metric measured with only the layers in T switched on.

    phi_i = sum over T ⊆ N\{i} of  |T|!(n−|T|−1)!/n! · [ v(T ∪ {i}) − v(T) ]

This needs all 8 cells of the 2×2×2 sub-grid at one aggressive level per factor. The full grid already contains them.

The interaction residual for a pair of layers is:

    I(i, j) = v({i, j}) − v({i}) − v({j}) + v(∅)

A positive I means the combined harm is larger than the sum of the parts.

### Oaxaca–Blinder decomposition

Each language l is compared with English.

- **Outcome:** item-level degradation.
- **Covariates:** token count, item length, FP16 log-probability of the gold answer.
  **Amended 31 Oct 2026, before research runs:** exclude script and any covariate combination
  that exactly identifies the language group in an English-versus-language fit. Such a
  design mechanically assigns the whole gap to endowment. Script can be used in separately
  specified exploratory analyses across languages. The split is descriptive, not causal.
- **Split:** endowment (the gap explained by differences in covariates) versus treatment (the gap explained by different returns to those covariates).
- **Reference coefficients:** pooled (Neumark).
- **Reporting:** both parts, with bootstrap CIs.

### ParityQuant objective

Notation:

- x(j, b) = 1 if layer j gets b bits.
- s(l, j, b) is the profiled NLL increase for language l.
- p(j) is the parameter count of layer j.
- B is the budget in average bits.

The problem:

    minimize t
    subject to  Σ_j Σ_b s(l,j,b)·x(j,b) ≤ t        for every language l
                Σ_b x(j,b) = 1                       for every layer j
                Σ_j Σ_b p(j)·b·x(j,b) ≤ B · Σ_j p(j)

This assumes layer effects add up. MixQuant shows that is only approximately true, and FR-14 measures what the approximation costs.

Variants to report:

- **Soft version:** replace the max with a softmax temperature, tracing the curve from average-optimal to worst-optimal.
- **Equity-constrained version:** minimize mean degradation subject to worst-language degradation ≤ δ.

### Statistical protocol

- **CIs:** paired bootstrap, 2,000 resamples over items, stratified by language.
- **Interactions:** mean model `correct ~ W*K*language`, originally written as a
  random-intercept logistic. **Amended 3 Nov 2026, before research runs:** confirmatory
  p-values are language-specific weight × KV contrasts from an item-clustered binomial
  GEE (Mancl–DeRouen SE, t on items−1, BH over that family). The variational
  `(1|item)` fit stays descriptive. See the preregistration amendment.
- **Multiple comparisons:** Benjamini–Hochberg at q = 0.05 across all language × config tests.
- **Seeds:** 3 calibration-subsample seeds per quantized cell; report the seed spread.
- **Pre-registration:** commit a dated PREREGISTRATION.md with these hypotheses and tests before M2 runs.

## Compute plan

The budget is about 145 T4 GPU-hours, including a 20% re-run buffer. These are planning estimates: M0 measures real throughput, and the table is re-planned at the M0 gate.

| Milestone | Weeks | GPU-h | Platform | Main GPU jobs |
| --- | --- | --- | --- | --- |
| M0 Harness | 1–2 | 4 | Colab free, Kaggle | FP16 baselines for 0.5B/1.5B/3B; lm-evaluation-harness cross-check |
| M1 Replication + probe | 3–4 | 16 | Kaggle | Uniform W8/W4/W3 × 2 calibration sets × 3 seeds on 1.5B; 3B spot-check; sensitivity probe on 1.5B (~3 h) |
| M2 Audit grid | 5–8 | 60 | Kaggle | 12 quality cells × 2 models (Belebele + MGSM); extra seeds on Belebele; speed grid of 24 cells on 1.5B and 4 on 3B |
| M3 Decomposition | 9–10 | 0 | M1 Air, CPU | Shapley, Oaxaca–Blinder, mixed-effects models |
| M4 ParityQuant | 11–14 | 27 | Kaggle | Profiling 1.5B and 3B (~9 h); 4 methods × 3 budgets × 2 models; 3 seeds on key cells |
| M5 Scale check + release | 15–16 | 12 | Kaggle T4×2 (or Colab Pro) | 7B: FP16, W4 and ParityQuant at 3.5 bits, on Belebele plus an MGSM subset |
| Buffer | — | 26 | Any | Re-runs and bug fixes |

**Throughput assumptions to check in M0** (rough T4 FP16 batched estimates):

| Task | 1.5B | 3B |
| --- | --- | --- |
| Belebele, 9,900 items (11 languages × 900) | ~15 min | ~30 min |
| MGSM generation, 2,750 problems | ~30 min | ~60 min |

Speculative decoding runs at batch size 1, which is why the speed grid uses a 30-problem subset. If M0 measures 2× slower than this, halve the MGSM seeds first.

### Run protocol

1. Write and unit-test code on the M1, using the 0.5B model (MPS) and a tiny random-weight model on CPU.
2. Push to GitHub. Every Kaggle or Colab notebook installs the repo at a pinned commit. Code is never edited inside a notebook.
3. Launch a job list. Each job is one config hash and is resumable (FR-8). On Kaggle, use "Save & Run All" (background, up to 12 h).
4. Flush results every 50 items to a private Hugging Face dataset, with Google Drive as an optional mirror on Colab. The HF token lives in Kaggle/Colab Secrets, never in code.
5. Pull the Parquet files to the M1 for all statistics and figures.

### What the M1 Air does and does not do

**Does:** all coding, unit tests, 0.5B smoke tests, all statistics (statsmodels, scipy), the ParityQuant solver (PuLP + CBC), figures, the dashboard and writing.

**Does not:** produce any reported measurement. Its numerics and speed differ from a T4, and mixing hardware would confound the results.

## Architecture

Data flows in two halves, with one loop back:

- **On the GPU (Kaggle or Colab T4):** a run config (YAML, hashed to a run ID) goes to the model loader (FP16, fake-quant, GPTQ or pre-quantized). The loader feeds the serving stack (weights W8/W4/W3, one of 3 KV policies, speculative decoding on/off). The stack feeds the task runners (Belebele, MGSM, FLORES NLL), which feed the results writer (JSONL, flush every 50 items, resumable). The writer pushes to the HF dataset (Parquet, private).
- **On the CPU (MacBook Air M1):** the HF dataset feeds Analysis (bootstrap CIs, mixed effects, Shapley, Oaxaca), which feeds Reports (figures, dashboard, disparity cards). The per-layer NLL profiles in the HF dataset also feed ParityQuant (minimax MILP solved with PuLP + CBC, budgets of 3.0–4.0 bits).
- **The loop back:** ParityQuant's output bit maps become new run configs, evaluated on the T4 like any other cell.

### Repository layout

```
whopays/
  configs/                    # YAML run configs, one per grid cell (FR-1)
  src/whopays/
    config.py                 # schema + stable hashing (FR-1)
    env.py                    # detect colab / kaggle / mps / cpu; paths; secrets
    models/loader.py          # FP16, fake-quant, GPTQ, pre-quantized (FR-2)
    quant/rtn.py              # round-to-nearest (FR-3)
    quant/gptq.py             # minimal GPTQ, per-layer bits (FR-3)
    quant/fakequant.py        # quantize-dequantize in FP16 (FR-3)
    kv/policies.py            # full, 4-bit cache, fixed-budget eviction (FR-4)
    decoding/speculative.py   # greedy draft-verify loop with logging (FR-5)
    tasks/belebele.py         # letter scoring (FR-6)
    tasks/mgsm.py             # generation + numeric extraction (FR-6)
    tasks/flores_nll.py       # per-token NLL (FR-6)
    io/results.py             # JSONL writer, resume, Parquet (FR-7, FR-8)
    io/hub_sync.py            # push/pull HF dataset (FR-7)
    tokens/premium.py         # token premium per language (FR-9)
    stats/bootstrap.py        # paired bootstrap CIs (FR-10)
    stats/mixed.py            # mixed-effects logistic (FR-10)
    stats/shapley.py          # attribution over the grid (FR-10)
    stats/oaxaca.py           # endowment vs treatment (FR-10)
    stats/inequality.py       # worst-case, spread, Gini, Atkinson (FR-11)
    parityquant/profile.py    # layer x bits x language sensitivity (FR-12)
    parityquant/allocate.py   # minimax MILP + baselines (FR-13)
    parityquant/refine.py     # closed-loop re-profiling (FR-14)
    report/figures.py         # paper figures (FR-15)
    report/card.py            # serving disparity cards (FR-16)
    run.py                    # run(job_list): the only entry point notebooks call
  notebooks/                  # thin runners, see docs/NOTEBOOKS.md
  space/app.py                # Gradio dashboard (FR-15)
  tests/                      # pytest: tiny random model, synthetic stats
  PREREGISTRATION.md
  requirements.lock
  README.md
  LICENSE                     # Apache-2.0
```

## Milestones and timeline

The plan is 16 weeks, starting Monday 26 Oct 2026. The audit preprint lands in week 10, before ParityQuant starts.

| Phase | Weeks | Dates | Content | Gate at end |
| --- | --- | --- | --- | --- |
| M0 Harness | 1–2 | Oct 26 – Nov 8 | Loader, GPTQ, tasks, results writer, CPU tests | K1: FP16 within 1 pt of lm-eval |
| M1 Replicate | 3–4 | Nov 9 – Nov 22 | Uniform W8/W4/W3, calibration language, layer probe | K2: layer sensitivity differs across languages, or M4 is dropped |
| M2 Audit grid | 5–8 | Nov 23 – Dec 20 | 12 cells × 2 sizes, speed subset | Grid complete as pre-registered |
| M3 Decompose | 9–10 | Dec 21 – Jan 3 | Shapley, Oaxaca, mixed effects; audit preprint | Preprint out |
| M4 ParityQuant | 11–14 | Jan 4 – Jan 31 | Profile, solve, 3 budgets × 2 models, 3 seeds | K3: CI excludes 0, or publish as a negative result |
| M5 Release | 15–16 | Feb 1 – Feb 14 | 7B check, dashboard, cards, final preprint | — |

## Risks, kill criteria, open questions

### Risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Quantization or KV libraries fail on T4 (Turing, sm75) or break with a transformers release | High | Medium | Our own fake-quant GPTQ (FR-3) and speculative loop (FR-5); pin versions in requirements.lock |
| Scooped (the field moved monthly through 2026) | Medium | High | Preprint the audit after M3 (week 10) |
| Kaggle quota or policy changes | Low | Medium | Colab free as fallback; Colab Pro for one crunch month |
| Sessions killed mid-run | High | Low | Resumable runs (FR-8); jobs under 10 h |
| Belebele letter scoring skewed by the model's prior for A/B/C/D | Medium | Medium | Also report calibrated accuracy; full-option log-likelihood on a 200-item subset |
| MGSM answer extraction fails more in some languages, creating fake gaps | Medium | High | Extract the final number only; hand-audit 50 items per language; report extraction failure rate per language |
| Fake quantization differs from real kernels | Low | Medium | Validate uniform W4 against a real GPTQ checkpoint; label memory figures as computed |
| Benchmark contamination | Medium | Low | Shifts levels but mostly not deltas; state it in limitations |
| Small models don't represent production | Medium | Medium | 7B check in M5; scope claims to the tested sizes |

### Kill and pivot criteria

- **K1 (end of week 2):** the harness is not within 1 point of lm-evaluation-harness. Stop all experiments until it is.
- **K2 (end of week 4):** the minimum pairwise Spearman correlation of layer-sensitivity vectors across languages is above 0.9. Drop M4 and extend the audit instead (add Llama-3.2).
- **K3 (week 13):** ParityQuant's gain has a CI that includes 0 at 2 or more budgets. Publish it as a negative result.
- **K4 (any time):** a close preprint appears. Re-read it within 48 h, keep only what it doesn't cover, and cite it.

### Open questions for week 1

- [x] Does GPTQModel load and run on a Kaggle/Colab T4? It is needed only for the FR-3 reference. *Yes on Colab, with `BACKEND.GPTQ_TORCH` in an isolated venv (DECISIONS 2026-10-21).*
- [x] Does transformers `QuantizedCache` (quanto or HQQ backend) work with Qwen2.5 on T4? *HQQ yes, with a patched crop for rollback; quanto no (DECISIONS 2026-10-21).*
- [x] Do NVIDIA kvpress or a hand-written fixed-budget eviction work with Qwen2.5 at the pinned transformers version? *Hand-written yes at 5.19.0; kvpress works only in an isolated venv on transformers 4.53.3 (DECISIONS 2026-10-21).*
- [x] Are 0.5B draft and 7B target token IDs aligned? Embedding sizes differ across Qwen2.5 sizes. *Yes: identical IDs on 20 multilingual probes for 0.5B, 1.5B, 3B and 7B.*
- [ ] Is AWQ worth adding as a second quantizer, given T4 support?
- [ ] Request Llama-3.2 access on Hugging Face (optional).
- [ ] Confirm the Apache-2.0 code licence; check dataset licences before redistributing per-item outputs.

### Deliverables

| Deliverable | Where | Due |
| --- | --- | --- |
| Code repo with tests and pinned env | GitHub (SVamseekar/whopays) | Week 2, then ongoing |
| Run-anywhere notebooks | Kaggle (public) + "Open in Colab" badges | Week 4 |
| PREREGISTRATION.md | Repo | Before M2 (week 5) |
| Per-item results dataset | Hugging Face dataset | Week 10 |
| Audit preprint | arXiv | Weeks 10–11 |
| ParityQuant code + preprint | GitHub + arXiv | Week 16 |
| Dashboard | HF Space (free CPU) | Week 16 |
| Serving disparity cards | Repo + dashboard | Week 16 |

## Sources

- [Chimoto et al., Calibrating Beyond English (EACL 2026)](https://arxiv.org/abs/2601.18306)
- [Marchisio et al., How Does Quantization Affect Multilingual LLMs?](https://arxiv.org/pdf/2407.03211)
- [Quantization and MT across 55 language pairs](https://arxiv.org/pdf/2508.20893)
- [Sandler et al., Disparate Impacts of Speculative Decoding](https://arxiv.org/abs/2510.02128)
- [Paudel et al., Speculative Decoding and the Curse of Multilinguality](https://arxiv.org/pdf/2605.30580)
- [Waheed et al., Speculative Decoding Shortcuts Can Undermine Safety Guarantees (preprint, Oct 2026)](https://www.researchsquare.com/article/rs-10796407/v1)
- [TASA: joint calibration and mixed-precision allocation](https://arxiv.org/pdf/2607.00908)
- [MixQuant: layer sensitivity depends on upstream bit-widths](https://arxiv.org/html/2607.23047v1)
- [CoopQ/SPQE: Shapley-based mixed-precision quantization](https://arxiv.org/abs/2509.15455) (previously listed as IMPQ)
- [FairQuant: fairness-aware mixed precision in medical imaging](https://arxiv.org/abs/2602.23192)
- [Colla-Q: minimax precision balancing across MoE experts](https://arxiv.org/abs/2609.18131)
- [Multilingual Brain Surgeon](https://arxiv.org/pdf/2404.04748)
- [Tokenizer equity study (Parity-aware BPE)](https://arxiv.org/html/2606.15044)
- [Free GPU limits, Oct 2026](https://gpuperhour.com/blog/free-cloud-gpus-and-credits) · [Colab FAQ](https://research.google.com/colaboratory/faq.html)
