# ParityQuant v2: choosing a novel worst-language method (research note, decision pending)

Date: 2026-11-15. Skill: `researching-design-choices`. Context: the owner reopened ParityQuant as
a method paper (DECISIONS 2026-11-15). The original allocator's K3 negative stands and is reported.

## Choice

Which method gives ParityQuant a defensible novel contribution: better worst-language quality at
a fixed bit budget in dense multilingual LLMs, using calibration data only.

## Options (written before searching)

- O1. Interaction-aware minimax bit allocation: evaluate candidate maps end to end per language, not by summed single-block sensitivities; widths of 3 bits or more.
- O2. A different loss signal: per-language KL to FP16 or a task-aligned likelihood instead of NLL increase.
- O3. Smooth worst-case objectives (CVaR, group DRO) over languages.
- O4. Minimax calibration mixture: weights on per-language calibration data chosen to minimise the worst language's loss.
- O5. Joint weight and KV-cache bit allocation for the worst language.
- O6. Language-conditional precision at serving time.
- O7. Worst-language error compensation inside GPTQ: per-language Hessians reweighted by multiplicative weights on worst-language loss.

## What our own work already established (reusable)

| Asset | Where | Use in v2 |
| --- | --- | --- |
| Layer sensitivity differs by language (K2 PASS, min Spearman 0.420, te–zh) | RUN_LOG 15 | Motivation: one allocation cannot be best for all languages |
| Additive single-block model breaks with 2-bit blocks (observed/predicted 1.3–3.8) | DECISIONS 2026-11-14 | Explains the v1 failure; v2 must not rely on it |
| NLL gains need not become accuracy (3.5-bit minimax at chance on Belebele) | `docs/preprint/PARITYQUANT_NEGATIVE.md` | v2's confirmatory test must include accuracy |
| GPTQ seed spread 5–7 nats (Telugu), 0.8–4.9 Belebele points | adversarial review 2026-11-14; RUN_LOG 20 | v2 must evaluate over several GPTQ seeds |
| Calibration language moves results (multi vs en: en −3, ru +2 points) | RUN_LOG 20 | Direct evidence that calibration weights are a lever |
| Own GPTQ with per-module Hessian hooks | `src/whopays/quant/fakequant.py`, `quant/gptq.py` | Per-language Hessians are a small change |
| Probe, solver, held-out FLORES evaluation, `h4_budget`, price-of-parity code | `src/whopays/parityquant/`, `report/hypotheses.py` | Reused for evaluation and comparisons |
| Replication grid: uniform W3/W4 × {en, multi} calibration × 3 seeds on HF | RUN_LOG 14 | Baselines already computed for 1.5B |

## Prior art found (2026-11-15 search: 25 OpenAlex queries, 8 web searches)

| Work | What it does | Gap left for us |
| --- | --- | --- |
| Group-robust LLM pruning ([arXiv 2608.02940](https://doi.org/10.48550/arxiv.2608.02940), Aug 2026) | Worst-group allocation for **pruning** dense 3B LLMs; groups are data sources (general, rare code, rare knowledge, safety); group-balanced calibration cuts worst-group damage 21.7–31.5%; shows pooled scores can pick the worse model; no quantization, no languages | Languages as groups; quantization; their selection-bias lesson applies to us |
| Fair-GPTQ ([TACL 2026](https://doi.org/10.1162/tacl.a.816)) | Group-fairness constraints added to the GPTQ rounding objective; stereotype generation (gender, race, religion) | Not languages; not worst-group quality; constraint on bias, not degradation |
| FairQuant ([arXiv 2602.23192](https://arxiv.org/abs/2602.23192)) | Group-aware mixed precision with fairness regularisation; medical images, ResNet/ViT | Not LLMs, not languages |
| Colla-Q ([arXiv 2609.18131](https://arxiv.org/abs/2609.18131)) | Balanced precision across MoE experts | Experts, not languages; MoE, not dense |
| Balancing Act ([arXiv 2310.20673](https://doi.org/10.48550/arxiv.2310.20673)) | Constrained pruning that bounds each subgroup's accuracy change | Pruning, vision/classification groups |
| Explaining How Quantization Disparately Skews a Model ([arXiv 2509.07222](https://doi.org/10.48550/arxiv.2509.07222)) | Mechanism of disparate quantization damage; fix via mixed-precision QAT with resampling | QAT (training), not calibration-only PTQ |
| Calibrating Beyond English ([EACL 2026](https://doi.org/10.18653/v1/2026.eacl-long.223)) | Eight fixed calibration mixes, GPTQ/AWQ, 10 languages; multilingual mixes best on average | Mixes are fixed and evaluated, not optimised for the worst language |
| Lost in Compression ([IMPACT-Speech 2026](https://doi.org/10.18653/2026.impactspeech-1.6)) | Speech LLM GPTQ: balanced calibration barely moves the accent gap; homogeneous calibration narrows it | Warns that mixture effects are non-monotone: an optimised mixture must be tested, not assumed |
| LangCompress ([IJCNLP-AACL 2025](https://aclanthology.org/2025.ijcnlp-long.112)) | Synthetic calibration data per target language for pruning and quantization | One target language at a time; no worst-language objective |
| Waskito et al. ([Findings AACL 2026](https://doi.org/10.18653/2026.findings-aacl.9)) | Calibration language × bit-width × mixed precision, 5 languages | Evaluation, no worst-language method |
| LCD ([arXiv 2608.11786](https://arxiv.org/abs/2608.11786)) | Per-language LoRA on INT3 models | Post-hoc adapters (extra parameters), not quantization itself |
| CASA ([arXiv 2609.25916](https://doi.org/10.48550/arxiv.2609.25916)); QEP ([NeurIPS 2025](https://doi.org/10.52202/085713-5080)) | Cross-layer refinement of mixed precision; error propagation across layers | Single objective; no groups |
| KL fidelity metrics ([arXiv 2609.07664](https://doi.org/10.48550/arxiv.2609.07664); [EMNLP 2026 industry](https://doi.org/10.18653/2026.emnlp-industry.178)) | KL tracks quality across a cohort but fails to rank near-lossless quantizations (ρ = +0.11 and +0.02) | Argues against O2 as the main signal |
| MixLLM, MixQuant, CoopQ, SliM-LLM | Average-objective mixed precision | No groups |

Web searches for "group DRO GPTQ", "worst-language quantization" and "language-fair compression minimax" found no method that optimises calibration weights or GPTQ's Hessian for the worst language.

## Assessment

| Option | Novelty | Evidence it can work | Cost (T4) | Verdict |
| --- | --- | --- | --- | --- |
| O7 + O4 (worst-language DRO inside GPTQ) | High: no indexed prior work; distinct from Fair-GPTQ (bias constraints), Calibrating Beyond English (fixed mixes), 2608.02940 (pruning) | Calibration language moves per-language results (ours, Chimoto, 2608.02940's 21.7–31.5%); Lost in Compression warns of non-monotone effects | Per 1.5B model about 650 s per GPTQ pass; 5–8 reweighting rounds, so about 1–1.5 GPU-hours per seed and width | **Lead** |
| O1 (measured, interaction-aware minimax allocation) | Medium: CASA-style refinement with a worst-language objective | Our 2-bit diagnosis; CASA | Each candidate map needs a quantize-and-evaluate pass | **Second stage**, optional, widths ≥ 3 |
| O3 (CVaR or soft minimax) | Low alone | — | Free | Use as the objective form inside O7 (temperature controls price of parity) |
| O2 (KL signal) | Low | KL fails to rank good quantizations | — | Reject as main signal; report as diagnostic |
| O5 (joint weights and KV) | Medium | Audit shows the 4-bit cache is nearly free, so little to allocate | High | Reject for now |
| O6 (language-conditional precision) | Medium | — | Systems work, memory overhead | Reject for now |

## Recommended design (to be preregistered before any evaluation)

**ParityGPTQ.** GPTQ's per-module Hessian becomes H(λ) = Σ_l λ_l H_l, with H_l from language l's
calibration windows (Wikipedia, as today). λ starts uniform (the equal-mix baseline, which is
2608.02940's group-balanced calibration). For rounds t = 1..T: quantize with H(λ), measure each
language's NLL increase on a **selection** split of calibration-side text, and update λ_l ∝
λ_l · exp(η · loss_l) (exponentiated gradient; η sets the trade-off between average and worst
language). The final λ is chosen on the selection split; evaluation uses held-out FLORES devtest
and Belebele, never the selection split (the selection-bias lesson of 2608.02940).
Optional stage 2: worst-language bit allocation over widths {3, 4, 8} evaluated end to end.

**Baselines.** English-only calibration; equal multilingual mix (group-balanced); the best fixed
mix from Calibrating Beyond English if reproducible; average-objective reweighting (same loop,
mean instead of max); v1 ParityQuant (negative).

**Confirmatory test (draft).** At W3 and W4, three GPTQ seeds each: worst-language held-out NLL
increase against the best baseline, with intervals that resample items **and** seeds; plus
Belebele above-chance retention with a non-inferiority margin. Price-of-parity curve over η.

**Models.** Qwen2.5-1.5B and 3B; Llama-3.2-1B/3B recommended for generality.

**Compute (projected, unmeasured).** 1.5B: 2 widths × 3 seeds × (5–8 rounds × about 11 min +
evaluation about 15 min) ≈ 10–15 GPU-hours, plus baselines, part already on HF. 3B about 2×.

## What would change this

A paper that optimises calibration weights or GPTQ Hessians for a worst group in multilingual
LLMs (K4). A pilot showing that reweighting does not move the worst language beyond seed spread
(then O1 becomes the lead or the paper becomes an analysis of why worst-group PTQ fails).

## Status

Research done; decision pending the owner's go-ahead on the design and a pilot. Nothing
preregistered or run yet.

## Positioning from lab and advert scan (2026-11-15)

- The closest framing in print is Parity-Aware BPE (Foroutan, Meister, Paul, Niklaus, Ahmadi, Bosselut, Sennrich; [ACL 2026](https://aclanthology.org/2026.acl-long.342/)): at each merge, use the statistics of the currently worst-compressed language; tokenization Gini falls by up to 89% at comparable global compression. ParityGPTQ is the same principle for quantization (reweight toward the worst language at each round), which makes the story easy to state and places it in an active research line (EPFL, ETH, Zurich).
- Language-targeted pruning and calibration language reach TACL (Kurz et al., [TACL 2026](https://aclanthology.org/2026.tacl-1.9)), so a journal is a realistic venue for either paper.
- Labs and programmes with matching stated focus: Cohere Labs (Multilingual, Efficiency, Inference, Social Impact; Scholars Program), IST Austria Alistarh group (GPTQ, low-precision LLMs), Edinburgh (efficient LLM inference studentship, Ponti; ML Systems CDT), Copenhagen DIKU (resource-efficient generative AI PhD), ELLIS (ML and Sustainability; multilingual NLP faculty in Copenhagen and Lisbon).
- The token-tax literature (Ahia et al., EMNLP 2023; "The Token Tax", arXiv 2509.05486) prices tokenization only; WhoPays extends the question to inference optimizations.
