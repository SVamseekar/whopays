# Literature: ParityQuant v2 (worst-language post-training quantization)

Saved 2026-11-15 for `docs/decisions/2026-11-15-parityquant-v2-method.md`. Found through OpenAlex
(`litsearch.py`, IDs verified) and web search. Grade A = full text read; B = abstract.

## Closest prior art (must cite and contrast)

| Grade | Work | What it does | Difference from ParityGPTQ |
| --- | --- | --- | --- |
| A | Group-robust LLM pruning, [arXiv 2608.02940](https://doi.org/10.48550/arxiv.2608.02940) (2026) | Worst-group mask selection for pruning 3B dense LLMs; groups are data sources; group-balanced calibration cuts worst-group damage 21.7–31.5%; pooled scores can select the worse model | Pruning, not quantization; no languages; mask menus, not Hessian reweighting |
| B | DRPruning, [ACL 2025](https://doi.org/10.18653/v1/2025.acl-long.1414) | DRO over domains while pruning with continued pretraining, monolingual and multilingual | Needs training; structured pruning; ours is training-free GPTQ |
| B | Fair-GPTQ, [arXiv 2509.15206](https://arxiv.org/abs/2509.15206) (TACL per OpenAlex) | Pair-difference bias penalty (he/she likelihood) added to the GPTQ objective | Stereotype bias, not worst-language quality; regulariser, not reweighting |
| B | FairQuant, [arXiv 2602.23192](https://arxiv.org/abs/2602.23192) | Group-aware mixed precision with fairness regularisation, medical images | Vision; trained bit allocation |
| B | Colla-Q, [arXiv 2609.18131](https://arxiv.org/abs/2609.18131) | Balanced precision across MoE experts | Experts, not languages |
| B | Balancing Act, [arXiv 2310.20673](https://doi.org/10.48550/arxiv.2310.20673) | Per-subgroup constrained pruning | Pruning, classification groups |
| B | Explaining How Quantization Disparately Skews a Model, [arXiv 2509.07222](https://doi.org/10.48550/arxiv.2509.07222) | Mechanism; fix with mixed-precision QAT and resampling | Training-based |

## Calibration data for multilingual PTQ

| Grade | Work | Use |
| --- | --- | --- |
| B | Calibrating Beyond English, [EACL 2026](https://doi.org/10.18653/v1/2026.eacl-long.223) | Fixed mixes; GPTQ more calibration-sensitive than AWQ; target-language calibration best per language |
| B | Lost in Compression, [IMPACT-Speech 2026](https://doi.org/10.18653/2026.impactspeech-1.6) | Balanced calibration barely moves an accent gap; homogeneous narrows it: mixture effects are non-monotone |
| B | LangCompress, [IJCNLP-AACL 2025](https://aclanthology.org/2025.ijcnlp-long.112) | Synthetic per-language calibration data |
| B | When Calibration Data Matter in Sub-4-Bit Quantization, [arXiv 2611.11079](https://doi.org/10.48550/arxiv.2611.11079) (Nov 2026) | Across task families, single-pass GPTQ responds weakly to calibration at 3 bits; iterative methods respond more. Risk to test early |
| B | Calibration Data Curation, [NeurIPS 2025](https://doi.org/10.52202/085713-1961) | Representativeness and diversity in activation space drive calibration quality |
| B | Waskito et al., [Findings AACL 2026](https://doi.org/10.18653/2026.findings-aacl.9) | Calibration language × bits × mixed precision, 5 languages |
| B | Language-targeted pruning, [TACL 2026](https://aclanthology.org/2026.tacl-1.9) | Calibration language effects in pruning; TACL as venue |

## Group DRO and mixture reweighting (algorithmic ancestry)

| Grade | Work | Use |
| --- | --- | --- |
| B | DoReMi, [NeurIPS 2023](https://doi.org/10.52202/075280-3059) | Group DRO on excess loss over a reference model sets domain weights; our NLL increase over FP16 is the analogous excess loss |
| B | CTC-DRO, [arXiv 2502.01777](https://doi.org/10.48550/arxiv.2502.01777) | Group DRO fails when group losses are not comparable across languages; smoothing the weight update fixes it (worst-language error −47.1%) |
| B | GRAPE, [NeurIPS 2025](https://doi.org/10.52202/085713-5281) | Minimax over target tasks for data mixtures |
| B | UniMax, [arXiv 2304.09151](https://doi.org/10.48550/arxiv.2304.09151) | Language sampling for multilingual pretraining |
| B | Parity-Aware BPE, [ACL 2026](https://aclanthology.org/2026.acl-long.342/) | Worst-language greedy step in tokenization (Gini −89%) |

## Mixed-precision structure (why v1 failed, what v2 can rely on)

| Grade | Work | Use |
| --- | --- | --- |
| B | Saturation Makes Quantization Error Additive, [arXiv 2607.12266](https://doi.org/10.48550/arxiv.2607.12266) | At 4-bit, per-layer effects explain 85–93% of loss variance; saturating coverage model. Our 2-bit result is super-additive (1.3–3.8×), the opposite regime: supports capping widths at ≥ 3 bits |
| B | CASA, [arXiv 2609.25916](https://doi.org/10.48550/arxiv.2609.25916); QEP, [NeurIPS 2025](https://doi.org/10.52202/085713-5080) | Cross-layer refinement; error propagation |
| B | KL fidelity: [arXiv 2609.07664](https://doi.org/10.48550/arxiv.2609.07664); [EMNLP 2026 industry](https://doi.org/10.18653/2026.emnlp-industry.178) | KL fails to rank near-lossless quantizations: not the main objective |

## WhoPays (analysis paper) additions

| Grade | Work | Use |
| --- | --- | --- |
| B | DEPART, [Findings EMNLP 2026](https://doi.org/10.18653/2026.findings-emnlp.2328) | Bayesian decomposition of base multilingual performance by language features; WhoPays decomposes compression-induced changes across techniques |
| B | MoEXBench, [arXiv 2608.21693](https://doi.org/10.48550/arxiv.2608.21693) | Composable compression interactions in MoE, not per language |
| B | Speculative Decoding and the Curse of Multilinguality, [arXiv 2605.30580](https://doi.org/10.48550/arxiv.2605.30580) | 11 languages, speculation only |
| B | Token Tax, [arXiv 2509.05486](https://arxiv.org/abs/2509.05486); Ahia et al. EMNLP 2023 | Tokenization cost disparities only |

Searches: about 41 OpenAlex queries and 15 web searches on 2026-11-15.

## 2026-11-18 additions: layer-local minimax inside GPTQ (A+B) and its neighbours

The lead moved from a global λ outer loop to per-layer minimax inside the GPTQ pass, scored
with an answer-letter Fisher (DECISIONS 2026-11-18). Nearest neighbours and how A+B differs:

| Grade | Work | What it does | Difference from A+B |
| --- | --- | --- | --- |
| B | Fair-GPTQ, [TACL 2026](https://doi.org/10.1162/tacl.a.816) | Adds a bias regularizer to GPTQ's objective, built from token-aligned stereotype/anti-stereotype pairs | Regularizes a bias gap, not worst-group quality; needs token-aligned pairs, which parallel text in different scripts does not give |
| B | RSQ, [arXiv 2503.01820](https://arxiv.org/abs/2503.01820) | Weights tokens (attention importance) inside one Hessian | One pooled Hessian; no groups, no worst-case weights |
| B | ADMM-Q, [arXiv 2605.11222](https://arxiv.org/abs/2605.11222) | Better solver for the layer-wise Hessian-weighted problem | Single Hessian; a solver choice, orthogonal to which Hessian |
| B | REAL-Q, [arXiv 2609.00049](https://arxiv.org/abs/2609.00049) | Separate Hessians per row group | Row groups of the weight, not input distributions |
| B | QEP, [NeurIPS 2025](https://doi.org/10.52202/085713-5080); GPTAQ / residual-error compensation, [ICLR 2026](https://arxiv.org/html/2604.07955v1) | Correct the target for upstream quantization error | Single input distribution; orthogonal, could be combined |
| B | Information Boundaries for Group-Robust LLM Pruning, [arXiv 2608.02940](https://arxiv.org/abs/2608.02940) | Group-balanced calibration cuts worst-group pruning damage 21.7% (Wanda) and 31.5% (SparseGPT); one compute-matched hard-max trajectory on OLMoE ends 32.7% worse than pooled | Pruning; supports anchoring λ to uniform and warns against hard-max trajectories |
| B | Language-Conditional Dequantization, [arXiv 2608.11786](https://arxiv.org/abs/2608.11786) | Per-language rank-2 LoRA on INT3; closes 63.5–69.1% of non-English perplexity gap but 17–28% of the GlobalMMLU gap | Post-hoc adapters and routing; its perplexity-versus-accuracy split matches ours |

Checked on 2026-11-18 against the pages (abstract or body text). Corrections to claims in the
owner's external strategy report (`docs/ParityQuant Novelty and Method Strategy.docx`):
2608.11786 does not find the same late hotspot band across models (Qwen2.5-3B blocks 18–30,
Llama-3.2-3B blocks 7–14); 2607.12266 measures 4-bit weight-and-activation precisions, not
"≥ 3 bits"; 2611.11079's "+0.06 pp" is a pooled iterative-versus-single-pass contrast, not a
GPTQ-specific one (its abstract says single-pass GPTQ and AWQ show "small mean responses at three
bits"); 2608.02940's 32.7% is one trajectory on OLMoE. Unverified (pages unreadable): the
GPTQ Cholesky-instability figures attributed to 2026.findings-aacl.8 and the GainGate
framework (OpenReview NJ6JEwvZn5). Do not cite these until read.

One web search (2026-11-18, "minimax layer-wise reconstruction GPTQ per-group Hessian worst-case
calibration") found no GPTQ variant that takes a worst case over calibration groups.
