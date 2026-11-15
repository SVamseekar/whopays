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
