# Literature: answer extraction for generated math answers

Saved 2026-11-15 for `docs/decisions/2026-11-15-mgsm-answer-extraction.md`. All entries passed
`litsearch.py check`. Grade: A = read in full, numbers taken from the body; B = abstract or
secondary summary only.

| Grade | Source | Link | Use |
| --- | --- | --- | --- |
| A | Peter, Vilar, Domhan, Malkin, Freitag. Mind the Gap… or Not? How Translation Errors and Evaluation Details Skew Multilingual Results. GEM 2026 | https://doi.org/10.18653/v1/2026.gem-main.22 | MGSM extraction alone moves scores by up to 41 points in one language (Gemma 3 27B Bengali 45.2 to 86.4); format in prompt recommended |
| A | Same Answer, New Box: How Much of RLVR's Out-of-Domain Gain Is Answer Format? 2026 | https://doi.org/10.48550/arxiv.2610.20639 | Treatment-dependent answer placement; second extractor, balanced abstention, LLM-reader check (98.6% agreement) |
| A | Right Answer, Wrong Score: Uncovering the Inconsistencies of LLM Evaluation in MCQA. Findings ACL 2025 | https://doi.org/10.18653/v1/2025.findings-acl.950 | Regex underestimates; LLM extractors closer to humans but systematically wrong in places |
| A | lm-evaluation-harness 0.4.13, `tasks/mgsm/native_cot` | https://pypi.org/project/lm-eval/0.4.13/ | strict-match phrases and flexible-extract regex |
| B | Shi et al. Language Models are Multilingual Chain-of-Thought Reasoners. 2022 | https://doi.org/10.48550/arxiv.2210.03057 | MGSM origin, few-shot design |
| B | Yu et al. xFinder. 2024 | https://doi.org/10.48550/arxiv.2405.11874 | 93.42% vs 74.38% extraction accuracy |
| B | Kojima et al. Large Language Models Are Zero-Shot Reasoners. NeurIPS 2022 | https://doi.org/10.52202/068431-1613 | Two-stage answer extraction prompt |
| B | Wang et al. Plan-and-Solve Prompting. ACL 2023 | https://doi.org/10.18653/v1/2023.acl-long.147 | Same extraction prompt reused |
| B | The Coupling Tax. 2026 | https://doi.org/10.48550/arxiv.2605.07686 | Last-number fallback on truncated text: under 6% vs 56.9% |
| B | Noise Injection Systemically Degrades LLM Safety Guardrails. 2025 | https://doi.org/10.48550/arxiv.2505.13500 | Last-number false matches on truncated fragments |
| B | GEMS. 2026 | https://doi.org/10.48550/arxiv.2606.19946 | Truncation as error; regex misses as extraction bugs |
| B | When Reasoning Meets Compression. 2025 | https://doi.org/10.48550/arxiv.2504.02010 | Compression changes reasoning behaviour |
| B | PolyMath. NeurIPS 2025 | https://doi.org/10.52202/085713-1431 | Language-dependent reasoning length |

Searches: 10 OpenAlex queries (answer extraction, MGSM extraction, regex errors, xFinder,
harness parsing, LLM-judge reliability, zero-shot CoT extraction, multilingual math evaluation,
truncation bias, quantization and verbosity) and 5 web searches (Mind the Gap, simple-evals MGSM,
Math-Verify, Kojima extraction prompt, truncated GSM8K outputs). Case against LLM readers: Right
Answer, Wrong Score. Case against regex rules: Mind the Gap, Same Answer New Box, our own sample.
Not found: a validated zero-shot multilingual MGSM extractor; simple-evals' MGSM parser source
(searched, not reached).
