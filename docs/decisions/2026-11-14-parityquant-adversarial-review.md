# Adversarial review: ParityQuant 1.5B claims (2026-11-14)

Run under the `adversarial-review` skill before any ParityQuant sentence enters the paper.
Evidence: `docs/figures/parityquant_1.5B_*.csv`, RUN_LOG 16, DECISIONS 2026-11-14, and two CPU
checks run for this review (GPTQ seed spread on held-out FLORES and on Belebele, from the
replication grid's three seeds per calibration set).

## Claims as first stated

1. Confirmatory: K3 NEGATIVE_RESULT; the average objective beats minimax at 3.5 and 4.0 bits.
2. Sensitivity-aware allocation cuts worst-language (Telugu) held-out NLL about 3× against uniform at 3.5–4.0 bits.
3. Exploratory: the additive single-block sensitivity model breaks in step with the number of 2-bit blocks.
4. Exploratory: NLL-optimal maps lose Belebele accuracy against uniform; the 3.5-bit minimax model is at chance in all 11 languages.

## Attacks

| Attack | Evidence | Verdict |
| --- | --- | --- |
| Statistics: is the effect larger than seed noise? | The H4 intervals bootstrap items only, one GPTQ seed per map. Replication grid, held-out items (n = 512), Telugu increase per sentence across seeds 0–2 with multilingual calibration: 83.4–90.5 nats at 3 bits (spread 7.1), 32.9–38.4 at 4 bits (5.4). K3 differences are +1.50, −1.89 and −2.23 nats. | Claim 1 must narrow: the differences are inside GPTQ seed variation, so "average beats minimax" is not supported. The preregistered verdict stands (it is defined on the item bootstrap), worded as "no minimax advantage". |
| Statistics, claim 2 | Telugu 20.7 (minimax) and 18.8 (average) against 56.0 (uniform) at 3.5 bits; 10.6 and 8.4 against 38.4 at 4.0. Gaps of 28–37 nats against a 5–7 nat seed spread. At 3.0 bits 64.6 against 85.4 (1.3×). | Survives on NLL at 3.5–4.0 bits; "3×" becomes "2.7–4.6× at 3.5–4.0 bits, 1.3× at 3.0". |
| Baselines | The 3.5-bit uniform comparator puts blocks 0–13 at 4 bits and 14–27 at 3 bits, with no sensitivity information. A reviewer will call it a weak baseline. At 3.0 and 4.0 bits the comparator is truly uniform. | Claim 2 is judged at 4.0 bits first; the 3.5-bit comparison is labelled as against a sensitivity-free split. |
| Selection / scope of claim 2 | Belebele accuracy at 4.0 bits: both allocators trail uniform by 6–11 points in en, de, fr, es, ru, zh and ja. Seed spread of uniform W4 Belebele is 0.8–4.0 points, so the gap exceeds it in those 7 languages (Thai marginal). | Claim 2 can never appear without claim 4 beside it: the NLL gain is not a quality gain. |
| Floor and ceiling, claim 4 | FP16 Belebele is 39.8 (bn), 33.1 (sw) and 29.9 (te): near the 25% floor. "Chance in all 11 languages" counts three languages that were near chance already. | Narrow to "at chance in all 8 languages where FP16 exceeds 60%". |
| Alternative explanation, claim 4 | Letter bias (one letter 46–72% of answers) is the symptom, not a cause. Batch shape is the same for all methods (Belebele scoring). Calibration set is the same (multi) for minimax, average and uniform_multi. | Survives as an observation. |
| Alternative explanation, claim 3 | Only 6 maps; the number of 2-bit blocks is confounded with budget (both 3.0-bit maps have 9). Only the 3.5-bit pair (4 against 1 block, same budget) separates them: ratio 1.3–1.6 against 0.9–1.2. | Narrow: "at a fixed 3.5-bit budget, the map with more 2-bit blocks departs further from the additive prediction; across budgets the pattern is consistent but confounded with average bits". |
| Novelty, claim 3 | Scalar per-layer sensitivity proxies are known to fail at low bits and across layers: CASA ([arXiv 2609.25916](https://doi.org/10.48550/arxiv.2609.25916), Sep 2026) proves multiplicative distortion of scalar proxies and reports the largest gains below 3 bits; inter-layer dependencies ([arXiv 2306.04879](https://doi.org/10.48550/arxiv.2306.04879)). | Not a contribution on its own: report it as a multilingual instance of a known failure and the reason the worst-case objective lost. |
| Novelty, claim 4 | LCD (arXiv 2608.11786, in RESEARCH_CONTEXT) reports a perplexity–accuracy disconnect for quantized multilingual models; Waskito et al. ([Findings of IJCNLP-AACL 2026](https://doi.org/10.18653/2026.findings-aacl.9)) find calibration-language gains "clear in perplexity, although rarely substantial in downstream tasks" on Qwen3-8B and Llama-3.1-8B, GPTQ 2–4 bits with SliM-LLM and TaCQ mixed precision. | Not new as a general caution. What is ours: the size of the divergence under a worst-language allocator (better worst-language NLL, chance-level reading comprehension). |
| Confirmatory or exploratory | Claim 1 is the sixth-amendment H4. Claims 3 and 4 came after the result. | Labels already correct in DECISIONS; keep them in the text. |
| Reproducibility | 396 runs plus 22 FP16 references on HF at `2a7caef`; `scripts/parity_eval.py 1.5B` regenerates the tables. Replication seed 0 reproduces the ParityQuant uniform values (Telugu 85.37 against 85.38 at 3 bits; 38.35 against 38.36 at 4 bits), from separate runs. | Survives. |
| Cost honesty | Average bits cover decoder linear layers only; the tied embedding stays FP16 (DECISIONS 2026-11-14). | Every bit figure must say "decoder projections". |
| Overclaim | One model (Qwen2.5-1.5B), fake quantization, T4 FP16, one seed per map, 11 languages. | All four claims are scoped to that. |

## Steelman of the opposite

A worst-case objective can help: it lost here because it was fed an additive single-block
sensitivity model that underprices 2-bit blocks, and it used that freedom more than the average
objective did. With widths of at least 3 bits, or a cross-layer sensitivity model (CASA-style),
minimax might win. Best evidence: at 3.0 bits, where both maps carry the same nine 2-bit
blocks, minimax was ahead (+1.50 nats, item CI excluding 0), and the additive model is accurate
(0.8–1.3) wherever no block sits at 2 bits. The owner chose not to run the width-capped test
(DECISIONS 2026-11-14), so the paper must say this remains open.

## Verdict and wording for the paper

- Claim 1, narrowed: "Under the preregistered test, a worst-language (minimax) bit allocation showed no advantage over an average-loss allocation at 3.0–4.0 average bits on Qwen2.5-1.5B (K3 negative). Differences of −2.2 to +1.5 nats per sentence lie within the 5–7 nat spread across GPTQ seeds."
- Claim 2, narrowed and paired: "Allocating bits by measured per-language sensitivity reduced Telugu's held-out loss increase 2.7–4.6× against uniform quantization at 3.5–4.0 bits, but lowered Belebele accuracy by 6–11 points in seven of eight high-resource languages at 4.0 bits."
- Claim 3, narrowed and cited: the multilingual instance of scalar-proxy failure (CASA; inter-layer dependency work), with the confound stated.
- Claim 4, narrowed and cited: "at chance in all eight languages with FP16 accuracy above 60%", positioned against LCD and Waskito et al.
- No new GPU work required for these wordings.
