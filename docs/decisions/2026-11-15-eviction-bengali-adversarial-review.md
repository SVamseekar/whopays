# Adversarial review: 3B eviction cost in Bengali and Thai (interim)

Date: 2026-11-15. Data: audit slices 0–2 (RUN_LOG 19–21), Kaggle T4, commits `3f6f753`,
`a3af608`, `e31b9d3`; torch 2.15.0+cu132, transformers 5.19.0. Interim: 3B quantized cells are
still running, and no hypothesis has been tested.

## Claim under review

"Fixed 256-token eviction costs Qwen2.5-3B 9.0 Belebele points in Bengali and 3.3 in Thai, much
more than at 1.5B." Intended place: audit results section, and possibly the abstract.

## Evidence

Paired comparison of the `evict` cell against the full cache, same weights (FP16), same 900
items, 2,000 item resamples. Benjamini–Hochberg is applied over all 418 interim within-model KV
contrasts (both models, both tasks, every finished weight, calibration and seed cell).

| Model | Language | Full-cache accuracy | Eviction delta (points) | 95% interval | Retained share | BH-adjusted p |
| --- | --- | --- | --- | --- | --- | --- |
| 3B | bn | 58.9 | −9.0 | [−12.0, −6.1] | 0.50 | 9.5e-7 |
| 3B | th | 74.7 | −3.3 | [−4.8, −2.0] | 0.86 | 5.9e-4 |
| 3B | te | 38.0 | −2.1 | not significant | 0.41 | — |
| 1.5B | bn | 39.8 | −2.0 | not significant | 0.55 | — |

Only these two of the 418 contrasts survive the correction. Bengali 3B: the 872 truncated items
lose 9.3 points; the 28 items that fit the budget lose 0.0. Size difference (3B minus 1.5B, paired
over items): bn −7.0 [−10.6, −3.4], th −2.1 [−3.8, −0.3], te −1.9 [−5.7, 1.9], sw +0.1
[−0.9, 1.1]. Full table: `docs/figures/audit_interim_kv_within_model.csv`.

## Attacks

| Attack | Evidence | Verdict |
| --- | --- | --- |
| Novelty | "Equal Tokens, Unequal Losses" (OpenReview `57mHZnnuEe`, read in full 2026-11-14) already shows that fixed eviction budgets penalize high-fertility languages on Belebele. An OpenAlex search (2026, five queries) found no other multilingual KV-eviction study; "The Pitfalls of KV Cache Compression" (ACL 2026) is English IFEval. | **Narrow.** The direction is theirs, so cite it. What remains ours: Qwen2.5 at two sizes, composition with weight quantization (pending), the attribution. |
| Overclaim | Two sizes of one family; FP16 weights only; one budget (256); one eviction policy (sink + recent); one task. | **Narrow** the wording to that setting. "Larger models lose more" is not supported in general. |
| Statistics | Paired bootstrap; BH over 418 contrasts. The raw p for bn is about 1e-9, which survives even Bonferroni over 1,000 tests, so the final, larger family will not remove it. FP16 3B is deterministic: no GPTQ seed noise in this cell. | **Survives.** |
| Confirmatory or exploratory | The per-language comparison against FP16 with BH is preregistered (Analysis section), so the 3B FP16 `evict` contrast is a preregistered comparison type. The final BH family is the full language × configuration set at audit end. The 1.5B versus 3B difference is not preregistered. | **Label:** per-language deltas confirmatory at audit end; size difference exploratory. |
| Floor and ceiling | Telugu keeps less context (41%) than Bengali (50%) yet loses less, because it sits 13 points above chance against Bengali's 34. Above-chance loss: bn 26.6% (3B) and 13.5% (1.5B); te 16.2% and 4.5%; th 6.7% and 3.4%. At 1.5B, Bengali is 15 points above chance, so part of the size difference is headroom. | **Narrow.** Report above-chance loss beside raw points. The size effect is partly a floor effect. |
| Alternative: batch-shape numerics | `evict` and full both run at batch 8 with prefill reuse. Languages that are never truncated show exactly 0.0 (en, de, zh at 3B), and the 28 untruncated Bengali items show 0.0. | **Survives.** The loss sits only where context is removed. |
| Alternative: tokenizer fertility | That is the mechanism: Bengali prompts run to a median of 499 tokens against 99 for English, so only high-fertility languages are truncated. | **Not a confound but the explanation.** Say so. |
| Alternative: which tokens go | The sink + recent policy keeps the question and options (at the end of the prompt) and drops the middle of the passage. | **Disclose** the policy; another policy could differ. |
| Selection | All 11 languages, all 418 contrasts and both sizes are in the CSV, including the 416 null contrasts. | **Survives** if the table is shown in full. |
| Reproducibility | Run IDs, configs and items are on the HF dataset. The table comes from `report.hypotheses.kv_within_model_table` (tested; it reproduces the first scratch analysis to 1e-16). | **Fixed.** |
| Baselines | The retention-matched comparison (`evict_share`) was uninformative at this budget (DECISIONS 2026-11-14). The reference paper's share-of-input remedy is the natural comparison. | **Gap:** no remedy arm. A sub-English budget design needs the owner's approval. |
| Cost honesty | No memory claim here. Eviction's memory saving is zero in languages that are never truncated. | **Disclose.** |

## Steelman of the opposite claim

"This is a known tokenizer-budget artefact, not a finding: Equal Tokens, Unequal Losses already
shows it, the 1.5B contrast is muddied by Bengali sitting near chance, and one 256-token budget
on a short-context benchmark says little about real long-context serving, where every language is
truncated." The best evidence for it: the identical sign and mechanism in the reference paper, and
Telugu's small loss despite heavier truncation.

## Verdict: narrow, then hold for the composed result

Wording allowed now (interim, internal): "On Belebele, fixed 256-token sink-plus-recent eviction
lowers Qwen2.5-3B (FP16) accuracy in Bengali by 9.0 points [−12.0, −6.1] and Thai by 3.3
[−4.8, −2.0], with no change in languages whose prompts fit the budget, in line with Equal
Tokens, Unequal Losses. The loss is larger at 3B than at 1.5B (Bengali −7.0 points, exploratory),
partly because 1.5B Bengali is nearer chance."

Not allowed: a general scale law, novelty of the fixed-budget penalty, or any claim about the
weight × eviction interaction before slices 3–6 finish and H2 is run on the full family.

Hold until: (1) 3B quantized eviction cells are done (slices 3–6, about 14 h of Kaggle session
time); (2) H2 is run once on the complete family.
