# ParityQuant: a preregistered negative result

Draft section for the audit paper. Wording follows the adversarial review of 2026-11-14
(`docs/decisions/2026-11-14-parityquant-adversarial-review.md`). Every number comes from logged
runs: RUN_LOG 15 (sensitivity probe) and 16 (evaluation, Kaggle T4×2, commit `2a7caef`). The
tables regenerate with `python scripts/parity_eval.py 1.5B`. Setting throughout: Qwen2.5-1.5B-
Instruct, GPTQ with fake quantization on a T4 in FP16, one GPTQ seed per bit map, 11 languages.
Average bits cover the decoder's linear projections only; the tied embedding stays FP16.

## Question

Mixed-precision quantization usually spends bits to minimize average loss. If languages differ in
which layers they depend on, an allocation that minimizes the worst language's loss could protect
low-resource languages at the same average bit-width. We tested whether such a worst-language
(minimax) allocation beats an average-loss allocation (H4).

## Method

**Sensitivity profile.** For each of the 28 decoder blocks and each width in {2, 3, 4, 8} bits,
we quantized that block alone and measured the increase in FLORES-200 NLL on 500 sentences per
language. Sensitivities are in nats per parallel sentence, so they do not depend on how finely a
tokenizer segments each language (sixth preregistration amendment). Gate K2 required that layer
sensitivities differ across languages (minimum pairwise Spearman correlation ≤ 0.9). It passed:
the minimum was 0.420 (Telugu, Chinese).

**Allocators.** Each block gets one width. At a fixed parameter-weighted average of 3.0, 3.5 or
4.0 bits:

- *minimax* minimizes the largest predicted language loss;
- *average* minimizes the mean predicted loss;
- *uniform* gives every block the same width (3 or 4 bits), with English or multilingual GPTQ
  calibration. At 3.5 bits, where no single width meets the budget, it is a sensitivity-free
  split (blocks 0–13 at 4 bits, 14–27 at 3).

The predicted loss of a bit map is the sum of single-block sensitivities.

**Preregistered test.** The primary measure is the increase in NLL per sentence against FP16 on
512 held-out FLORES devtest sentences per language (sentences the profile did not use). H4 is
falsified if, at two or more budgets, the 95% paired bootstrap interval of the worst-language
improvement (best baseline minus minimax; 2,000 item resamples) includes zero, or the mean
Belebele accuracy cost exceeds one point. The test and its unit were fixed before any
allocation was run (sixth amendment, 2026-11-14).

## Result (confirmatory): no minimax advantage

| Budget (bits) | Best baseline | Worst-language increase, minimax (nats) | Worst-language increase, baseline (nats) | Improvement (nats) | 95% interval |
| --- | --- | --- | --- | --- | --- |
| 3.0 | average | 64.6 | 66.1 | +1.50 | [0.06, 2.98] |
| 3.5 | average | 20.7 | 18.8 | −1.89 | [−2.69, −1.20] |
| 4.0 | average | 10.6 | 8.4 | −2.23 | [−2.70, −1.80] |

The worst language is Telugu in every cell. The improvement interval excludes zero in favour of
minimax only at 3.0 bits, so H4 is falsified (K3 negative).

On Qwen2.5-1.5B, a worst-language (minimax) bit allocation showed no advantage over an
average-loss allocation at 3.0–4.0 average bits. The differences, −2.2 to +1.5 nats per
sentence, lie within the 5–7 nat spread of Telugu's held-out loss across three GPTQ seeds of
uniform quantization (83.4–90.5 nats at 3 bits, 32.9–38.4 at 4 bits). We therefore do not claim
that the average objective is better either; the intervals above resample items, not GPTQ seeds.

Held-out NLL increase per sentence (nats), all 11 languages:

| Budget | Method | en | de | fr | es | ru | zh | ja | th | bn | sw | te |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.0 | minimax | 36.6 | 51.7 | 36.8 | 35.5 | 40.4 | 42.0 | 36.3 | 41.1 | 57.7 | 37.8 | 64.6 |
| 3.0 | average | 47.3 | 53.2 | 41.9 | 43.0 | 44.0 | 46.0 | 37.7 | 43.1 | 58.1 | 44.4 | 66.1 |
| 3.0 | uniform, multilingual | 22.7 | 45.3 | 35.4 | 32.4 | 39.6 | 28.7 | 32.0 | 51.4 | 73.3 | 36.1 | 85.4 |
| 3.0 | uniform, English | 19.0 | 50.1 | 41.3 | 37.7 | 60.5 | 30.9 | 53.7 | 87.7 | 149.0 | 49.8 | 200.4 |
| 3.5 | minimax | 10.7 | 17.0 | 12.8 | 11.6 | 14.0 | 9.3 | 11.5 | 14.9 | 20.2 | 15.0 | 20.7 |
| 3.5 | average | 6.3 | 11.1 | 7.8 | 7.5 | 9.3 | 6.6 | 7.8 | 10.7 | 16.3 | 9.3 | 18.8 |
| 3.5 | split, multilingual | 11.3 | 22.1 | 17.0 | 14.6 | 25.8 | 18.9 | 20.6 | 28.1 | 36.2 | 18.9 | 56.0 |
| 3.5 | split, English | 10.3 | 28.2 | 22.2 | 22.0 | 35.4 | 23.1 | 33.5 | 46.9 | 74.4 | 35.7 | 114.8 |
| 4.0 | minimax | 3.3 | 7.0 | 5.3 | 4.9 | 4.8 | 2.3 | 4.0 | 5.7 | 9.3 | 7.6 | 10.6 |
| 4.0 | average | 2.6 | 4.7 | 3.0 | 2.9 | 3.4 | 2.1 | 3.0 | 3.9 | 7.1 | 5.8 | 8.4 |
| 4.0 | uniform, multilingual | 6.6 | 12.5 | 11.5 | 8.7 | 18.2 | 12.2 | 13.6 | 19.6 | 27.0 | 8.3 | 38.4 |
| 4.0 | uniform, English | 7.6 | 14.7 | 15.0 | 14.8 | 22.7 | 15.6 | 18.9 | 27.8 | 43.0 | 14.9 | 65.5 |

## Sensitivity-aware allocation lowers NLL but not reading accuracy (exploratory)

Both sensitivity-using allocators cut Telugu's held-out loss increase 2.7–4.6× against uniform
multilingual quantization at 3.5–4.0 bits (1.3× at 3.0 bits). The 3.5-bit comparison is against
a sensitivity-free split, a weak comparator, so the 4.0-bit comparison is the one to read.

The NLL gain is not a quality gain. Belebele accuracy (%, 900 items per language, chance 25%):

| Budget | Method | en | de | fr | es | ru | zh | ja | th | bn | sw | te |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 | — | 78.9 | 70.4 | 71.8 | 70.8 | 71.1 | 76.3 | 63.7 | 60.8 | 39.8 | 33.1 | 29.9 |
| 3.0 | minimax | 22.9 | 25.0 | 24.3 | 24.4 | 24.8 | 23.0 | 24.7 | 23.2 | 23.7 | 23.4 | 24.7 |
| 3.0 | average | 23.9 | 23.6 | 27.0 | 24.7 | 24.7 | 20.3 | 23.7 | 25.3 | 25.0 | 22.6 | 25.6 |
| 3.0 | uniform, multilingual | 27.3 | 27.7 | 27.0 | 27.4 | 30.6 | 33.6 | 24.8 | 23.8 | 24.9 | 26.9 | 24.1 |
| 3.0 | uniform, English | 42.0 | 28.1 | 32.8 | 28.3 | 31.4 | 39.1 | 25.4 | 25.3 | 23.2 | 22.4 | 20.8 |
| 3.5 | minimax | 23.4 | 23.9 | 24.4 | 24.4 | 24.9 | 23.7 | 24.6 | 24.5 | 23.0 | 23.3 | 22.8 |
| 3.5 | average | 41.6 | 35.7 | 36.0 | 33.2 | 39.0 | 43.4 | 35.3 | 27.7 | 30.1 | 23.6 | 25.1 |
| 3.5 | split, multilingual | 49.0 | 48.6 | 46.9 | 33.1 | 45.3 | 60.9 | 43.3 | 37.4 | 29.1 | 27.3 | 25.1 |
| 3.5 | split, English | 52.2 | 41.0 | 49.9 | 56.7 | 47.0 | 52.9 | 43.9 | 40.2 | 26.0 | 27.9 | 23.1 |
| 4.0 | minimax | 59.5 | 52.3 | 52.2 | 51.6 | 51.7 | 63.5 | 51.3 | 45.4 | 28.6 | 25.6 | 23.6 |
| 4.0 | average | 64.0 | 54.1 | 58.3 | 55.8 | 54.2 | 62.2 | 52.2 | 51.3 | 33.8 | 28.3 | 23.0 |
| 4.0 | uniform, multilingual | 70.3 | 64.2 | 62.3 | 61.4 | 65.8 | 73.2 | 59.6 | 54.3 | 33.3 | 29.3 | 29.0 |
| 4.0 | uniform, English | 73.8 | 63.1 | 67.8 | 67.2 | 62.8 | 70.4 | 56.4 | 52.9 | 34.1 | 29.9 | 26.6 |

At 4.0 bits, in the eight languages whose FP16 accuracy exceeds 60%, minimax trails uniform
multilingual quantization by 8.3–14.1 points and the average allocator by 3.0–11.6. Each gap
exceeds that language's spread across three GPTQ seeds of uniform multilingual W4 (0.8–4.0
points; RUN_LOG 20), except the average allocator in French (4.0 against 4.0). The 3.5-bit minimax model is at chance in all eight languages
whose FP16 accuracy exceeds 60%. Bengali, Swahili and Telugu were near chance at FP16 already, so
they say nothing here. At 3.0 bits every method, uniform included, is near chance.

This direction is known in general: perplexity and downstream accuracy can disconnect for
quantized multilingual models (LCD, arXiv 2608.11786), and calibration-language gains are "clear
in perplexity, although rarely substantial in downstream tasks" (Waskito et al., Findings of
IJCNLP-AACL 2026). What this result adds is the size of the divergence under a worst-language
allocator: better worst-language NLL alongside chance-level reading comprehension.

## Why minimax lost (exploratory)

The allocators predict a map's loss as the sum of single-block sensitivities. A CPU check on the
six evaluated maps compared observed held-out loss with that additive prediction. The ratio of
observed to predicted loss was 0.8–1.3 for maps with no 2-bit block, 0.9–1.2 with one, 1.3–1.6
with four and 1.6–3.8 with nine. Single 2-bit middle blocks cost little alone (about 0.02 nats
per token), so the additive model prices them cheaply. Minimax used that freedom more than the
average objective did.

The evidence is limited. The number of 2-bit blocks is confounded with the budget: both 3.0-bit
maps carry nine. Only the 3.5-bit pair separates them at one budget: four 2-bit blocks gave
ratios of 1.3–1.6, one gave 0.9–1.2. The failure is a multilingual instance of a known problem.
Scalar per-layer sensitivity proxies distort multiplicatively at low bit-widths (CASA, arXiv
2609.25916), and layer contributions interact (arXiv 2306.04879). It is not a contribution on
its own.

## What stays open

A worst-case objective may still help with widths capped at 3 bits or more, or with a
cross-layer sensitivity model. The strongest evidence for this is at 3.0 bits, where both maps
carry the same nine 2-bit blocks: there minimax was ahead (+1.50 nats, interval excluding zero).
Also, the additive model is accurate wherever no block sits at 2 bits. We did not run the
width-capped test or a 3B evaluation; the project stopped ParityQuant at the K3 result.

## Limitations

One model (Qwen2.5-1.5B), one GPTQ seed per map, fake quantization (memory from bit counts,
decoder projections only), T4 FP16, 11 languages. Belebele passages come from FLORES-200 and may
overlap the profile's sentences, so Belebele is not fully held out from the allocator. The
2-bit analysis and the Belebele comparison were done after the K3 result.

## Data

396 evaluation runs and 22 FP16 references on the HF dataset `Heavy-Dervish1205/whopays-results`
(commit `2a7caef`); bit maps in `configs/parityquant_1.5B.allocations.json`; tables in
`docs/figures/parityquant_1.5B_{h4,languages,belebele}.csv`. The replication grid's seed-0
uniform runs reproduce the ParityQuant uniform values from separate runs (Telugu 85.37 against
85.38 nats at 3 bits; 38.35 against 38.36 at 4 bits).
