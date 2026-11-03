# Who pays for faster inference? A per-language audit of LLM serving optimizations

**Status: methods draft. It contains no results.** Every result section waits for logged GPU runs. Design values below come from `PREREGISTRATION.md`.

## Question

Serving teams routinely combine weight quantization, KV-cache compression and speculative decoding. Each is usually evaluated in aggregate, or on English. This audit asks how each one changes quality and speed per language, whether the effects add up when combined, and how much of each language's gap is explained by its items rather than by the optimization treating it differently.

## Design

**Models.** Qwen2.5-Instruct 1.5B and 3B, with 0.5B as the speculative draft model.

**Languages.** English, German, French, Spanish, Russian, Chinese, Japanese, Thai, Bengali, Swahili and Telugu.

**Tasks.** Belebele reading comprehension is the primary quality outcome. MGSM provides generation quality and speed. FLORES-200 dev supplies calibration only, and devtest supplies held-out NLL.

**Factors.**
- Weights: FP16, W8 RTN, W4 GPTQ and W3 GPTQ. Quantization is fake: weights are quantized and dequantized to FP16, so memory savings are computed from bit counts and never measured.
- Calibration: English or a multilingual mix, with three subsample seeds, replicated on Belebele.
- KV cache: full, 4-bit quantized, or eviction to a fixed 256-token budget that keeps the first 4 tokens and the most recent tokens. The policy applies to the context before the question is scored, so it never sees the question.
- Speculative decoding: off, or greedy with four draft tokens. Output must equal plain greedy token for token; any difference is treated as a harness bug.

## Validity checks

K1 compares the FP16 harness with lm-evaluation-harness item by item, and must agree within one point per task. FR-3 compares fake quantization with a real GPTQ checkpoint on FLORES perplexity, within 2%. FR-5 checks speculative exactness under every cache policy, across 11 languages.

## Analysis

All effects are changes against FP16 on identical items, with paired bootstrap 95% intervals over 2,000 resamples, and Benjamini–Hochberg correction at q = 0.05 across language × configuration comparisons.

- **H1.** Language gaps grow with token premium and with lower FP16 accuracy. Falsified if both Spearman intervals include zero.
- **H2.** At least one language has a nonzero weight × KV interaction on the logit of correctness, after correction. Tests use an item-clustered binomial GEE with the Mancl–DeRouen covariance and a t reference on items − 1. This is a population-averaged contrast. The pinned software has no frequentist likelihood for the random-intercept model originally written.
- **H3.** A pooled Oaxaca–Blinder decomposition of item degradation, English against each language, using token count, item length and FP16 gold log-probability. Endowment below half of the gap in at least three non-Latin-script languages supports H3. The split is descriptive, not causal. Script is excluded because it exactly identifies the group in a pairwise fit.
- **Attribution.** Exact Shapley values over each eight-cell factorial. Missing cells are an error.
- **Disparity.** Worst-language degradation, max–min spread, and Gini and Atkinson indices over nonnegative harms.

## Results

Pending. No treatment result exists at the time of writing.

## Limitations to state with the results

Fake quantization measures quality, not real kernels. One model family is audited. Eleven languages give a narrow base for correlations across languages, which is why intervals are reported. Fixed-budget eviction is partly a token-premium treatment by construction. The prior-art boundaries in `docs/REVIEW_2026-10-31.md` must be re-read before submission.
