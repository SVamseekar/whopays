# WhoPays + ParityQuant preregistration

Draft date: 2026-10-22. Amended: 2026-10-31, 2026-11-03 (twice). Status: prospective protocol;
the original file was uncommitted at this review. Registration requires a recorded commit
before M2 execution; this file's existence alone is not evidence of registration.
Source of truth: docs/PRD.md and the dated decisions log. This document contains
hypotheses and planned thresholds, not measured results.

## Scope

Qwen2.5-Instruct 1.5B and 3B; 0.5B draft. Languages: en, de, fr, es, ru, zh,
ja, th, bn, sw, te. Belebele is the primary quality outcome; MGSM provides
generation quality and speed; FLORES dev supplies calibration only and devtest
supplies held-out NLL and sensitivity measurements.

Quality: FP16, W8 RTN, W4 GPTQ, W3 GPTQ crossed with full, quant4 and
256-token eviction caches. W4/W3 use English and multilingual calibration with
three subsample seeds. Speed compares speculative off/on with four draft tokens
on the fixed first 30 MGSM items per language. Greedy token identity is mandatory.

## Confirmatory hypotheses

- H1: language gaps grow with token premium and lower FP16 accuracy. Falsified
  when both Spearman correlation confidence intervals include zero.
- H2: at least one pairwise interaction is nonzero after Benjamini–Hochberg
  correction; weight precision × fixed-budget eviction is the leading candidate.
- H3: endowment explains less than half of the gap for at least three non-Latin
  languages. Report signed components; shares are undefined for zero total gaps.
- H4: minimax allocation improves worst-language degradation at equal average
  bits (3.0, 3.5, 4.0), with mean accuracy cost no greater than one percentage
  point. Falsified if improvement intervals include zero at two or more budgets.

## Analysis

Report changes against FP16 on identical items with paired 95% bootstrap
intervals using 2,000 resamples; preserve item pairing across parallel languages
and compared configurations. Report calibration seed spread separately; do not
count seed repetitions as independent questions. Correct language × configuration
comparisons with Benjamini–Hochberg at q=0.05.

Fit mixed-effects logistic regression `correct ~ W*K*language + (1|item)`.
Use all eight cells of each binary factorial sub-grid for Shapley attribution,
and report pairwise interaction residuals. Missing cells are an analysis error.
Oaxaca–Blinder uses pooled reference coefficients with token count, item length,
and FP16 gold log-probability; report descriptive endowment and treatment
components with paired bootstrap intervals and disclose rank deficiencies.

Report worst-language degradation, max–min spread, Gini and Atkinson (epsilon=1).
Nonnegative-index conventions and treatment of improvements must be disclosed.
ParityQuant compares minimax, average objective, uniform and multilingual
calibration baselines at matching parameter-weighted bit budgets. A null result
is publishable. Other analyses, refinements and 7B confirmation are exploratory.

## Gates and integrity

Apply K1–K4 exactly as specified in docs/PRD.md. No M2 execution before K1
validation and the preregistration commit. Never treat local CPU/MPS checks as
research measurements. Store item-level results, stable config hashes, code and
library versions, GPU identity and resume metadata. Report theoretical weight
storage separately from measured memory and speed. Protocol amendments must be
dated, committed, justified and distinguished from this original protocol.

## Amendment — 2026-10-31, before research results

Exclude script from English-versus-language Oaxaca fits, and reject any design whose
covariate span contains the language-group indicator. In those comparisons, differing
scripts identify group exactly, making pooled least squares fit both group means and
forcing the treatment residual to zero. This is an identification failure, not evidence
against H3. H3 and its threshold remain unchanged; script analyses require a separate
exploratory specification across languages. The implementation now rejects such designs.

Two specifications still need resolution before the relevant analyses: the current
variational mixed logistic implementation yields credible intervals, not the frequentist
interaction p-values required for H2/BH; a uniform baseline cannot express a fractional
average bit budget, so the sensitivity-independent matched-budget comparator must be
defined before H4 evaluation. Neither requirement is considered complete in this
31 Oct amendment. Both are specified by the 3 Nov amendment below.

## Amendment — 2026-11-03, before research results

H2's confirmatory family is one test per language of the weight × KV interaction
on the logit of item correctness. The mean formula remains
`correct ~ W * K * language`. Item dependence uses a binomial GEE with an
independence working correlation and the Mancl–DeRouen bias-reduced covariance
clustered by item. The contrast in language l is
logit(W=1,K=1,l) − logit(W=1,K=0,l) − logit(W=0,K=1,l) + logit(W=0,K=0,l).
Two-sided p-values use a t reference with degrees of freedom equal to the number
of items minus one. Benjamini–Hochberg at q=0.05 applies to this family only.
If any of the four cells in a language has no outcome variation, the logit
interaction is undefined and the analysis stops; that language is not dropped.
The variational random-intercept fit stays a descriptive conditional summary and
is not an input to BH.

This is the population-averaged logit interaction, not the conditional
coefficient of `correct ~ W*K*language + (1|item)`. The pinned statsmodels
release has no frequentist likelihood for that binomial GLMM. Its conditional
logit, the fixed-item-intercept alternative, did not return a covariance on this
saturated within-item design. The substitution is declared here, before any M2
result is analysed. A later frequentist GLMM likelihood can replace the GEE
p-values only by a further dated amendment.

H4's sensitivity-independent comparator is `matched_uniform`. Among assignments
that meet the exact parameter-weighted bit budget, it minimises the
parameter-weighted mean squared deviation of layer widths from the target
average. Ties break toward higher widths on earlier layers. The objective does
not read sensitivities. Uniform remains the comparator when the target is an
allowed width. At 3.5 bits, uniform is undefined and `matched_uniform` is the
baseline. Minimax, average and softmax remain the sensitivity-using objectives.

## Second amendment — 2026-11-03, before research results

No quantized, KV or speculative result exists; the results dataset holds FP16 baselines and
two labelled smoke runs only. These specify points the protocol above left open.

**Grid.** The PRD's "12 quality cells × 2 models, extra seeds on Belebele" is operationalized:
MGSM runs the 12 cells at multilingual calibration, seed 0. Calibration language and the three
seeds replicate on Belebele only. The speed grid runs those 12 cells draft off and on for 1.5B
(24 cells). 3B runs FP16 and W4 with a full cache, draft off and on (4 cells).

**K2.** The vector is one language's 4-bit sensitivities across decoder blocks. The PRD
threshold (minimum pairwise Spearman at most 0.9) stays literal. The profile uses two disjoint
text halves per language, and their Spearman-Brown-corrected reliability is reported as a
separate `noise_limited` warning. The warning does not change the gate.

**Speculative decoding.** FR-5 exactness runs on 10 prompts per language (110 in total) under
each cache policy, past EOS, up to 256 tokens. Compressed-cache block verification may replace
the labelled sequential control only after that gate passes on a T4.

**Oaxaca covariates.** Item length is the English token count of the parallel item, so it
carries no token premium. Token count is the language's own count for that item.

## Third amendment — 2026-11-06, before research results

K1 passed on 2026-11-06. No quantized, KV or speculative result exists; the results dataset
holds FP16 baselines, the lm-eval reference and two labelled smoke runs only.

**Compute.** Logged FP16 throughput projects the 1,188-run audit at 80.1 T4 GPU-hours against
the PRD's 60. Calibration language and the three subsample seeds now replicate on 1.5B
Belebele only. 3B runs the 12 quality cells (four weight levels × three cache policies) on
Belebele and MGSM at multilingual calibration, seed 0. The audit becomes 858 runs and about
61 GPU-hours, of which 1.4 are FP16 baselines already logged.

**Why this cut.** H1 to H3 and the Shapley attribution need every cell of each factorial on both
models, and this keeps them. Calibration language is not a confirmatory hypothesis. A 3B
calibration contrast would need its own seed spread to be interpretable, and that spread is
the cost being removed. Dropping 3B MGSM eviction instead would remove a cell of the weight ×
eviction factorial that H2 names as its leading candidate. 3B calibration effects are out of
scope and are not reported.

## Fourth amendment — 2026-11-07, after the FR-3 replication result, before audit results

**Disclosure.** This amendment follows a failed FR-3 check. On 2026-11-06 notebook 02 ran the
1.5B replication grid. Our W4 differed from Qwen's official GPTQ-Int4 checkpoint by 7.5% mean
FLORES devtest perplexity (limit 2%). Those W4 and W3 replication runs are superseded, kept in
the results dataset and reported as a deviation. No audit (M2) result exists.

**Calibration data.** GPTQ calibration changes from 128 single FLORES dev sentences (a few
thousand tokens, below the 8,960 inputs of the widest layer, so its Hessians were
rank-deficient) to 128 windows of 2,048 tokens (262,144 tokens). The windows come from native-language Wikipedia
(`wikimedia/wikipedia`, dump 20231101, pinned commit `b04c8d1c`), using at most 512 tokens per
article. English and multilingual sets have equal token budgets. The multilingual set gives each
of the 11 languages 11 or 12 windows; the seed rotates the remainder. Seeds select the shard, row
groups and articles. FLORES dev is no longer used for calibration. FLORES devtest stays the
held-out NLL set.

**FR-3 reference.** The primary reference is GPTQModel 7.6.0 quantizing the same model on the
same windows. Its settings are pinned to ours: asymmetric, group 128, no act-order, no
group-aware reordering, damping 0.01, true-sequential. The pass rule is unchanged: mean
absolute perplexity gap ≤ 2%. The official checkpoint is reported descriptively only.

**Unchanged.** Hypotheses, grids, budgets, gates and analysis are unchanged. Every GPTQ cell
runs with the new calibration. FP16 and W8 RTN runs do not use calibration and are reused.

**Correction, 2026-11-07, after the first FR-3 run under this amendment.** That run failed
(mean gap 4.32%). The GPTQModel reference had run in bfloat16 and saved its asymmetric zero
points in the v1 format, which GPTQModel's own source says overflows. The reference now runs in
FP16 and saves `gptq_v2`; its runs get new IDs and the earlier ones are kept as superseded. Our
quantizer, the calibration and the 2% rule are unchanged.

