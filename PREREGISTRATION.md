# WhoPays + ParityQuant preregistration

Draft date: 2026-10-22. Amended: 2026-10-31. Status: prospective protocol;
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
defined before H4 evaluation. Neither requirement is considered complete.
