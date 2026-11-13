# Decision: how ParityQuant measures a language's degradation (allocator and H4)

Date: 2026-11-13. Skill: `researching-design-choices`. Status: **proposed, awaiting owner approval**
(it changes how a preregistered hypothesis is measured).

## Choice

Which quantity defines "a language's degradation" for (a) the ParityQuant allocator's
sensitivities s(l, j, b) and (b) the H4 / K3 test of worst-language degradation.

## Options (written before searching)

- **A.** Keep the preregistered wording: H4 on raw Belebele accuracy drop; allocator on mean
  per-token NLL increase (PRD metric table, "NLL change: mean per-token NLL change").
- **B.** Above-chance relative loss on calibrated Belebele accuracy, as H1 under the fifth
  amendment: 1 − (acc_q − 0.25)/(acc_fp16 − 0.25).
- **C.** Relative increase of total NLL on held-out FLORES devtest sentences (the fifth
  amendment's replication measure).
- **D.** Absolute increase of total NLL per parallel FLORES sentence (nats per sentence) on
  held-out sentences, for both allocator and H4 primary; Belebele as the accuracy check.
- **E.** Calibrated raw Belebele accuracy drop (letter prior removed, no headroom correction).

## Local constraints

- PRD success row and RQ4: worst-language degradation lower than the best baseline with a CI
  excluding 0 at 2 of 3 budgets (3.0, 3.5, 4.0 bits), mean accuracy cost ≤ 1 point.
- The allocator minimizes max over languages of Σ s(l, j, b); s is the probe's **mean per-token**
  NLL increase (`parityquant/profile.py`: `nll - baseline`, both `loss / token_count`).
- The probe profile (`probe-6f52db8b4007`, K2 PASS) uses FLORES devtest items 0–499 by item ID.
  Items 500–1011 are unused by the allocator.
- Belebele at FP16 (1.5B): te 29.9%, sw 33.1%, bn 39.8% against 25% chance (RUN_LOG entry 14).
- **Already seen:** 1.5B uniform W4 and W3 GPTQ results on Belebele and on all 1,012 FLORES
  devtest sentences (replication grid). These are the uniform baselines at budgets 3.0 and 4.0.
  Not seen: any minimax, average or matched-uniform allocation; no H4 statistic computed.
- Cost: T4-hours only; a held-out FLORES NLL pass for 1.5B is about 9 minutes per bit map
  (logged: 10,120 sentences in about 16 min at batch 8, DECISIONS 2026-11-03).

## What the probe inputs show (allocation inputs, not results)

4-bit summed single-block sensitivity per language from the K2 profile, three units:

| lang | tokens per sentence | per-token rank | per-sentence rank | relative rank |
| --- | --- | --- | --- | --- |
| en | 28.8 | 4 | 11 | 8 |
| de | 44.3 | 2 | 5 | 5 |
| fr | 44.7 | 8 | 7 | 7 |
| es | 43.2 | 7 | 8 | 10 |
| ru | 50.5 | 3 | 4 | 1 |
| zh | 30.0 | 1 | 9 | 6 |
| ja | 40.5 | 5 | 6 | 9 |
| th | 71.9 | 6 | 3 | 2 |
| bn | 140.2 | 10 | 2 | 3 |
| sw | 54.3 | 9 | 10 | 11 |
| te | 191.1 | 11 | 1 | 4 |

Per-token units put Telugu and Bengali last because their text splits into 4.9–6.6× more
tokens than English, each individually easier to predict. **A minimax over per-token NLL would
protect Chinese, German and Russian, not the languages the project is about.**

## Evidence

| # | Source | Type | Claim used | Grade | Our setting |
| --- | --- | --- | --- | --- | --- |
| 1 | Mielke et al. 2019, ACL, [doi](https://doi.org/10.18653/v1/p19-1491) | peer-reviewed | Compare languages by surprisal of translations of the same sentences (multi-text), so content is held fixed | Strong | Partly (pre-LLM LMs) |
| 2 | Petrov et al. 2023, NeurIPS, [doi](https://doi.org/10.52202/075280-1608) | peer-reviewed | The same text tokenizes up to about 15× longer across languages; byte/char models still >4× | Strong | Yes |
| 3 | "Apples to Apples?" 2026, EMNLP, [arXiv 2608.25089](https://arxiv.org/abs/2608.25089) | peer-reviewed | Several normalized metrics carry cross-lingual biases from tokenization, encoding, orthography; proposes sentence-level NLL over semantically equivalent sequences | Moderate (abstract read) | Yes |
| 4 | Tsvetkov & Kipnis 2024, Findings EMNLP, [ACL](https://aclanthology.org/2024.findings-emnlp.468/) | peer-reviewed | Information Parity (compression efficiency vs a reference language) correlates better with task benchmarks than other metrics they know | Moderate | Partly |
| 5 | Marchisio et al. 2024, Findings EMNLP, [doi](https://doi.org/10.18653/v1/2024.findings-emnlp.935) | peer-reviewed | Automatic metrics underestimate quantization harm (Japanese 1.7% automatic vs 16.0% human); non-Latin scripts worst | Strong | Yes |
| 6 | Jin et al. 2024, Findings ACL, [arXiv 2402.16775](https://arxiv.org/abs/2402.16775) | peer-reviewed | Perplexity vs benchmark scores of GPTQ/SpQR 3- and 4-bit Qwen-Chat: mean absolute Pearson 0.7895 (Table 3) | Moderate | Yes (Qwen, GPTQ) |
| 7 | Dutta et al. 2024, NeurIPS, [arXiv 2407.09141](https://arxiv.org/abs/2407.09141) | peer-reviewed | Accuracy changes within 0–2% hide up to 13.62% answer flips (Llama2-7B GPTQ W4A16, MMLU); report KL or flips with accuracy | Strong | Partly |
| 8 | Jaiswal et al. 2024, ICLR, [arXiv 2310.01382](https://arxiv.org/abs/2310.01382) | peer-reviewed | Perplexity misses knowledge-intensive losses of compressed LLMs | Strong | Partly |
| 9 | LCD 2026, [arXiv 2608.11786](https://arxiv.org/abs/2608.11786) | preprint | INT3 GPTQ Qwen2.5-3B: non-English perplexity degradation 2–4× English; correction recovers 70–83% of the perplexity gap but 17–28% of the GlobalMMLU gap (perplexity–accuracy disconnect) | Weak–moderate | Yes (Qwen2.5, GPTQ) |
| 10 | Rababah et al. 2026, [arXiv 2607.08734](https://arxiv.org/abs/2607.08734) | preprint | Behaviour shifts even when accuracy and perplexity are preserved | Weak | Partly |
| 11 | Open LLM Leaderboard normalization, [HF docs](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/normalization) | docs | Subtract the 1/k random baseline and rescale, so gains above chance are comparable | Moderate | Yes |
| 12 | Zheng et al. 2023, [arXiv 2309.03882](https://arxiv.org/abs/2309.03882) | preprint, widely cited | Multiple-choice selection bias from option-ID token priors; PriDe removes the prior | Moderate | Yes |
| 13 | Sagawa et al. 2019/2020, ICLR, [arXiv 1911.08731](https://arxiv.org/abs/1911.08731) | peer-reviewed | Worst-group metric is reported alongside average; groups predefined | Strong | Partly |
| 14 | Bertsimas et al. 2011, Operations Research, [doi](https://doi.org/10.1287/opre.1100.0865) | peer-reviewed | Price of fairness for max-min allocation; tight bounds over convex utility sets | Strong | Partly (our set is discrete) |
| 15 | Hake's g versus Cohen's d, [arXiv 1612.09180](https://arxiv.org/abs/1612.09180) | preprint | Ceiling-normalized gains over-correct and favour high-pretest groups; disputed by others | Weak | Analogy only |
| 16 | Ogueji et al. 2022, EMNLP, [doi](https://doi.org/10.18653/v1/2022.emnlp-main.619); Ahia et al. 2021, Findings EMNLP, [arXiv 2110.03036](https://arxiv.org/abs/2110.03036) | peer-reviewed | Compression effects on low-resource languages depend on regime; harm concentrates on rare inputs | Moderate | Partly |
| 17 | FairQuant 2026 [arXiv 2602.23192](https://arxiv.org/abs/2602.23192); Colla-Q 2026 [arXiv 2609.18131](https://arxiv.org/abs/2609.18131) | preprint / peer-reviewed | Adjacent prior art: report worst-group performance under shared budgets; abstracts do not define the unit | — | Novelty boundary |

All 17 entries pass `litsearch.py check`.

## Case for and against each option

**A. Raw Belebele accuracy; per-token NLL allocator.**
For: as preregistered; accuracy is what users feel (Marchisio [5]).
Against: the floor caps the measurable drop of te, sw and bn (6.7–16.3 points of calibrated
headroom), so the worst language by raw points is a high-resource one; letter collapse adds
noise ([12]). Per-token units rank te and bn last (table above; [2], [3]). The allocator and the
test would both favour the wrong languages. **Rejected.**

**B. Above-chance retention on Belebele.**
For: corrects the floor (leaderboard practice [11]); consistent with H1's fifth-amendment loss.
Against: the denominator is tiny for te (6.7 points calibrated headroom on 900 items, binomial
SE about 1.5 points), so the ratio's standard error is about 0.2–0.3 for te alone. A max over
11 noisy ratios selects the noisiest language, and the H4 interval would be wide for reasons
unrelated to the method. Ceiling-normalized gains are known to over-correct ([15], disputed).
**Kept as a secondary, descriptive measure.**

**C. Relative total NLL increase (held-out).**
For: tokenization-invariant (ratio over the same tokens); the fifth amendment's replication
measure; low noise.
Against: dividing by baseline NLL makes a poorly modeled language look robust: Swahili
(4.84 nats per token at FP16) ranks last, the same floor problem in another form. Optimizing and
testing on NLL risks a perplexity–accuracy disconnect ([6] supports correlation, [9], [10]
dispute it). **Reported as a sensitivity analysis.**

**D. Absolute total NLL increase per parallel sentence (held-out).**
For: −log p(string) on translations of the same sentence holds content fixed and does not depend
on how the string is tokenized ([1], [3]); its increase is the information the quantized model
loses on that content. It ranks te, bn, th, ru as most damaged and en least, the published
direction ([5]). Continuous and paired, so a max over 11 languages is not dominated by noise.
Held-out items 500–1011 keep the test out of the allocator's sample.
Against: still an NLL measure, so the perplexity–accuracy disconnect applies ([8], [9], [10]):
a minimax win in nats may not carry to task accuracy. Mitigation: H4 keeps the PRD's accuracy
condition, and Belebele worst-language retention is reported with it. Belebele passages are
drawn from FLORES-200, which may overlap the probe's devtest items; the split was not
confirmed, so Belebele is not fully held out from the allocator's text.

**E. Calibrated raw accuracy drop.** Fixes letter bias but not the floor. **Rejected.**

## Where sources disagree

Perplexity tracks benchmark scores well for Qwen-Chat GPTQ at 3–4 bits ([6]) but not for
corrections to quantized Qwen2.5-3B ([9]) or knowledge-heavy tasks ([8]). The difference is
plausibly what is being changed: uniform bit-width changes move perplexity and accuracy together,
while targeted changes can repair likelihood without repairing decisions. ParityQuant is a
targeted change, so the accuracy check is required, not optional.

## Decision (proposed)

**D**, as one unit for allocator and test:
1. Allocator sensitivity s(l, j, b) = per-token NLL increase × mean tokens per probe sentence
   of language l (nats per parallel sentence). Same profile, no new GPU work.
2. H4 / K3 primary: worst-language mean increase in total NLL per sentence on FLORES devtest
   items 500–1011, versus the best baseline, paired bootstrap over items (parallel sentences
   shared across languages), 2,000 resamples. Pass rule unchanged: CI excludes 0 at 2 of 3
   budgets.
3. H4 accuracy condition unchanged: mean calibrated Belebele accuracy cost ≤ 1 point.
4. Secondary, descriptive: worst-language above-chance retention on Belebele (B), relative NLL
   (C), and answer flips ([7]).

Confidence: medium-high for rejecting A; medium for D over C.

**Cost:** one held-out FLORES NLL pass per bit map, about 9 T4-minutes each; 14 bit maps at
1.5B is about 2.1 T4-hours on top of Belebele.

**Protocol impact:** changes how H4 is measured and the allocator's units. Needs a sixth
amendment before any ParityQuant run. Disclose that uniform W3/W4 baselines (budgets 3.0 and
4.0) were seen on Belebele and on all FLORES devtest sentences, and that the per-token ranking
above was seen on the K2 profile before choosing.

**What would reverse it:** an owner decision to keep the original wording (then A runs as
preregistered and the units issue is reported as a limitation); or evidence that per-sentence
NLL ranks languages against human judgements of quantization damage.

## Novelty note (K4)

LCD (arXiv 2608.11786, 2026) corrects quantized Qwen2.5-3B per language with LoRA adapters. It
is not mixed precision and has no worst-language objective; add it to RESEARCH_CONTEXT as
adjacent work.

## Round 2 (owner asked for more research on the most efficient solution)

Further options considered: bits per byte (BPB), bits per character (BPC), Information Parity,
excess-loss (baseline-subtracted) group DRO, and a CPU robustness check across units.

| # | Source | Type | Claim used | Grade |
| --- | --- | --- | --- | --- |
| 18 | Yang, Wilcox & Arnett, "Apples to Apples?", EMNLP 2026, [arXiv 2608.25089](https://arxiv.org/abs/2608.25089) (full text read, §3 and Limitations) | peer-reviewed | "Normalization over tokens, characters, or bytes introduces systematic biases and is unsuitable for cross-tokenizer or cross-lingual comparison." BPB and BPC suit cross-tokenizer comparison within one language only; Information Parity and BPEC are transformations of Sent-NLL. Sent-NLL = sum of token NLL over parallel text, "normalizes by meaning rather than representation units". Limitation: needs high-quality parallel text (FLORES provides it); Sent-NLL rankings vary more across datasets, which the authors argue reflects real content differences | Strong |
| 19 | Zhou et al., Distributionally Robust Multilingual MT, EMNLP 2021, [doi](https://doi.org/10.18653/v1/2021.emnlp-main.458) (full text read) | peer-reviewed | Worst-case over language pairs uses token-level loss with a baseline loss subtracted before the max (after Oren et al. 2019); "what are the right baseline losses?" is left open; the choice of baseline changed which dataset benefited | Strong |
| 20 | Oren et al., Distributionally Robust Language Modeling, EMNLP 2019, [doi](https://doi.org/10.18653/v1/d19-1432) | peer-reviewed | Subtract a per-group baseline before the worst case, so groups with high irreducible loss do not dominate | Strong |
| 21 | DoReMi, Xie et al., NeurIPS 2023, [doi](https://doi.org/10.52202/075280-3059) | peer-reviewed | Group DRO over domains on excess loss relative to a reference model | Moderate (abstract read) |
| 22 | Vowel Signs Are Not Letters, 2026, [arXiv 2608.26449](https://arxiv.org/abs/2608.26449) | preprint | Per-token loss across different segmentations measures different quantities | Weak |

**What this adds.**
- BPB and BPC are rejected: both carry script bias across languages ([18]); BPC penalizes
  logographic Chinese.
- Our degradation (quantized minus FP16 on the same text) is already an excess loss with the
  FP16 model as the baseline, which is the construction the multilingual DRO literature uses to
  stop high-loss groups dominating a max ([19], [20], [21]). The open question those papers
  leave, the right baseline, is answered here by the FP16 model itself.
- The remaining choice is the unit of that excess loss. Token-level (as in [19]) is biased
  across languages by segmentation ([2], [18], [22]); sentence-level on parallel text is the
  only unit [18] finds suitable across languages.

**Efficiency.** Option D costs no new probe GPU time: the per-token profile converts to
per-sentence units exactly with the probe's own token counts, computed on the M1. Its held-out
test adds about 9 T4-minutes per bit map. BPB, BPC and Information Parity would cost the same
but are biased or redundant. A free robustness check: solve the allocations under per-sentence,
relative and per-token units on the M1 and report how much the bit maps differ, so a reviewer
can see how much the unit choice matters before any GPU run.

**Updated recommendation:** unchanged, D, with the CPU robustness check added. Confidence
raised to medium-high.
