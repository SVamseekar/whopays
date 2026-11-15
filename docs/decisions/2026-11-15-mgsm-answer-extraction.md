# MGSM answer extraction for zero-shot native chain of thought (exploratory re-score)

Date: 2026-11-15. Skill: `researching-design-choices`. Corpus: `docs/literature/answer-extraction.md`.

## Choice

How to read the final answer from zero-shot MGSM outputs that often run to the 256-token cap and
restate question numbers around the answer, so that compression contrasts are not driven by
where a treatment's answers put their last number.

## Options (written before searching)

- A. Last number only (lm-eval flexible-extract; the K1-validated primary score).
- B. lm-eval strict-match (first number after a per-language answer phrase).
- C. Multilingual answer-word rule: first number after the last answer word, falling back to the last number.
- D. Cut at continuation markers (a new question in any language), then the last number.
- E. Independent LLM reader, blind to gold and treatment, validated against human labels.
- F. Regenerate with a larger cap or an answer-format instruction.
- G. Most frequent number among final statements.

## Local constraints

- Flexible-extract is the preregistered, K1-validated score (lm-eval agreement 98.4%, DECISIONS 2026-11-03); changing the primary after seeing results is ruled out.
- Results are seen, so any second score is exploratory and must be disclosed as written after seeing outputs.
- F costs GPU time and changes the generation protocol; ruled out for the audit.
- E needs a reader model: an API costs money and needs the owner's approval; a Qwen-family reader judging Qwen outputs risks self-preference; a 7B reader on the M1 would take days.

## Evidence from our outputs (slices 0–4, 54,000 MGSM answers)

- Cap hits: 21.6% (1.5B en) to 100% per cell; weight quantization and Telugu eviction raise them (DECISIONS 2026-11-15).
- B: the strict phrase appears in 0–12.4% of 1.5B FP16 answers (strict accuracy 0–7.6% against 0.8–61.2%).
- A fails when the answer sentence restates question numbers after the answer: "Answer: Claire will eat 7 dozens of eggs in 4 weeks" scores 4; "8 centimeters shorter after burning from 1:00 PM to 5:00 PM" scores 0. 3B FP16 English (52.0%) is below 1.5B (61.2%) while 3B is ahead in every other language; 55 items flip right-to-wrong and 32 wrong-to-right.
- C (implemented and tested as `tasks.mgsm.extract_after_answer_word`; answer words taken from lm-eval's own phrases) changes 2,370 of 54,000 decisions: 1,789 wrong-to-right, 581 right-to-wrong. It fails the mirror way: "Ответ: через 10 лет будет 109 лет" scores 10; answer words reappear in verification text ("Esta respuesta se puede verificar…") and in prompt echoes ("Jibu la Hatua…"). It finds an answer word in only 36% of 3B English answers, so it barely moves the 3B English figure (52.0 to 53.6). Largest shifts: 1.5B FP16 Russian 20.8 to 36.4, Thai 11.2 to 18.0, German 22.4 to 27.2.
- D: no sampled failure involved a new question; the errors sit inside the answer sentence, so D alone fixes little.

## What the field does

| Source | Type | Claim used |
| --- | --- | --- |
| Shi et al., MGSM ([arXiv 2210.03057](https://doi.org/10.48550/arxiv.2210.03057)) | Benchmark paper | Few-shot exemplars that end in a stated answer; extraction designed for that format. |
| Peter et al., "Mind the Gap… or Not?" ([GEM 2026](https://doi.org/10.18653/v1/2026.gem-main.22); [arXiv 2511.05162](https://arxiv.org/abs/2511.05162)) | Workshop paper, read in full | No standard MGSM extraction. Relaxing the regex alone moves GPT-5 French 80.0 to 90.0 and Gemma 3 27B Bengali 45.2 to 86.4; with corrected data the language gap "mostly disappears". Recommends stating the format in the prompt and counting non-compliance as error. |
| "Same Answer, New Box" ([arXiv 2610.20639](https://doi.org/10.48550/arxiv.2610.20639)) | Preprint, read in full | Two treatments that put answers in different places get different extractor scores: 82% of a reported gain was format. Across 112 extractor combinations on identical outputs, reported gains differ by a median 15.1 points. Recommends scoring the same outputs with a second extractor, keeping abstention balanced, and checking with independent LLM readers (98.6% agreement with their rule). |
| "Right Answer, Wrong Score" ([Findings ACL 2025](https://doi.org/10.18653/v1/2025.findings-acl.950)) | Conference paper, read in full | Regex extraction underestimates models; LLM extractors align better with humans but make systematic errors. Format constraints ease extraction but may hurt reasoning. |
| xFinder ([arXiv 2405.11874](https://doi.org/10.48550/arxiv.2405.11874)) | Preprint | Fine-tuned extractor: 93.42% extraction accuracy against 74.38% for the best regex framework. Languages and numeric math answers not covered on the abstract page. |
| Kojima et al. ([NeurIPS 2022](https://doi.org/10.52202/068431-1613)); Plan-and-Solve ([ACL 2023](https://doi.org/10.18653/v1/2023.acl-long.147)) | Conference papers | Zero-shot CoT uses a second prompt ("Therefore, the answer (arabic numerals) is") to extract the answer; needs a second forward pass (GPU). |
| lm-eval 0.4.13 `mgsm_native_cot` source | Code, pinned version read | Two filters: strict-match (phrase) and flexible-extract (last number); no handling of restated numbers. |
| "The Coupling Tax" ([arXiv 2605.07686](https://doi.org/10.48550/arxiv.2605.07686)) | Preprint | A last-number fallback on truncated reasoning changes accuracy from under 6% to 56.9%, so reported accuracies are "optimistic". |
| Noise injection study ([arXiv 2505.13500](https://doi.org/10.48550/arxiv.2505.13500)); GEMS ([arXiv 2606.19946](https://doi.org/10.48550/arxiv.2606.19946)) | Preprints | Last-number false matches on truncated text; truncation counted as error and regex misses counted as extraction bugs. |
| "When Reasoning Meets Compression" ([arXiv 2504.02010](https://doi.org/10.48550/arxiv.2504.02010)) | Preprint | Compression changes reasoning behaviour, not only accuracy (supports treatment-dependent output form). |
| PolyMath ([NeurIPS 2025](https://doi.org/10.52202/085713-1431)) | Benchmark paper | Thinking length differs by language; output-language control changes scores. |

## Balanced synthesis

- **A, case for:** standard (lm-eval), validated at K1, deterministic, symmetric across treatments. **Against:** biased whenever a treatment changes how answers end; our 3B English and the literature (Coupling Tax, Same Answer New Box) show the size can be large.
- **B, for:** standard, phrase-anchored. **Against:** written for few-shot outputs; covers 0–12% here.
- **C, for:** cheap, deterministic, cue words not invented. **Against:** mirror-image failure on restated numbers; cue words recur in verification and echoes; 581 right-to-wrong flips in our sample; not validated.
- **D, for:** removes runaway continuations. **Against:** the errors we sampled are inside the answer sentence.
- **E, for:** best human alignment in the literature; reads stated answers in any form. **Against:** systematic LLM-extractor errors (Right Answer, Wrong Score); cost or self-preference; needs human validation.
- **F, for:** removes the cause. **Against:** GPU cost; changes the protocol after results.
- **G, for:** robust to one stray number. **Against:** ad hoc; no source uses it.
- Sources agree that extraction choice can move MGSM by more than the effects we measure (Mind the Gap: up to 41 points in one language), and that the remedy is a second extractor plus validation, not replacing the primary.

## Decision

1. Primary stays A (flexible-extract), unchanged.
2. Every MGSM contrast is also scored with C (`extract_after_answer_word`) as an **extraction-sensitivity check**, as "Same Answer, New Box" recommends. A and C fail in opposite directions on restated numbers, so their gap per cell bounds how much extraction can move it. An MGSM claim needs the same sign and BH significance under both; disagreement is reported, not resolved by choosing one.
3. C is not a corrected score: it is not validated and has known failure modes. It is labelled exploratory and written after seeing outputs.
4. Validation by an independent reader (option E) on a stratified sample (for example 20 disagreements per language plus 20 agreements) is the next step if MGSM claims go into the paper. It needs either the owner's approval for a small API spend or owner hand-labels. Not started.
5. The 3B-versus-1.5B English MGSM comparison is not reported as a model difference: both extractors leave 3B below 1.5B, but 64% of 3B English answers carry no answer word and the sampled losses are extraction errors.

## Rejected

B (coverage), D alone (misses the observed errors), F (GPU, protocol), G (ad hoc, unsourced), C as a replacement (unvalidated, mirror failures).

## Protocol impact

None on preregistered measures. The sensitivity check is exploratory, written after seeing outputs, and disclosed as such.

## What would reverse it

A validated reader (E) showing that C agrees with humans far better than A on our outputs would justify C (or the reader) as the corrected exploratory score. A published zero-shot multilingual MGSM extractor validated against human labels would replace C.
