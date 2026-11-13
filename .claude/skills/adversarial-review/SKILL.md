---
name: adversarial-review
description: Use in WhoPays before any claim leaves the lab notebook — a README or paper section, an abstract, a figure caption, a novelty statement, a hypothesis verdict (H1–H4), a gate verdict reported as a finding, or when the user asks "is this result solid" or "what would reviewers say".
---

# Adversarial review

## Overview

Attack the claim the way a hostile ACL, EMNLP or efficient-ML reviewer would, before a reviewer
does. Each attack ends as one of: survives (with evidence), needs a fix, or needs a narrower
claim. Done inline, without subagents, unless the user asks for them.

## Steps

1. **Write the claim** as one sentence, plus where it will appear.
2. **Run every attack** below and write one line each: the attack, the evidence for or against
   it (file, run, table, citation), and the verdict.

   | Attack | Question |
   | --- | --- |
   | Novelty | Is it already in `docs/RESEARCH_CONTEXT.md` prior art, or in work published since? Run a quick search under `references/literature-protocol.md` of `researching-design-choices` for the last 6 months. |
   | Overclaim | Does the wording go beyond the setting: model sizes, Qwen only, 11 languages, T4 FP16, fake quantization? |
   | Statistics | Interval, test and multiple-comparison correction stated? Effect size larger than seed and item noise? Paired where it should be? |
   | Confirmatory or exploratory | Was this test preregistered, or added by an amendment after the data was seen? Is it labelled so? |
   | Floor and ceiling | Could chance level or headroom explain it (the Belebele floor lesson)? |
   | Alternative explanation | Tokenizer premium, prompt format, letter bias, batch-shape numerics, calibration data, decoding settings? |
   | Selection | Are all 11 languages, all budgets and all seeds shown, or only the ones that support the claim? |
   | Reproducibility | Run IDs, commit, config and data on the HF dataset? Can a reviewer regenerate the number? |
   | Baselines | Is the strongest fair baseline included and tuned equally (FairQuant-style, average-objective mixed precision)? |
   | Cost honesty | Are memory savings computed from bit counts and labelled so, not claimed as measured? |

3. **Steelman the opposite claim** in two or three sentences, with its best evidence.
4. **Verdict**: keep, narrow (give the new wording), or hold (give the missing check and its
   GPU cost).
5. **Record** in the decision note or `docs/DECISIONS.md` when the wording or plan changes.
   Commit by exact path, push.

## Common mistakes

- Reviewing only the attacks that are easy to answer.
- Answering an attack from memory instead of a file or run.
- Narrowing the claim silently instead of saying what was cut and why.
