# Research context and novelty boundaries

Read this before writing anything that claims novelty: README, paper or comments.

## How we got here (21 Oct 2026)

- **AegisServe** (an adaptive control plane for LLM serving, built in a simulator) was **rejected as the flagship.** SLO- and goodput-aware scheduling is crowded (SOLA, Apt-Serve, JITServe, GoodServe, Equinox), and a simulator-only result is weak evidence.
- **"On-device compression + adaptation pipeline"** was **rejected.** It had no research question, AWQ, GPTQ and QLoRA already cover it, and it can't be won on Colab compute.
- **Chosen: WhoPays + ParityQuant.** It fits the owner's fairness, EU AI Act assurance and econometrics background, inference-engineering roles (quantization, speculative decoding, KV cache) and PhD groups working on trustworthy or efficient ML.

## What is NOT new (cite it, don't claim it)

| Prior result | Source |
| --- | --- |
| Quantization hurts non-Latin and low-resource languages more | Marchisio et al. 2024 (arXiv 2407.03211); MT study (2508.20893) |
| Multilingual or target-language calibration beats English-only calibration | Chimoto et al., EACL 2026 (2601.18306) |
| Speculative decoding acceptance drops in low-resource languages | Sandler et al. (2510.02128); Paudel et al. (2605.30580) |
| Multilingual drafters | AdaSpec (AAAI 2026) |
| Composing weight × KV × speculative transformations in a factorial design, for **safety** | Waheed et al., Research Square preprint, Oct 2026 |
| Mixed-precision bit allocation, including Shapley-based and calibration-aware variants | IMPQ (2509.15455), TASA (2607.00908), MixQuant (2607.23047), SliM-LLM, GAMMA |
| Language-proportional calibration sampling for pruning | Multilingual Brain Surgeon (2404.04748) |
| Worst-language objective in tokenizers | Parity-aware BPE |

## What we claim is new (keep claims this narrow)

1. **A per-language audit of the *composed* stack, with quality and speed measured as separate channels.**
2. **Layer-level attribution of language gaps,** using Shapley values over the factorial grid and an Oaxaca–Blinder endowment/treatment split.
3. **ParityQuant:** a minimax (worst-language) objective for mixed-precision bit allocation, with a price-of-parity curve.

## Re-check before publishing

Search arXiv for "worst-group" OR "minimax" mixed-precision quantization, and for multilingual compression fairness. Run the search again at M3 and M4. If something close appears, apply kill criterion K4 (PRD).

## Target venues

- **Audit:** ACL/EMNLP (main or Findings), or FAccT.
- **Method:** efficient-ML or trustworthy-ML workshops.

## Why this matters for the owner

- **PhD applications:** ELLIS, IMPRS-IS, MCML; groups at Copenhagen DIKU, Aalto and TU Delft.
- **Industry roles:** inference and quantization engineering (NVIDIA, Nebius, Mistral) and AI assurance work (EU AI Act).
- **CV bullets:** written only after results exist. Never use invented numbers.
