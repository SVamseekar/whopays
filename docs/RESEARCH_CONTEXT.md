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
| Mixed-precision bit allocation, including Shapley-based and calibration-aware variants | [CoopQ/SPQE](https://arxiv.org/abs/2509.15455) (previously listed here as IMPQ), TASA (2607.00908), MixQuant (2607.23047), SliM-LLM, GAMMA |
| Group-aware mixed precision under bit budgets and fairness objectives | [FairQuant](https://arxiv.org/abs/2602.23192), medical image classification; not a multilingual LLM audit |
| Minimax precision balancing across MoE experts | [Colla-Q](https://arxiv.org/abs/2609.18131); experts are the protected units, rather than languages |
| Language-proportional calibration sampling for pruning | Multilingual Brain Surgeon (2404.04748) |
| Worst-language objective in tokenizers | Parity-aware BPE |
| Per-language correction of a quantized LLM (rank-2 LoRA per language on INT3 GPTQ Qwen2.5-3B / Llama-3.2-3B); perplexity–accuracy disconnect | [LCD](https://arxiv.org/abs/2608.11786), 2026 preprint; post-hoc adapters, not mixed precision, no worst-language objective |

## Candidate contributions (novelty remains provisional)

1. **A per-language audit of the *composed* stack, with quality and speed measured as separate channels.**
2. **Layer-level attribution of language gaps,** using Shapley values over the factorial grid and an Oaxaca–Blinder endowment/treatment split.
3. **ParityQuant:** calibration-only mixed-precision allocation for the worst language's degradation in dense multilingual LLMs, with a price-of-parity curve. Do not claim the first fairness-aware or minimax mixed-precision quantizer generally; FairQuant and Colla-Q are adjacent prior art.

## Re-check before publishing

Search arXiv for "worst-group" OR "minimax" mixed-precision quantization, and for multilingual compression fairness. Run the search again at M3 and M4. If something close appears, apply kill criterion K4 (PRD).

- **21 Oct 2026:** searches for per-language or multilingual evaluation of KV-cache compression (eviction or quantization) found none. Benchmarks such as kvpress/RULER, SCBench and LongBench-based studies are English or long-context only. Keep checking: this underpins the KV channel of claim 1.
- **31 Oct 2026 correction:** that negative search is not a defensible novelty claim.
  The indexed primary paper [Towards Dynamic KV-Cache Compression: Fine-Grained Evaluation
  of Key and Value Ranks in LLMs](https://openreview.net/pdf?id=0noVRtnzJt) includes multilingual
  VisR-Bench analyses. Its full text was blocked by OpenReview's browser challenge during
  this review; inspect the paper before specifying exactly what it does not cover. The
  candidate contribution is a paired language-disparity audit of the composed stack,
  rather than the first multilingual KV study.
- **31 Oct 2026:** [Lightweight but Linguistically Fair?](https://arxiv.org/abs/2610.18343)
  reports model- and method-dependent language effects. Treat disproportionate harm as a
  falsifiable hypothesis, rather than expecting every low-resource language to lose more.
  This and the FairQuant/Colla-Q overlaps narrow the claims under K4; they do not establish
  that the composed multilingual audit or the specific ParityQuant objective is covered.

## Target venues

- **Audit:** ACL/EMNLP (main or Findings), or FAccT.
- **Method:** efficient-ML or trustworthy-ML workshops.

## Why this matters for the owner

- **PhD applications:** ELLIS, IMPRS-IS, MCML; groups at Copenhagen DIKU, Aalto and TU Delft.
- **Industry roles:** inference and quantization engineering (NVIDIA, Nebius, Mistral) and AI assurance work (EU AI Act).
- **CV bullets:** written only after results exist. Never use invented numbers.
