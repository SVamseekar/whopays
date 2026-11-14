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
| Fixed KV-cache eviction budgets penalize high-fertility languages; retention-matched (share-of-input) budgets remove the mean gap | "Equal Tokens, Unequal Losses", anonymous ACL submission 2026 (OpenReview 57mHZnnuEe); eviction only |
| Speculative decoding widens cross-lingual latency gaps (21 languages) | "No Refund for the Token Tax", Oct 2026 preprint; speculation only |
| Per-language correction of a quantized LLM (rank-2 LoRA per language on INT3 GPTQ Qwen2.5-3B / Llama-3.2-3B); perplexity–accuracy disconnect | [LCD](https://arxiv.org/abs/2608.11786), 2026 preprint; post-hoc adapters, not mixed precision, no worst-language objective |
| Calibration language × bit-width × mixed precision (GPTQ 2–4 bit, SliM-LLM, TaCQ) on Qwen3-8B, Llama-3.1-8B, Aya; 5 languages (en, ta, id, sw, zh); gains "clear in perplexity, although rarely substantial in downstream tasks" | Waskito et al., [Findings of IJCNLP-AACL 2026](https://doi.org/10.18653/2026.findings-aacl.9); weights only, no KV cache, speculation, composition or attribution |
| Scalar per-layer sensitivity proxies distort mixed-precision allocation, worst below 3 bits; cross-layer refinement fixes it | CASA, [arXiv 2609.25916](https://doi.org/10.48550/arxiv.2609.25916) (Sep 2026); inter-layer dependencies, [arXiv 2306.04879](https://doi.org/10.48550/arxiv.2306.04879) |

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
- **14 Nov 2026 (K4 review, from the owner's inference-bottlenecks report):**
  [No Refund for the Token Tax](https://www.alphaxiv.org/abs/2610.speculative-decoding-token-tax-languages)
  (Oct 2026) shows speculative decoding widens cross-lingual latency gaps across 21 languages
  (English acceptance highest; Finnish relative latency 1.55x to 3.02x on vLLM). Speculative
  decoding alone is not new ground; it covers no quantization, KV compression or composition.
  "Equal Tokens, Unequal Losses" (anonymous ACL submission, OpenReview `57mHZnnuEe`) was read in
  full on 14 Nov (K4 resolved within 48 h). It establishes, for KV eviction alone on Belebele, that
  fixed token budgets penalize high-fertility languages (+27.2 points at half the English length),
  that differential retention explains it, and that budgets as shares of input remove it on
  average. **Do not claim** a per-language fixed-budget eviction penalty, or the share-of-input
  remedy, as ours; cite it. Still open for WhoPays: eviction composed with weight quantization and
  speculative decoding (H2's weight × eviction interaction), attribution of the composed gap,
  Qwen2.5 and MGSM generation, and ParityQuant.
  Corpus: docs/literature/inference-bottlenecks.md.

## Target venues

- **Audit:** ACL/EMNLP (main or Findings), or FAccT.
- **Method:** efficient-ML or trustworthy-ML workshops.

## Why this matters for the owner

- **PhD applications:** ELLIS, IMPRS-IS, MCML; groups at Copenhagen DIKU, Aalto and TU Delft.
- **Industry roles:** inference and quantization engineering (NVIDIA, Nebius, Mistral) and AI assurance work (EU AI Act).
- **CV bullets:** written only after results exist. Never use invented numbers.
