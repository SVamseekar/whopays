# KV-cache quantization: recipe for the `quant4` cell (2026-11-14)

Gathered under `investigate-then-research` after the 4-bit cache collapsed MGSM on Qwen2.5-1.5B
(RUN_LOG 17). Evidence grades: A = read in full or the relevant section read, B = abstract or
secondary summary only.

| Source | Claim used | Grade |
| --- | --- | --- |
| KIVI, Liu et al., ICML 2024, [arXiv 2402.02750](https://arxiv.org/abs/2402.02750) | Key outliers concentrate in fixed channels, so keys are quantized per channel ("confine the error to each individual channel"); values per token. Section 4.1: "group size G ... set as 32 across all experiments"; "residual length R ... set to 128". Round-to-nearest with zero point and scale. KIVI-4 stays close to 16-bit on most models. | A (sections 3.1, 4.1, 4.2) |
| KVQuant, Hooper et al., NeurIPS 2024, [doi](https://doi.org/10.52202/079017-0040) | Per-channel key quantization (pre-RoPE in their method) for outlier channels. | B |
| KVTuner, [arXiv 2502.04420](https://arxiv.org/abs/2502.04420) | Qwen2.5-7B-Instruct is unusually sensitive to low-bit keys; uniform per-token 4-bit and 2-bit keys cause large attention-score errors in some layers; argues against one static precision. Case against plain KIVI-4 on Qwen. | B |
| [Hurwitz quaternion KV quantization, arXiv 2605.27646](https://arxiv.org/pdf/2605.27646) | Qwen2.5-7B key channels with K_max/K_med above 250× in some layers. | B |
| transformers 5.19 `cache_utils.QuantizedLayer` (source read at the pinned version) | Quantizes the whole prompt at the first update and re-quantizes the full dequantized history at every residual flush; docstring says per-channel but axis 1 in HQQ groups per token. | A |
| Community report, [InventiveHQ benchmark](https://inventivehq.com/blog/kv-cache-quantization-quality-benchmark) | q4_0 KV on a Qwen2.5-Coder model drops output similarity to 8.3%. Informal; pointer only. | B |

## Decision

KIVI as published (per-channel keys, per-token values, G = 32, R = 128, 4 bits, asymmetric RTN),
implemented as a fake-quantized cache layer (`kv.policies.KIVILayer`), each token quantized once.
Rejected: HQQ per-token (collapses on Qwen2.5); HQQ axis 0 (groups strided across heads, tokens
and batch rows); smaller per-token groups (an outlier still sets the scale of its group);
mixed-precision keys per layer as in KVTuner (a different, tuned method; the audit asks what a
standard 4-bit cache does). Against: KVTuner's evidence that even KIVI-4 errs on some Qwen2.5
heads; the audit reports the `quant4` cell as KIVI-4, not as "any 4-bit cache".
