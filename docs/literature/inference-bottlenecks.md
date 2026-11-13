# Literature corpus: multilingual inference bottlenecks (owner's research report)

Source: the owner's report "Multilingual LLM Inference Bottlenecks" (PDF in the working tree,
not committed), received 2026-11-14. It is a secondary synthesis: several of its tables are
analytical models attributed to a source listed as "unknown_url", and some citations point to
vendor blogs or forums. Only items checked against a primary source are graded here.

| Work | Status of check | Grade | Relevance to WhoPays |
| --- | --- | --- | --- |
| Yang, "No Refund for the Token Tax: Speculative Decoding Across 21 Languages", Oct 2026, [alphaXiv](https://www.alphaxiv.org/abs/2610.speculative-decoding-token-tax-languages) | Abstract page read 2026-11-14 | Weak-moderate (single author, preprint) | K4, speculative channel. Built-in drafters of Qwen3.5-9B and Gemma 4 E4B, 200 FLORES devtest sentences, 21 languages. English has the highest acceptance length (Qwen 3.16 vs 1.89-2.81). vLLM, one A100, batch 1: Gemma speedup 2.38x English vs 1.23x Finnish; Finnish relative latency 1.55x to 3.02x with speculation. Translation only, one run, no quantization or KV compression. |
| "Equal Tokens, Unequal Losses: Fixed KV-Cache Budgets Penalize ..." (OpenReview `57mHZnnuEe`, ACL ARR Oct 2026 per the report) | **Not verified**: OpenReview returned a browser check, its API returned nothing, exact-title search found nothing | Unknown | K4, KV channel, potentially close: per the report it recommends per-language results and budgets as shares of input length. Owner to open the link and send the abstract. |
| KV-CoRE, arXiv 2602.05929 (W7128024358) | Search snippet | Weak | Adjacent: KV low-rank compressibility varies across languages |
| Chimoto et al., EACL 2026 (already cited) | Report's summary | Strong | GPTQ swings up to 3.52 perplexity across calibration languages, AWQ under 0.35; error concentrated in first and last layers |
| Petrov et al., NeurIPS 2023 (already cited) | Read earlier | Strong | Up to 15x token premium |
| Paudel et al. 2026, arXiv 2605.30580 (already cited) | Report's summary | Moderate | Speculative speedups small or negative on low-resource languages |

Out of scope for WhoPays (dense Qwen2.5, single T4): MoE routing and expert parallelism,
disaggregated serving, grammar-constrained decoding, streaming detokenization, byte-level
architectures.

Actions taken from this report are in docs/DECISIONS.md (2026-11-14).
