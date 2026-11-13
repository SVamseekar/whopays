# Literature corpus: measuring per-language degradation under compression

Last searched 2026-11-13 (OpenAlex via `litsearch.py`, plus web search). Start here next time and
search only for work newer than this date. Grades follow
`.claude/skills/researching-design-choices/references/literature-protocol.md`.

| Work | Year | Venue | Grade | Relevance |
| --- | --- | --- | --- | --- |
| Mielke et al., What Kind of Language Is Hard to Language-Model? (W2951412882) | 2019 | ACL | Strong | Surprisal on parallel sentences for cross-language comparison |
| Petrov et al., Language Model Tokenizers Introduce Unfairness Between Languages (W7133205974) | 2023 | NeurIPS | Strong | Up to ~15x token-length disparity for the same text |
| Apples to Apples? Comparable Crosslingual LM Evaluation (W7204468137, arXiv 2608.25089) | 2026 | EMNLP | Moderate | Sentence-level NLL over equivalent sequences; normalized metrics biased |
| Tsvetkov & Kipnis, Information Parity (W4404781974) | 2024 | Findings EMNLP | Moderate | Compression efficiency vs reference language predicts task scores |
| Marchisio et al., How Does Quantization Affect Multilingual LLMs? (W4404781102) | 2024 | Findings EMNLP | Strong | Automatic metrics underestimate; non-Latin worst |
| Jin et al., Comprehensive Evaluation of Quantization Strategies (W4402683730) | 2024 | Findings ACL | Moderate | Perplexity-benchmark |r| 0.7895 for Qwen-Chat GPTQ/SpQR |
| Dutta et al., Accuracy is Not All You Need (W4415795507) | 2024 | NeurIPS | Strong | Flips and KL alongside accuracy |
| Jaiswal et al., Compressing LLMs: The Truth is Rarely Pure (W4387322997) | 2024 | ICLR | Strong | Perplexity misses knowledge losses |
| Language-Conditional Dequantization (W7202380372, arXiv 2608.11786) | 2026 | preprint | Weak-moderate | Perplexity-accuracy disconnect on Qwen2.5-3B INT3; adjacent (K4) |
| Rababah et al., Illusion of Equivalency (W7167932887) | 2026 | preprint | Weak | Behaviour shifts with preserved accuracy/perplexity |
| Zheng et al., LLMs Are Not Robust Multiple Choice Selectors (W4386555476) | 2023 | preprint | Moderate | Letter prior; PriDe |
| Sagawa et al., Group DRO (W2991497298) | 2019 | ICLR 2020 | Strong | Worst-group metric conventions |
| Bertsimas et al., The Price of Fairness (W2013518095) | 2011 | Operations Research | Strong | Price of fairness for max-min |
| Hake's g vs Cohen's d (W2963662961) | 2016 | preprint | Weak | Ceiling-normalized gains over-correct (disputed) |
| Ogueji et al., Intriguing Properties of Compression on Multilingual Models (W4385573538) | 2022 | EMNLP | Moderate | Regime-dependent effects on low-resource languages |
| Ahia et al., The Low-Resource Double Bind (W3203385474) | 2021 | Findings EMNLP | Moderate | Harm concentrates on rare inputs |
| Chimoto et al., Calibrating Beyond English (W7140085483) | 2026 | EACL | Strong | Calibration language matters for multilingual GPTQ |
| FairQuant (W7131843171); Colla-Q (W7213505222) | 2026 | preprint; EMNLP | — | Adjacent prior art for worst-group mixed precision |

Screened and dropped: Open LLM Leaderboard normalization (docs, kept in the decision note only);
physics-education normalized-gain debate beyond 1612.09180 (analogy only); mixed-precision
proxy papers RAMP 2609.28262 and TASA 2607.00908 (vision or task-specific calibration, not
cross-lingual units).

Added 2026-11-13 (round 2):

| Work | Year | Venue | Grade | Relevance |
| --- | --- | --- | --- | --- |
| Zhou et al., Distributionally Robust Multilingual MT (W3197336992) | 2021 | EMNLP | Strong | Baseline-subtracted worst-case over languages; right baseline left open |
| Oren et al., Distributionally Robust Language Modeling (W2970789589) | 2019 | EMNLP | Strong | Per-group baseline before the max |
| Xie et al., DoReMi (W7133216852) | 2023 | NeurIPS | Moderate | Group DRO on excess loss vs reference model |
| Vowel Signs Are Not Letters (arXiv 2608.26449) | 2026 | preprint | Weak | Per-token loss not comparable across segmentations |
