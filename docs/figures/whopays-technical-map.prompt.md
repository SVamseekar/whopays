Use case: infographic-diagram / scientific-educational
Asset type: a detailed technical project map and interview preparation poster for WhoPays + ParityQuant.
Primary request: explain what this actual research project does, what is happening now, its implementation architecture, its planned experiments and gates, and defensible interview answers. Produce ONE polished, large, vertically scrollable technical infographic. Use the highest available native resolution; aim for a tall 4K portrait research poster. Dense but extremely readable, crisp professional typography, generous panel padding, accurate mathematical labels, no invented data. Use diagrams, paths and a compact factor matrix rather than long prose wherever possible.

Style: clean engineering whiteboard / research-poster design on an off-white background with dark navy text and hairline connectors. Functional color coding: teal = software verified, amber = running or evidence pending, violet = planned experiments, red = stop/pivot gates. Flat schematic components, clear arrows, no photorealism, no glossy 3D, no decorative clutter. Make headings much larger than body text. Fit every panel without cropping. Preserve case and spelling in the copy. No fabricated charts, accuracy bars, numerical benefits, benchmark results, logos, flags or extra claims.

Composition: clear top-to-bottom numbered reading order. Header and legend. A prominent current-status band. A full-width execution architecture with a feedback loop. Two side-by-side serving/data panels. A metrics/analysis band. A ParityQuant allocation band. A six-stage roadmap. Interview question/answer cards at the bottom. Keep semantic color contrast accessible. Factor attribution W/K/S must be visually distinct from transformer-block sensitivity j.

The following is the exact concise content to render. Break it into readable labels/cards, using flow arrows and small icons in place of connective text. All numerical thresholds below are PLANNED TESTS, not measured benefits.

TITLE
"WhoPays + ParityQuant"
"Who pays for efficient LLM inference?"
"Technical architecture • current progress • research roadmap • interview brief"

01 — WHY THIS PROJECT
"Measure quality and speed costs across languages; then test worst-language bit allocation."
"Empirical paired audit on modest GPUs. Earlier simulator-only serving ideas were set aside."
"Candidate novelty: composed-stack language disparities + language-aware minimax allocation. Prior art exists; novelty remains provisional."

02 — CURRENT STATUS: M0 BASELINES RUNNING
"Latest shared Colab T4 log: Qwen2.5-1.5B loaded."
"Belebele completed: English, German, French, Spanish — 900 items each. Russian started next."
"Software verification: 210 CPU tests passed; 5 skipped."
"K1 still pending. Job completion is not an accuracy result."
Small status legend: "Implemented ≠ GPU validated ≠ research finding"

03 — EXECUTION AND FEEDBACK LOOP
Main arrow chain: "YAML config" → "Canonical SHA-256 run ID" → "Model loader" → "Serving stack: W × K × S" → "Parallel task items" → "JSONL → Parquet" → "Hugging Face results"
Under JSONL: "Flush every 50 items • resume by item ID"
Under storage: "Config + code hash + library versions + GPU"
Lower loop: "GPU block-sensitivity profiles" → "CPU minimax solver" → "Per-block bit map" → arrow back to "YAML config"
Machine badges: "M1 Mac: code, CPU tests, statistics, solver" / "Colab T4: short GPU runs" / "Kaggle T4: main grid"
Constraint strip: "FP16 + PyTorch SDPA on T4 • no training • no new CUDA kernels"

04 — DATA AND MODELS
"11 languages: en de fr es ru zh ja th bn sw te"
"Belebele: paired reading accuracy"
"MGSM: greedy math + decoding cost"
"FLORES dev: calibration; devtest: NLL profiling"
"Targets: Qwen2.5 1.5B / 3B; draft: 0.5B; later confirmation: 7B"

05 — THREE SERVING FACTORS
"W / WEIGHTS: FP16 • W8 RTN • W4 GPTQ • W3 GPTQ"
"GPTQ: English vs multilingual calibration; 3 subsample seeds"
"Fake quantization measures distortion. Storage savings are theoretical bit counts."
"K / CACHE: full • HQQ 4-bit • fixed 256-token sink + recent"
"Prefill context → compress cache → score query; preserve original RoPE positions."
"S / SPECULATION: 0.5B drafts k=4 → target verifies → accept prefix / correct mismatch"
"Greedy output must match exactly. Compressed-cache verifier is currently sequential: no target acceleration."

06 — WHAT WE MEASURE AND ATTRIBUTE
"Quality change: Δacc = accuracy(config) − accuracy(FP16)"
"Worst degradation: max over languages of −Δacc"
"Speed: accepted/proposed draft tokens; output tokens/actual target forward; same-GPU latency"
"Token premium: tokens(language) / tokens(English), on parallel items"
"Paired bootstrap CIs • BH correction • Shapley over W/K/S • descriptive Oaxaca"
"Open: valid H2 interaction inference. Group-identifying script covariates are excluded."

07 — PARITYQUANT: CONDITIONAL METHOD EXTENSION
"Profile NLL increase by language × transformer block × bits."
"Choose bits in {2,3,4,8} to minimize the maximum predicted language loss."
"One bit width per block; parameter-weighted bit budget matched exactly when feasible."
"Planned budgets: 3.0 / 3.5 / 4.0 average bits."
"Compare minimax vs mean allocation; measure the price of parity."
"Risk: block effects may interact. Validate actual models on independent outcomes."

08 — FUTURE ROADMAP / EVIDENCE GATES
"M0 — Finish FP16 + lm-eval cross-check"
"K1: full coverage; mean absolute gap ≤1 percentage point per task"
"M1 — Replicate quantization + profile block sensitivity"
"K2: if minimum pairwise Spearman ρ >0.9, drop ParityQuant"
"M2 — Audit 12 nominal quality cells; 24 nominal speed cells; seeds/calibration add runs"
"M3 — Attribution + audit manuscript"
"M4 — Equal-budget ParityQuant evaluation"
"K3: null at ≥2 budgets → publish negative result"
"M5 — 7B confirmation + data, figures, cards, dashboard, papers"
"K4 throughout: check close prior art and narrow claims"

09 — INTERVIEW ANSWERS TO REMEMBER
"Why FP16 first? Validate the harness before interventions."
"Why parallel items? Compare the same questions across languages and configurations."
"Why separate quality and speed? Exact greedy speculation changes cost, not answers."
"What does parity mean? Worst degradation, not equal absolute accuracy."
"Biggest open risks? GPU validation, inference gaps, additive sensitivity, compute budget."
"What can you claim today? Tested infrastructure and partial baseline execution."

FOOTER
"Snapshot: 31 Oct 2026 • latest shared log, not live monitoring • No final research results yet"
"Full formulas, caveats and interview follow-ups: docs/INTERVIEW_GUIDE.md"

Critical invariants: Do not mark K1 passed. Do not depict ParityQuant as proven. Do not depict compressed-cache sequential verification as a speedup. Do not show fake quantization as measured packed-weight memory savings. No invented accuracy, fairness or latency improvements. Do not label future experiments complete merely because CPU code exists. No real experimental plots.
