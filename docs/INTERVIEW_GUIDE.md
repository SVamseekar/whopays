# WhoPays + ParityQuant — technical interview guide

Snapshot: 31 October 2026. Status uses the latest output shared in this chat, not live
Colab monitoring. Read alongside the [technical map](figures/whopays-technical-map-2026-10-31.png).
The picture uses built-in image generation with a saved [generation brief](figures/whopays-technical-map.prompt.md)
and a targeted [arrow correction](figures/whopays-technical-map.edit.md).

## The answer to “What are you building?”

“I am building WhoPays, a reproducible audit of how weight quantization, KV-cache
compression and speculative decoding affect quality and computational cost across
languages. I evaluate their composition on parallel multilingual tasks and attribute
disparities to serving-stack factors. ParityQuant is the conditional method extension:
allocate precision to minimize the worst language's predicted degradation at a fixed
parameter-weighted bit budget. The baseline evaluation is running; improved fairness
or speed has not yet been established.”

## 1. Why this project?

**What is the question?** A serving optimization can improve an average metric while
its costs fall unevenly across languages. The project asks how much quality or speed
changes for each language, whether optimizations interact, and whether allocation can
protect the worst degradation at equal bits.

**Why this direction?** The original decisions rejected a simulator-led serving-control
project because the field was crowded and the evidence would be weak. A generic
compression/adaptation pipeline lacked a distinct research question. This project connects
econometrics, trustworthy ML and inference engineering to experiments feasible on modest
GPUs. These are design motivations, not findings.

**What is new?** Novelty is provisional and narrow: a composed multilingual quality/speed
audit, attribution across serving factors, and the specific worst-language degradation
objective for dense multilingual LLM allocation. Quantization language disparities,
multilingual calibration, language-dependent speculative efficiency, general mixed
precision, fairness-aware allocation and minimax balancing all have prior art.

**Does “parity” mean equal accuracy?** No. The objective concerns degradation relative
to each language's own FP16 reference. It does not equalize absolute accuracies or fix
weak pretraining coverage. Baseline weakness and floor effects need explicit analysis.

| Question | Preregistered hypothesis / falsification |
| --- | --- |
| RQ1: who bears individual optimization costs? | H1 relates gaps to token premium and lower FP16 accuracy; falsified when both relevant Spearman correlation intervals include zero |
| RQ2: do serving factors interact? | H2 requires a nonzero interaction after BH correction; all indistinguishable interactions falsify it |
| RQ3: what is covariate composition versus residual? | H3 predicts an endowment share below half for at least three non-Latin languages; shares are undefined for zero total gaps and the split is descriptive |
| RQ4: does minimax allocation protect degradation? | H4 tests equal-bit worst-language benefit at the planned budgets with mean-accuracy cost at most one percentage point; null intervals at two or more budgets falsify it |

## 2. What has happened so far?

| Evidence level | Current state | What this establishes |
| --- | --- | --- |
| CPU software checks | Latest local suite: 210 passed, 5 download-dependent tests skipped; Ruff and generated-notebook checks passed | Tested software behavior; no research quality/speed conclusion |
| Environment work | Colab optional torchvision/torchaudio import failures identified; repair instructions and setup checks added | The text-model loader can progress past those reported failures |
| Latest shared T4 log | Model weights loaded; English, German, French and Spanish Belebele jobs each report 900 completed items; Russian dataset loaded next | Partial baseline execution only; log does not report accuracies or full completion |
| K1 validation | Pending full FP16 and lm-eval comparison plus throughput review | Harness agreement has not yet been established |
| Optimization research | CPU implementations exist; full GPU acceptance, audit and method evaluation remain pending | No demonstrated language gap, speedup or ParityQuant gain yet |

Do not turn job elapsed times or completed-item counts into accuracy claims. The log's
`done` means that particular job finished, not that the research gate passed. The
`No files matched allow_patterns` warning alone means that requested remote checkpoint
was absent; it is expected for a new run. A traceback identifies an actual failure.

## 3. Explain the architecture

The experiment path is:

`YAML → RunConfig → canonical hash → model loader → W/K/S serving policies → task → JSONL → Parquet → Hugging Face`

The analysis and feedback path is:

`released Parquet → paired statistics / figures / cards`

`GPU sensitivity profile → CPU allocation solver → per-block bit map → new YAML evaluation → GPU`

**Where does code live?** Research logic is in `src/whopays/`; notebooks install a pinned
package, configure jobs, call `whopays.run(...)` and display summaries. Notebook source is
percent-format Python, regenerated by `scripts/build_notebooks.py`.

**What runs where?** The M1 Mac writes/tests code, runs statistics and solves allocation.
It does not produce headline experiment measurements. Colab T4 handles short GPU runs;
Kaggle handles measured-cost slices of the main grid. The user starts those GPU notebooks.
The setup uses FP16 and PyTorch SDPA on T4, respecting its memory limits. The 7B check is
a later confirmation requiring an appropriate multi-GPU or prequantized setup.

**What models?** Qwen2.5 Instruct: 1.5B and 3B primary targets; 0.5B draft and local smoke
model; 7B confirmation later. The tokenizer alignment was probed previously. Matching
token IDs is required for drafting; equal embedding-table sizes are not the requirement.

## 4. Why these tasks and languages?

The planned language set is English, German, French, Spanish, Russian, Chinese, Japanese,
Thai, Bengali, Swahili and Telugu (`en de fr es ru zh ja th bn sw te`). English is the
comparison reference. The set covers different scripts and tokenization burdens, but
is not a random sample of the world's languages.

| Task | Technical protocol | Role |
| --- | --- | --- |
| Belebele | Parallel reading questions; zero-shot raw prompt; next-token log-probabilities for A/B/C/D; 900 questions/language | Primary paired quality measurement |
| MGSM | Greedy multilingual math generation; language-specific reasoning cue and stop marker; numeric extraction; 250 problems/language | Generative quality and decoding cost |
| FLORES dev | Seeded English or multilingual sampling | Quantizer calibration, not scored evaluation |
| FLORES devtest | Sentence NLL with token/character/byte counts; per-block sensitivity | Profiling and NLL comparison, with allocation-selection caveat |

**Why parallel items?** Compare the same underlying question across languages and the
same item across configurations. Match IDs before analysis; different per-language file
orders must not produce different limited subsets. Paired comparisons reduce question
difficulty confounding.

**Why raw prompts?** Matching lm-evaluation-harness's templates makes K1 a meaningful
comparison. MGSM explicitly overrides Qwen's repetition penalty to obtain pure greedy
decoding. Native decimal digits are parsed so a correct Bengali/Telugu/Thai answer is
not rejected just because of its script. An ASCII scoring column supports the lm-eval
comparison; extraction failure rates and manual audits remain necessary.
The protocol calls for manually checking extraction on 50 items per language.

**Why not compare raw perplexity across languages?** Tokens represent different amounts
of text. Use within-language changes as primary comparisons, log text/token counts, and
consider bits per byte as a complementary normalization.

## 5. Explain each serving factor

### W — weight precision

The nominal levels are FP16, W8 RTN, W4 GPTQ and W3 GPTQ. RTN rounds within grouped
asymmetric quantizers. GPTQ uses calibration activations and a Hessian proxy to compensate
for sequential rounding error; the specified setup uses group 128, damping 0.01 and no
activation ordering. English-only versus multilingual calibration and three subsample
seeds are crossed with GPTQ levels.

**Why fake quantization?** Quantize and dequantize into FP16 to test low-bit weight
distortion without depending on fragile T4 mixed-bit kernels. It does not deliver low-bit
kernel acceleration or physically packed model-memory savings. Report theoretical weight
bits separately from measured runtime memory. Validate uniform W4 perplexity against
the GPTQModel reference before relying on the approximation.

### K — KV-cache policy

The policies are full cache, HQQ 4-bit cache and a fixed 256-token sink-plus-recent
eviction budget. The latter keeps initial sink tokens and recent context, retaining
original rotary-position coordinates for later queries.

**Why two passes for Belebele?** Prefill the passage context, apply compression, then
score the question/options against the resulting cache. A single prefill can use
uncompressed K/V for its own attention, making a purported cache-compression condition
a no-op. The full-cache reference uses the same two-pass structure.

**Why could fixed budgets create a language disparity?** A high token premium uses more
cache slots for the same parallel content. A fixed token budget can therefore remove a
larger share of the passage. This is a mechanism hypothesis, not a finding already shown.
The proposed ratio-matched control would separate eviction dose from sensitivity.

**What was the rollback issue?** Quantized history and the FP16 residual buffer are
different states. Cropping residual tensors without updating cumulative length breaks
subsequent attention masks. The implementation validates residual-only rollback and
repairs the counter; rollback through already-compressed rejected tokens is refused.

### S — speculative decoding

A 0.5B draft proposes up to four tokens. The target verifies the prefix; accept matching
tokens, replace the first mismatch with the target's greedy token, or emit a bonus token
when the whole proposal is accepted. Stop strings, EOS and output limits must be handled
inside accepted blocks. Counters include wasted proposals and actual model forwards.

**Why must output quality stay unchanged?** The reference is the same target with the
same weight/cache policy and pure greedy protocol. Token disagreement is a correctness
failure, not a quality effect to report as speculative harm. FP16 block/sequential
numerics still need the multilingual T4 exactness check.
The FR-5 acceptance check specifies token identity on 100 prompts in all 11 languages.

**What is implemented versus accelerated?** Full cache has block verification. Compressed
caches currently verify the k-proposal prefix sequentially to preserve per-token eviction
and quantization state, rolling back only the full draft cache. They explicitly log
`target_acceleration_enabled=False`. This is a correctness control, not accelerated
compressed-stack speed evidence. The default batched FP16 MGSM path is preserved for K1;
consistent baseline forward instrumentation is still needed for complete speed comparisons.

## 6. How do you measure effects?

For language `l` and configuration `c`:

```text
quality change:       Δacc(l,c) = acc(l,c) − acc(l,FP16)
worst degradation:    D(c) = max_l [−Δacc(l,c)]
flip rate:           fraction correct at FP16 and wrong under c
acceptance:          α = accepted draft tokens / proposed draft tokens
tokens per forward:  committed output tokens / actual target forwards
token premium:       TP(l,i) = tokens(l,i) / tokens(English,i)
```

Keep quality and speed as separate channels. Token-level counters are useful across
hardware, while latency comparisons require the same pinned GPU/protocol. Forward counts
are a cost proxy; they do not price draft work or different prefill/decode call costs
equally. Include draft calls, wall time and policy overhead in an honest speed assessment.

The nominal quality design is four weight levels × three cache levels = 12 cells per
model. Speed adds speculative on/off = 24 nominal cells, using a fixed MGSM subset.
Calibration settings and seeds expand the actual run count; they are not extra independent
questions. The full primary-model speed grid and any second-model subset need a measured
compute plan before launch.

## 7. Explain the statistical design

**How are intervals built?** Paired bootstrap over aligned items, preserving dependence
across languages/configurations; 2,000 resamples and 95% intervals are specified. Report
calibration seed spread separately. Recompute the worst-language statistic inside each
resample rather than treating the observed worst language as fixed.

**What does Shapley attribute?** W, K and S are serving-stack factors. All eight coalitions
of a binary sub-grid are required to allocate the measured disparity change. These are
not the transformer blocks profiled by ParityQuant. Pairwise interaction is
`v(i,j) − v(i) − v(j) + v(empty)`; it tests departure from additivity.

**What does Oaxaca do?** A descriptive pooled-coefficient decomposition of a language's
gap relative to English into differences in measured covariates and a residual component.
Token count, item length and FP16 gold-answer log-probability are candidate covariates.
It does not causally identify discrimination or independently establish a serving mechanism.

**Why remove script from a pairwise fit?** Differing group-constant scripts encode the
language-group indicator. Pooled least squares then fits both group means, mechanically
assigning the total gap to endowment and forcing the residual to zero. The protocol was
amended before results; the implementation rejects any covariate span encoding group.

**How is H2 tested?** One contrast per language:
`logit11 − logit10 − logit01 + logit00` for weight × KV. The fit is a binomial GEE
with Mancl–DeRouen standard errors clustered by item, t(items − 1) p-values, and
Benjamini–Hochberg at q=0.05 on that family only. These are population-averaged
logit interactions. The variational random-intercept fit is descriptive and is not
fed to BH. A language cell with no outcome variation stops the analysis.

## 8. Explain ParityQuant on a whiteboard

Profile one transformer block at one precision while other blocks stay at reference
precision. Let `s(l,j,b)` be the language's NLL increase for block `j` at bit width `b`;
`p(j)` counts quantized parameters; binary `x(j,b)` selects one bit width per block.

```text
minimize t
subject to:
    Σ_j Σ_b s(l,j,b) x(j,b) ≤ t           for every language l
    Σ_b x(j,b) = 1                       for every block j
    Σ_j Σ_b p(j) b x(j,b) = B Σ_j p(j)    exact matched budget in the implementation
    x(j,b) ∈ {0,1},  b ∈ {2,3,4,8}
```

The PRD writes a budget upper bound; the implemented comparator solver imposes exact
matching and rejects infeasible targets. PuLP/CBC runs on CPU. Average bits cover the
weights actually quantized; excluded FP16 embeddings/heads and quantizer metadata must
be included separately when discussing whole-model storage.

**What baselines?** Mean-sensitivity allocation, uniform precision where the budget
is an allowed width, multilingual-calibration baselines and a softmax objective.
At a fractional budget such as 3.5, the sensitivity-free comparator is
`matched_uniform`: exact total bits, minimum parameter-weighted squared deviation
from the target average, higher widths on earlier layers when that deviation ties.
Planned budgets are 3.0, 3.5 and 4.0 average bits.

**What is the price of parity?** Mean-accuracy loss of the minimax solution minus that
of the mean-objective allocation at matched bits. This cost and the worst-language benefit
are measured outcomes, not quantities already established by the solver's surrogate.

**What is the main assumption?** Independently profiled block effects add. Actual block
interactions can invalidate the preferred allocation. Validate the assembled model and
consider closed-loop reprofiling as an ablation. Profiling FLORES devtest also makes those
sentences allocation-development data; independent downstream tasks and a prospective
profile/validation split are needed for honest generalization claims.

## 9. How is a run reproducible and resumable?

- Canonical YAML config determines a SHA-256-derived 12-hex run ID; irrelevant fields are
  excluded deliberately, and schema changes require versioning.
- The loaded-weight identity enables reuse across compatible jobs; task/language/cache
  configurations still have separate result folders.
- Item JSONL flushes every 50 items and compacts to Parquet. Resume skips durable IDs and
  preserves original batch membership to avoid FP16 padding/composition drift.
- Rows record configuration, code identity, torch/transformers/CUDA and GPU. The resume
  guard rejects incompatible code/hardware/library environments rather than mixing them.
- Hugging Face mirrors need a token and successful uploads. The lm-eval reference uses
  consistent SQLite/WAL snapshots during execution; failed uploads leave local data intact.
- Offline wheels carry the lock, use content hashes and force reinstall the package itself.
  Changed builds require fresh kernels; optional media imports are checked in subprocesses.
- Sensitivity-profile resume now refuses a different code version, GPU, torch,
  transformers build, CUDA build or model identity, matching the item-run pin.

## 10. What happens next, and when would you stop?

| Phase | Work and deliverable | Gate / evidence |
| --- | --- | --- |
| M0 — current | Finish all language/task FP16 baselines; lm-eval comparison; extraction health; throughput | K1: per task, mean absolute language accuracy gap ≤1 percentage point on full coverage; inspect flagged languages; replan if projected runtime exceeds 2× the planning estimate |
| M1 | Uniform quantization replication, W4 reference-checkpoint comparison and per-language block sensitivity | W4 perplexity within 2% of the reference; K2 sensitivity diversity |
| M2 | Preregistered quality/speed grid on T4; measured-cost session slices; all item results | K1 satisfied; registration committed; compressed acceleration and exactness validated before accelerated claims |
| M3 | Paired statistics, interaction inference, Shapley/Oaxaca figures; audit manuscript | Resolve inference gaps; regenerate from Parquet; publish supported findings, including nulls |
| M4 | Allocate at matched budgets; evaluate actual models; price-of-parity curve | K3: test worst-language benefit and mean-accuracy cost; do not substitute surrogate loss for measured task outcomes |
| M5 | 7B confirmation; final dataset, dashboard, disparity cards and papers | Reproducibility, license review and measured regeneration checks |

**K1:** Stop downstream experiments if reference agreement fails. It is still pending.
**K2:** If minimum pairwise sensitivity-vector Spearman correlation exceeds 0.9, drop
ParityQuant and extend the audit; the method needs language-specific allocation structure.
**K3:** If the improvement CI includes zero at two or more budgets, publish the method as
a negative result. The planned success criterion also constrains mean accuracy cost.
**K4:** Read close prior work within 48 hours, narrow claims to the uncovered contribution
and cite it. A workshop or paper acceptance is a goal, not a guaranteed deliverable.

## 11. Strong follow-up questions and honest answers

**“Why not train a better draft or retrain the quantized model?”** No training/fine-tuning
in v1. The goal is controlled diagnosis of inference choices within the stated compute
constraints; training would change the intervention and expand the budget.

**“How do you know the language causes the effect?”** The paired audit measures
configuration-dependent differences. Tokenization, translation, baseline ability and
benchmark design remain entangled; factorial interventions identify serving-factor effects
under the protocol, not every causal explanation for language inequality.

**“Could low baseline accuracy hide degradation?”** Yes. Include uncertainty-conditioned
flip analysis, baseline performance and task limitations. A null delta on a weak model
is not proof of robust capability.

**“Can you claim production performance?”** Scope claims to tested models, hardware,
batching and implementations. Fake quantization is a quality proxy; compressed speculative
fallback is a correctness reference. Production serving and new CUDA kernels are non-goals.

**“What would make the project valuable if ParityQuant fails?”** A valid, reproducible
composed-stack language audit, a released per-item dataset, clear attribution and documented
limits. Nulls can falsify hypotheses and guide deployment choices.

**“What can you confidently claim today?”** Implemented and CPU-tested infrastructure,
specific environment/debugging lessons, and partial GPU baseline execution shown in logs.
Do not claim K1 passed, measured optimization speedups, statistically significant language
harm, improved fairness, published papers or completed release artifacts.

## 12. References and source-of-truth files

- Local specification: `docs/PRD.md`, `docs/BUILD_PLAN.md`, `docs/DECISIONS.md`,
  `PREREGISTRATION.md`, `docs/REVIEW_2026-10-31.md` and current `src/whopays/`.
- [Marchisio et al.: multilingual quantization effects](https://arxiv.org/abs/2407.03211).
- [Chimoto et al.: multilingual calibration](https://aclanthology.org/2026.eacl-long.223/).
- [Paudel et al.: multilingual speculative decoding](https://arxiv.org/abs/2605.30580).
- [FairQuant: fairness-aware mixed precision in medical imaging](https://arxiv.org/abs/2602.23192).
- [Colla-Q: minimax precision balancing across MoE experts](https://arxiv.org/abs/2609.18131).
- [CoopQ/SPQE: interaction-aware mixed precision](https://arxiv.org/abs/2509.15455).
- [MixQuant: sensitivity depends on other blocks' precision](https://arxiv.org/abs/2607.23047).
- [Dynamic KV compression with multilingual analyses](https://openreview.net/pdf?id=0noVRtnzJt):
  full-text review remains necessary before asserting a narrow priority distinction.

The illustration is an explanatory snapshot. Text and equations in this guide are the
readable reference; neither artifact is an experimental results figure.
