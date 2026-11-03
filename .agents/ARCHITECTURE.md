# Architecture

WhoPays measures how weight quantization, KV-cache compression and speculative
decoding change quality and speed across 11 languages. ParityQuant then assigns
per-layer bit widths to minimise the worst language's profiled loss at a fixed
average bit budget. Notebooks only call the package. Numbers in the docs must
come from logged runs. K1, the FP16 T4 baseline, is still the first research gate.

## Stack

Python 3.11 package `whopays` (`src/whopays`). Pydantic configs, PyTorch plus
Transformers for models, statsmodels/scipy for analysis, PuLP/CBC for allocation.
CPU and MPS run tests and the 0.5B smoke path. Reported measurements come from
Colab or Kaggle T4 notebooks, FP16 only, PyTorch SDPA, fake quantization by default.

## Flow

1. A YAML `RunConfig` (`config.py`) hashes to a run id. Irrelevant fields are
   dropped so equivalent cells share one results folder.
2. `run.py` skips finished runs, refuses mixed code/GPU/library resumes, loads
   one model bundle per weight setup (`models/loader.py`) and dispatches a task.
3. Tasks write item JSONL through `io/results.py` (flush every 50, resume by id),
   then Parquet. `io/hub_sync.py` mirrors a private Hugging Face dataset when a
   token exists.
4. The M1 side reads Parquet. It does not produce table numbers.

## Modules

| Path | Role |
| --- | --- |
| `env.py` | Colab/Kaggle/local, device, secrets, versions, text-runtime repair |
| `languages.py` | The 11 language codes |
| `tasks/` | Belebele letter logits, MGSM generation, FLORES token NLL |
| `quant/` | Group-wise RTN, minimal GPTQ, fake-quant, calibration samples |
| `kv/policies.py` | Full cache, optional HQQ 4-bit, sink+recent eviction |
| `decoding/speculative.py` | Greedy draft-verify; compressed caches verify sequentially |
| `parityquant/profile.py` | One-block NLL sensitivity, manifest-checked resume |
| `parityquant/allocate.py` | Minimax, average, softmax, uniform, matched-uniform MILP |
| `stats/` | Paired bootstrap, BH, McNemar, Shapley, Oaxaca, inequality, logit fits |
| `report/`, `space/` | Cards, figures, Gradio reader over released Parquet |
| `grids.py` | Job-list generation for the audit grid |
| `notebooks/src/` | Sources for generated notebooks 00 and 01 |

## Statistics that are specified

- Paired percentile bootstrap over aligned items, 2,000 resamples.
- H2: language-specific weight × KV population-averaged logit interactions from
  an item-clustered binomial GEE (Mancl–DeRouen SE, t with items−1 df, BH on
  that family). The variational GLMM is descriptive only.
- Oaxaca rejects any covariate span that contains the language-group indicator.
- `matched_uniform` is the sensitivity-free baseline at fractional bit budgets.

## Open gates

K1 FP16 agreement and throughput, FR-3 GPTQ reference, K2 sensitivity diversity,
multilingual speculative exactness on T4, accelerated compressed-cache
verification, H4/K3 allocation benefit, and the release regeneration check.
Passing unit tests do not close those gates.
