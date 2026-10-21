# First prompt to paste into Claude Code

Open a terminal and run:

```bash
cd ~/Projects/whopays
claude
```

Then paste:

---

Read CLAUDE.md, docs/PRD.md, docs/BUILD_PLAN.md, docs/NOTEBOOKS.md and docs/DECISIONS.md.

Then do the M0 "Project skeleton" and "Config and results" tasks from BUILD_PLAN.md:
- pyproject, env.py, CI, build_notebooks.py
- config.py, io/results.py, io/hub_sync.py, run.py

Write tests first, and run pytest until it passes.

After that, generate `notebooks/00_env_check.ipynb`. Make it self-contained: it should NOT need the whopays package yet. It should check, on a Colab T4:
- the GPU name
- torch / transformers / CUDA versions
- whether GPTQModel installs and loads a Qwen2.5-1.5B GPTQ-Int4 checkpoint
- whether transformers QuantizedCache works with Qwen2.5-0.5B
- whether kvpress imports and runs on Qwen2.5-0.5B
- whether the Qwen2.5-0.5B and Qwen2.5-7B tokenizers produce identical IDs on 20 multilingual sentences, and the two models' vocab sizes

It should print one PASS/FAIL table at the end.

Initialise git and make one commit. Don't push.

Finish by telling me exactly how to open the notebook in Colab and what output to paste back to you.

---
