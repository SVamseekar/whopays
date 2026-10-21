# WhoPays + ParityQuant

**Who pays for fast inference?** This project measures how LLM inference optimizations (weight quantization, KV-cache compression and speculative decoding) change quality and speed differently across languages. It attributes each gap to the layer that causes it. It also introduces **ParityQuant**, a mixed-precision quantizer that minimizes the worst language's degradation at a fixed memory budget.

Status: harness under construction (M0). No results yet.

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -c requirements.lock -e ".[dev,stats]"
pytest -q && ruff check src tests scripts
WHOPAYS_SLOW=1 pytest -q -m slow           # also tests that download data
python -m whopays.cli smoke --device mps   # 0.5B, 20 Belebele items x en/te
python scripts/build_notebooks.py          # regenerate notebooks/*.ipynb
python -m whopays.cli hash configs/x.yaml  # print run IDs for a job list
```

- Spec: `docs/PRD.md`
- Build order: `docs/BUILD_PLAN.md`
- Decisions: `docs/DECISIONS.md`
- Running on Colab / Kaggle: `docs/NOTEBOOKS.md`
- Instructions for Claude Code: `CLAUDE.md`

Licence: Apache-2.0 (code).
