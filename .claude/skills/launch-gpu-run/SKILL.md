---
name: gpu-handoff
description: Use in WhoPays whenever the user must run a notebook on a GPU (Kaggle or Google Colab), for example "give me the steps", "run notebook 02", "next GPU run", or after a fix that needs GPU verification. Produces exact platform-specific steps, the expected setup-cell output, a time estimate, check-in points, known errors and what to send back. Uses the platform this session is already on; otherwise picks by job length.
---

# GPU handoff (Kaggle or Colab)

Claude cannot run notebooks on a GPU. The user runs them, so the steps must be exact.

## 1. Pick the platform

1. **Use the platform this session is already on.** If the user's last run, log or setup output came from Kaggle (`platform: kaggle`, a `.log` download, "Save Version"), use Kaggle. If it came from Colab (`platform: colab`, "Runtime → Run all"), use Colab. Say which, in one line.
2. **No earlier run in this session:** use Kaggle if the estimate is over about 2.5 h, or if it must run unattended. Use Colab for short debugging runs under about 2.5 h. Free Colab sessions die when the tab is idle or after about 3 h.
3. If the estimate is over 11 h, split the job (slices, or a re-run that resumes). Kaggle's limit is 12 h per session and about 30 GPU-hours a week.

## 2. Before giving any steps

- Check that the work the notebook needs is committed and pushed to `main`; the setup cell installs `WHOPAYS_REF = "main"`. Note the short SHA (`git log -1 --format=%h`), which the user should see as `code_version`.
- Notebooks must be regenerated (`python scripts/build_notebooks.py`) and committed.
- **Estimate the time from logged evidence**, never from guesses:
  - Find finished runs on the HF dataset: `hf_fs find hf://datasets/Heavy-Dervish1205/whopays-results --name DONE`, compared against the notebook's run IDs. Finished runs take about 5 s each.
  - Use the per-item rates from the logged throughput tables in `docs/DECISIONS.md`.
  - Add setup time (about 4 min) and any unmeasured steps (GPTQ quantization, lm-eval, reference installs), and label those as unmeasured.
  - Give a range plus the single riskiest step.
- Say where the run is likely to fail first, and that a re-run resumes.

## 3. Steps: Kaggle

Give these numbered, with the notebook name filled in:

1. kaggle.com → **Create → New Notebook**.
2. **File → Import Notebook**, paste `https://github.com/SVamseekar/whopays/blob/main/notebooks/<NAME>.ipynb`. If the dialog only takes files, download the raw `.ipynb` from GitHub and drag it in. An imported notebook never updates itself: after any code fix, import again.
3. Right panel → **Session options**: Accelerator **GPU T4 x2**, Internet **On**. Phone verification is required once per account.
4. **Add-ons → Secrets** → tick `HF_TOKEN` **for this notebook**. Secrets are attached per notebook, so every new notebook needs the tick.
5. Run **only the first code cell** and check its output against section 5.
6. **Save Version → Save & Run All (Commit) → Save**. The tab can be closed.
7. Progress: **Your Work → the notebook → Versions → the running version → Log**.
8. If the run dies or reaches 12 h, do Save & Run All again. Finished runs skip.
9. When it finishes: download the log (it lands in `~/Downloads` as `notebookXXXX.log`; a second `download*.txt` is the same file) or **⋮ → Download** the notebook. Tell Claude the paths.

## 4. Steps: Google Colab

1. Open `https://colab.research.google.com/github/SVamseekar/whopays/blob/main/notebooks/<NAME>.ipynb`. Opening from GitHub always gives the current version.
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. Key icon (**Secrets**): `HF_TOKEN` present with **Notebook access** on. This is once per Google account.
4. **Runtime → Run all**; click **Grant access** if asked.
5. Check the first cell's output against section 5.
6. Keep the tab open and the laptop awake. Long steps (lm-eval, GPTQ) print nothing for a long time; that is normal.
7. If the session dies: reopen the notebook and **Runtime → Run all**. Finished runs skip.
8. When it finishes: **File → Download → Download .ipynb** (lands in `~/Downloads`). Tell Claude the path.

## 5. Expected setup output

- `platform: kaggle | GPU: Tesla T4, Tesla T4` (Kaggle) or `platform: colab | GPU: Tesla T4` (Colab).
- **No** `secret HF_TOKEN not available` line. A `GH_TOKEN` line is fine; the repo is public.
- A version table ending `code_version: <SHA from step 2>` or a later commit.
- Pip "dependency resolver" warnings and "Removing optional packages ... torchvision, torchaudio" are normal.

## 6. Known errors and their fixes

| Output | Meaning | Fix |
|---|---|---|
| `secret HF_TOKEN not available (BackendError)` (Kaggle) | Secret not attached to this notebook | Add-ons → Secrets → tick `HF_TOKEN`, rerun cell 1 |
| `secret HF_TOKEN not available (TimeoutException)` on Kaggle | Stale import from before commit `200cea5`, which misdetected Kaggle as Colab | Import the notebook again |
| `No CUDA GPU attached` | No accelerator selected | Kaggle: GPU T4 x2. Colab: Runtime → T4 GPU |
| `Restart the runtime before loading a different WhoPays build` | Code changed inside a live kernel | Restart the session, run all |
| `No files matched allow_patterns=[...]` | That run has no remote checkpoint yet | Normal |
| `GateError` / K1 not passed | Gate could not rebuild from HF (token or network) | Check `HF_TOKEN`, Internet on |
| `OutOfMemoryError` | Out of GPU memory | Send the last 20 lines; do not just retry |
| `Invalid user token` | HF token revoked or wrong | New **Write** token at huggingface.co/settings/tokens; update the Kaggle/Colab secret and the local `.env` |

## 7. Check-ins and what to send back

- Give one check-in time (usually 30 min) and the exact line that shows progress for this notebook. Examples: `[   done] ... belebele`, `Running loglikelihood requests` past a few thousand, or the FR-3 table.
- Say what Claude needs back (status counts, gate tables, verdict lines), and that Claude can read results from the HF dataset if the local `.env` token works. Test it without printing it.
- When the files arrive, follow the "When the user pastes notebook output back" section of `CLAUDE.md`. If a check fails, use the `investigate-then-research` skill.

## Style

Steps are numbered, short and imperative, in plain English: no compressed or caveman style for steps, because order matters. Give the time estimate before the steps. Say "close the tab" only for Kaggle.
