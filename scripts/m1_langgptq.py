"""Local ParityQuant v2 diagnostics on Qwen2.5-0.5B (exploratory; never results tables).

Measures, with per-language Wikipedia calibration windows:
  (a) per-language Hessian scale: trace and output-energy shares per layer;
  (b) subspace geometry: top-k eigenvector overlap between languages, how much of each
      language's activation energy the pooled top-k subspace captures, channel dilution;
  (c) time split of a GPTQ pass: block forward passes against per-layer solves, and one
      layer's solve timed alone;
  (d) variants at W3 (and two at W4) on layer error, held-out FLORES NLL and Belebele:
      equal mix (pooled GPTQ), energy-normalized mix, one-step and 4-step layer-local
      minimax, the diagonal envelope, and minimax scored with an answer-letter Fisher (A+B);
  (e) a layer-band ablation: W4 everywhere except one third of the blocks at W3.

Results go to runs/m1/langgptq.json (git-ignored), written after every variant; finished
variants are skipped on rerun. DECISIONS 2026-11-18.

    python scripts/m1_langgptq.py --device mps
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
GEOMETRY_MODULES = ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj")


def calibration(tokenizer, langs, seed, n, window, cache: Path):
    from whopays.quant.calibration import pack_windows, wikipedia_articles

    out = {}
    for lang in langs:
        path = cache / f"calib_{lang}_s{seed}_{n}x{window}.json"
        if path.exists():
            windows = json.loads(path.read_text())
        else:
            windows, _ = pack_windows(tokenizer, wikipedia_articles(lang, seed), n, window)
            path.write_text(json.dumps(windows))
        out[lang] = [{"input_ids": torch.tensor([w])} for w in windows]
    return out


def summarize_errors(report) -> dict:
    errs = report["errors"]
    langs = list(next(iter(errs.values())))
    per_lang = {x: float(np.mean([e[x] for e in errs.values()])) for x in langs}
    worst = float(np.mean([max(e.values()) for e in errs.values()]))
    ratio = float(np.mean([max(e.values()) / np.mean(list(e.values())) for e in errs.values()]))
    lam = {x: float(np.mean([w[x] for w in report["weights"].values()])) for x in langs}
    return {"layer_error_mean": per_lang, "layer_error_worst": worst,
            "worst_over_mean": ratio, "lambda_mean": lam, "solves": report["solves"],
            "timing": report["timing"]}


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from whopays.datasets import belebele, flores
    from whopays.languages import CODES
    from whopays.models.loader import ModelBundle
    from whopays.parityquant import explore as ex
    from whopays.parityquant import langgptq as lg
    from whopays.quant.fakequant import linear_layers
    from whopays.quant.gptq import quantize_weight

    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--windows", type=int, default=16)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--flores", type=int, default=200)
    p.add_argument("--belebele", type=int, default=400)
    p.add_argument("--fisher-items", type=int, default=32)
    p.add_argument("--out", default=str(ROOT / "runs" / "m1" / "langgptq.json"))
    args = p.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    def save():
        out_path.write_text(json.dumps(results, indent=1))

    langs = list(CODES)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                 attn_implementation="sdpa")
    model = model.to(args.device).eval()
    bundle = ModelBundle(model=model, tokenizer=tok, device=args.device, dtype=torch.float32)
    layers = linear_layers(model)
    original = {n: m.weight.detach().cpu().clone() for n, m in layers.items()}
    n_blocks = model.config.num_hidden_layers
    results["setup"] = {"model": MODEL, "device": args.device, "dtype": "float32",
                        "windows": args.windows, "window": args.window,
                        "flores_heldout": args.flores, "belebele_items": args.belebele,
                        "torch": torch.__version__}

    def restore():
        with torch.no_grad():
            for n, m in layers.items():
                m.weight.copy_(original[n])

    calib = {s: calibration(tok, langs, s, args.windows, args.window, out_path.parent)
             for s in (0, 1)}
    flores_items = {x: sorted(flores(x, "devtest"), key=lambda r: r["item_id"])
                    [500:500 + args.flores] for x in langs}
    bele_all = {x: belebele(x) for x in langs}
    bele_eval = {x: v[:args.belebele] for x, v in bele_all.items()}
    probe = {x: v[800:800 + args.fisher_items] for x, v in bele_all.items()}

    t0 = time.perf_counter()
    reference = ex.score(bundle, flores_items, bele_eval)
    results["fp_scoring_s"] = time.perf_counter() - t0
    results["fp"] = {x: {"raw": 100 * float((r["letter_lp"].argmax(1) == r["gold"]).mean()),
                         "nll_mean": float(r["nll"].mean())} for x, r in reference.items()}
    save()
    print("FP reference done", round(results["fp_scoring_s"]), "s")

    # (a, b) geometry of the unquantized model, and one layer's Hessians for timing (c)
    kept = {}
    if "geometry" not in results:
        geo = {}

        def on_module(name, w, hs):
            if name.endswith(GEOMETRY_MODULES):
                geo[name] = lg.geometry({x: h.cpu() for x, h in hs.items()}, w.cpu())
            if name in ("model.layers.12.mlp.down_proj", "model.layers.12.mlp.gate_proj"):
                kept[name] = (w.cpu(), {x: h.cpu() for x, h in hs.items()})

        t0 = time.perf_counter()
        lg.quantize_by_language(model, 3, calib[0], quantize=False, on_module=on_module)
        results["geometry"] = geo
        results["geometry_s"] = time.perf_counter() - t0
        save()
        print("geometry done", round(results["geometry_s"]), "s")

    if kept and "solve_timing" not in results:
        timing = {}
        for name, (w, hs) in kept.items():
            h = lg.mix(hs, dict.fromkeys(hs, 1 / len(hs)), w, normalize=False)
            t0 = time.perf_counter()
            quantize_weight(w, h, 3, 128)
            timing[name] = {"cpu_solve_s": time.perf_counter() - t0, "shape": list(w.shape)}
        results["solve_timing"] = timing
        save()

    # (d, e) variants
    fisher = None
    variants = [
        ("w3_equal", 3, {"normalize": False}, 0),
        ("w3_equal_seed1", 3, {"normalize": False}, 1),
        ("w3_energy", 3, {}, 0),
        ("w3_equal_k4", 3, {"normalize": False, "steps": 4}, 0),
        ("w3_energy_1step", 3, {"steps": 1, "disparity": 1.2}, 0),
        ("w3_energy_k4", 3, {"steps": 4}, 0),
        ("w3_energy_k4_env", 3, {"steps": 4, "envelope": 0.5}, 0),
        ("w3_AB_k4", 3, {"steps": 4, "fisher": True}, 0),
        ("w4_equal", 4, {"normalize": False}, 0),
        ("w4_AB_k4", 4, {"steps": 4, "fisher": True}, 0),
    ]
    thirds = np.array_split(np.arange(n_blocks), 3)
    for label, band in zip(("early", "middle", "late"), thirds, strict=True):
        variants.append((f"w4_band_{label}_w3", ex.band_bitmap(n_blocks, 4, band.tolist(), 3),
                         {"normalize": False}, 0))
    for name, bits, kw, seed in variants:
        if name in results.get("variants", {}):
            continue
        kw = dict(kw)
        if kw.pop("fisher", False):
            if fisher is None:
                restore()
                t0 = time.perf_counter()
                fisher = lg.output_fisher(
                    model, {x: ex.belebele_batches(tok, probe[x]) for x in langs},
                    ex.letter_fisher_loss(tok))
                results["fisher_s"] = time.perf_counter() - t0
            kw["row_weights"] = fisher
        restore()
        t0 = time.perf_counter()
        report = lg.quantize_by_language(model, bits, calib[seed], solve_device="cpu", **kw)
        quant_s = time.perf_counter() - t0
        metrics = ex.compare(ex.score(bundle, flores_items, bele_eval), reference)
        results.setdefault("variants", {})[name] = {
            "bits": bits if isinstance(bits, int) else "band", "seed": seed,
            "options": {k: v for k, v in kw.items() if k != "row_weights"},
            "fisher": "row_weights" in kw, "quantize_s": quant_s,
            **summarize_errors(report), "metrics": metrics}
        save()
        worst = max(metrics.values(), key=lambda m: m["nll_increase"])
        print(f"{name}: {quant_s:.0f}s, worst NLL increase {worst['nll_increase']:.2f}, "
              f"mean calibrated {np.mean([m['calibrated'] for m in metrics.values()]):.1f}")
    restore()


if __name__ == "__main__":
    main()
