"""Fast late-block diagnostics for the ParityQuant v2 redesign (exploratory; never results tables).

Qwen2.5-0.5B-Instruct on CPU, float32. Only `self_attn.o_proj` and `mlp.down_proj` of blocks
16-23 are quantized (W3, group 128); everything else stays at full precision, so the inputs to
block 16 are cached once and every variant re-runs only blocks 16-23 plus the LM head.

Calibration: FLORES-200 dev, the same parallel sentence indices in every language, drawn by a
seed (seed = which sentences, not a numerical seed). Evaluation: FLORES devtest sentences,
scored after Qwen's prefix token as in `tasks.flores_nll`; metric = mean per-sentence NLL
increase over FP (nats), per language.

Diagnostics (owner request 2026-11-19):
  A  channel dilution: per-language Hessian diagonals on FP inputs, Telugu's top channels,
     how much the equal mix and the supremum envelope keep of them; Hessian condition numbers.
  B  seed noise: baseline GPTQ (float32, damping 0.01, no act-order) against float64,
     damping 0.03 and act-order with static groups, over calibration seeds.
  C  ablation ladder: English-only, equal mix, envelope (rho 0.5, 1.0), smooth layer DRO,
     multilingual QEP (with H_pool^-1 or H_tilde^-1 in the correction).

The GPTQ solver here is a copy of `quant.gptq.quantize_weight` with three switches (dtype,
damping, act-order with static groups). With defaults it reproduces `quantize_weight`
exactly (checked at start). Nothing here feeds a logged run.

    python scripts/m1_pq2_latediag.py --part A
    python scripts/m1_pq2_latediag.py --part B --seeds 0 1 2
    python scripts/m1_pq2_latediag.py --part C --seeds 0 1
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from whopays.quant.rtn import parameters, reconstruct

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODULES = ("self_attn.o_proj", "mlp.down_proj")


# --------------------------------------------------------------------------- GPTQ variant
@torch.no_grad()
def gptq(weight, hessian, bits=3, group_size=128, damping=0.01, dtype=torch.float32,
         act_order=False):
    """`quant.gptq.quantize_weight` plus dtype and act-order (static groups) switches.

    act_order: columns are processed in descending diag(H) order; group scales are fixed up
    front from the original contiguous 128-column groups (GPTQ `static_groups`), so the stored
    format (one scale/zero per contiguous group) and the bit count are unchanged.
    """
    w = weight.to(dtype).clone()
    h = hessian.to(dtype).clone()
    n = w.shape[1]
    dead = h.diag() == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    static = None
    if act_order:
        static = [parameters(w[:, s:s + group_size], bits) for s in range(0, n, group_size)]
        perm = torch.argsort(h.diag(), descending=True)
        w, h = w[:, perm], h[perm][:, perm]
    else:
        perm = torch.arange(n)
    h.diagonal().add_(damping * h.diag().mean())
    inv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    q = torch.empty_like(w)
    for start in range(0, n, group_size):
        end = min(start + group_size, n)
        block = w[:, start:end].clone()
        errors = torch.zeros_like(block)
        if not act_order:
            scale, zero = parameters(block, bits)
        for local, j in enumerate(range(start, end)):
            if act_order:
                scale, zero = static[int(perm[j]) // group_size]
            q[:, j:j + 1] = reconstruct(block[:, local:local + 1], scale, zero, bits)
            error = (block[:, local] - q[:, j]) / inv[j, j]
            errors[:, local] = error
            block[:, local:] -= error[:, None] * inv[j, j:end][None, :]
        w[:, end:] -= errors @ inv[start:end, end:]
    if act_order:
        q = q[:, torch.argsort(perm)]
    return q.float()


# --------------------------------------------------------------------------- data and model
def load(langs, n_calib, n_eval, first_block):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from whopays.datasets import flores
    from whopays.quant.fakequant import capture_block_inputs

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                 attn_implementation="sdpa").eval()
    prefix = model.config.bos_token_id
    prefix = prefix if prefix is not None else tok.eos_token_id

    def batches(texts):
        return [{"input_ids": torch.tensor([[prefix, *tok(t, add_special_tokens=False)
                                             ["input_ids"]]])} for t in texts]

    dev = {x: [r["text"] for r in flores(x, "dev")] for x in langs}
    test = {x: [r["text"] for r in flores(x, "devtest")][:n_eval] for x in langs}
    block = model.model.layers[first_block]
    t0 = time.perf_counter()
    evals = {x: capture_block_inputs(model, block, batches(test[x])) for x in langs}
    ids = {x: [b["input_ids"] for b in batches(test[x])] for x in langs}
    capture_s = time.perf_counter() - t0

    def calib(seed):
        idx = np.random.default_rng(seed).permutation(len(dev[langs[0]]))[:n_calib]
        return {x: capture_block_inputs(model, block, batches([dev[x][i] for i in idx]))
                for x in langs}

    return tok, model, evals, ids, calib, capture_s


def run_blocks(blocks, args, kwargs):
    hidden = args[0]
    for b in blocks:
        out = b(hidden, *args[1:], **kwargs)
        hidden = out[0] if isinstance(out, tuple) else out
    return hidden


@torch.no_grad()
def eval_nll(model, blocks, evals, ids):
    """Per language, per sentence NLL sum (nats) from cached block inputs."""
    out = {}
    for lang, items in evals.items():
        vals = []
        for (args, kwargs), tokens in zip(items, ids[lang], strict=True):
            hidden = model.model.norm(run_blocks(blocks, args, kwargs))
            logits = model.lm_head(hidden)[0, :-1]
            vals.append(float(torch.nn.functional.cross_entropy(
                logits, tokens[0, 1:], reduction="sum")))
        out[lang] = np.array(vals)
    return out


class _Stop(Exception):
    pass


def module_inputs(block, name, stream):
    """Inputs of one module over a stream; each forward stops right after the capture."""
    mod = block.get_submodule(name)
    xs = []

    def hook(m, a):
        xs.append(a[0].detach().reshape(-1, m.in_features))
        raise _Stop

    handle = mod.register_forward_pre_hook(hook)
    try:
        for args, kwargs in stream:
            with contextlib.suppress(_Stop):
                block(*args, **kwargs)
    finally:
        handle.remove()
    return torch.cat(xs)


@torch.no_grad()
def fp_inputs(fp_blocks, block_ids, calib):
    """FP-stream module inputs for every quantized module: identical for all configs."""
    out = {}
    stream = {x: list(s) for x, s in calib.items()}
    for b, fpb in zip(block_ids, fp_blocks, strict=True):
        for x, items in stream.items():
            xs = {n: [] for n in MODULES}
            hooks = [fpb.get_submodule(n).register_forward_pre_hook(
                lambda m, a, n=n, xs=xs: xs[n].append(a[0].detach().reshape(-1, m.in_features)))
                for n in MODULES]
            nxt = []
            try:
                for a, k in items:
                    nxt.append(((run_blocks([fpb], a, k), *a[1:]), k))
            finally:
                for h in hooks:
                    h.remove()
            for n in MODULES:
                out[(b, n, x)] = torch.cat(xs[n])
            stream[x] = nxt
    return out


# --------------------------------------------------------------------------- one quantization pass
@torch.no_grad()
def quantize_pass(model, fp_cache, block_ids, calib, cfg):
    """Sequential pass over the late blocks with FP and quantized streams per language.

    cfg keys: mix ('en' | 'equal' | 'dro'), rho (envelope, 0 = off), qep_alpha (0 = off),
    qep_inv ('pool' | 'tilde'), dtype, damping, act_order, beta, eta.
    """
    dtype = torch.float64 if cfg.get("fp64") else torch.float32
    langs = list(calib)
    lam = {x: 1 / len(langs) for x in langs}
    hat = {x: list(s) for x, s in calib.items()}
    log = {"lambda": {}, "drift": {}, "layer_err": {}, "solve_s": 0.0, "fwd_s": 0.0}
    for b in block_ids:
        qb = model.model.layers[b]
        for name in MODULES:
            full = f"{b}.{name}"
            t0 = time.perf_counter()
            xh = {x: module_inputs(qb, name, hat[x]).to(dtype) for x in langs}
            xf = {x: fp_cache[(b, name, x)].to(dtype) for x in langs}
            log["fwd_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            w = qb.get_submodule(name).weight.detach().to(dtype)
            drift = {}
            for x in langs:
                ref = (xf[x] @ w.T).pow(2).sum()
                drift[x] = float((((xf[x] - xh[x]) @ w.T).pow(2).sum()) / (ref + 1e-12))
            if cfg["mix"] == "en":
                lam = {x: float(x == "en") for x in langs}
            elif cfg["mix"] == "equal":
                lam = {x: 1 / len(langs) for x in langs}
            elif cfg["mix"] == "dro":
                mean = sum(drift.values()) / len(drift)
                raw = {x: lam[x] ** cfg.get("beta", 0.5)
                       * float(np.exp(cfg.get("eta", 1.0) * drift[x] / mean if mean > 0 else 0))
                       for x in langs}
                tot = sum(raw.values())
                lam = {x: v / tot for x, v in raw.items()}
            # memory trick: one weighted matrix plus per-language diagonals
            h = None
            diags = {}
            cross = None
            for x in langs:
                n = len(xh[x])
                diags[x] = xh[x].pow(2).sum(0) / n
                if lam[x] == 0:
                    continue
                term = (lam[x] / n) * (xh[x].T @ xh[x])
                h = term if h is None else h + term
                if cfg.get("qep_alpha"):
                    c = (lam[x] / n) * ((xf[x] - xh[x]).T @ xh[x])
                    cross = c if cross is None else cross + c
            h_tilde = h
            if cfg.get("rho"):
                d_pool = h.diag()
                d_max = torch.stack(list(diags.values())).amax(0)
                d_sup = (1 - cfg["rho"]) * d_pool + cfg["rho"] * d_max
                s = torch.where(d_pool > 0, (d_sup / d_pool.clamp_min(1e-30)).sqrt(),
                                torch.ones_like(d_pool))
                h_tilde = s[:, None] * h * s[None, :]
            target = w
            if cfg.get("qep_alpha"):
                base = h if cfg.get("qep_inv", "pool") == "pool" else h_tilde
                reg = base.clone()
                reg.diagonal().add_(cfg.get("damping", 0.01) * reg.diag().mean())
                # W* = W + alpha W C H^-1, C = sum_l lam_l/N_l delta_l^T xhat_l
                shift = torch.linalg.solve(reg, (w @ cross).T).T
                target = w + cfg["qep_alpha"] * shift
            q = gptq(target, h_tilde, cfg.get("bits", 3), 128, cfg.get("damping", 0.01),
                     dtype, cfg.get("act_order", False))
            # layer error against the FP target on the quantized-flow inputs, per language
            err = {}
            for x in langs:
                ref = (xf[x] @ w.T).pow(2).sum()
                err[x] = float(((xf[x] @ w.T - xh[x] @ q.to(dtype).T).pow(2).sum()) / ref)
            qb.get_submodule(name).weight.copy_(q)
            log["solve_s"] += time.perf_counter() - t0
            log["lambda"][full] = {x: round(v, 4) for x, v in lam.items()}
            log["drift"][full] = {x: round(v, 6) for x, v in drift.items()}
            log["layer_err"][full] = {x: round(v, 6) for x, v in err.items()}
            del xh, xf, h, h_tilde, cross
        t0 = time.perf_counter()
        for x in langs:
            hat[x] =[((run_blocks([qb], a, k), *a[1:]), k) for a, k in hat[x]]
        log["fwd_s"] += time.perf_counter() - t0
    return log


# --------------------------------------------------------------------------- parts
def part_a(model, calib, block_ids, out):
    langs = list(calib)
    res = {}
    for b in (block_ids[0], block_ids[-1]):
        for name in MODULES:
            t0 = time.perf_counter()
            blk = model.model.layers[b]
            xs = {x: module_inputs(blk, name, calib[x]).double() for x in langs}
            d = {x: v.pow(2).mean(0) for x, v in xs.items()}
            stack = torch.stack([d[x] for x in langs])
            d_eq = stack.mean(0)
            d_max = stack.amax(0)
            others = torch.stack([d[x] for x in langs if x != "te"]).amax(0)
            top = d["te"].topk(50).indices
            spec = (d["te"] / others.clamp_min(1e-30))
            spec_top = spec.topk(50).indices
            row = {
                "te_top50_te_over_en_median": float((d["te"][top] / d["en"][top]).median()),
                "te_top50_equal_over_te_mean": float((d_eq[top] / d["te"][top]).mean()),
                "te_specific_top50_te_over_maxother_median": float(spec[spec_top].median()),
                "te_specific_top50_equal_over_te_mean": float(
                    (d_eq[spec_top] / d["te"][spec_top]).mean()),
                "channels_te_over_maxother_gt2": int((spec > 2).sum()),
                "channels_te_over_maxother_gt5": int((spec > 5).sum()),
                "channels_any_lang_over_equal_gt5": int((d_max / d_eq > 5).sum()),
                "max_over_equal_max": float((d_max / d_eq).max()),
                "d_in": int(d_eq.numel()),
            }
            for rho in (0.5, 1.0):
                d_sup = (1 - rho) * d_eq + rho * d_max
                row[f"rho{rho}_sup_over_te_te_specific_top50"] = float(
                    (d_sup[spec_top] / d["te"][spec_top]).mean())
                row[f"rho{rho}_mean_diag_inflation"] = float((d_sup / d_eq).mean())
            # condition numbers (float64, undamped and damped 0.01 / 0.03 x mean diag)
            for label, hm in (("te", xs["te"].T @ xs["te"] / len(xs["te"])),
                              ("en", xs["en"].T @ xs["en"] / len(xs["en"])),
                              ("equal", sum(v.T @ v / len(v) for v in xs.values()) / len(xs))):
                ev = torch.linalg.eigvalsh(hm)
                mean_d = float(hm.diag().mean())
                row[f"cond_{label}"] = {
                    "undamped": float(ev[-1] / ev[0].clamp_min(1e-300)),
                    "min_eig": float(ev[0]),
                    "d0.01": float((ev[-1] + 0.01 * mean_d) / (ev[0] + 0.01 * mean_d)),
                    "d0.03": float((ev[-1] + 0.03 * mean_d) / (ev[0] + 0.03 * mean_d))}
            row["tokens"] = {x: len(v) for x, v in xs.items()}
            row["seconds"] = round(time.perf_counter() - t0, 1)
            res[f"{b}.{name}"] = row
            skip = ("cond_te", "cond_en", "cond_equal", "tokens")
            brief = {k: v for k, v in row.items() if k not in skip}
            print(f"A {b}.{name}: {json.dumps(brief)}", flush=True)
            print(f"   cond te {row['cond_te']} | equal {row['cond_equal']}", flush=True)
    out["A"] = res


def summarize(nll, ref):
    inc = {x: float((nll[x] - ref[x]).mean()) for x in nll}
    rel = {x: float((nll[x] - ref[x]).sum() / ref[x].sum()) for x in nll}
    worst = max(inc, key=inc.get)
    worst_rel = max(rel, key=rel.get)
    return {"inc": {x: round(v, 3) for x, v in inc.items()},
            "rel_inc": {x: round(v, 5) for x, v in rel.items()},
            "worst": worst, "worst_inc": round(inc[worst], 3),
            "worst_rel": worst_rel, "worst_rel_inc": round(rel[worst_rel], 5),
            "mean_inc": round(float(np.mean(list(inc.values()))), 3)}


CONFIGS = {
    # part B: numerical stabilization of the equal-mix baseline
    "eq_base": {"mix": "equal"},
    "eq_base_jit": {"mix": "equal", "damping": 0.0100001},  # numerical-chaos floor
    "eq_fp64": {"mix": "equal", "fp64": True},
    "eq_fp64_d03": {"mix": "equal", "fp64": True, "damping": 0.03},
    "eq_fp64_act": {"mix": "equal", "fp64": True, "act_order": True},
    "eq_fp64_d03_act": {"mix": "equal", "fp64": True, "damping": 0.03, "act_order": True},
    # part C: one knob at a time on the stabilized baseline from part B
    # (float64, damping 0.03, act-order with static groups)
    **{k: {"fp64": True, "damping": 0.03, "act_order": True, **v} for k, v in {
        "en": {"mix": "en"},
        "equal": {"mix": "equal"},
        "env05": {"mix": "equal", "rho": 0.5},
        "env10": {"mix": "equal", "rho": 1.0},
        "dro": {"mix": "dro", "beta": 0.5, "eta": 1.0},
        "qep": {"mix": "equal", "qep_alpha": 0.5},
        "env05_qep_pool": {"mix": "equal", "rho": 0.5, "qep_alpha": 0.5, "qep_inv": "pool"},
        "env05_qep_tilde": {"mix": "equal", "rho": 0.5, "qep_alpha": 0.5, "qep_inv": "tilde"},
    }.items()},
}
PARTS = {"B": ["eq_base", "eq_base_jit", "eq_fp64", "eq_fp64_d03", "eq_fp64_act",
               "eq_fp64_d03_act"],
         "C": ["en", "equal", "env05", "env10", "dro", "qep", "env05_qep_pool",
               "env05_qep_tilde"]}


def main() -> None:
    from whopays.languages import CODES
    from whopays.quant.gptq import quantize_weight

    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=("A", "B", "C"), required=True)
    p.add_argument("--configs", nargs="*")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--langs", nargs="+", default=list(CODES))
    p.add_argument("--calib", type=int, default=32)
    p.add_argument("--eval", type=int, default=48)
    p.add_argument("--first", type=int, default=16)
    p.add_argument("--out", default=str(ROOT / "runs" / "m1" / "pq2_latediag.json"))
    args = p.parse_args()
    torch.manual_seed(0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(out_path.read_text()) if out_path.exists() else {}

    # the copied solver must match the project's solver with defaults
    g = torch.Generator().manual_seed(0)
    x = torch.randn(300, 256, generator=g)
    w0 = torch.randn(32, 256, generator=g)
    assert torch.equal(gptq(w0, x.T @ x / 300), quantize_weight(w0, x.T @ x / 300, 3, 128))

    t_start = time.perf_counter()
    tok, model, evals, ids, calib_fn, cap_s = load(args.langs, args.calib, args.eval, args.first)
    n_blocks = model.config.num_hidden_layers
    block_ids = list(range(args.first, n_blocks))
    print(f"loaded + eval capture {time.perf_counter() - t_start:.1f}s (capture {cap_s:.1f}s)",
          flush=True)
    out["setup"] = {"model": MODEL, "device": "cpu", "dtype": "float32", "langs": args.langs,
                    "calib_sentences": args.calib, "eval_sentences": args.eval,
                    "blocks": block_ids, "modules": MODULES, "bits": 3, "group": 128,
                    "torch": torch.__version__}

    if args.part == "A":
        t0 = time.perf_counter()
        calib = calib_fn(args.seeds[0])
        part_a(model, calib, block_ids, out)
        out["A_seconds"] = round(time.perf_counter() - t0, 1)
        out_path.write_text(json.dumps(out, indent=1))
        return

    layers = model.model.layers
    original = {b: copy.deepcopy(layers[b]) for b in block_ids}
    fp_blocks = [original[b] for b in block_ids]
    t0 = time.perf_counter()
    ref = eval_nll(model, [layers[b] for b in block_ids], evals, ids)
    print(f"FP eval {time.perf_counter() - t0:.1f}s; mean NLL/sentence "
          f"{ {x: round(float(v.mean()), 1) for x, v in ref.items()} }", flush=True)
    out["fp_nll"] = {x: round(float(v.mean()), 3) for x, v in ref.items()}
    results = out.setdefault(args.part, {})
    for seed in args.seeds:
        t0 = time.perf_counter()
        calib = calib_fn(seed)
        print(f"seed {seed}: calib capture {time.perf_counter() - t0:.1f}s, tokens "
              f"{ {x: sum(a[0].shape[1] for a, _ in s) for x, s in calib.items()} }", flush=True)
        todo = [n for n in (args.configs or PARTS[args.part]) if f"{n}|s{seed}" not in results]
        if not todo:
            continue
        t0 = time.perf_counter()
        fp_cache = fp_inputs(fp_blocks, block_ids, calib)
        print(f"seed {seed}: FP stream cache {time.perf_counter() - t0:.1f}s", flush=True)
        for name in todo:
            key = f"{name}|s{seed}"
            for b in block_ids:
                layers[b].load_state_dict(original[b].state_dict())
            t0 = time.perf_counter()
            log = quantize_pass(model, fp_cache, block_ids, calib, CONFIGS[name])
            q_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            nll = eval_nll(model, [layers[b] for b in block_ids], evals, ids)
            e_s = time.perf_counter() - t0
            summ = summarize(nll, ref)
            le = log["layer_err"]
            summ["layer_err_mean"] = {x: round(float(np.mean([v[x] for v in le.values()])), 5)
                                      for x in args.langs}
            summ["layer_err_worst_mean"] = round(float(np.mean([max(v.values())
                                                                for v in le.values()])), 5)
            results[key] = {"cfg": CONFIGS[name], **summ, "per_sentence_delta": {
                x: np.round(nll[x] - ref[x], 3).tolist() for x in nll},
                "lambda_final": log["lambda"][f"{block_ids[-1]}.mlp.down_proj"],
                "lambda": log["lambda"], "drift": log["drift"], "layer_err": le,
                "quant_s": round(q_s, 1), "solve_s": round(log["solve_s"], 1),
                "fwd_s": round(log["fwd_s"], 1), "eval_s": round(e_s, 1)}
            out_path.write_text(json.dumps(out, indent=1))
            print(f"{key}: quant {q_s:.0f}s (solve {log['solve_s']:.0f}) eval {e_s:.0f}s | "
                  f"worst {summ['worst']} {summ['worst_inc']} | mean {summ['mean_inc']} | "
                  f"te {summ['inc'].get('te')} en {summ['inc'].get('en')}", flush=True)
    for b in block_ids:
        layers[b].load_state_dict(original[b].state_dict())


if __name__ == "__main__":
    main()
