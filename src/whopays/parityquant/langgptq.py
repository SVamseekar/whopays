"""Per-language GPTQ: Hessian geometry, mixing and layer-local minimax (ParityQuant v2 work).

Exploratory machinery for the v2 redesign (DECISIONS 2026-11-18). GPTQ quantizes each linear
layer W against an input Hessian H = E[x x^T]. With calibration text in several languages, H is
a mixture of per-language Hessians H_l. This module keeps the H_l apart, so that

- `geometry` can describe them: each language's share of the trace and of the output energy
  tr(W H_l W^T), overlap of their top eigenvector subspaces, how much of each language's
  activation energy the pooled top subspace captures, and how much the pooled diagonal
  dilutes each language's largest channels;
- `mix` can recombine them with weights λ, optionally after normalizing each H_l to equal
  output energy (`normalized`) and optionally raising the diagonal towards the per-channel
  maximum over languages (`envelope`, a congruence transform that keeps H positive
  semi-definite);
- `minimax_quantize` can choose λ per layer: quantize, measure each language's relative output
  error tr(D H_l D^T) / tr(W H_l W^T) (D = W − Q), take an exponentiated-gradient step towards
  the worst language, anchored to uniform, and keep the iterate with the smallest maximum
  error. Optional per-language output-channel weights (for example an output Fisher from
  `output_fisher`) change how errors are scored, not how one solve rounds;
- `quantize_by_language` runs the sequential GPTQ pass of `quant.fakequant` with
  per-language input streams, so later blocks see each language's quantized-model inputs.

With steps=0, normalize=False and equal token counts per language, `quantize_by_language`
gives the weights of pooled GPTQ (tests check this). Nothing here is used by logged runs.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch

from whopays.quant.fakequant import (
    _move,
    _stages,
    capture_block_inputs,
    linear_layers,
    resolve_bitmap,
)
from whopays.quant.gptq import quantize_weight

Hessians = Mapping[str, torch.Tensor]


def energy(w: torch.Tensor, h: torch.Tensor) -> float:
    """tr(W H W^T): the layer's mean squared output on the inputs that made H."""
    return float(((w @ h) * w).sum())


def normalized(hessians: Hessians, w: torch.Tensor) -> dict[str, torch.Tensor]:
    """Rescale each H_l so every language has the same output energy (their mean)."""
    energies = {lang: energy(w, h) for lang, h in hessians.items()}
    mean = sum(energies.values()) / len(energies)
    return {lang: h * (mean / energies[lang]) for lang, h in hessians.items()}


def mix(hessians: Hessians, weights: Mapping[str, float], w: torch.Tensor, *,
        normalize: bool = True, envelope: float = 0.0) -> torch.Tensor:
    """Σ_l λ_l H_l, optionally energy-normalized and diagonal-enveloped.

    envelope γ in [0, 1] rescales H to S H S with S = diag((e / d)^(γ/2)), d = diag(H) and
    e = the per-channel maximum of diag(H_l) over languages: γ = 1 sets the diagonal to e.
    """
    hs = normalized(hessians, w) if normalize else dict(hessians)
    out = None
    for lang, h in hs.items():
        term = weights[lang] * h
        out = term if out is None else out + term
    if envelope:
        d = out.diag().clamp_min(torch.finfo(out.dtype).tiny)
        e = torch.stack([h.diag() for h in hs.values()]).amax(0)
        s = (e / d).clamp_min(0).pow(envelope / 2)
        out = s[:, None] * out * s[None, :]
    return out


def relative_errors(w: torch.Tensor, q: torch.Tensor, hessians: Hessians,
                    row_weights: Mapping[str, torch.Tensor] | None = None) -> dict[str, float]:
    """Per language, tr(D H_l D^T) / tr(W H_l W^T), rows optionally weighted."""
    d = w - q.float()
    out = {}
    for lang, h in hessians.items():
        num, den = ((d @ h) * d).sum(1), ((w @ h) * w).sum(1)
        if row_weights is not None:
            r = row_weights[lang].to(num)
            num, den = r * num, r * den
        out[lang] = float(num.sum() / den.sum()) if float(den.sum()) > 0 else 0.0
    return out


@dataclass
class MinimaxResult:
    q: torch.Tensor
    weights: dict[str, float]
    errors: dict[str, float]
    solves: int
    history: list[dict[str, float]] = field(default_factory=list)


def minimax_quantize(w: torch.Tensor, hessians: Hessians, *, bits: int, group_size: int = 128,
                     steps: int = 0, eta: float = 1.0, anchor: float = 0.1,
                     disparity: float | None = None, normalize: bool = True,
                     envelope: float = 0.0,
                     row_weights: Mapping[str, torch.Tensor] | None = None) -> MinimaxResult:
    """GPTQ on a λ-mixed Hessian, with up to `steps` worst-language reweighting rounds.

    Round t quantizes with H(λ_t) and scores each language's relative error. The next λ is
    λ_l · exp(η · r_l / mean r), renormalized, then mixed with uniform by `anchor`. Rounds stop
    early when max r / mean r falls below `disparity`. Returns the iterate with the smallest
    maximum error (round 0 is the uniform mix, so the result is never worse than it on the
    calibration data).
    """
    langs = list(hessians)
    lam = dict.fromkeys(langs, 1.0 / len(langs))
    best, solves, history = None, 0, []
    for t in range(steps + 1):
        q = quantize_weight(w, mix(hessians, lam, w, normalize=normalize, envelope=envelope),
                            bits, group_size)
        solves += 1
        err = relative_errors(w, q, hessians, row_weights)
        history.append(err)
        if best is None or max(err.values()) < max(best.errors.values()):
            best = MinimaxResult(q, dict(lam), err, 0)
        mean = sum(err.values()) / len(err)
        if t == steps or mean <= 0 or (disparity is not None
                                       and max(err.values()) / mean < disparity):
            break
        raw = {lang: lam[lang] * float(torch.exp(torch.tensor(eta * err[lang] / mean)))
               for lang in langs}
        total = sum(raw.values())
        lam = {lang: (1 - anchor) * raw[lang] / total + anchor / len(langs) for lang in langs}
    best.solves, best.history = solves, history
    return best


def top_eigvecs(h: torch.Tensor, k: int, *, exact_below: int = 1024, oversample: int = 16,
                iters: int = 8, seed: int = 0) -> torch.Tensor:
    """Orthonormal top-k eigenvectors of a symmetric PSD matrix (columns), float32.

    Exact `eigh` in float64 up to `exact_below` rows, else randomized subspace iteration.
    """
    n = h.shape[0]
    k = min(k, n)
    h64 = h.detach().double().cpu()
    if n <= exact_below:
        _, v = torch.linalg.eigh(h64)
        return v[:, -k:].flip(1).float()
    g = torch.Generator().manual_seed(seed)
    basis = torch.randn(n, min(n, k + oversample), generator=g, dtype=torch.float64)
    for _ in range(iters):
        basis, _ = torch.linalg.qr(h64 @ basis)
    _, v = torch.linalg.eigh(basis.T @ h64 @ basis)
    return (basis @ v[:, -k:]).flip(1).float()


def subspace_overlap(u: torch.Tensor, v: torch.Tensor) -> float:
    """Mean squared cosine of the principal angles between two orthonormal bases."""
    k = min(u.shape[1], v.shape[1])
    return float((u.T @ v).pow(2).sum() / k)


def geometry(hessians: Hessians, w: torch.Tensor, *, ks=(16, 64),
             latin=("en", "de", "fr", "es"), top_channels: int = 50) -> dict:
    """Scale, energy, subspace and channel-dilution summaries for one layer."""
    langs = list(hessians)
    traces = {lang: float(h.trace()) for lang, h in hessians.items()}
    energies = {lang: energy(w.float(), h) for lang, h in hessians.items()}
    pooled = sum(hessians.values()) / len(hessians)
    out = {"trace_share": {x: v / sum(traces.values()) for x, v in traces.items()},
           "energy_share": {x: v / sum(energies.values()) for x, v in energies.items()},
           "overlap": {}, "mix_capture": {}, "dilution": {}}
    for k in ks:
        bases = {lang: top_eigvecs(h, k) for lang, h in hessians.items()}
        pair = {(a, b): subspace_overlap(bases[a], bases[b])
                for i, a in enumerate(langs) for b in langs[i + 1:]}

        def mean(cond, pair=pair):
            vals = [v for (a, b), v in pair.items() if cond(a, b)]
            return sum(vals) / len(vals) if vals else float("nan")

        out["overlap"][k] = {
            "pairs": {f"{a}-{b}": v for (a, b), v in pair.items()},
            "latin_within": mean(lambda a, b: a in latin and b in latin),
            "latin_vs_other": mean(lambda a, b: (a in latin) != (b in latin)),
            "other_within": mean(lambda a, b: a not in latin and b not in latin)}
        u = top_eigvecs(pooled, k)
        out["mix_capture"][k] = {
            lang: float((u.T @ h.float().cpu() @ u).trace() / h.trace().cpu())
            for lang, h in hessians.items()}
    pooled_diag = pooled.diag()
    for lang, h in hessians.items():
        d = h.diag()
        top = d.topk(min(top_channels, d.numel())).indices
        out["dilution"][lang] = float((pooled_diag[top] / d[top]).mean())
    return out


def nll_loss(model, batch) -> torch.Tensor:
    """Mean next-token NLL of a batch: the self-supervised loss behind an output Fisher."""
    ids = batch["input_ids"].to(next(model.parameters()).device)
    logits = model(input_ids=ids, use_cache=False).logits[:, :-1].float()
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                             ids[:, 1:].reshape(-1))


def output_fisher(model, batches_by_lang: Mapping[str, list], loss_fn: Callable,
                  ) -> dict[str, dict[str, torch.Tensor]]:
    """Per language and linear layer: mean squared loss gradient per output channel.

    A diagonal (empirical) Fisher of each layer's output, from one backward pass per batch.
    Parameters get no gradients: only the embedding output is made to require grad.
    """
    layers = linear_layers(model)
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    sums: dict[str, torch.Tensor] = {}
    hooks = []

    def grad_hook(name):
        def hook(_module, _grad_in, grad_out):
            g = grad_out[0].detach().float()
            g = g.reshape(-1, g.shape[-1]).pow(2).sum(0)
            sums[name] = g if name not in sums else sums[name] + g
        return hook

    def needs_grad(_module, _args, output):
        return output.requires_grad_(True)

    hooks.append(model.get_input_embeddings().register_forward_hook(needs_grad))
    hooks += [m.register_full_backward_hook(grad_hook(n)) for n, m in layers.items()]
    out = {}
    try:
        with torch.enable_grad():
            for lang, batches in batches_by_lang.items():
                sums.clear()
                for batch in batches:
                    loss_fn(model, batch).backward()
                out[lang] = {n: (sums[n] / len(batches)).cpu() for n in layers}
    finally:
        for hook in hooks:
            hook.remove()
        for n, p in model.named_parameters():
            p.requires_grad_(flags[n])
    return out


@torch.no_grad()
def quantize_by_language(model, bits, batches_by_lang: Mapping[str, list], *,
                         group_size: int = 128, steps: int = 0, eta: float = 1.0,
                         anchor: float = 0.1, disparity: float | None = None,
                         normalize: bool = True, envelope: float = 0.0,
                         row_weights: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
                         quantize: bool = True, solve_device: str | None = None,
                         on_module: Callable | None = None) -> dict:
    """Sequential block-by-block GPTQ with one input stream per language.

    Mirrors `quant.fakequant._gptq_blocks` (true-sequential stages, outputs of each
    quantized block feed the next) but accumulates a separate Hessian per language and
    quantizes each layer with `minimax_quantize`. `on_module(name, W, {lang: H_l})` sees
    every layer before it is quantized; with quantize=False the weights stay unchanged and
    the pass only serves `on_module` (geometry of the unquantized model). `solve_device`
    moves each layer's solve (Cholesky and rounding) to another device, for example the CPU
    when the model runs on MPS, whose float32 Cholesky rejects these Hessians.
    """
    bitmap = resolve_bitmap(model, bits)
    layers = linear_layers(model)
    blocks: dict[str, list[str]] = {}
    for name in bitmap:
        parts = name.split(".")
        blocks.setdefault(".".join(parts[:3]), []).append(name)
    order = sorted(blocks, key=lambda b: int(b.split(".")[2]))
    first = model.get_submodule(order[0])
    device = next(first.parameters()).device
    timing = {"forward_s": 0.0, "solve_s": 0.0}
    t0 = time.perf_counter()
    inputs = {lang: [(_move(a, device), _move(k, device))
                     for a, k in capture_block_inputs(model, first, batches)]
              for lang, batches in batches_by_lang.items()}
    timing["capture_s"] = time.perf_counter() - t0
    errors, weights, solves = {}, {}, 0
    current = {"lang": None}
    for index, block_name in enumerate(order):
        block = model.get_submodule(block_name)
        for stage in _stages(block_name, blocks[block_name]):
            hs = {n: {lang: torch.zeros(layers[n].in_features, layers[n].in_features,
                                        device=layers[n].weight.device, dtype=torch.float32)
                      for lang in inputs} for n in stage}
            counts = {n: dict.fromkeys(inputs, 0) for n in stage}

            def hook_for(n, hs=hs, counts=counts):
                def collect(module, inp):
                    x = inp[0].detach().reshape(-1, module.in_features).float()
                    hs[n][current["lang"]].addmm_(x.T, x)
                    counts[n][current["lang"]] += len(x)
                return collect

            hooks = [layers[n].register_forward_pre_hook(hook_for(n)) for n in stage]
            t0 = time.perf_counter()
            try:
                for lang, stream in inputs.items():
                    current["lang"] = lang
                    for args, kwargs in stream:
                        block(*args, **kwargs)
            finally:
                for hook in hooks:
                    hook.remove()
            _sync(device)
            timing["forward_s"] += time.perf_counter() - t0
            for n in stage:
                per = {lang: hs[n].pop(lang) / counts[n][lang] for lang in inputs}
                w = layers[n].weight.float()
                if solve_device is not None:
                    per = {lang: h.to(solve_device) for lang, h in per.items()}
                    w = w.to(solve_device)
                if on_module is not None:
                    on_module(n, w, per)
                if quantize:
                    t0 = time.perf_counter()
                    rows = ({lang: row_weights[lang][n] for lang in inputs}
                            if row_weights is not None else None)
                    res = minimax_quantize(w, per, bits=bitmap[n], group_size=group_size,
                                           steps=steps, eta=eta, anchor=anchor,
                                           disparity=disparity, normalize=normalize,
                                           envelope=envelope, row_weights=rows)
                    _sync(device)
                    timing["solve_s"] += time.perf_counter() - t0
                    errors[n], weights[n] = res.errors, res.weights
                    solves += res.solves
                    layers[n].weight.copy_(res.q.to(layers[n].weight.dtype))
                del per, w
        if index + 1 < len(order):
            t0 = time.perf_counter()
            for lang, stream in inputs.items():
                outputs = []
                for args, kwargs in stream:
                    out = block(*args, **kwargs)
                    hidden = out[0] if isinstance(out, tuple) else out
                    outputs.append(((hidden, *args[1:]), kwargs))
                inputs[lang] = outputs
            _sync(device)
            timing["forward_s"] += time.perf_counter() - t0
    return {"errors": errors, "weights": weights, "solves": solves, "timing": timing}


def _sync(device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
