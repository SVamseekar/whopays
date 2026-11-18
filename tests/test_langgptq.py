"""Per-language GPTQ (ParityQuant v2 exploration): Hessian geometry, mixing and minimax."""
import pytest
import torch

from whopays.parityquant import langgptq as lg
from whopays.quant.gptq import quantize_weight


def _spd(n, rank, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rank, n, generator=g) * scale
    return x.T @ x / rank


def test_top_eigvecs_matches_eigh_subspace():
    q, _ = torch.linalg.qr(torch.randn(300, 300, generator=torch.Generator().manual_seed(0)))
    h = q @ torch.diag(torch.exp(-torch.arange(300.0) / 10)) @ q.T
    u = lg.top_eigvecs(h, 8, exact_below=0)  # force the randomized path
    _, v = torch.linalg.eigh(h.double())
    exact = v[:, -8:].float()
    assert lg.subspace_overlap(u, exact) > 0.99
    assert lg.subspace_overlap(lg.top_eigvecs(h, 8), exact) > 0.999


def test_subspace_overlap_bounds():
    q, _ = torch.linalg.qr(torch.randn(50, 10))
    assert lg.subspace_overlap(q[:, :5], q[:, :5]) == pytest.approx(1.0, abs=1e-5)
    assert lg.subspace_overlap(q[:, :5], q[:, 5:]) == pytest.approx(0.0, abs=1e-5)


def test_mix_equal_unnormalized_is_mean():
    hs = {"a": _spd(16, 40, 1), "b": _spd(16, 40, 2, 3.0)}
    w = torch.randn(8, 16)
    mixed = lg.mix(hs, {"a": 0.5, "b": 0.5}, w, normalize=False)
    assert torch.allclose(mixed, (hs["a"] + hs["b"]) / 2, atol=1e-6)


def test_energy_normalization_equalizes_output_energy():
    hs = {"a": _spd(16, 40, 1), "b": _spd(16, 40, 2, 10.0)}
    w = torch.randn(8, 16)
    normed = lg.normalized(hs, w)
    energies = [lg.energy(w, h) for h in normed.values()]
    assert energies[0] == pytest.approx(energies[1], rel=1e-5)


def test_envelope_keeps_psd_and_raises_diagonal_to_worst_language():
    hs = {"a": _spd(12, 30, 1), "b": _spd(12, 30, 2, 4.0)}
    w = torch.randn(6, 12)
    base = lg.mix(hs, {"a": 0.5, "b": 0.5}, w, normalize=False)
    env = lg.mix(hs, {"a": 0.5, "b": 0.5}, w, normalize=False, envelope=1.0)
    top = torch.maximum(hs["a"].diag(), hs["b"].diag())
    assert torch.allclose(env.diag(), top, rtol=1e-4)
    assert torch.linalg.eigvalsh(env.double()).min() > -1e-6
    assert (env.diag() >= base.diag() - 1e-6).all()


def test_minimax_zero_steps_is_plain_gptq_on_the_mix():
    hs = {"a": _spd(32, 64, 1), "b": _spd(32, 64, 2, 2.0)}
    w = torch.randn(16, 32)
    out = lg.minimax_quantize(w, hs, bits=3, group_size=16, steps=0, normalize=False)
    ref = quantize_weight(w, (hs["a"] + hs["b"]) / 2, 3, 16)
    assert torch.equal(out.q, ref)
    assert out.solves == 1


def test_minimax_never_worse_than_start_and_reports_errors():
    hs = {"a": _spd(32, 64, 1), "b": _spd(32, 64, 2), "c": _spd(32, 8, 3, 5.0)}
    w = torch.randn(16, 32)
    start = lg.minimax_quantize(w, hs, bits=3, group_size=16, steps=0)
    many = lg.minimax_quantize(w, hs, bits=3, group_size=16, steps=4, eta=2.0)
    assert set(many.errors) == set(hs)
    assert max(many.errors.values()) <= max(start.errors.values()) + 1e-9
    assert sum(many.weights.values()) == pytest.approx(1.0)
    assert many.solves <= 5


def test_one_step_skips_resolve_below_disparity_threshold():
    hs = {"a": _spd(32, 64, 1), "b": _spd(32, 64, 2)}
    w = torch.randn(16, 32)
    out = lg.minimax_quantize(w, hs, bits=3, group_size=16, steps=1, disparity=1e9)
    assert out.solves == 1


def test_row_weights_change_errors_not_unweighted_solution():
    hs = {"a": _spd(32, 64, 1), "b": _spd(32, 64, 2)}
    w = torch.randn(16, 32)
    rows = {"a": torch.ones(16), "b": torch.linspace(0.1, 3, 16)}
    plain = lg.minimax_quantize(w, hs, bits=3, group_size=16, steps=0)
    weighted = lg.minimax_quantize(w, hs, bits=3, group_size=16, steps=0, row_weights=rows)
    assert torch.equal(plain.q, weighted.q)
    assert plain.errors["a"] == pytest.approx(weighted.errors["a"], rel=1e-5)
    assert plain.errors["b"] != pytest.approx(weighted.errors["b"], rel=1e-3)


def _lang_batches(n_per_lang=2, length=24):
    g = torch.Generator().manual_seed(0)
    return {lang: [{"input_ids": torch.randint(1, 200, (1, length), generator=g)}
                   for _ in range(n_per_lang)] for lang in ("en", "te")}


def test_equal_mix_matches_pooled_gptq(tiny_model_dir):
    from whopays.config import RunConfig
    from whopays.models.loader import load
    from whopays.quant.fakequant import apply

    cfg = RunConfig(model=str(tiny_model_dir), task="belebele", lang="en")
    batches = _lang_batches()
    pooled, per_lang = load(cfg, "cpu").model, load(cfg, "cpu").model
    apply(pooled, 3, method="gptq", group_size=16,
          calibration=[b for bs in batches.values() for b in bs])
    report = lg.quantize_by_language(per_lang, 3, batches, group_size=16, steps=0,
                                     normalize=False)
    for (name, a), (_, b) in zip(pooled.named_parameters(), per_lang.named_parameters(),
                                 strict=True):
        assert torch.allclose(a, b, atol=1e-5, rtol=0), name
    assert set(report["errors"]) == set(report["weights"])
    assert all(set(e) == {"en", "te"} for e in report["errors"].values())
    assert report["timing"]["solve_s"] > 0 and report["timing"]["forward_s"] > 0


def test_geometry_pass_leaves_weights_unchanged(tiny_model_dir):
    from whopays.config import RunConfig
    from whopays.models.loader import load

    model = load(RunConfig(model=str(tiny_model_dir), task="belebele", lang="en"), "cpu").model
    before = {n: p.clone() for n, p in model.named_parameters()}
    seen = []
    lg.quantize_by_language(model, 3, _lang_batches(), group_size=16, quantize=False,
                            on_module=lambda name, w, hs: seen.append((name, set(hs))))
    assert seen and all(langs == {"en", "te"} for _, langs in seen)
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]), n


def test_geometry_summary_fields():
    hs = {lang: _spd(20, 50, i, 1 + i) for i, lang in enumerate(("en", "de", "te"))}
    w = torch.randn(10, 20)
    g = lg.geometry(hs, w, ks=(4,), latin=("en", "de"))
    assert set(g["trace_share"]) == set(hs)
    assert sum(g["trace_share"].values()) == pytest.approx(1.0)
    assert 0 <= g["overlap"][4]["latin_vs_other"] <= 1
    assert set(g["mix_capture"][4]) == set(hs)
    assert g["dilution"]["te"] > 0


def test_output_fisher_shapes(tiny_model_dir):
    from whopays.config import RunConfig
    from whopays.models.loader import load

    model = load(RunConfig(model=str(tiny_model_dir), task="belebele", lang="en"), "cpu").model
    batches = _lang_batches()
    fisher = lg.output_fisher(model, batches, lg.nll_loss)
    layers = {n: m for n, m in model.named_modules()
              if isinstance(m, torch.nn.Linear) and n != "lm_head"}
    assert set(fisher) == {"en", "te"}
    for per in fisher.values():
        assert set(per) == set(layers)
        for name, f in per.items():
            assert f.shape == (layers[name].out_features,)
            assert (f >= 0).all() and f.sum() > 0
    assert all(not p.requires_grad or p.grad is None for p in model.parameters())
