"""GPTQ error compensation with damped activation Hessian and act-order disabled."""
import torch

from whopays.quant.rtn import parameters, reconstruct


@torch.no_grad()
def quantize_weight(weight, hessian, bits=4, group_size=128, damping=0.01):
    if weight.ndim != 2 or group_size < 1 or damping <= 0:
        raise ValueError('invalid weight shape, group size or damping')
    if hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError('Hessian shape must match input features')
    if not torch.isfinite(weight).all() or not torch.isfinite(hessian).all():
        raise ValueError('weights and Hessian must be finite')
    w = weight.float().clone()
    h = hessian.to(device=w.device, dtype=torch.float32).clone()
    dead = h.diag() == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h.diagonal().add_(damping * h.diag().mean())
    inv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    q = torch.empty_like(w)
    # Process a group at a time: local sequential corrections, then one GEMM for
    # all remaining columns. This avoids a full-width rank-one update per column.
    for start in range(0, w.shape[1], group_size):
        end = min(start + group_size, w.shape[1])
        block = w[:, start:end].clone()
        errors = torch.zeros_like(block)
        scale, zero = parameters(block, bits)
        for local, j in enumerate(range(start, end)):
            q[:, j:j+1] = reconstruct(block[:, local:local+1], scale, zero, bits)
            error = (block[:, local] - q[:, j]) / inv[j, j]
            errors[:, local] = error
            block[:, local:] -= error[:, None] * inv[j, j:end][None, :]
        w[:, end:] -= errors @ inv[start:end, end:]
    return q.to(weight.dtype)
