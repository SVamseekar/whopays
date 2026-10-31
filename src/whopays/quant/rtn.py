"""Per-output-channel, input-group asymmetric round-to-nearest quantization."""
import torch


def parameters(weight: torch.Tensor, bits: int):
    if bits not in (2, 3, 4, 8):
        raise ValueError('bits must be one of 2, 3, 4, 8')
    lo = weight.amin(dim=1, keepdim=True).clamp(max=0)
    hi = weight.amax(dim=1, keepdim=True).clamp(min=0)
    scale = (hi - lo) / (2**bits - 1)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zero = torch.round(-lo / scale)
    return scale, zero


def reconstruct(weight, scale, zero, bits):
    return (torch.round(weight / scale + zero).clamp(0, 2**bits - 1) - zero) * scale


@torch.no_grad()
def quantize(weight: torch.Tensor, bits: int, group_size: int = 128) -> torch.Tensor:
    if weight.ndim != 2 or group_size < 1 or not torch.isfinite(weight).all():
        raise ValueError('expected a finite matrix and positive group_size')
    out = weight.float().clone()
    for start in range(0, out.shape[1], group_size):
        group = out[:, start:start + group_size]
        scale, zero = parameters(group, bits)
        group.copy_(reconstruct(group, scale, zero, bits))
    return out.to(weight.dtype)
