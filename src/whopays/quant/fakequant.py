"""Apply quantized/dequantized weights in place, retaining the model's float dtype.

Linear projections within decoder blocks are quantized; embeddings, norms and lm_head
remain at reference precision. Memory numbers are theoretical bit counts, not allocations.
"""
from contextlib import suppress

import torch

from whopays.quant.gptq import quantize_weight
from whopays.quant.rtn import quantize


def linear_layers(model):
    return {name: module for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and name != 'lm_head'}


def resolve_bitmap(model, bits):
    layers = linear_layers(model)
    if isinstance(bits, int):
        result = dict.fromkeys(layers, bits)
    else:
        result = {}
        used = set()
        for name in layers:
            matches = [k for k in bits if name == k or name.startswith(k + '.')]
            if not matches:
                raise ValueError(f'bit map missing {name}')
            key = max(matches, key=len)
            result[name] = bits[key]
            used.add(key)
        if set(bits) - used:
            raise ValueError(f'unknown or shadowed bit map keys: {set(bits) - used}')
    if not result or not set(result.values()) <= {2, 3, 4, 8}:
        raise ValueError('bit map must contain supported bits for decoder projections')
    return result


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, tuple):
        return tuple(_move(v, device) for v in value)
    if isinstance(value, list):
        return [_move(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _move(v, device) for k, v in value.items()}
    return value


@torch.no_grad()
def capture_block_inputs(model, block, batches):
    """Cache one block's reference inputs on CPU; stop before executing that block."""
    class Captured(Exception):
        pass

    captured = []
    device = next(model.parameters()).device

    def collect(_module, args, kwargs):
        captured.append((_move(args, 'cpu'), _move(kwargs, 'cpu')))
        raise Captured

    hook = block.register_forward_pre_hook(collect, with_kwargs=True)
    try:
        for batch in batches:
            with suppress(Captured):
                model(**_move(batch, device), use_cache=False)
    finally:
        hook.remove()
    if len(captured) != len(batches):
        raise ValueError('calibration did not reach decoder block')
    return captured


@torch.no_grad()
def activation_hessian(model, module, batches, *, block=None, captured=None):
    device = module.weight.device
    h = torch.zeros(module.in_features, module.in_features, device=device, dtype=torch.float32)
    count = 0

    def collect(_module, args):
        nonlocal count
        x = args[0].detach().reshape(-1, module.in_features).float()
        h.addmm_(x.T, x)
        count += len(x)

    hook = module.register_forward_pre_hook(collect)
    try:
        if captured is not None:
            for args, kwargs in captured:
                block(*_move(args, device), **_move(kwargs, device))
        else:
            for batch in batches:
                model(**{k: v.to(device) for k, v in batch.items()}, use_cache=False)
    finally:
        hook.remove()
    if not count:
        raise ValueError('empty calibration activations')
    return h / count


@torch.no_grad()
def apply(model, bits, method='rtn', calibration=None, group_size=128):
    bitmap = resolve_bitmap(model, bits)
    if method not in ('rtn', 'gptq'):
        raise ValueError('method must be rtn or gptq')
    if method == 'gptq' and not calibration:
        raise ValueError('GPTQ requires calibration batches')
    layers = linear_layers(model)
    modules = dict(model.named_modules())
    current_block, captured = None, None
    for name, bit in bitmap.items():
        module = layers[name]
        block_name = '.'.join(name.split('.')[:3])
        block = modules.get(block_name)
        if method == 'gptq' and block is not None and block_name != current_block:
            captured = capture_block_inputs(model, block, calibration)
            current_block = block_name
        if method == 'gptq':
            h = activation_hessian(model, module, calibration, block=block, captured=captured)
            q = quantize_weight(module.weight, h, bit, group_size)
            del h
        else:
            q = quantize(module.weight, bit, group_size)
        module.weight.copy_(q)
    quantized = sum(layers[n].weight.numel() for n in bitmap)
    total = sum(p.numel() for p in model.parameters())
    return {'bitmap': bitmap, 'method': method, 'memory_kind': 'computed_weight_bits',
            'weight_bits': sum(layers[n].weight.numel() * b for n, b in bitmap.items())
            + (total - quantized) * 16}
