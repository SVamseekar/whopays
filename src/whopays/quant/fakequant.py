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


# Projections whose inputs do not depend on each other, quantized after one shared pass, as in
# GPTQ's true-sequential order (and GPTQModel's): attention inputs, attention output, MLP
# inputs, MLP output. Projections not listed get a stage of their own, in module order.
STAGES = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), ("self_attn.o_proj",),
          ("mlp.gate_proj", "mlp.up_proj"), ("mlp.down_proj",))


def _stages(block_name, names):
    local = [n[len(block_name) + 1:] for n in names]
    stages = [[f"{block_name}.{m}" for m in stage if m in local] for stage in STAGES]
    known = {m for stage in STAGES for m in stage}
    stages += [[f"{block_name}.{m}"] for m in local if m not in known]
    return [s for s in stages if s]


def _hessian_hook(hessians, counts, name):
    def hook(module, args):
        x = args[0].detach().reshape(-1, module.in_features).float()
        hessians[name].addmm_(x.T, x)
        counts[name] += len(x)
    return hook


@torch.no_grad()
def _gptq_blocks(model, bitmap, calibration, group_size):
    """Layer-by-layer GPTQ: capture the first block's inputs once, then feed each quantized
    block's outputs to the next (Frantar et al., 2023). Equivalent to recapturing every
    block's inputs from the partly quantized model, at one forward pass instead of L."""
    layers = linear_layers(model)
    blocks: dict[str, list[str]] = {}
    for name in bitmap:
        parts = name.split('.')
        if len(parts) < 4 or not parts[2].isdigit():
            raise ValueError(f'GPTQ supports decoder-block projections only: {name}')
        blocks.setdefault('.'.join(parts[:3]), []).append(name)
    order = sorted(blocks, key=lambda b: int(b.split('.')[2]))
    kinds = {getattr(model.get_submodule(b), 'attention_type', None) for b in order}
    if len(kinds) > 1:
        raise ValueError('blocks with different attention types need per-block masks')
    first = model.get_submodule(order[0])
    device = next(first.parameters()).device
    inputs = [(_move(a, device), _move(k, device))
              for a, k in capture_block_inputs(model, first, calibration)]
    for index, block_name in enumerate(order):
        block = model.get_submodule(block_name)
        if index and int(block_name.split('.')[2]) != int(order[index - 1].split('.')[2]) + 1:
            raise ValueError('GPTQ blocks must be consecutive to propagate inputs')
        for stage in _stages(block_name, blocks[block_name]):
            hessians = {n: torch.zeros(layers[n].in_features, layers[n].in_features,
                                       device=layers[n].weight.device, dtype=torch.float32)
                        for n in stage}
            counts = dict.fromkeys(stage, 0)

            hooks = [layers[n].register_forward_pre_hook(_hessian_hook(hessians, counts, n))
                     for n in stage]
            try:
                for args, kwargs in inputs:
                    block(*args, **kwargs)
            finally:
                for hook in hooks:
                    hook.remove()
            for n in stage:
                if not counts[n]:
                    raise ValueError(f'empty calibration activations for {n}')
                h = hessians.pop(n) / counts[n]
                layers[n].weight.copy_(quantize_weight(layers[n].weight, h, bitmap[n],
                                                       group_size))
                del h
        if index + 1 < len(order):
            outputs = []
            for args, kwargs in inputs:
                out = block(*args, **kwargs)
                hidden = out[0] if isinstance(out, tuple) else out
                outputs.append(((hidden, *args[1:]), kwargs))
            inputs = outputs


@torch.no_grad()
def apply(model, bits, method='rtn', calibration=None, group_size=128):
    bitmap = resolve_bitmap(model, bits)
    if method not in ('rtn', 'gptq'):
        raise ValueError('method must be rtn or gptq')
    if method == 'gptq' and not calibration:
        raise ValueError('GPTQ requires calibration batches')
    layers = linear_layers(model)
    if method == 'gptq':
        _gptq_blocks(model, bitmap, calibration, group_size)
    else:
        for name, bit in bitmap.items():
            layers[name].weight.copy_(quantize(layers[name].weight, bit, group_size))
    quantized = sum(layers[n].weight.numel() for n in bitmap)
    total = sum(p.numel() for p in model.parameters())
    return {'bitmap': bitmap, 'method': method, 'memory_kind': 'computed_weight_bits',
            'weight_bits': sum(layers[n].weight.numel() * b for n, b in bitmap.items())
            + (total - quantized) * 16}
