"""Single-decoder-block NLL sensitivity with per-observation durable resume.

All non-probed blocks remain at reference precision. Input texts must be FLORES devtest;
calibration batches (if GPTQ) must come from FLORES dev. Runtime is a GPU acceptance gate.
"""
import json
from pathlib import Path

import torch

from whopays.config import stable_hash
from whopays.env import env_info
from whopays.quant.fakequant import activation_hessian, capture_block_inputs, linear_layers
from whopays.quant.gptq import quantize_weight
from whopays.quant.rtn import quantize
from whopays.tasks.flores_nll import nll_batch


def _score(bundle, texts, batch_size):
    loss, count = 0.0, 0
    for start in range(0, len(texts), batch_size):
        for nll, tokens in nll_batch(bundle, texts[start:start + batch_size]):
            loss += nll
            count += tokens
    if count == 0:
        raise ValueError('profile language has no scored tokens')
    return loss / count


def _model_identity(bundle):
    """Architecture and source id, so a reused run_id cannot mix two models."""
    config = bundle.model.config
    return {
        'model_type': getattr(config, 'model_type', None),
        'name_or_path': getattr(config, 'name_or_path', None),
        'hidden_size': getattr(config, 'hidden_size', None),
        'num_hidden_layers': getattr(config, 'num_hidden_layers', None),
        'vocab_size': getattr(config, 'vocab_size', None),
    }


@torch.no_grad()
def profile(bundle, texts_by_language, bits=(2, 3, 4, 8), *, method='rtn',
            calibration=None, group_size=128, batch_size=8, output_path=None, run_id=None):
    """Return long-form layer × bits × language rows suitable for allocation.

    run_id identifies the reference model/config and is required for persistence. Resume
    verifies settings, texts, model identity and the pinned code/GPU/library environment,
    and trims an interrupted final JSON line. A mismatch refuses the resume instead of
    appending rows from another environment.
    """
    if method not in ('rtn', 'gptq') or (method == 'gptq' and not calibration):
        raise ValueError('expected rtn or gptq with calibration batches')
    if not bits or not set(bits) <= {2, 3, 4, 8} or batch_size < 1 or group_size < 1:
        raise ValueError('invalid bits, batch_size or group_size')
    if not texts_by_language or any(not t for t in texts_by_language.values()):
        raise ValueError('each language needs nonempty evaluation texts')
    layers = linear_layers(bundle.model)
    groups = {}
    for name in layers:
        # Qwen decoder block names are model.layers.N.
        parts = name.split('.')
        if len(parts) < 4 or parts[1] != 'layers':
            raise ValueError(f'unsupported decoder projection: {name}')
        groups.setdefault('.'.join(parts[:3]), []).append(name)
    pinned = {key: env_info()[key]
              for key in ('code_version', 'gpu', 'torch', 'transformers', 'cuda')}
    manifest = {'run_id': run_id, 'texts': stable_hash(texts_by_language), 'bits': list(bits),
                'method': method, 'group_size': group_size, 'batch_size': batch_size,
                'layers': list(groups), 'device': bundle.device, 'dtype': str(bundle.dtype),
                'env': pinned, 'model': _model_identity(bundle),
                'calibration': stable_hash([{k: v.tolist() for k, v in batch.items()}
                                             for batch in (calibration or [])])}
    rows = []
    path = Path(output_path) if output_path else None
    if path:
        if not run_id:
            raise ValueError('run_id is required to persist a profile')
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = path.with_suffix('.manifest.json')
        if meta.exists() and json.loads(meta.read_text()) != manifest:
            raise ValueError('profile manifest differs; choose a new output path')
        if path.exists() and not meta.exists():
            raise ValueError('existing profile has no manifest')
        meta.write_text(json.dumps(manifest, sort_keys=True) + '\n')
        if path.exists():
            data = path.read_bytes()
            end = data.rfind(b'\n') + 1
            if end != len(data):
                path.write_bytes(data[:end])
            rows = [json.loads(line) for line in data[:end].splitlines()]
    done = {(r['layer'], r['bits'], r['lang']) for r in rows}
    expected = {(layer, bit, lang) for layer in groups for bit in bits
                for lang in texts_by_language}
    if done == expected:
        return rows
    baseline = {lang: _score(bundle, texts, batch_size)
                for lang, texts in texts_by_language.items()}
    for layer, names in groups.items():
        originals = {name: layers[name].weight.detach().cpu().clone() for name in names}
        block = bundle.model.get_submodule(layer)
        captured = (capture_block_inputs(bundle.model, block, calibration)
                    if method == 'gptq' else None)
        try:
            for bit in bits:
                pending = [lang for lang in texts_by_language if (layer, bit, lang) not in done]
                if not pending:
                    continue
                for name in names:
                    layers[name].weight.copy_(originals[name])
                for name in names:
                    module = layers[name]
                    if method == 'gptq':
                        h = activation_hessian(bundle.model, module, calibration,
                                               block=block, captured=captured)
                        q = quantize_weight(module.weight, h, bit, group_size)
                        del h
                    else:
                        q = quantize(module.weight, bit, group_size)
                    module.weight.copy_(q)
                for lang in pending:
                    nll = _score(bundle, texts_by_language[lang], batch_size)
                    row = {'run_id': run_id, 'layer': layer, 'bits': bit, 'lang': lang,
                           'baseline_nll': baseline[lang], 'nll': nll,
                           'torch': pinned['torch'], 'transformers': pinned['transformers'],
                           'cuda': pinned['cuda'], 'code_version': pinned['code_version'],
                           'gpu': pinned['gpu'],
                           'sensitivity': nll - baseline[lang],
                           'parameters': sum(layers[n].weight.numel() for n in names)}
                    if path:
                        with path.open('a') as f:
                            f.write(json.dumps(row, sort_keys=True) + '\n')
                            f.flush()
                    rows.append(row)
        finally:
            for name in names:
                layers[name].weight.copy_(originals[name])
    return rows
