"""Optional parameter freezing; preserve the existing forward and dropout behavior."""
import hashlib
import math

import torch


def optimizer_parameter_groups(model, learning_rate, head_learning_rate,
                               weight_decay, layer_lr_decay=1.0):
    """Preserve legacy grouping at 1; otherwise assign explicit DeBERTa depths."""
    if (type(layer_lr_decay) not in (int, float) or not math.isfinite(layer_lr_decay)
            or not 0 < layer_lr_decay <= 1):
        raise ValueError('invalid layer_lr_decay')
    layers = getattr(getattr(model.text_encoder, 'encoder', None), 'layer', None)
    if layer_lr_decay != 1 and not isinstance(layers, torch.nn.ModuleList):
        raise ValueError('LLRD requires encoder.layer')
    groups, records = {}, {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        text, decay = name.startswith('text_encoder.'), parameter.ndim >= 2
        depth = None
        if text and layer_lr_decay != 1:
            local = name[len('text_encoder.'):]
            if local.startswith('encoder.layer.'):
                index = local.split('.')[2]
                if not index.isdigit() or not 0 <= int(index) < len(layers):
                    raise ValueError(f'unrecognized text parameter: {name}')
                depth = len(layers) - 1 - int(index)
            elif local.startswith(('embeddings.', 'encoder.rel_embeddings.', 'encoder.LayerNorm.')):
                depth = len(layers)
            elif local.startswith('encoder.conv.'):
                depth = len(layers) - 1
            else:
                raise ValueError(f'unrecognized text parameter: {name}')
        key = (text, decay, depth)
        lr = learning_rate * layer_lr_decay ** (depth or 0) if text else head_learning_rate
        if key not in groups:
            groups[key] = dict(params=[], lr=lr, weight_decay=weight_decay if decay else 0.)
            records[key] = dict(names=[], numel=0, lr=lr, weight_decay=weight_decay if decay else 0.)
        groups[key]['params'].append(parameter)
        records[key]['names'].append(name)
        records[key]['numel'] += parameter.numel()
    return list(groups.values()), list(records.values())


def freeze_text_layers(model, count):
    if type(count) is not int or count < 0:
        raise ValueError('text_freeze_layers must be a nonnegative integer')
    if count == 0:
        return None
    text = model.text_encoder
    encoder = getattr(text, 'encoder', None)
    layers = getattr(encoder, 'layer', None)
    embeddings = getattr(text, 'embeddings', None)
    if not isinstance(layers, torch.nn.ModuleList) or not isinstance(embeddings, torch.nn.Module):
        raise ValueError('text backbone must expose embeddings and encoder.layer')
    if count >= len(layers):
        raise ValueError('text_freeze_layers must leave at least one upper layer trainable')
    modules = [embeddings, *layers[:count]]
    # DeBERTa shares relative positions across all layers. Its optional convolution
    # follows the first layer; freeze it too so lower-layer parameters remain fixed.
    shared = []
    for name in ('rel_embeddings', 'LayerNorm', 'conv'):
        module = getattr(encoder, name, None)
        if module is not None:
            if not isinstance(module, torch.nn.Module):
                raise ValueError(f'unsupported encoder.{name}')
            modules.append(module)
            shared.append('encoder.' + name)
    selected = {id(p) for module in modules for p in module.parameters()}
    names = []
    for name, parameter in model.named_parameters():
        if id(parameter) in selected:
            parameter.requires_grad_(False)
            names.append(name)
    return dict(bottom_layers=count, total_layers=len(layers), frozen_names=names,
                shared_modules=shared, dropout='unchanged_train_mode',
                frozen_parameters=sum(p.numel() for p in model.parameters() if id(p) in selected),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))


def parameter_hash(model, names):
    parameters = dict(model.named_parameters())
    digest = hashlib.sha256()
    for name in sorted(names):
        value = parameters[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
