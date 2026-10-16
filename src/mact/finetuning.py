"""Optional parameter freezing; preserve the existing forward and dropout behavior."""
import hashlib

import torch


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
