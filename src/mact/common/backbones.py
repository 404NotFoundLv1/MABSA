"""Native encoder loaders for this pinned, isolated backbone experiment."""
from transformers import AutoConfig, AutoModel, CLIPVisionConfig, CLIPVisionModel


def load_text(path, local_files_only=True):
    config = AutoConfig.from_pretrained(path, local_files_only=local_files_only)
    if config.model_type == 'modernbert':
        config.reference_compile = False
    return AutoModel.from_pretrained(path, config=config, local_files_only=local_files_only,
                                     attn_implementation='eager')


def text_from_config(raw, attention='eager'):
    raw = dict(raw)
    kind = raw.pop('model_type')
    config = AutoConfig.for_model(kind, **raw)
    if kind == 'modernbert': config.reference_compile = False
    return AutoModel.from_config(config, attn_implementation=attention)


def load_vision(path, local_files_only=True):
    config = AutoConfig.from_pretrained(path, local_files_only=local_files_only)
    if config.model_type in ('clip', 'clip_vision_model'):
        return CLIPVisionModel.from_pretrained(path, local_files_only=local_files_only,
                                               attn_implementation='eager')
    if config.model_type in ('siglip', 'siglip_vision_model'):
        from transformers import SiglipVisionModel
        return SiglipVisionModel.from_pretrained(path, local_files_only=local_files_only,
                                                 attn_implementation='eager')
    raise ValueError(f'unsupported vision type: {config.model_type}')


def vision_from_config(raw, attention='eager'):
    kind = raw.get('model_type', 'clip_vision_model')
    if kind == 'clip_vision_model':
        config = CLIPVisionConfig.from_dict(raw)
        config._attn_implementation = attention
        return CLIPVisionModel(config)
    if kind == 'siglip_vision_model':
        from transformers import SiglipVisionConfig, SiglipVisionModel
        config = SiglipVisionConfig.from_dict(raw)
        config._attn_implementation = attention
        return SiglipVisionModel(config)
    raise ValueError(f'unsupported saved vision type: {kind}')
