"""Read-only, provenance-checked raw frozen visual tokens."""
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def model_hashes(path):
    path = Path(path)
    files = sorted(set(path.glob('*.safetensors')) | set(path.glob('*.bin'))
                   | {path/'config.json', path/'preprocessor_config.json'})
    return {p.name:file_hash(p) for p in files if p.is_file()}


def open_cache(path, vision_model, image_root, examples):
    path = Path(path)
    meta = json.loads((path/'manifest.json').read_text())
    def identity(value):
        value = Path(value)
        return (value if value.is_absolute() else path / value).resolve()
    if (meta.get('format_version') != 1 or
        identity(meta['vision_model']) != Path(vision_model).resolve() or
        identity(meta['image_root']) != Path(image_root).resolve()):
        raise ValueError('cache identity mismatch')
    if meta['model_sha256'] != model_hashes(vision_model):
        raise ValueError('cache backbone hash mismatch')
    if file_hash(path/'features.pt') != meta['features_sha256']:
        raise ValueError('cache feature hash mismatch')
    for image in {e.image_id for e in examples}:
        if image not in meta['image_sha256']:
            raise ValueError(f'cache missing image: {image}')
        if file_hash(Path(image_root)/image) != meta['image_sha256'][image]:
            raise ValueError(f'cache image hash mismatch: {image}')
    features = torch.load(path/'features.pt', map_location='cpu', weights_only=True)
    for image in {e.image_id for e in examples}:
        value = features.get(image)
        if (not isinstance(value, torch.Tensor) or list(value.shape) != meta['token_shape']
                or value.dtype != torch.float32 or value.requires_grad
                or not torch.isfinite(value).all()):
            raise ValueError(f'invalid raw feature: {image}')
    return features, meta


class CachedVisionDataset(Dataset):
    def __init__(self, base, features):
        if base.image_processor is not None:
            raise ValueError('cached dataset must not process images online')
        self.base, self.features = base, features

    def __len__(self): return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        item['visual_features'] = self.features[item['example'].image_id]
        return item
