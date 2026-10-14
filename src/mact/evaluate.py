"""Explicit checkpoint evaluation or unlabeled prediction; never selects on Test."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .common.cache import open_cache, file_hash
from .common.data import load_examples
from .data import load_labeled, SequenceCollator
from .labels import get_tags, aspects_to_tags
from .runtime import load_checkpoint, evaluate_model, write_json
from .train import make_dataset


def run_evaluation(checkpoint, data, output, device='cpu', batch_size=1,
                   force_null=False, predict_only=False, visual_cache=None, resource_paths=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('evaluation output must be fresh')
    model, metadata = load_checkpoint(checkpoint, device)
    cfg = dict(metadata['run_config'])
    label_scheme = model.config.label_scheme
    tags = get_tags(label_scheme)
    cfg['model'] = {**cfg.get('model', {}), 'label_scheme': label_scheme}
    if resource_paths:
        if set(resource_paths) - {'vision_model', 'image_root'}:
            raise ValueError('only vision_model and image_root may be relocated')
        cfg.update(resource_paths)
    torch.set_num_threads(cfg['num_threads'])
    tokenizer = AutoTokenizer.from_pretrained(Path(checkpoint).parent / 'tokenizer', local_files_only=True)
    examples, _ = load_examples(data) if predict_only else load_labeled(data)
    if not examples or batch_size < 1:
        raise ValueError('nonempty input and positive batch size required')
    features, cache_meta = None, None
    cache_path = visual_cache or cfg.get('visual_cache')
    if model.config.mode != 'text':
        features, cache_meta = open_cache(cache_path, cfg['vision_model'], cfg['image_root'], examples)
    loader = DataLoader(make_dataset(examples, tokenizer, cfg, features), batch_size=batch_size,
                        collate_fn=SequenceCollator(tokenizer.pad_token_id))
    metrics, predictions = evaluate_model(model, loader, device, force_null=force_null)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent
    provenance = {
        'checkpoint': str(Path(checkpoint).resolve()), 'checkpoint_sha256': file_hash(checkpoint),
        'data': str(Path(data).resolve()), 'data_sha256': file_hash(data),
        'force_null': force_null, 'predict_only': predict_only, 'tags': list(tags),
        'resource_paths': {key: str(cfg.get(key)) for key in ('vision_model', 'image_root')},
        'examples': len(examples), 'device': str(device), 'batch_size': batch_size,
        'cache_manifest_sha256': file_hash(Path(cache_path) / 'manifest.json') if cache_meta else None,
        'source_sha256': {str(p.relative_to(source)): file_hash(p) for p in sorted(source.rglob('*.py'))},
    }
    if label_scheme == 'seven':
        provenance['label_scheme'] = label_scheme
    if cfg.get('ocr_cache'):
        provenance['ocr']=dict(cache=cfg['ocr_cache'],manifest_sha256=cfg['ocr_manifest_sha256'],
                               max_tokens=cfg['ocr_max_tokens'],supervised_positions='original_words_only')
    write_json(output / 'provenance.json', provenance)
    write_json(output / 'predictions.json', predictions)
    write_json(output / 'word_tags.json', [
        {'id': ex.id, 'words': ex.words,
         'tags': [tags[t] for t in aspects_to_tags(len(ex.words), predictions[ex.id], label_scheme)],
         'aspects': predictions[ex.id]} for ex in examples])
    if not predict_only:
        write_json(output / 'metrics.json', metrics)
    return None if predict_only else metrics, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'data', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--visual-cache')
    parser.add_argument('--force-null', action='store_true')
    parser.add_argument('--predict-only', action='store_true')
    args = parser.parse_args()
    run_evaluation(**vars(args))


if __name__ == '__main__':
    main()
