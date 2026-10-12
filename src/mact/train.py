"""Train five-tag MABSA; select exclusively by Dev exact joint F1."""

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .common.backbones import load_text
from .common.cache import CachedVisionDataset, open_cache, file_hash, model_hashes
from .data import SequenceDataset, SequenceCollator, load_labeled
from .labels import TAGS
from .logging import TrainingLogger
from .model import SequenceModel, ModelConfig
from .runtime import train_epoch, evaluate_model, save_checkpoint, write_json


DEFAULTS = dict(seed=42, epochs=10, batch_size=1, eval_batch_size=1, accumulation=8,
                learning_rate=2e-5, head_learning_rate=1e-4, weight_decay=.01,
                warmup_ratio=.1, gradient_clip=1., max_length=256, device='cuda:0',
                num_workers=0, num_threads=4, log_every=50, train_limit=None, dev_limit=None,
                max_updates=None, text_model=None, vision_model=None, visual_cache=None,
                train_data=None, dev_data=None, image_root=None, model={})


def resolve_config(raw):
    if not isinstance(raw, dict) or set(raw) - DEFAULTS.keys():
        raise ValueError(f'unknown configuration keys: {set(raw) - DEFAULTS.keys() if isinstance(raw, dict) else raw}')
    cfg = {**DEFAULTS, **raw}
    try:
        cfg['model'] = asdict(ModelConfig(**cfg['model']))
    except (TypeError, ValueError) as error:
        raise ValueError(f'invalid model configuration: {error}') from error
    for key in ('epochs', 'batch_size', 'eval_batch_size', 'accumulation', 'max_length', 'num_threads'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be positive integer')
    for key in ('seed', 'num_workers', 'log_every'):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f'{key} must be nonnegative integer')
    for key in ('train_limit', 'dev_limit', 'max_updates'):
        if cfg[key] is not None and (type(cfg[key]) is not int or cfg[key] < 1):
            raise ValueError(f'{key} must be null or positive integer')
    for key in ('learning_rate', 'head_learning_rate', 'weight_decay', 'gradient_clip', 'warmup_ratio'):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f'invalid {key}')
    if not 0 <= cfg['warmup_ratio'] < 1 or min(cfg['learning_rate'], cfg['head_learning_rate']) <= 0:
        raise ValueError('invalid learning rate or warmup ratio')
    return cfg


def make_dataset(examples, tokenizer, cfg, features):
    dataset = SequenceDataset(examples, tokenizer, max_length=cfg['max_length'])
    return CachedVisionDataset(dataset, features) if features is not None else dataset


def run_training(raw, output):
    cfg = resolve_config(raw)
    started = time.monotonic()
    for name in ('text_model', 'train_data', 'dev_data'):
        if not cfg[name]:
            raise ValueError(f'{name} is required')
    if Path(cfg['train_data']).resolve() == Path(cfg['dev_data']).resolve():
        raise ValueError('Train and Dev must be different files')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be fresh')
    train, train_audit = load_labeled(cfg['train_data'])
    dev, dev_audit = load_labeled(cfg['dev_data'])
    if file_hash(cfg['train_data']) == file_hash(cfg['dev_data']):
        raise ValueError('Train and Dev have identical content')
    train, dev = train[:cfg['train_limit']], dev[:cfg['dev_limit']]
    torch.set_num_threads(cfg['num_threads'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing a silent CPU training fallback')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(cfg['text_model'], use_fast=True, local_files_only=True)
    features, cache_meta = None, None
    if cfg['model']['mode'] != 'text':
        if not all(cfg[key] for key in ('visual_cache', 'vision_model', 'image_root')):
            raise ValueError('multimodal model requires visual_cache, vision_model, image_root')
        features, cache_meta = open_cache(cfg['visual_cache'], cfg['vision_model'], cfg['image_root'], train + dev)
        if cache_meta['token_shape'][-1] != cfg['model']['visual_size']:
            raise ValueError('visual size does not match verified cache')
    train_ds, dev_ds = [make_dataset(examples, tokenizer, cfg, features) for examples in (train, dev)]
    collator = SequenceCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              generator=torch.Generator().manual_seed(cfg['seed']),
                              collate_fn=collator, num_workers=cfg['num_workers'])
    dev_loader = DataLoader(dev_ds, batch_size=cfg['eval_batch_size'], shuffle=False,
                            collate_fn=collator, num_workers=cfg['num_workers'])
    model = SequenceModel(load_text(cfg['text_model']), ModelConfig(**cfg['model'])).to(device)
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups.setdefault((name.startswith('text_encoder.'), parameter.ndim >= 2), []).append(parameter)
    optimizer = torch.optim.AdamW([{'params': params,
        'lr': cfg['learning_rate'] if backbone else cfg['head_learning_rate'],
        'weight_decay': cfg['weight_decay'] if decay else 0.}
        for (backbone, decay), params in groups.items()])
    planned = math.ceil(len(train_loader) / cfg['accumulation']) * cfg['epochs']
    if cfg['max_updates'] is not None:
        planned = min(planned, cfg['max_updates'])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(planned * cfg['warmup_ratio']), planned)
    output.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(output)
    logger.info(f"MABSA five-tag CRF | {Path(cfg['train_data']).parent.name}", pad=True)
    for key in ('seed', 'epochs', 'batch_size', 'accumulation', 'learning_rate',
                'head_learning_rate', 'warmup_ratio', 'device'):
        logger.info(f'{key}: {cfg[key]}')
    logger.info(f"model: {cfg['model']['mode']}; selection: Dev Joint F1; metrics: percent")
    auxiliary_weight = cfg['model']['aspect_sentiment_weight']
    if auxiliary_weight > 0:
        logger.info(f'Loss = CRF NLL + {auxiliary_weight:g} * Aspect CE; sentence mean.')
        logger.info('Aspect CE: mean over gold training aspects per sentence; zero for empty sentences.')
    else:
        logger.info('Step counts training mini-batches; Loss is mean CRF NLL per sentence.')
    logger.info(f'Train examples: {len(train)} | Dev examples: {len(dev)} | Optimizer updates: {planned}')
    write_json(output / 'config.json', cfg)
    tokenizer.save_pretrained(output / 'tokenizer')
    source = Path(__file__).parent
    is_smoke = any(cfg[k] is not None for k in ('train_limit', 'dev_limit', 'max_updates'))
    provenance = {
        'is_smoke': is_smoke, 'tags': list(TAGS), 'selection_metric': 'dev_joint_f1',
        'precision': 'fp32', 'objective': 'sentence_mean_crf_nll', 'vision_frozen': True,
        'torch': str(torch.__version__), 'transformers': transformers.__version__,
        'train_sha256': file_hash(cfg['train_data']), 'dev_sha256': file_hash(cfg['dev_data']),
        'train_audit': train_audit, 'dev_audit': dev_audit,
        'used_train_examples': len(train), 'used_dev_examples': len(dev),
        'text_model_sha256': model_hashes(cfg['text_model']),
        'cache_manifest_sha256': file_hash(Path(cfg['visual_cache']) / 'manifest.json') if cache_meta else None,
        'cache_features_sha256': cache_meta['features_sha256'] if cache_meta else None,
        'source_sha256': {str(p.relative_to(source)): file_hash(p) for p in sorted(source.rglob('*.py'))},
    }
    if auxiliary_weight > 0:
        provenance.update(objective='sentence_mean_crf_nll_plus_aspect_sentiment_ce',
                          aspect_sentiment_weight=auxiliary_weight,
                          aspect_pooling='mean_projected_text_words',
                          aspect_loss_reduction='mean_aspects_per_sentence_empty_zero',
                          auxiliary_labels='train_only', decoding='five_tag_crf')
    write_json(output / 'provenance.json', provenance)
    initial_head = model.text_head.weight.detach().clone()
    history, best, updates = [], -1., 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    logger.event({'event': 'ready', 'train_examples': len(train), 'dev_examples': len(dev),
                  'planned_updates': planned, 'mode': cfg['model']['mode'], 'is_smoke': is_smoke})
    for epoch in range(1, cfg['epochs'] + 1):
        logger.info(f'Epoch {epoch}', pad=True)
        result = train_epoch(model, train_loader, optimizer, device, cfg['accumulation'], cfg['gradient_clip'],
                             scheduler, planned - updates, cfg['log_every'], logger=logger,
                             epoch=epoch, epochs=cfg['epochs'])
        result['epoch_complete'] = result['examples'] == len(train)
        updates += result['updates']
        metrics, predictions = evaluate_model(model, dev_loader, device)
        record = {'epoch': epoch, 'train': result, 'dev': metrics, 'optimizer_updates': updates,
                  'elapsed_seconds': time.monotonic() - started}
        history.append(record)
        write_json(output / 'history.json', history)
        logger.event({'event': 'epoch', **record})
        if auxiliary_weight > 0:
            logger.info(f"Epoch [{epoch}/{cfg['epochs']}], Mean Loss: {result['sentence_loss']:.4f}, "
                        f"CRF NLL: {result['sentence_nll']:.4f}, Aspect CE: {result['aspect_sentiment_loss']:.4f}")
        else:
            logger.info(f"Epoch [{epoch}/{cfg['epochs']}], Mean Loss: {result['sentence_nll']:.4f}")
        logger.joint_metrics('DEV', metrics['joint'])
        logger.info(f"DEV AE  ae_p:{metrics['mate']['precision'] * 100:.2f} "
                    f"ae_r:{metrics['mate']['recall'] * 100:.2f} ae_f:{metrics['mate']['f1'] * 100:.2f}")
        if metrics['joint']['f1'] > best:
            best, best_epoch, best_metrics = metrics['joint']['f1'], epoch, metrics
            save_checkpoint(output / 'best.pt', model, cfg, epoch, metrics)
            write_json(output / 'best_dev_predictions.json', predictions)
            logger.info(f'Saved best Dev checkpoint at epoch {epoch}.')
        if updates >= planned:
            break
    save_checkpoint(output / 'last.pt', model, cfg, epoch, metrics)
    summary = {
        'best_epoch': best_epoch, 'best_dev': best_metrics, 'last_epoch': epoch, 'last_dev': metrics,
        'optimizer_updates': updates, 'planned_updates': planned, 'is_smoke': is_smoke,
        'completed_epochs': sum(r['train']['epoch_complete'] for r in history),
        'evaluated_epochs': len(history), 'selection_metric': 'dev_joint_f1',
        'elapsed_seconds': time.monotonic() - started,
        'checkpoint_sha256': file_hash(output / 'best.pt'),
        'emission_head_delta_l2': float((model.text_head.weight.detach() - initial_head).norm()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
    }
    write_json(output / 'summary.json', summary)
    logger.info(f'Training complete in: {summary["elapsed_seconds"] / 60:.2f} minutes', pad=True)
    logger.info('BEST DEV:-----')
    logger.info(f'Best epoch: {best_epoch}')
    logger.joint_metrics('BEST DEV', best_metrics['joint'])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device')
    parser.add_argument('--mode', choices=['text', 'fusion', 'evidence'])
    for name in ('seed', 'epochs', 'train-limit', 'dev-limit', 'max-updates'):
        parser.add_argument('--' + name, type=int)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    for key in ('device', 'seed', 'epochs', 'train_limit', 'dev_limit', 'max_updates'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.mode:
        cfg.setdefault('model', {})['mode'] = args.mode
    run_training(cfg, args.output)


if __name__ == '__main__':
    main()
