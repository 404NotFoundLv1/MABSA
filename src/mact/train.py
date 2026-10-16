"""Train word-tag MABSA; select exclusively by Dev exact joint F1."""

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
from .labels import get_tags
from .logging import TrainingLogger
from .model import SequenceModel, ModelConfig
from .ocr import open_ocr
from .runtime import train_epoch, evaluate_model, save_checkpoint, write_json
from .finetuning import freeze_text_layers, parameter_hash, optimizer_parameter_groups


DEFAULTS = dict(seed=42, epochs=10, batch_size=1, eval_batch_size=1, accumulation=8,
                learning_rate=2e-5, head_learning_rate=1e-4, weight_decay=.01,
                warmup_ratio=.1, gradient_clip=1., max_length=256, device='cuda:0',
                num_workers=0, num_threads=4, log_every=50, train_limit=None, dev_limit=None,
                max_updates=None, text_model=None, vision_model=None, visual_cache=None,
                train_data=None, dev_data=None, image_root=None, model={}, rdrop_weight=0.,
                ocr_cache=None, ocr_max_tokens=64, ocr_manifest_sha256=None, text_freeze_layers=0,
                layer_lr_decay=1., ema_decay=0.)


def resolve_config(raw):
    if not isinstance(raw, dict) or set(raw) - DEFAULTS.keys():
        raise ValueError(f'unknown configuration keys: {set(raw) - DEFAULTS.keys() if isinstance(raw, dict) else raw}')
    cfg = {**DEFAULTS, **raw}
    try:
        cfg['model'] = asdict(ModelConfig(**cfg['model']))
    except (TypeError, ValueError) as error:
        raise ValueError(f'invalid model configuration: {error}') from error
    for key in ('epochs', 'batch_size', 'eval_batch_size', 'accumulation', 'max_length', 'num_threads', 'ocr_max_tokens'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be positive integer')
    for key in ('seed', 'num_workers', 'log_every', 'text_freeze_layers'):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f'{key} must be nonnegative integer')
    for key in ('train_limit', 'dev_limit', 'max_updates'):
        if cfg[key] is not None and (type(cfg[key]) is not int or cfg[key] < 1):
            raise ValueError(f'{key} must be null or positive integer')
    for key in ('learning_rate', 'head_learning_rate', 'weight_decay', 'gradient_clip', 'warmup_ratio', 'rdrop_weight'):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f'invalid {key}')
    if not 0 <= cfg['warmup_ratio'] < 1 or min(cfg['learning_rate'], cfg['head_learning_rate']) <= 0:
        raise ValueError('invalid learning rate or warmup ratio')
    if cfg['rdrop_weight'] > 0:
        m = cfg['model']
        if (m['mode'] not in ('text', 'fusion') or m['label_scheme'] != 'five'
                or m['aspect_sentiment_weight'] or m['visual_gate'] or m['word_context']):
            raise ValueError('R-Drop must be an independent five-tag baseline training control')
    if cfg['ocr_cache'] is not None:
        m=cfg['model']
        if (not isinstance(cfg['ocr_cache'],str) or not cfg['ocr_cache']
                or cfg['max_length']+cfg['ocr_max_tokens']+1>512
                or m['mode']!='fusion' or m['label_scheme']!='five' or cfg['rdrop_weight']
                or m['aspect_sentiment_weight'] or m['visual_gate'] or m['word_context']):
            raise ValueError('OCR requires an independent five-tag fusion baseline and total token budget <=512')
    if cfg['text_freeze_layers']:
        m = cfg['model']
        if (m['mode'] not in ('text', 'fusion') or m['label_scheme'] != 'five'
                or cfg['rdrop_weight'] or cfg['ocr_cache'] is not None
                or m['aspect_sentiment_weight'] or m['visual_gate'] or m['word_context']):
            raise ValueError('partial freezing must be an independent five-tag baseline control')
    for key in ('layer_lr_decay', 'ema_decay'):
        value = cfg[key]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not (0 < value <= 1 if key == 'layer_lr_decay' else 0 <= value < 1)):
            raise ValueError(f'invalid {key}')
    if cfg['layer_lr_decay'] != 1 or cfg['ema_decay']:
        m = cfg['model']
        if (m['mode'] not in ('text', 'fusion') or m['label_scheme'] != 'five'
                or cfg['rdrop_weight'] or cfg['ocr_cache'] is not None or cfg['text_freeze_layers']
                or m['aspect_sentiment_weight'] or m['visual_gate'] or m['word_context']):
            raise ValueError('LLRD/EMA require independent five-tag baseline training')
    return cfg


def make_dataset(examples, tokenizer, cfg, features):
    texts,ocr_meta = (open_ocr(cfg['ocr_cache'],cfg['image_root'],examples,cfg.get('ocr_manifest_sha256'))
                      if cfg.get('ocr_cache') else (None,None))
    dataset = SequenceDataset(examples, tokenizer, max_length=cfg['max_length'],
                              label_scheme=cfg.get('model', {}).get('label_scheme', 'five'),
                              ocr_texts=texts,ocr_max_tokens=cfg.get('ocr_max_tokens',64))
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
    ocr_meta=None
    if cfg['ocr_cache']:
        _,ocr_meta=open_ocr(cfg['ocr_cache'],cfg['image_root'],train+dev,cfg['ocr_manifest_sha256'])
        cfg['ocr_manifest_sha256']=ocr_meta['manifest_sha256']
    train_ds, dev_ds = [make_dataset(examples, tokenizer, cfg, features) for examples in (train, dev)]
    collator = SequenceCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              generator=torch.Generator().manual_seed(cfg['seed']),
                              collate_fn=collator, num_workers=cfg['num_workers'])
    dev_loader = DataLoader(dev_ds, batch_size=cfg['eval_batch_size'], shuffle=False,
                            collate_fn=collator, num_workers=cfg['num_workers'])
    model = SequenceModel(load_text(cfg['text_model']), ModelConfig(**cfg['model'])).to(device)
    freeze_policy = freeze_text_layers(model, cfg['text_freeze_layers'])
    frozen_before = parameter_hash(model, freeze_policy['frozen_names']) if freeze_policy else None
    groups, group_records = optimizer_parameter_groups(model, cfg['learning_rate'],
        cfg['head_learning_rate'], cfg['weight_decay'], cfg['layer_lr_decay'])
    optimizer = torch.optim.AdamW(groups)
    planned = math.ceil(len(train_loader) / cfg['accumulation']) * cfg['epochs']
    if cfg['max_updates'] is not None:
        planned = min(planned, cfg['max_updates'])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(planned * cfg['warmup_ratio']), planned)
    output.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(output)
    label_scheme = model.config.label_scheme
    logger.info(f"MABSA {label_scheme}-tag CRF | {Path(cfg['train_data']).parent.name}", pad=True)
    for key in ('seed', 'epochs', 'batch_size', 'accumulation', 'learning_rate',
                'head_learning_rate', 'warmup_ratio', 'device'):
        logger.info(f'{key}: {cfg[key]}')
    logger.info(f"model: {cfg['model']['mode']}; selection: Dev Joint F1; metrics: percent")
    if freeze_policy:
        logger.info(f"Text freeze: embeddings + bottom {freeze_policy['bottom_layers']}/{freeze_policy['total_layers']} layers "
                    f"+ {freeze_policy['shared_modules']}; dropout unchanged.")
        logger.info(f"Frozen parameters: {freeze_policy['frozen_parameters']} | Trainable parameters: {freeze_policy['trainable_parameters']}")
    if ocr_meta:
        logger.info(f"OCR context: up to {cfg['ocr_max_tokens']} subwords; labels and CRF decode on original words only.")
    if cfg['model']['visual_gate']:
        logger.info(f"Visual gate: per-word null-anchored residual; initial gate: {cfg['model']['visual_gate_init']:g}")
    if cfg['model']['word_context']:
        logger.info(f"Word context: fused-word attention; size:{cfg['model']['word_context_size']} "
                    f"heads:{cfg['model']['word_context_heads']} distance:{cfg['model']['word_context_max_distance']}; "
                    'zero-initialized five-score residual.')
    auxiliary_weight = cfg['model']['aspect_sentiment_weight']
    if cfg['rdrop_weight'] > 0:
        logger.info(f"Loss = mean(two CRF NLLs) + {cfg['rdrop_weight']:g} * symmetric sequence KL; sentence mean.")
        logger.info('R-Drop: shared whole-image dropout, independent neural dropout; two training passes, one inference pass.')
    elif auxiliary_weight > 0:
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
        'optimizer_groups': group_records,
        'is_smoke': is_smoke, 'tags': list(get_tags(label_scheme)), 'selection_metric': 'dev_joint_f1',
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
    if ocr_meta:
        provenance['ocr']=dict(**ocr_meta,max_tokens=cfg['ocr_max_tokens'],added_parameters=0,
                               supervised_positions='original_words_only')
    if freeze_policy:
        provenance['text_freeze'] = dict(**freeze_policy, initial_frozen_sha256=frozen_before,
            initialization='original_pretrained_text_backbone_and_seeded_task_heads')
    if cfg['rdrop_weight'] > 0:
        provenance.update(objective='sentence_mean_two_pass_crf_nll_plus_symmetric_sequence_kl',
            rdrop=dict(weight=cfg['rdrop_weight'], kl='0.5*(KL(P||Q)+KL(Q||P))',
                distribution='constrained_crf_sequences', nll_reduction='mean_of_two_passes',
                length_normalization=False, marginal_gradients=True, shared_visual_dropout=True,
                neural_passes_per_train_batch=2, inference_passes=1, added_parameters=0))
    if label_scheme == 'seven':
        provenance.update(label_scheme=label_scheme,
                          initialization_strategy='five_tag_inside_parameter_expansion')
    if auxiliary_weight > 0:
        provenance.update(objective='sentence_mean_crf_nll_plus_aspect_sentiment_ce',
                          aspect_sentiment_weight=auxiliary_weight,
                          aspect_pooling='mean_projected_text_words',
                          aspect_loss_reduction='mean_aspects_per_sentence_empty_zero',
                          auxiliary_labels='train_only', decoding=f'{label_scheme}_tag_crf')
    if cfg['model']['visual_gate']:
        provenance['visual_gate'] = dict(formula='E_null + sigmoid(linear([word, attention_image-attention_null])) * (E_image-E_null)',
            initial_value=cfg['model']['visual_gate_init'], shared_text_and_visual_head=True,
            supervision='existing_crf_objective', uses_gold_inputs=False)
    if cfg['model']['word_context']:
        provenance['word_context'] = dict(feature_source='visual_head_hidden',
            size=cfg['model']['word_context_size'], heads=cfg['model']['word_context_heads'],
            max_distance=cfg['model']['word_context_max_distance'],
            relative_bias='learned_signed_clipped_key_minus_query',
            relative_bias_initialization='-abs(clipped_distance)/max_distance',
            output_initialization='zeros', initialization_preserves_baseline_rng=True,
            supervision='existing_crf_objective', uses_gold_inputs=False, decoding='five_tag_crf')
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
                             epoch=epoch, epochs=cfg['epochs'], rdrop_weight=cfg['rdrop_weight'])
        result['epoch_complete'] = result['examples'] == len(train)
        updates += result['updates']
        metrics, predictions = evaluate_model(model, dev_loader, device)
        record = {'epoch': epoch, 'train': result, 'dev': metrics, 'optimizer_updates': updates,
                  'elapsed_seconds': time.monotonic() - started}
        history.append(record)
        write_json(output / 'history.json', history)
        logger.event({'event': 'epoch', **record})
        if cfg['rdrop_weight'] > 0:
            logger.info(f"Epoch [{epoch}/{cfg['epochs']}], Mean Loss: {result['sentence_loss']:.4f}, "
                        f"CRF NLL: {result['sentence_nll']:.4f}, Symmetric KL: {result['rdrop_symmetric_kl']:.4f}, "
                        f"KL Weight: {cfg['rdrop_weight']:g}")
        elif auxiliary_weight > 0:
            logger.info(f"Epoch [{epoch}/{cfg['epochs']}], Mean Loss: {result['sentence_loss']:.4f}, "
                        f"CRF NLL: {result['sentence_nll']:.4f}, Aspect CE: {result['aspect_sentiment_loss']:.4f}")
        else:
            logger.info(f"Epoch [{epoch}/{cfg['epochs']}], Mean Loss: {result['sentence_nll']:.4f}")
        if 'visual_gate' in result:
            stats = result['visual_gate']
            logger.info(f"TRAIN Gate mean:{stats['mean']:.4f} std:{stats['std']:.4f} "
                        f"below_0.1:{stats['fraction_below_0_1'] * 100:.2f}% "
                        f"above_0.9:{stats['fraction_above_0_9'] * 100:.2f}%")
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
    if freeze_policy and parameter_hash(model, freeze_policy['frozen_names']) != frozen_before:
        raise ValueError('frozen text parameters changed during training')
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
    if freeze_policy:
        summary['text_freeze'] = dict(bottom_layers=freeze_policy['bottom_layers'],
            frozen_exact=True, frozen_sha256=frozen_before,
            frozen_parameters=freeze_policy['frozen_parameters'], dropout=freeze_policy['dropout'])
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
