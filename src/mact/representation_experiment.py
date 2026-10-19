"""Train target representations, select solely on fixed-fusion predicted-span Dev."""
import gc
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .calibrated_fusion import fuse_labels
from .common.backbones import load_text
from .common.cache import file_hash
from .common.metrics import evaluate_predictions
from .data import load_labeled
from .experiment import code_identity
from .fusion_experiment import load_verified, _same, _verify_hashes
from .runtime import write_json
from .target_context import add_markers, relabel
from .target_context_train import pilot_config as legacy_config
from .target_representation import (VARIANTS, RepresentationDataset, RepresentationCollator,
    RepresentationClassifier, save_representation, load_representation)

SELECTION = 'fixed_fusion_predicted_boundary_dev_joint_f1'


def is_better(row, best_f1):
    return row['fusion']['joint']['f1'] > best_f1


def evaluate_scores(cache, logits, gold):
    scores = torch.as_tensor(logits, dtype=torch.float64).reshape(-1, 3)
    if len(scores) != len(cache['keys']) or not torch.isfinite(scores).all():
        raise ValueError('invalid or misaligned target scores')
    old = torch.tensor([p for v in cache['baseline'].values() for _, _, p in v], dtype=torch.long)
    new = scores.argmax(-1)
    fused = fuse_labels(torch.tensor(cache['base_logits']).reshape(-1, 3), scores, old, new, 1., 1., .5)
    keys = [tuple(k) for k in cache['keys']]
    predictions = {name: relabel(cache['baseline'], keys, labels.tolist())
                   for name, labels in [('target', new), ('fusion', fused)]}
    metrics = {name: evaluate_predictions(gold, p) for name, p in predictions.items()}
    return metrics, predictions


@torch.no_grad()
def predict_scores(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        scores = model(**inputs)['logits']
        if not torch.isfinite(scores).all():
            raise FloatingPointError('nonfinite inference scores')
        rows.extend(scores.cpu().tolist())
    return rows


def pilot_config(dataset, variant, score_cache, smoke=False):
    cfg = legacy_config(dataset, smoke)
    cfg.update(dataset=dataset, variant=variant, attention_dim=128,
               score_cache=str(Path(score_cache).resolve()))
    return cfg


def validate_config(cfg):
    allowed = {'text_model', 'train_data', 'dev_data', 'baseline_predictions', 'baseline_summary',
        'baseline_checkpoint', 'seed', 'epochs', 'batch_size', 'eval_batch_size', 'device',
        'num_threads', 'learning_rate', 'head_learning_rate', 'weight_decay', 'warmup_ratio',
        'gradient_clip', 'dropout', 'max_length', 'train_limit', 'dataset', 'variant',
        'attention_dim', 'score_cache'}
    if set(cfg) != allowed:
        raise ValueError(f'unsupported/missing config keys: {set(cfg) ^ allowed}')
    if cfg['dataset'] not in ('twitter2015', 'twitter2017') or cfg['variant'] not in VARIANTS:
        raise ValueError('invalid dataset or variant')
    for key in ('epochs', 'batch_size', 'eval_batch_size', 'num_threads', 'max_length', 'attention_dim'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'invalid {key}')
    if type(cfg['seed']) is not int or cfg['seed'] < 0:
        raise ValueError('invalid seed')
    if cfg['train_limit'] is not None and (type(cfg['train_limit']) is not int or cfg['train_limit'] < 1):
        raise ValueError('invalid train_limit')
    for key in ('learning_rate', 'head_learning_rate', 'gradient_clip'):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'invalid {key}')
    for key in ('dropout', 'warmup_ratio'):
        if not math.isfinite(cfg[key]) or not 0 <= cfg[key] < 1:
            raise ValueError(f'invalid {key}')
    if not math.isfinite(cfg['weight_decay']) or cfg['weight_decay'] < 0:
        raise ValueError('invalid weight_decay')
    for key in ('train_data', 'dev_data'):
        if 'test' in Path(cfg[key]).name.lower():
            raise ValueError('Test data is forbidden')


def _load_inputs(cfg):
    cache = load_verified(cfg['score_cache'], cfg['dataset'], cfg['seed'])
    bound = cache['provenance']['input_sha256']
    for key in ('dev_data', 'baseline_checkpoint', 'baseline_summary', 'baseline_predictions'):
        path = str(Path(cfg[key]).resolve())
        if path not in bound or file_hash(path) != bound[path]:
            raise ValueError(f'config/cache input identity mismatch: {key}')
    train, _ = load_labeled(cfg['train_data'])
    dev, _ = load_labeled(cfg['dev_data'])
    gold = {e.id: e.aspects for e in dev}
    if not _same(gold, cache['gold']):
        raise ValueError('cache gold differs from official Dev')
    baseline_summary = json.loads(Path(cfg['baseline_summary']).read_text())
    if (not _same(cache['baseline'], json.loads(Path(cfg['baseline_predictions']).read_text()))
            or evaluate_predictions(gold, cache['baseline']) != baseline_summary['best_dev']
            or file_hash(cfg['baseline_checkpoint']) != baseline_summary['checkpoint_sha256']):
        raise ValueError('baseline provenance mismatch')
    if file_hash(cfg['train_data']) == file_hash(cfg['dev_data']):
        raise ValueError('Train and Dev must differ')
    return cache, train, dev, gold


def verify_completed(output, cfg):
    output = Path(output)
    status_path = output.with_name(output.name + '.status.json')
    if not status_path.is_file() or json.loads(status_path.read_text())['state'] != 'completed':
        raise ValueError('job is incomplete; refusing reuse')
    if not _same(json.loads((output / 'config.json').read_text()), cfg):
        raise ValueError('completed job config mismatch')
    artifact_hashes = json.loads((output / 'artifact-sha256.json').read_text())
    required = {'best.pt', 'summary.json', 'config.json', 'provenance.json', 'history.json',
        'best_fusion_predictions.json', 'best_target_predictions.json', 'best_scores.json',
        'reload-verification.json'}
    if not required.issubset(artifact_hashes):
        raise ValueError('incomplete artifact manifest')
    for name, sha in artifact_hashes.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file() or file_hash(path) != sha:
            raise ValueError(f'completed artifact changed: {name}')
    provenance = json.loads((output / 'provenance.json').read_text())
    if provenance['code']['source_sha256'] != code_identity()['source_sha256']:
        raise ValueError('source changed since completed run')
    _verify_hashes(provenance['input_sha256'])
    summary = json.loads((output / 'summary.json').read_text())
    if summary['completed_epochs'] != cfg['epochs'] or summary['selection_metric'] != SELECTION:
        raise ValueError('incomplete or incompatible summary')
    return summary


def run_training(config, output):
    cfg = dict(config)
    validate_config(cfg)
    output = Path(output).resolve()
    status_path = output.with_name(output.name + '.status.json')
    if output.exists() or status_path.exists():
        raise FileExistsError('output and status must be fresh')
    output.mkdir(parents=True)
    started = time.time()
    status = dict(state='running', started_at=started, completed_epochs=0, variant=cfg['variant'])
    write_json(status_path, status)
    try:
        identity = code_identity()
        cache, train, dev, gold = _load_inputs(cfg)
        hashes = dict(cache['provenance']['input_sha256'])
        paths = [Path(cfg[k]).resolve() for k in ('train_data', 'score_cache')]
        paths.append(Path(cfg['score_cache']).with_suffix('.sha256').resolve())
        paths += sorted(p for p in Path(cfg['text_model']).resolve().rglob('*') if p.is_file())
        hashes.update({str(p): file_hash(p) for p in paths})
        torch.set_num_threads(cfg['num_threads'])
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(cfg['device'])
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; refusing CPU fallback')
        tokenizer = AutoTokenizer.from_pretrained(cfg['text_model'], use_fast=True, local_files_only=True)
        add_markers(tokenizer)
        tokenizer.save_pretrained(output / 'tokenizer')
        train_ds = RepresentationDataset(train, {e.id: e.aspects for e in train}, tokenizer, True, cfg['max_length'])
        if cfg['train_limit'] is not None:
            train_ds = Subset(train_ds, range(min(cfg['train_limit'], len(train_ds))))
        if not len(train_ds):
            raise ValueError('empty training aspects')
        pred_ds = RepresentationDataset(dev, cache['baseline'], tokenizer, False, cfg['max_length'])
        if pred_ds.keys != [tuple(k) for k in cache['keys']]:
            raise ValueError('score keys not aligned with predicted spans')
        collate = RepresentationCollator(tokenizer.pad_token_id)
        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
            generator=torch.Generator().manual_seed(cfg['seed']), collate_fn=collate)
        pred_loader = DataLoader(pred_ds, batch_size=cfg['eval_batch_size'], shuffle=False, collate_fn=collate)
        encoder = load_text(cfg['text_model'])
        encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model = RepresentationClassifier(encoder, cfg['variant'], cfg['dropout'], cfg['attention_dim']).to(device)
        groups = []
        for head in (False, True):
            for decay in (False, True):
                params = [p for name, p in model.named_parameters()
                    if (not name.startswith('encoder.')) == head
                    and (not (name.endswith('.bias') or 'LayerNorm' in name)) == decay]
                if params:
                    groups.append(dict(params=params, lr=cfg['head_learning_rate'] if head else cfg['learning_rate'],
                                       weight_decay=cfg['weight_decay'] if decay else 0.))
        optimizer = torch.optim.AdamW(groups)
        updates = len(train_loader) * cfg['epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer, int(updates * cfg['warmup_ratio']), updates)
        write_json(output / 'config.json', cfg)
        reference, reference_predictions = evaluate_scores(cache, cache['target_logits'], gold)
        write_json(output / 'provenance.json', dict(code=identity, input_sha256=hashes,
            historical_reference=reference, train_aspects=len(train_ds), dev_sentences=len(dev),
            predicted_dev_aspects=len(pred_ds), selection_metric=SELECTION, test_used=False,
            is_smoke=cfg['train_limit'] is not None))
        history, best_f1, best_epoch = [], -1., None
        from .representation_diagnostic import error_transitions, matched_diagnostic
        for epoch in range(1, cfg['epochs'] + 1):
            epoch_started = time.time()
            model.train()
            loss_sum, seen = 0., 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = model(**{k: v.to(device) for k, v in batch.items()})['loss']
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                optimizer.step(); scheduler.step()
                count = len(batch['labels']); loss_sum += loss.item() * count; seen += count
            logits = predict_scores(model, pred_loader, device)
            metrics, predictions = evaluate_scores(cache, logits, gold)
            if metrics['fusion']['mate'] != reference['fusion']['mate']:
                raise ValueError('frozen boundary invariant violated')
            if is_better(metrics, best_f1):
                best_f1, best_epoch = metrics['fusion']['joint']['f1'], epoch
                save_representation(output / 'best.pt', model, dict(epoch=epoch, metrics=metrics, config=cfg))
                for name, values in predictions.items():
                    write_json(output / f'best_{name}_predictions.json', values)
                write_json(output / 'best_scores.json', dict(keys=cache['keys'], logits=logits))
            row = dict(epoch=epoch, train_ce=loss_sum / seen, **metrics,
                matched_sentiment=matched_diagnostic(gold, predictions['fusion']),
                vs_historical=error_transitions(gold, reference_predictions['fusion'], predictions['fusion']),
                seconds=time.time()-epoch_started, elapsed_seconds=time.time()-started)
            history.append(row)
            write_json(output / 'history.json', history)
            status.update(completed_epochs=epoch, best_epoch=best_epoch, best_joint_f1=best_f1,
                          last_epoch_seconds=row['seconds'])
            write_json(status_path, status)
            print(json.dumps(row), flush=True)
        del optimizer, scheduler, model, encoder
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
        restored, metadata = load_representation(output / 'best.pt', device)
        restored_tokenizer = AutoTokenizer.from_pretrained(output / 'tokenizer', local_files_only=True)
        reload_ds = RepresentationDataset(dev, cache['baseline'], restored_tokenizer, False, cfg['max_length'])
        if reload_ds.keys != pred_ds.keys:
            raise ValueError('reload aspect order changed')
        reload_loader = DataLoader(reload_ds, batch_size=cfg['eval_batch_size'], shuffle=False,
            collate_fn=RepresentationCollator(restored_tokenizer.pad_token_id))
        reload_logits = predict_scores(restored, reload_loader, device)
        metrics, predictions = evaluate_scores(cache, reload_logits, gold)
        if not _same(metadata['config'], cfg) or metadata['epoch'] != best_epoch or metadata['metrics'] != metrics:
            raise ValueError('checkpoint metadata/metric replay mismatch')
        for name, values in predictions.items():
            if not _same(values, json.loads((output / f'best_{name}_predictions.json').read_text())):
                raise ValueError('full Dev prediction reload mismatch')
        saved = json.loads((output / 'best_scores.json').read_text())
        if saved['keys'] != cache['keys'] or not np.allclose(saved['logits'], reload_logits, rtol=0, atol=1e-5):
            raise ValueError('full Dev score reload mismatch')
        _verify_hashes(hashes)
        if code_identity()['source_sha256'] != identity['source_sha256']:
            raise ValueError('source changed during training')
        write_json(output / 'reload-verification.json', dict(full_dev=True, predictions_exact=True,
            metrics_exact=True, scores_atol=1e-5, inputs_unchanged=True, source_unchanged=True,
            boundaries_unchanged=True, test_used=False))
        summary = dict(variant=cfg['variant'], dataset=cfg['dataset'], seed=cfg['seed'],
            best_epoch=best_epoch, best_fusion=metrics['fusion'], best_target=metrics['target'],
            historical_reference=reference, selection_metric=SELECTION, completed_epochs=len(history),
            matched_sentiment=matched_diagnostic(gold, predictions['fusion']),
            vs_historical=error_transitions(gold, reference_predictions['fusion'], predictions['fusion']),
            is_smoke=cfg['train_limit'] is not None, elapsed_seconds=time.time()-started,
            checkpoint_sha256=file_hash(output / 'best.pt'), test_used=False)
        write_json(output / 'summary.json', summary)
        artifacts = {str(p.relative_to(output)): file_hash(p) for p in output.rglob('*') if p.is_file()}
        write_json(output / 'artifact-sha256.json', artifacts)
        status.update(state='completed', ended_at=time.time())
        write_json(status_path, status)
        return summary
    except BaseException as error:
        status.update(state='failed', error=repr(error), ended_at=time.time())
        write_json(status_path, status)
        raise
