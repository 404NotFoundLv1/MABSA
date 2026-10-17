"""Train/Dev-only target sentiment pilot on immutable MACT boundaries."""
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

from .common.backbones import load_text
from .common.cache import file_hash
from .common.metrics import evaluate_predictions
from .data import load_labeled
from .experiment import ROOT, code_identity
from .runtime import write_json
from .target_context import (add_markers, TargetDataset, TargetCollator, TargetClassifier,
                             predict_targets, relabel, save_target, load_target)


def is_better(metrics, best_f1):
    return metrics['joint']['f1'] > best_f1


def gold_diagnostic(gold, predicted):
    metrics = evaluate_predictions(gold, predicted)
    confusion = [[0] * 3 for _ in range(3)]
    for key, triples in gold.items():
        labels = {(s, e): p for s, e, p in predicted[key]}
        for s, e, p in triples:
            confusion[p][labels[s, e]] += 1
    scores = []
    for p in range(3):
        denominator = sum(confusion[p]) + sum(row[p] for row in confusion)
        scores.append(2 * confusion[p][p] / denominator if denominator else 0.)
    return dict(accuracy=metrics['joint']['f1'], macro_f1=sum(scores) / 3,
                confusion=confusion, count=metrics['joint']['gold'])


def run_training(config, output):
    cfg = dict(config)
    allowed = {'text_model', 'train_data', 'dev_data', 'baseline_predictions', 'baseline_summary',
               'baseline_checkpoint', 'seed', 'epochs', 'batch_size', 'eval_batch_size', 'device',
               'num_threads', 'learning_rate', 'head_learning_rate', 'weight_decay', 'warmup_ratio',
               'gradient_clip', 'dropout', 'max_length', 'train_limit'}
    if set(cfg) != allowed:
        raise ValueError(f'unsupported or missing pilot config keys: {set(cfg) ^ allowed}')
    for key in ('epochs', 'batch_size', 'eval_batch_size', 'num_threads', 'max_length'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'invalid {key}')
    if cfg['train_limit'] is not None and (type(cfg['train_limit']) is not int or cfg['train_limit'] < 1):
        raise ValueError('invalid train_limit')
    for key in ('train_data', 'dev_data'):
        if Path(cfg[key]).resolve().stem.lower() == 'test':
            raise ValueError('Test data is forbidden in this pilot')
    output = Path(output).resolve()
    status_path = output.with_name(output.name + '.status.json')
    if output.exists() or status_path.exists():
        raise FileExistsError('output and status must be fresh')
    output.mkdir(parents=True)
    started = time.time()
    status = dict(state='running', started_at=started, completed_epochs=0)
    write_json(status_path, status)
    try:
        identity = code_identity()
        paths = [Path(cfg[k]).resolve() for k in ('train_data', 'dev_data', 'baseline_predictions',
                                                'baseline_summary', 'baseline_checkpoint')]
        paths += sorted(p for p in Path(cfg['text_model']).resolve().rglob('*') if p.is_file())
        hashes = {str(p): file_hash(p) for p in paths}
        if hashes[str(Path(cfg['train_data']).resolve())] == hashes[str(Path(cfg['dev_data']).resolve())]:
            raise ValueError('Train and Dev must differ')
        train, _ = load_labeled(cfg['train_data'])
        dev, _ = load_labeled(cfg['dev_data'])
        gold = {e.id: e.aspects for e in dev}
        baseline = json.loads(Path(cfg['baseline_predictions']).read_text())
        baseline_summary = json.loads(Path(cfg['baseline_summary']).read_text())
        baseline_metrics = evaluate_predictions(gold, baseline)
        if (baseline_metrics != baseline_summary['best_dev'] or
                hashes[str(Path(cfg['baseline_checkpoint']).resolve())] != baseline_summary['checkpoint_sha256']):
            raise ValueError('baseline predictions/checkpoint do not reproduce archived summary')
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
        collate = TargetCollator(tokenizer.pad_token_id)
        train_ds = TargetDataset(train, {e.id: e.aspects for e in train}, tokenizer, True, cfg['max_length'])
        if cfg['train_limit'] is not None:
            train_ds = Subset(train_ds, range(min(cfg['train_limit'], len(train_ds))))
        if not len(train_ds):
            raise ValueError('Train requires at least one aspect')
        pred_ds = TargetDataset(dev, baseline, tokenizer, False, cfg['max_length'])
        gold_ds = TargetDataset(dev, gold, tokenizer, False, cfg['max_length'])
        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
            generator=torch.Generator().manual_seed(cfg['seed']), collate_fn=collate)
        def dev_loader(ds):
            return DataLoader(ds, batch_size=cfg['eval_batch_size'], shuffle=False, collate_fn=collate)
        pred_loader, gold_loader = dev_loader(pred_ds), dev_loader(gold_ds)
        encoder = load_text(cfg['text_model'])
        encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model = TargetClassifier(encoder, cfg['dropout']).to(device)
        groups = []
        for head in (False, True):
            for decay in (False, True):
                params = [p for name, p in model.named_parameters()
                    if name.startswith('classifier.') == head
                    and (not (name.endswith('.bias') or 'LayerNorm' in name)) == decay]
                if params:
                    groups.append(dict(params=params, lr=cfg['head_learning_rate'] if head else cfg['learning_rate'],
                                       weight_decay=cfg['weight_decay'] if decay else 0.))
        optimizer = torch.optim.AdamW(groups)
        planned_updates = len(train_loader) * cfg['epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer, int(planned_updates * cfg['warmup_ratio']), planned_updates)
        write_json(output / 'config.json', cfg)
        write_json(output / 'provenance.json', dict(code=identity, input_sha256=hashes,
            baseline_metrics=baseline_metrics, train_aspects=len(train_ds), dev_sentences=len(dev),
            predicted_dev_aspects=len(pred_ds), gold_dev_aspects=len(gold_ds),
            test_used=False, is_smoke=cfg['train_limit'] is not None))
        history, best_f1, best_epoch, best_metrics, best_gold = [], -1., None, None, None
        for epoch in range(1, cfg['epochs'] + 1):
            epoch_started = time.time()
            model.train()
            loss_sum, seen = 0., 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = model(**{k: v.to(device) for k, v in batch.items()})['loss']
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite classification loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                optimizer.step(); scheduler.step()
                size = len(batch['labels'])
                loss_sum += loss.item() * size; seen += size
            predictions = relabel(baseline, pred_ds.keys, predict_targets(model, pred_loader, device))
            metrics = evaluate_predictions(gold, predictions)
            if metrics['mate'] != baseline_metrics['mate']:
                raise ValueError('frozen boundary invariant violated')
            diagnostic = gold_diagnostic(gold, relabel(gold, gold_ds.keys, predict_targets(model, gold_loader, device)))
            if is_better(metrics, best_f1):
                best_f1, best_epoch, best_metrics, best_gold = metrics['joint']['f1'], epoch, metrics, diagnostic
                save_target(output / 'best.pt', model, dict(epoch=epoch, dev_metrics=metrics, config=cfg))
                write_json(output / 'best_dev_predictions.json', predictions)
            row = dict(epoch=epoch, train_ce=loss_sum / seen, dev=metrics, gold_boundary=diagnostic,
                delta_joint_f1_pp=100 * (metrics['joint']['f1'] - baseline_metrics['joint']['f1']),
                seconds=time.time() - epoch_started, elapsed_seconds=time.time() - started)
            history.append(row)
            write_json(output / 'history.json', history)
            status.update(completed_epochs=epoch, best_epoch=best_epoch, best_joint_f1=best_f1,
                          last_epoch_seconds=row['seconds'])
            write_json(status_path, status)
            print(json.dumps(row), flush=True)
        del optimizer, scheduler, model, encoder
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
        restored, metadata = load_target(output / 'best.pt', device)
        restored_tokenizer = AutoTokenizer.from_pretrained(output / 'tokenizer', local_files_only=True)
        reload_ds = TargetDataset(dev, baseline, restored_tokenizer, False, cfg['max_length'])
        reload_loader = DataLoader(reload_ds, batch_size=cfg['eval_batch_size'], shuffle=False,
                                  collate_fn=TargetCollator(restored_tokenizer.pad_token_id))
        reloaded = relabel(baseline, reload_ds.keys, predict_targets(restored, reload_loader, device))
        if (json.loads(json.dumps(reloaded)) != json.loads((output / 'best_dev_predictions.json').read_text())
                or evaluate_predictions(gold, reloaded) != best_metrics or metadata['epoch'] != best_epoch):
            raise ValueError('full Dev checkpoint reload mismatch')
        for path, sha in hashes.items():
            if file_hash(path) != sha: raise ValueError(f'input changed during run: {path}')
        if code_identity()['source_sha256'] != identity['source_sha256']:
            raise ValueError('source changed during run')
        write_json(output / 'reload-verification.json', dict(full_dev=True, predictions_exact=True,
            metrics_exact=True, inputs_unchanged=True, source_unchanged=True, boundaries_unchanged=True))
        summary = dict(best_epoch=best_epoch, best_dev=best_metrics, best_gold_boundary=best_gold,
            baseline_dev=baseline_metrics, delta_joint_f1_pp=100 * (best_f1-baseline_metrics['joint']['f1']),
            selection_metric='predicted_boundary_dev_joint_f1', completed_epochs=len(history),
            is_smoke=cfg['train_limit'] is not None, elapsed_seconds=time.time()-started,
            checkpoint_sha256=file_hash(output / 'best.pt'), test_used=False)
        write_json(output / 'summary.json', summary)
        status.update(state='completed', ended_at=time.time())
        write_json(status_path, status)
        return summary
    except BaseException as error:
        status.update(state='failed', error=repr(error), ended_at=time.time())
        write_json(status_path, status)
        raise


def pilot_config(dataset, smoke=False):
    if dataset not in ('twitter2015', 'twitter2017'):
        raise ValueError('unknown pilot dataset')
    cfg = json.loads((ROOT / 'configs/target_context.json').read_text())
    baseline = ROOT / 'outputs/llrd_ema_s42_43_44_20261016/uniform' / dataset / 'seed42'
    cfg.update(text_model=str(ROOT / 'assets/models/deberta-v3-base'),
        train_data=str(ROOT / 'assets/data' / dataset / 'train.json'),
        dev_data=str(ROOT / 'assets/data' / dataset / 'dev.json'),
        baseline_predictions=str(baseline / 'best_dev_predictions.json'),
        baseline_summary=str(baseline / 'summary.json'), baseline_checkpoint=str(baseline / 'best.pt'))
    if smoke: cfg.update(epochs=1, train_limit=16)
    return cfg
