"""Verified three-fold Train-only feature generation and correction experiment."""
import gc
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .common.cache import file_hash, model_hashes, open_cache
from .common.metrics import evaluate_predictions
from .correction_experiment import (DEFAULTS, INPUTS, source_hashes, validate_payload,
    save_cache, load_cache, train_pair, _fit_verified_pair, extract_pilot, predict)
from .data import load_labeled
from .experiment import ROOT, code_identity
from .fusion_experiment import _hashes, _verify_hashes, _same, target_run
from .oof_data import prepare_dataset, verify_splits
from .runtime import load_checkpoint, write_json
from .sentiment_correction import extract_features, CorrectionHead
from .target_context import load_target, relabel
from .train import run_training as train_base
from .target_context_train import run_training as train_target

LIMITATION = ('OOF Train features exclude each held group from both expert training and inner selection. '
    'Fold experts use less training data and their learned feature coordinates may differ from full-Train '
    'experts used on Dev; this is not a pure causal saturation experiment. Dev was used previously and '
    'selects head epochs. No independent generalization, significance, novelty or Test claim.')


def read(path): return json.loads(Path(path).read_text())


def make_job(dataset, fold, splits, base_config, target_config):
    splits = Path(splits).resolve()
    partition, manifest = verify_splits(splits)
    if dataset != manifest['dataset'] or fold not in range(3): raise ValueError('wrong fold identity')
    output = splits.parent/f'fold{fold}'
    files = splits/f'fold{fold}'
    base = {**base_config, 'train_data': str(files/'fit.json'), 'dev_data': str(files/'inner.json')}
    target = {**target_config, 'train_data': base['train_data'], 'dev_data': base['dev_data'],
              'baseline_predictions': str(output/'base/best_dev_predictions.json'),
              'baseline_summary': str(output/'base/summary.json'), 'baseline_checkpoint': str(output/'base/best.pt')}
    return dict(format='oof_job_v1', dataset=dataset, seed=42, fold=fold, splits=str(splits), output=str(output),
        held_data=str(files/'held.json'), membership=partition['folds'][fold],
        base_config=base, target_config=target, pretrained_sha256=model_hashes(base['text_model']),
        input_sha256={**manifest['input_sha256'], str(splits/'manifest.json'): file_hash(splits/'manifest.json')},
        test_used=False)


def validate_job(job):
    if job['format'] != 'oof_job_v1' or job['seed'] != 42 or job['test_used']:
        raise ValueError('invalid OOF job')
    partition, manifest = verify_splits(job['splits'])
    if job['dataset'] != manifest['dataset'] or job['membership'] != partition['folds'][job['fold']]:
        raise ValueError('job membership mismatch')
    _verify_hashes(job['input_sha256'])
    files = Path(job['splits'])/f'fold{job["fold"]}'
    out = Path(job['output'])
    if job['held_data'] != str(files/'held.json'): raise ValueError('wrong held-out data')
    for cfg in (job['base_config'], job['target_config']):
        if (cfg['train_data'] != str(files/'fit.json') or cfg['dev_data'] != str(files/'inner.json')
                or cfg['seed'] != 42 or Path(cfg['text_model']).resolve() != ROOT/'assets/models/deberta-v3-base'
                or model_hashes(cfg['text_model']) != job['pretrained_sha256'] or cfg['train_limit'] is not None):
            raise ValueError('fold expert must start from original pretrained model and fit/inner data')
    b, t = job['base_config'], job['target_config']
    if b['dev_limit'] is not None or b['max_updates'] is not None: raise ValueError('partial fold training forbidden')
    expected = {'baseline_predictions': out/'base/best_dev_predictions.json',
                'baseline_summary': out/'base/summary.json', 'baseline_checkpoint': out/'base/best.pt'}
    if any(t[k] != str(v) for k, v in expected.items()): raise ValueError('wrong target selection baseline')


def verify_expert(job, kind):
    """Check the on-disk training run, not just copied cache metadata."""
    out = Path(job['output'])/kind
    cfg = job['base_config' if kind == 'base' else 'target_config']
    summary, actual, provenance = read(out/'summary.json'), read(out/'config.json'), read(out/'provenance.json')
    if (actual != cfg or summary['completed_epochs'] != cfg['epochs'] or summary['is_smoke']
            or summary['checkpoint_sha256'] != file_hash(out/'best.pt')):
        raise ValueError('fold expert did not complete the declared configuration')
    if kind == 'base':
        source = Path(__file__).parent
        current_source = {str(p.relative_to(source)): file_hash(p) for p in sorted(source.rglob('*.py'))}
        if (provenance['train_sha256'] != file_hash(cfg['train_data'])
                or provenance['dev_sha256'] != file_hash(cfg['dev_data'])
                or provenance['text_model_sha256'] != job['pretrained_sha256']
                or provenance['source_sha256'] != current_source):
            raise ValueError('base fold provenance mismatch')
    else:
        _verify_hashes(provenance['input_sha256'])
        if provenance['code']['source_sha256'] != code_identity()['source_sha256']:
            raise ValueError('target fold source mismatch')
    checkpoint = torch.load(out/'best.pt', map_location='cpu', weights_only=True)
    recorded = checkpoint['run_config'] if kind == 'base' else checkpoint['metadata']['config']
    epoch = checkpoint['epoch'] if kind == 'base' else checkpoint['metadata']['epoch']
    if recorded != cfg or epoch != summary['best_epoch']: raise ValueError('fold checkpoint metadata mismatch')
    return summary


def load_fold(path, job):
    validate_job(job)
    path = Path(path)
    if file_hash(path) != path.with_suffix('.sha256').read_text().strip(): raise ValueError('corrupt OOF fold cache')
    c = torch.load(path, weights_only=True, map_location='cpu')
    m = c['meta']
    if (m['format'] != 'oof_fold_features_v1' or m['job'] != job or m['source_sha256'] != source_hashes()
            or not m['inner_replay_exact'] or m['test_used'] or set(c['gold']) != set(job['membership']['held_ids'])):
        raise ValueError('wrong or incomplete held-out feature cache')
    _verify_hashes(m['input_sha256'])
    expected = _hashes([Path(job['output'])/k/'best.pt' for k in ('base', 'target')])
    if m['checkpoints'] != expected: raise ValueError('wrong fold checkpoint identity')
    for kind in ('base', 'target'): verify_expert(job, kind)
    gold, _ = load_labeled(job['held_data'])
    if not _same(c['gold'], {e.id: e.aspects for e in gold}): raise ValueError('held gold differs from original Train')
    validate_payload(c)
    return c


def run_fold(job):
    validate_job(job)
    # The preceding CPU head fitter enables strict deterministic algorithms.
    # Restore the original encoder training policy before the next CUDA stage.
    torch.use_deterministic_algorithms(False)
    output = Path(job['output'])
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output/'held_features.pt'
    if (output/'done.json').exists():
        load_fold(cache_path, job)
        done = read(output/'done.json')
        if done['cache_sha256'] != file_hash(cache_path): raise ValueError('stale completed fold')
        return done
    identity = source_hashes()
    for kind, trainer in [('base', train_base), ('target', train_target)]:
        cfg = job['base_config' if kind == 'base' else 'target_config']
        write_json(output/'status.json', dict(state='training', expert=kind, fold=job['fold']))
        if not (output/kind/'summary.json').exists():
            trainer(cfg, output/kind)
        verify_expert(job, kind)
        if source_hashes() != identity: raise ValueError('source changed during fold training')
        gc.collect()
        if cfg['device'].startswith('cuda'): torch.cuda.empty_cache()
    write_json(output/'status.json', dict(state='extracting', fold=job['fold']))
    cfg = job['base_config']; device = cfg['device']
    base, _ = load_checkpoint(output/'base/best.pt', device)
    target, _ = load_target(output/'target/best.pt', device)
    bt = AutoTokenizer.from_pretrained(output/'base/tokenizer', local_files_only=True)
    tt = AutoTokenizer.from_pretrained(output/'target/tokenizer', local_files_only=True)
    inner, _ = load_labeled(cfg['dev_data'])
    held, _ = load_labeled(job['held_data'])
    features, _ = open_cache(cfg['visual_cache'], cfg['vision_model'], cfg['image_root'], inner+held)
    replay = extract_features(base, target, inner, bt, tt, cfg, features, device)
    for kind, pred in [('base', replay['baseline']), ('target', relabel(replay['baseline'], replay['keys'], replay['target_labels'].tolist()))]:
        if (not _same(pred, read(output/kind/'best_dev_predictions.json'))
                or evaluate_predictions(replay['gold'], pred) != read(output/kind/'summary.json')['best_dev']):
            raise ValueError('inner checkpoint reload did not reproduce archived predictions')
    cache = extract_features(base, target, held, bt, tt, cfg, features, device)
    paths = [p for kind in ('base', 'target') for p in (output/kind).rglob('*') if p.is_file()]
    # Include effective visual cache and original pretrained identity in the certificate.
    paths += [Path(cfg['visual_cache'])/name for name in ('features.pt', 'manifest.json')]
    paths += [p for p in Path(cfg['text_model']).rglob('*') if p.is_file()]
    hashes = {**job['input_sha256'], **_hashes(paths)}
    checkpoints = _hashes([output/k/'best.pt' for k in ('base', 'target')])
    cache['meta'] = dict(format='oof_fold_features_v1', job=job, source_sha256=identity,
        input_sha256=hashes, checkpoints=checkpoints, inner_replay_exact=True, test_used=False)
    validate_payload(cache)
    _verify_hashes(hashes)
    if source_hashes() != identity: raise ValueError('source changed during fold extraction')
    save_cache(cache_path, cache)
    load_fold(cache_path, job)
    done = dict(cache=str(cache_path), cache_sha256=file_hash(cache_path), inner_replay_exact=True,
        held_sentences=len(cache['gold']), predicted_aspects=len(cache['keys']), test_used=False)
    write_json(output/'done.json', done)
    write_json(output/'status.json', dict(state='completed', fold=job['fold']))
    del base, target, features
    gc.collect()
    if device.startswith('cuda'): torch.cuda.empty_cache()
    return done


def aggregate_oof(jobs, paths, output):
    if len(jobs) != 3 or len(paths) != 3 or sorted(j['fold'] for j in jobs) != [0, 1, 2]:
        raise ValueError('require exactly three distinct completed folds')
    if len({(j['dataset'], j['splits'], j['seed']) for j in jobs}) != 1: raise ValueError('mixed OOF cohort')
    caches = [load_fold(p, j) for p, j in zip(paths, jobs)]
    merged = dict(keys=[], gold={}, baseline={})
    hashes, checkpoints = {}, {}
    for c in caches:
        if set(merged['gold']) & set(c['gold']): raise ValueError('duplicate OOF sentences')
        merged['keys'].extend(c['keys']); merged['gold'].update(c['gold']); merged['baseline'].update(c['baseline'])
        for p, sha in c['meta']['input_sha256'].items():
            if p in hashes and hashes[p] != sha: raise ValueError('inconsistent OOF input versions')
            hashes[p] = sha
        checkpoints.update(c['meta']['checkpoints'])
    for k in INPUTS+('base_labels', 'target_labels', 'fixed_labels', 'labels'):
        merged[k] = torch.cat([c[k] for c in caches])
    partition, _ = verify_splits(jobs[0]['splits'])
    if set(merged['gold']) != set(partition['groups']): raise ValueError('incomplete OOF sentence coverage')
    hashes.update(_hashes(paths))
    merged['meta'] = dict(format='oof_train_features_v1', dataset=jobs[0]['dataset'], seed=42, split='train',
        fold_paths=[str(Path(p).resolve()) for p in paths], jobs=jobs, input_sha256=hashes,
        checkpoints=checkpoints, source_sha256=source_hashes(), test_used=False)
    validate_payload(merged)
    save_cache(output, merged)
    return merged


def load_oof(path):
    path = Path(path)
    if file_hash(path) != path.with_suffix('.sha256').read_text().strip(): raise ValueError('corrupt OOF training cache')
    c = torch.load(path, weights_only=True, map_location='cpu'); m = c['meta']
    if (m['format'] != 'oof_train_features_v1' or m['split'] != 'train' or m['seed'] != 42
            or m['source_sha256'] != source_hashes() or m['test_used'] or len(m['jobs']) != 3
            or sorted(j['fold'] for j in m['jobs']) != [0, 1, 2] or len(m['fold_paths']) != 3):
        raise ValueError('invalid OOF format or cohort')
    _verify_hashes(m['input_sha256'])
    if any(j['dataset'] != m['dataset'] or j['splits'] != m['jobs'][0]['splits'] for j in m['jobs']):
        raise ValueError('mixed OOF dataset')
    folds = [load_fold(p, j) for p, j in zip(m['fold_paths'], m['jobs'])]
    for name in INPUTS+('base_labels', 'target_labels', 'fixed_labels', 'labels'):
        if not torch.equal(c[name], torch.cat([f[name] for f in folds])): raise ValueError('aggregate feature mismatch')
    keys = [key for f in folds for key in f['keys']]
    gold = {sid: values for f in folds for sid, values in f['gold'].items()}
    base = {sid: values for f in folds for sid, values in f['baseline'].items()}
    checkpoints = {p: sha for f in folds for p, sha in f['meta']['checkpoints'].items()}
    if (c['keys'] != keys or not _same(c['gold'], gold) or not _same(c['baseline'], base)
            or m['checkpoints'] != checkpoints or set(gold) != set(verify_splits(m['jobs'][0]['splits'])[0]['groups'])):
        raise ValueError('aggregate coverage or provenance mismatch')
    validate_payload(c)
    return c


def fit_oof_pair(train_path, dev_path, output, config=None):
    cfg = {**DEFAULTS, **(config or {})}
    if set(cfg) != set(DEFAULTS) or cfg['seed'] != 42 or cfg['epochs'] < 1 or cfg['batch_size'] < 1:
        raise ValueError('invalid head configuration')
    train, dev = load_oof(train_path), load_cache(dev_path)
    if (dev['meta']['split'] != 'dev' or train['meta']['dataset'] != dev['meta']['dataset']
            or set(train['gold']) & set(dev['gold'])
            or any(train[k].shape[1] != dev[k].shape[1] for k in INPUTS)):
        raise ValueError('incompatible OOF Train and full-expert Dev')
    # Explicitly verify the intended full-Train inference pair, not any seed42 pair.
    run = target_run(dev['meta']['dataset'], 42); original = read(run/'config.json')
    expected = _hashes([original['baseline_checkpoint'], run/'best.pt'])
    if dev['meta']['checkpoints'] != expected: raise ValueError('wrong full-Train inference experts')
    return _fit_verified_pair(train, dev, train_path, dev_path, output, cfg, limitation=LIMITATION,
        provenance_extra=dict(training_format='oof_train_features_v1', dev_checkpoints=expected,
                              outer_folds=3, inner_selection_only=True))


def verify_fit(fit, train_path, dev_path, kind, config=None):
    """Bind a completed head stage to its actual feature sources and protocol."""
    if kind not in ('in_sample', 'oof'): raise ValueError('unknown fitting protocol')
    cfg = {**DEFAULTS, **(config or {})}
    train = load_oof(train_path) if kind == 'oof' else load_cache(train_path)
    dev = load_cache(dev_path)
    if train['meta']['split'] != 'train' or dev['meta']['split'] != 'dev': raise ValueError('wrong fitting splits')
    expected = dict(cache_sha256=_hashes([train_path, dev_path]), source_sha256=source_hashes(),
                    checkpoints=train['meta']['checkpoints'], config=cfg, dataset=train['meta']['dataset'], test_used=False)
    if kind == 'oof':
        expected.update(training_format='oof_train_features_v1', dev_checkpoints=dev['meta']['checkpoints'],
                        outer_folds=3, inner_selection_only=True)
    fit = Path(fit)
    if read(fit/'protocol.json') != expected: raise ValueError('fit protocol/provenance differs from expected training')
    status = read(fit/'status.json'); result = read(fit/'result.json')
    if (status.get('state') != 'completed' or status.get('test_used') is not False or result.get('test_used') is not False
            or result['dataset'] != train['meta']['dataset'] or result['seed'] != 42
            or set(result['variants']) != {'plain', 'bounded'}):
        raise ValueError('head stage incomplete or wrong experiment')
    dimensions = dict(base_dim=train['base_features'].shape[1], target_dim=train['target_features'].shape[1],
                      hidden=cfg['hidden'], dropout=cfg['dropout'])
    for variant, row in result['variants'].items():
        if ([r['epoch'] for r in row['history']] != list(range(cfg['epochs']+1))
                or read(fit/variant/'history.json') != row['history'] or not row['reload_exact']):
            raise ValueError('incomplete head history')
        for name in ('best', 'best_trained'):
            saved = torch.load(fit/variant/(name+'.pt'), weights_only=True, map_location='cpu')
            if saved['provenance'] != expected or saved['variant'] != variant or saved['dimensions'] != dimensions:
                raise ValueError('saved head has stale or mixed training provenance')
    return result, dev


def audit_fit(fit, train_path, dev_path, kind, config=None):
    """Independently recompute exact tuples/counts from saved predictions and reload heads."""
    fit = Path(fit); r, dev = verify_fit(fit, train_path, dev_path, kind, config)
    gold = {(sid, *t) for sid, values in dev['gold'].items() for t in values}
    gb = {t[:3] for t in gold}
    fixed = {(sid, s, e, int(p)) for (sid, s, e), p in zip(dev['keys'], dev['fixed_labels'])}
    result = {}
    for variant, row in r['variants'].items():
        for name, key, predfile, changes_key in [('best', 'selected_metrics', 'predictions', 'selected_changes'),
                ('best_trained', 'best_trained_metrics', 'best_trained_predictions', 'best_trained_changes')]:
            p = read(fit/variant/(predfile+'.json'))
            pred = {(sid, *t) for sid, values in p.items() for t in values}
            if set(p) != set(dev['gold']) or {t[:3] for t in pred} != {t[:3] for t in fixed}:
                raise ValueError('saved predictions changed boundary coverage')
            tp = len(pred & gold); n = len(pred); g = len(gold)
            f1 = 2*tp/(n+g) if n+g else 0.
            m = row[key]['joint']
            if (m['tp'], m['predicted'], m['gold']) != (tp, n, g) or abs(m['f1']-f1) > 1e-12:
                raise ValueError('independent tuple metric mismatch')
            corrected = len((pred & gold)-(fixed & gold)); regressed = len((fixed & gold)-(pred & gold))
            before = {t[:3]: t[3] for t in fixed}; after = {t[:3]: t[3] for t in pred}
            base = dict(zip(dev['keys'], dev['base_labels'].tolist())); target = dict(zip(dev['keys'], dev['target_labels'].tolist()))
            shared = sum(base.get(t[:3]) != t[3] and target.get(t[:3]) != t[3] for t in pred & gold)
            changes = dict(wrong_to_correct=corrected, correct_to_wrong=regressed, shared_errors_rescued=shared,
                           changed_predictions=sum(before[k] != after[k] for k in before))
            if changes != row[changes_key]: raise ValueError('independent change-count mismatch')
            checkpoint = torch.load(fit/variant/(name+'.pt'), weights_only=True, map_location='cpu')
            model = CorrectionHead(**checkpoint['dimensions'], variant=variant)
            model.load_state_dict(checkpoint['state_dict'])
            replay = relabel(dev['baseline'], dev['keys'], predict(model, dev).tolist())
            if not _same(replay, p): raise ValueError('saved head predictions differ from tuples')
            history = row['history'] if name == 'best' else row['history'][1:]
            chosen = max(history, key=lambda h: h['dev']['joint']['f1'])
            if checkpoint['epoch'] != chosen['epoch'] or chosen['dev'] != row[key]:
                raise ValueError('incorrect earliest-best selection')
            result[variant+'_'+name] = dict(epoch=chosen['epoch'], f1_pp=f1*100, changes=changes)
    write_json(fit/'audit.json', dict(passed=True, results=result, test_used=False))
    return result


def replay_reference_paths(dataset):
    directory = ROOT/'outputs/sentiment_correction_s42_20261018'/dataset/'fit'
    return [directory/'result.json']+[directory/v/(name+'.json') for v in ('plain', 'bounded')
            for name in ('predictions', 'best_trained_predictions')]


def verify_control_replay(current, reference):
    """Require exact selected outcomes, while recording non-selection float drift."""
    current, reference = Path(current), Path(reference)
    a, b = read(current/'result.json'), read(reference/'result.json')
    if a['controls'] != b['controls'] or a['train_controls'] != b['train_controls']:
        raise ValueError('original control predictions/metrics changed')
    differences = {}
    keys = ('selected_epoch', 'selected_metrics', 'selected_changes', 'best_trained_epoch',
            'best_trained_metrics', 'best_trained_changes', 'parameter_count', 'initial_sha256')
    for variant in ('plain', 'bounded'):
        x, y = a['variants'][variant], b['variants'][variant]
        if any(x[k] != y[k] for k in keys): raise ValueError('control selected outcomes differ from locked reference')
        for name in ('predictions', 'best_trained_predictions'):
            if read(current/variant/(name+'.json')) != read(reference/variant/(name+'.json')):
                raise ValueError('control selected predictions differ from locked reference')
        if len(x['history']) != len(y['history']): raise ValueError('control epoch budgets differ')
        differences[variant] = {k: max(abs(i[k]-j[k]) for i, j in zip(x['history'], y['history']))
                                for k in ('train_ce', 'train_residual_rms')}
    report = dict(selected_epochs_exact=True, selected_predictions_exact=True, selected_metrics_exact=True,
                  diagnostic_max_absolute_difference=differences,
                  diagnostic_policy='record only; diagnostics do not select or invalidate exact selected outcomes')
    write_json(current/'reference_replay.json', report)
    return report


def compare_dataset(directory):
    directory = Path(directory)
    control, oof = read(directory/'control_fit/result.json'), read(directory/'oof_fit/result.json')
    if control['controls'] != oof['controls']: raise ValueError('Dev controls differ between head experiments')
    fixed = control['controls']['fixed']['joint']['f1']*100
    arms = {}
    for variant in ('plain', 'bounded'):
        c, o = control['variants'][variant], oof['variants'][variant]
        arms[variant] = dict(control_selected_f1_pp=100*c['selected_metrics']['joint']['f1'],
            oof_selected_f1_pp=100*o['selected_metrics']['joint']['f1'],
            oof_best_trained_f1_pp=100*o['best_trained_metrics']['joint']['f1'],
            selected_epoch=o['selected_epoch'], best_trained_epoch=o['best_trained_epoch'],
            changes=o['best_trained_changes'])
    report = dict(dataset=control['dataset'], fixed_f1_pp=fixed, arms=arms,
        in_sample_train=control['train_controls'], oof_train=oof['train_controls'],
        limitation=LIMITATION, test_used=False)
    # Descriptive overlap audit; no edits to official data and no Test input.
    train, _ = load_labeled(ROOT/'assets/data'/control['dataset']/'train.json')
    dev, _ = load_labeled(ROOT/'assets/data'/control['dataset']/'dev.json')
    images = read(ROOT/'assets/cache'/control['dataset']/'train_dev/manifest.json')['image_sha256']
    texts = {tuple(w.casefold() for w in e.words) for e in train}; imgs = {images[e.image_id] for e in train}
    report['official_train_dev_overlap'] = dict(dev_same_text=sum(tuple(w.casefold() for w in e.words) in texts for e in dev),
        dev_same_image_bytes=sum(images[e.image_id] in imgs for e in dev))
    write_json(directory/'comparison.json', report)
    return report


def prepare_pipeline(root, device):
    root = Path(root).resolve()
    if root.exists(): raise FileExistsError('OOF experiment must be fresh')
    root.mkdir(parents=True)
    jobs = []
    for ds in ('twitter2015', 'twitter2017'):
        splits = prepare_dataset(ds, root)
        original = read(target_run(ds, 42)/'config.json')
        base = read(Path(original['baseline_summary']).parent/'config.json')
        base['device'] = device
        target = {**original, 'device': device}
        for fold in range(3):
            job = make_job(ds, fold, splits, base, target)
            if job['base_config']['epochs'] != 35 or job['target_config']['epochs'] != 35:
                raise ValueError('pilot requires original 35-epoch experts')
            jobs.append(job)
    paths = [p for p in root.rglob('*.json')]
    for ds in ('twitter2015', 'twitter2017'):
        run = target_run(ds, 42); cfg = read(run/'config.json')
        paths += [run/'config.json', run/'best.pt', cfg['baseline_checkpoint'], cfg['baseline_summary']]
        paths += replay_reference_paths(ds)
    launch = dict(format='oof_correction_launch_v1', seed=42, datasets=['twitter2015', 'twitter2017'],
        jobs=jobs, code=code_identity(), input_sha256=_hashes(paths), config=DEFAULTS, test_used=False,
        limitation=LIMITATION, device=device, created_at=time.time())
    write_json(root/'launch.json', launch)
    (root/'launch.sha256').write_text(file_hash(root/'launch.json')+'\n')
    write_json(root/'status.json', dict(state='prepared', completed_fold_models=0, total_fold_models=12, test_used=False))
    return launch


def verify_launch(root):
    root = Path(root)
    if file_hash(root/'launch.json') != (root/'launch.sha256').read_text().strip(): raise ValueError('launch protocol changed')
    launch = read(root/'launch.json')
    if launch['code']['source_sha256'] != code_identity()['source_sha256']: raise ValueError('code/config changed during OOF queue')
    required = {str(p.resolve()) for ds in launch['datasets'] for p in replay_reference_paths(ds)}
    if not required <= set(launch['input_sha256']): raise ValueError('prior control replay references are not locked')
    _verify_hashes(launch['input_sha256'])
    return launch


def run_pipeline(root):
    root = Path(root).resolve()
    launch = verify_launch(root)
    # Process-wide lock prevents duplicate workers. Keep the file as PID evidence.
    import fcntl
    with (root/'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        status = dict(state='running', pid=os.getpid(), started_at=started, completed_fold_models=0,
                      total_fold_models=12, test_used=False)
        write_json(root/'status.json', status)
        try:
            reports = {}
            for ds in launch['datasets']:
                directory = root/ds
                jobs = [j for j in launch['jobs'] if j['dataset'] == ds]
                paths = []
                for job in jobs:
                    verify_launch(root)
                    status.update(dataset=ds, fold=job['fold'], stage='fold_training_and_extraction')
                    write_json(root/'status.json', status)
                    done = run_fold(job); paths.append(done['cache'])
                    status['completed_fold_models'] += 2
                    write_json(root/'status.json', status)
                verify_launch(root)
                oof_path = directory/'oof_train.pt'
                if oof_path.exists(): load_oof(oof_path)
                else: aggregate_oof(jobs, paths, oof_path)
                status.update(stage='original_controls_and_dev_features')
                write_json(root/'status.json', status)
                if (directory/'features/status.json').exists() and read(directory/'features/status.json')['state'] == 'completed':
                    load_cache(directory/'features/train.pt'); load_cache(directory/'features/dev.pt')
                else: extract_pilot(ds, 42, launch['device'], root)
                dev_path = directory/'features/dev.pt'
                status.update(stage='head_fitting_and_audit'); write_json(root/'status.json', status)
                if not (directory/'control_fit/result.json').exists():
                    train_pair(directory/'features/train.pt', dev_path, directory/'control_fit')
                audit_fit(directory/'control_fit', directory/'features/train.pt', dev_path, 'in_sample')
                verify_control_replay(directory/'control_fit', replay_reference_paths(ds)[0].parent)
                if not (directory/'oof_fit/result.json').exists(): fit_oof_pair(oof_path, dev_path, directory/'oof_fit')
                audit_fit(directory/'oof_fit', oof_path, dev_path, 'oof')
                reports[ds] = compare_dataset(directory)
            verify_launch(root)
            gates = {variant: all(r['arms'][variant]['oof_selected_f1_pp'] >= r['fixed_f1_pp']+.2-1e-10 and
                       r['arms'][variant]['oof_selected_f1_pp'] >= r['arms'][variant]['control_selected_f1_pp']-1e-10
                       for r in reports.values()) for variant in ('plain', 'bounded')}
            write_json(root/'comparison.json', dict(datasets=reports, screening_gate=gates,
                test_used=False, main_candidate='fixed_equal_fusion_pending_further_validation', limitation=LIMITATION))
            status.update(state='completed', ended_at=time.time(), screening_gate=gates)
            write_json(root/'status.json', status)
        except BaseException as error:
            status.update(state='failed', error=repr(error), ended_at=time.time())
            write_json(root/'status.json', status)
            raise
