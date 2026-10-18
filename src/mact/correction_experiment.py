"""Train/Dev-only frozen-feature correction pilot with replay and provenance gates."""
import gc
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer

from .calibrated_fusion import fuse_labels
from .common.cache import file_hash, open_cache
from .common.metrics import evaluate_predictions
from .data import load_labeled
from .experiment import ROOT
from .fusion_experiment import target_run, _same, _hashes, _verify_hashes
from .labels import aspects_to_tags
from .runtime import load_checkpoint, write_json
from .sentiment_correction import CorrectionHead, corrected_labels, extract_features
from .target_context import load_target, relabel

DEFAULTS = dict(seed=42, epochs=30, batch_size=64, hidden=128, dropout=.1,
                learning_rate=.001, weight_decay=.01, gradient_clip=1., residual_l2=.01)
LIMITATION = ('Exploratory seed42 Train/Dev pilot. Frozen encoders already saw Train and '
              'were selected on this Dev; in-sample features may be saturated. '
              'No Test, independent validation, significance or novelty claim.')
INPUTS = ('base_features', 'target_features', 'base_logits', 'target_logits')


def source_hashes():
    paths = sorted((ROOT/'src/mact').rglob('*.py')) + [ROOT/'scripts/run_sentiment_correction.py']
    return _hashes(paths)


def save_cache(path, cache):
    path = Path(path)
    if path.exists() or path.with_suffix('.sha256').exists():
        raise FileExistsError('feature cache must be fresh')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(cache, temporary)
    temporary.replace(path)
    path.with_suffix('.sha256').write_text(file_hash(path)+'\n')


def validate_cache(c):
    m = c['meta']
    if (m['format'] != 'sentiment_correction_v1' or m['dataset'] not in ('twitter2015', 'twitter2017')
            or m['seed'] != 42 or m['split'] not in ('train', 'dev') or m['test_used']
            or not m['dev_replay_exact'] or Path(m['data_path']).name != m['split']+'.json'):
        raise ValueError('unsupported or unverified Train/Dev cache')
    if m['source_sha256'] != source_hashes(): raise ValueError('stale feature source')
    _verify_hashes(m['input_sha256'])
    if (len(m['checkpoints']) != 2 or any(m['input_sha256'].get(p) != sha for p, sha in m['checkpoints'].items())
            or m['data_path'] not in m['input_sha256']):
        raise ValueError('checkpoint/data provenance mismatch')
    if (len(set(m['sentence_ids'])) != len(m['sentence_ids']) or set(m['sentence_ids']) != set(c['gold'])
            or set(c['gold']) != set(c['baseline'])):
        raise ValueError('incomplete sentence identities')
    expected = []
    for sid, spans in c['baseline'].items():
        for values in (spans, c['gold'][sid]):
            aspects_to_tags(max([e for _, e, _ in values]+[1]), values)
        expected.extend((sid, s, e) for s, e, _ in spans)
    if c['keys'] != expected or len(set(expected)) != len(expected):
        raise ValueError('missing/duplicate/misordered aspect keys')
    n = len(expected)
    for name in INPUTS:
        v = c[name]
        if (not isinstance(v, torch.Tensor) or v.ndim != 2 or v.shape[0] != n or v.shape[1] < 1
                or v.dtype != torch.float32 or v.requires_grad or not torch.isfinite(v).all()
                or ('logits' in name and v.shape[1] != 3)):
            raise ValueError('invalid feature or score matrix: '+name)
    for name in ('base_labels', 'target_labels', 'fixed_labels', 'labels'):
        v = c[name]
        if v.shape != (n,) or v.dtype != torch.long:
            raise ValueError('invalid label matrix')
    base = torch.tensor([p for spans in c['baseline'].values() for _, _, p in spans], dtype=torch.long)
    target = c['target_logits'].argmax(-1)
    fixed = fuse_labels(c['base_logits'], c['target_logits'], base, target, 1., 1., .5)
    matched = {(sid, s, e): p for sid, spans in c['gold'].items() for s, e, p in spans}
    labels = torch.tensor([matched.get(k, -100) for k in expected], dtype=torch.long)
    if any(not torch.equal(c[name], value) for name, value in
           [('base_labels', base), ('target_labels', target), ('fixed_labels', fixed), ('labels', labels)]):
        raise ValueError('cached labels contradict scores, boundaries or gold')


def load_cache(path):
    path = Path(path)
    if file_hash(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('corrupted feature cache')
    cache = torch.load(path, map_location='cpu', weights_only=True)
    validate_cache(cache)
    return cache


def extract_pilot(dataset, seed, device, root):
    if seed != 42: raise ValueError('only the approved seed42 pilot is supported')
    run = target_run(dataset, seed)
    output = Path(root)/dataset/'features'
    if output.exists(): raise FileExistsError('extraction destination must be fresh')
    output.mkdir(parents=True)
    def read(p): return json.loads(Path(p).read_text())
    cfg, summary = read(run/'config.json'), read(run/'summary.json')
    base_summary = read(cfg['baseline_summary'])
    if cfg['seed'] != seed or cfg['max_length'] != 258 or summary['is_smoke']:
        raise ValueError('wrong target configuration')
    examples = {}
    for split in ('train', 'dev'):
        p = Path(cfg[split+'_data']).resolve()
        if p != (ROOT/'assets/data'/dataset/(split+'.json')).resolve():
            raise ValueError('unexpected dataset split')
        examples[split], _ = load_labeled(p)
    if set(e.id for e in examples['train']) & set(e.id for e in examples['dev']):
        raise ValueError('Train/Dev overlap')
    archived_path = ROOT/'outputs/calibrated_fusion_20261018'/dataset/'seed42.json'
    if file_hash(archived_path) != archived_path.with_suffix('.sha256').read_text().strip():
        raise ValueError('corrupt archived Dev controls')
    archived = read(archived_path)
    _verify_hashes(archived['provenance']['input_sha256'])
    if archived['dataset'] != dataset or archived['seed'] != seed or archived['provenance']['test_used']:
        raise ValueError('wrong archived control')
    comparison_path = ROOT/'outputs/calibrated_fusion_20261018/comparison.json'
    comparison = next(r for r in read(comparison_path)['results'][dataset]['rows'] if r['seed'] == seed)
    tokenizer_path = Path(cfg['baseline_summary']).parent/'tokenizer'
    resources = dict(visual_cache=str(ROOT/'assets/cache'/dataset/'train_dev'),
                     vision_model=str(ROOT/'assets/models/siglip2'), image_root=str(ROOT/'assets/images'/dataset))
    paths = list(archived['provenance']['input_sha256']) + [cfg['train_data'], archived_path, comparison_path]
    hashes = _hashes(paths)
    sources = source_hashes()
    checkpoints = _hashes([cfg['baseline_checkpoint'], run/'best.pt'])
    if (checkpoints[str(Path(cfg['baseline_checkpoint']).resolve())] != base_summary['checkpoint_sha256']
            or checkpoints[str((run/'best.pt').resolve())] != summary['checkpoint_sha256']):
        raise ValueError('checkpoint identity mismatch')
    protocol = dict(dataset=dataset, seed=seed, config=DEFAULTS, resources=resources,
        input_sha256=hashes, source_sha256=sources, checkpoints=checkpoints, test_used=False,
        score_tolerance=1e-5, limitation=LIMITATION,
        advancement='bounded >= fixed + .20 F1pp on BOTH datasets and >= plain on each')
    write_json(output/'protocol.json', protocol)
    write_json(output/'status.json', dict(state='extracting', split='loading'))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    base, base_meta = load_checkpoint(cfg['baseline_checkpoint'], device)
    target, target_meta = load_target(run/'best.pt', device)
    if (base_meta['run_config']['seed'] != seed or target_meta['config']['seed'] != seed
            or target_meta['epoch'] != summary['best_epoch']):
        raise ValueError('mixed checkpoint metadata')
    bt = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tt = AutoTokenizer.from_pretrained(run/'tokenizer', local_files_only=True)
    features, _ = open_cache(resources['visual_cache'], resources['vision_model'], resources['image_root'],
                             examples['train']+examples['dev'])
    # Verify Dev before spending time on Train. No cache is published until both finish.
    caches = {}
    for split in ('dev', 'train'):
        write_json(output/'status.json', dict(state='extracting', split=split))
        c = extract_features(base, target, examples[split], bt, tt,
                             {**base_meta['run_config'], **resources}, features, device)
        if c['base_features'].shape[1] != 256 or c['target_features'].shape[1] != 768:
            raise ValueError('unexpected production feature dimensions')
        caches[split] = c
        if split == 'dev':
            target_pred = relabel(c['baseline'], c['keys'], c['target_labels'].tolist())
            fixed_pred = relabel(c['baseline'], c['keys'], c['fixed_labels'].tolist())
            if (not _same(c['keys'], archived['keys']) or not _same(c['gold'], archived['gold'])
                    or not _same(c['baseline'], archived['baseline']) or not _same(target_pred, archived['target'])):
                raise ValueError('archived Dev predictions do not replay exactly')
            differences = {k: float((c[k]-torch.tensor(archived[k])).abs().max())
                           for k in ('base_logits', 'target_logits')}
            if max(differences.values()) > 1e-5: raise ValueError('Dev scores exceed replay tolerance')
            for name, pred in [('baseline', c['baseline']), ('target', target_pred), ('raw_half', fixed_pred)]:
                if evaluate_predictions(c['gold'], pred) != comparison[name]:
                    raise ValueError('Dev metric replay failed: '+name)
            write_json(output/'dev_replay.json', dict(exact_predictions=True, exact_metrics=True,
                score_max_absolute_difference=differences, controls={k: comparison[k] for k in ('baseline', 'target', 'raw_half')}))
        print(json.dumps(dict(dataset=dataset, split=split, sentences=len(c['gold']), aspects=len(c['keys']))), flush=True)
    _verify_hashes(hashes)
    if sources != source_hashes(): raise ValueError('source changed during extraction')
    for split, c in caches.items():
        c['meta'] = dict(format='sentiment_correction_v1', dataset=dataset, seed=seed, split=split,
            data_path=str(Path(cfg[split+'_data']).resolve()), sentence_ids=[e.id for e in examples[split]],
            input_sha256=hashes, source_sha256=sources, checkpoints=checkpoints,
            dev_replay_exact=True, test_used=False)
        validate_cache(c)
        save_cache(output/(split+'.pt'), c)
    del base, target, features
    gc.collect()
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    write_json(output/'status.json', dict(state='completed', splits=['train', 'dev'], test_used=False))
    return output


@torch.no_grad()
def predict(model, cache):
    model.eval()
    out = model(*(cache[k] for k in INPUTS))
    labels = corrected_labels(out['scores'], cache['fixed_labels'])
    # Fixed fusion also has a consensus rule. Preserve it even if roundoff makes
    # its historical decision differ from the conditional-score argmax.
    zero = (out['delta'] == 0).all(-1)
    labels[zero] = cache['fixed_labels'][zero]
    return labels


def assess(cache, labels):
    pred = relabel(cache['baseline'], cache['keys'], labels.tolist())
    metrics = evaluate_predictions(cache['gold'], pred)
    matched = cache['labels'] >= 0
    before = cache['fixed_labels'] == cache['labels']
    after = labels == cache['labels']
    shared = (cache['base_labels'] != cache['labels']) & (cache['target_labels'] != cache['labels'])
    changes = dict(wrong_to_correct=int((matched & ~before & after).sum()),
                   correct_to_wrong=int((matched & before & ~after).sum()),
                   shared_errors_rescued=int((matched & shared & after).sum()),
                   changed_predictions=int((labels != cache['fixed_labels']).sum()))
    return metrics, changes, pred


def state_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()): h.update(k.encode()); h.update(v.numpy().tobytes())
    return h.hexdigest()


def train_pair(train_cache, dev_cache, output, config=None):
    output = Path(output)
    if output.exists(): raise FileExistsError('training destination must be fresh')
    config = {**DEFAULTS, **(config or {})}
    if (set(config) != set(DEFAULTS) or config['seed'] != 42 or config['epochs'] < 1
            or config['batch_size'] < 1): raise ValueError('invalid training configuration')
    train, dev = load_cache(train_cache), load_cache(dev_cache)
    a, b = train['meta'], dev['meta']
    if (a['split'] != 'train' or b['split'] != 'dev' or set(a['sentence_ids']) & set(b['sentence_ids'])
            or any(a[k] != b[k] for k in ('dataset', 'seed', 'checkpoints', 'source_sha256', 'input_sha256'))
            or any(train[k].shape[1] != dev[k].shape[1] for k in INPUTS)):
        raise ValueError('incompatible Train/Dev pair')
    indices = torch.where(train['labels'] >= 0)[0]
    if not len(indices): raise ValueError('no supervised Train aspects')
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    provenance = dict(cache_sha256=_hashes([train_cache, dev_cache]), source_sha256=source_hashes(),
                      checkpoints=a['checkpoints'], config=config, dataset=a['dataset'], test_used=False)
    write_json(output/'protocol.json', provenance)
    controls = {name: assess(dev, dev[key])[0] for name, key in
                [('baseline', 'base_labels'), ('target', 'target_labels'), ('fixed', 'fixed_labels')]}
    train_controls = {name: assess(train, train[key])[0] for name, key in
                      [('baseline', 'base_labels'), ('target', 'target_labels'), ('fixed', 'fixed_labels')]}
    result = dict(dataset=a['dataset'], seed=42, controls=controls, train_controls=train_controls,
        matched_train_aspects=len(indices), variants={}, limitation=LIMITATION, test_used=False)
    dimensions = dict(base_dim=train['base_features'].shape[1], target_dim=train['target_features'].shape[1],
                      hidden=config['hidden'], dropout=config['dropout'])
    for variant in ('plain', 'bounded'):
        torch.manual_seed(config['seed'])
        model = CorrectionHead(**dimensions, variant=variant)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        arm = output/variant
        arm.mkdir()
        initial = state_hash(model.state_dict())
        history, best, trained_best = [], None, None
        for epoch in range(config['epochs']+1):
            if epoch:
                model.train()
                order = indices[torch.randperm(len(indices))]
                for batch in order.split(config['batch_size']):
                    optimizer.zero_grad(set_to_none=True)
                    out = model(*(train[k][batch] for k in INPUTS))
                    loss = nn.functional.cross_entropy(out['scores'], train['labels'][batch])
                    loss = loss + config['residual_l2']*out['delta'].square().mean()
                    if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
                    optimizer.step()
            labels = predict(model, dev)
            metrics, changes, predictions = assess(dev, labels)
            train_labels = predict(model, train)
            train_metrics = assess(train, train_labels)[0]
            with torch.no_grad():
                train_out = model(*(train[k][indices] for k in INPUTS))
                train_ce = float(nn.functional.cross_entropy(train_out['scores'], train['labels'][indices]))
                residual_rms = float(train_out['delta'].square().mean().sqrt())
            row = dict(epoch=epoch, dev=metrics, changes=changes, train=train_metrics,
                       train_ce=train_ce, train_residual_rms=residual_rms)
            history.append(row)
            if epoch == 0 and not torch.equal(labels, dev['fixed_labels']):
                raise ValueError('zero residual did not reproduce fixed fusion')
            if best is None or metrics['joint']['f1'] > best['dev']['joint']['f1']:
                best = row
                torch.save(dict(state_dict=model.state_dict(), dimensions=dimensions, variant=variant,
                                epoch=epoch, provenance=provenance), arm/'best.pt')
                write_json(arm/'predictions.json', predictions)
            if epoch and (trained_best is None or metrics['joint']['f1'] > trained_best['dev']['joint']['f1']):
                trained_best = row
                torch.save(dict(state_dict=model.state_dict(), dimensions=dimensions, variant=variant,
                                epoch=epoch, provenance=provenance), arm/'best_trained.pt')
                write_json(arm/'best_trained_predictions.json', predictions)
            write_json(arm/'history.json', history)
            write_json(output/'status.json', dict(state='training', variant=variant, epoch=epoch))
            print(json.dumps(dict(dataset=a['dataset'], variant=variant, epoch=epoch,
                                  dev_f1_pp=metrics['joint']['f1']*100, train_ce=train_ce)), flush=True)
        for name, selected, prediction_name in [('best', best, 'predictions'),
                                               ('best_trained', trained_best, 'best_trained_predictions')]:
            saved = torch.load(arm/(name+'.pt'), map_location='cpu', weights_only=True)
            reloaded = CorrectionHead(**saved['dimensions'], variant=saved['variant'])
            reloaded.load_state_dict(saved['state_dict'], strict=True)
            m, changes, p = assess(dev, predict(reloaded, dev))
            if (m != selected['dev'] or changes != selected['changes']
                    or not _same(p, json.loads((arm/(prediction_name+'.json')).read_text()))):
                raise ValueError('reloaded head changed predictions or metrics')
        result['variants'][variant] = dict(selected_epoch=best['epoch'], selected_metrics=best['dev'],
            selected_changes=best['changes'], best_trained_epoch=trained_best['epoch'],
            best_trained_metrics=trained_best['dev'], best_trained_changes=trained_best['changes'],
            initial_sha256=initial, parameter_count=sum(p.numel() for p in model.parameters()),
            reload_exact=True, history=history)
    if result['variants']['plain']['initial_sha256'] != result['variants']['bounded']['initial_sha256']:
        raise ValueError('paired initialization differs')
    _verify_hashes(provenance['cache_sha256'])
    _verify_hashes(a['input_sha256'])
    if provenance['source_sha256'] != source_hashes(): raise ValueError('source changed during training')
    write_json(output/'result.json', result)
    write_json(output/'status.json', dict(state='completed', test_used=False))
    return result


def summarize(root):
    root = Path(root)
    rows = {}
    for ds in ('twitter2015', 'twitter2017'):
        r = json.loads((root/ds/'fit/result.json').read_text())
        values = {k: v['joint']['f1']*100 for k, v in r['controls'].items()}
        values.update({k: v['selected_metrics']['joint']['f1']*100 for k, v in r['variants'].items()})
        rows[ds] = dict(f1_pp=values, gain_bounded_vs_fixed=values['bounded']-values['fixed'],
                       gain_bounded_vs_plain=values['bounded']-values['plain'])
    passed = all(r['gain_bounded_vs_fixed'] >= .2-1e-10 and r['gain_bounded_vs_plain'] >= -1e-10 for r in rows.values())
    result = dict(rows=rows, advancement_gate_passed=passed, test_used=False, limitation=LIMITATION)
    write_json(root/'comparison.json', result)
    return result
