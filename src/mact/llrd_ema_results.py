"""Strict paired Dev aggregation for the fixed LLRD/EMA experiment."""
import json
import math
from pathlib import Path
import statistics

from .common.cache import file_hash
from .common.metrics import evaluate_predictions
from .data import load_labeled
from .runtime import write_json

DATASETS = ('twitter2015', 'twitter2017')
ARMS = {('uniform','online'):'A', ('uniform','ema'):'B', ('llrd','online'):'C', ('llrd','ema'):'D'}


def common_boundary_sentiment(gold, baseline, treatment):
    def mapping(rows):
        return {(key,s,e):p for key,triples in rows.items() for s,e,p in triples}
    g,a,b = map(mapping, (gold,baseline,treatment))
    common = g.keys() & a.keys() & b.keys()
    n = len(common)
    ca,cb = [sum(pred[k] == g[k] for k in common) for pred in (a,b)]
    return dict(count=n,baseline_correct=ca,treatment_correct=cb,
                baseline_accuracy=ca/n if n else None,treatment_accuracy=cb/n if n else None)


def summarize(rows, seeds):
    seeds = tuple(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('unique integer paired seeds required')
    expected = {(ds,s,arm) for ds in DATASETS for s in seeds for arm in 'ABCD'}
    by_key = {}
    for row in rows:
        key = (row['dataset'],row['seed'],row['arm'])
        f1 = row['joint_f1']
        if key not in expected or key in by_key or not math.isfinite(f1) or not 0 <= f1 <= 1:
            raise ValueError('duplicate, unexpected or invalid result row')
        by_key[key] = row
    result = dict(state='completed' if by_key.keys() == expected else 'incomplete', rows=rows,
                  missing=[list(k) for k in sorted(expected-by_key.keys())], aggregate=None,
                  selected_candidate=None, units='F1 percentage points', selection='Dev Joint F1 only')
    if result['missing']:
        return result
    aggregate = {}
    for ds in DATASETS:
        aggregate[ds] = {}
        for arm in 'ABCD':
            values = [by_key[ds,s,arm]['joint_f1']*100 for s in seeds]
            differences = [(by_key[ds,s,arm]['joint_f1']-by_key[ds,s,'A']['joint_f1'])*100 for s in seeds]
            aggregate[ds][arm] = dict(mean=statistics.mean(values), sample_std=statistics.stdev(values),
                paired_deltas_pp=dict(zip(map(str,seeds),differences)), mean_delta_pp=statistics.mean(differences),
                delta_sample_std_pp=statistics.stdev(differences), wins=sum(d>0 for d in differences),
                ties=sum(d==0 for d in differences), losses=sum(d<0 for d in differences))
    candidates = [arm for arm in 'BCD' if all(aggregate[ds][arm]['mean_delta_pp'] > 0 and
                  aggregate[ds][arm]['wins'] >= 2 for ds in DATASETS)]
    result.update(aggregate=aggregate, eligible_candidates=candidates,
                  selected_candidate=max(candidates,key=lambda a:sum(aggregate[d][a]['mean_delta_pp'] for d in DATASETS))
                  if candidates else None, confirmatory=False)
    return result


def collect(root, seeds=(42,43,44)):
    root = Path(root)
    launch = json.loads((root/'launch.json').read_text())
    if list(seeds) != launch['seeds']:
        raise ValueError('seed manifest mismatch')
    expected = {(ds,s,t) for ds in DATASETS for s in seeds for t in ('uniform','llrd')}
    jobs = launch['jobs']
    keys = [(j['dataset'],j['seed'],j['trajectory']) for j in jobs]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('incomplete or duplicate launch manifest')
    rows, predictions, golds, states = [], {}, {}, []
    for job in jobs:
        ds,seed,trajectory = job['dataset'],job['seed'],job['trajectory']
        run = root/trajectory/ds/f'seed{seed}'
        status_path = run.with_name(run.name+'.status.json')
        status = json.loads(status_path.read_text()) if status_path.exists() else {'state':'pending'}
        states.append(dict(dataset=ds,seed=seed,trajectory=trajectory,state=status['state']))
        if status['state'] != 'completed':
            continue
        cfg = json.loads((run/'config.json').read_text())
        if cfg != job['run_config'] or cfg['seed'] != seed or cfg['epochs'] != 35 or cfg['ema_decay'] != .99:
            raise ValueError(f'configuration mismatch: {run}')
        if cfg['layer_lr_decay'] != (1. if trajectory=='uniform' else .9):
            raise ValueError('trajectory mismatch')
        if any(cfg[k] is not None for k in ('train_limit','dev_limit','max_updates')):
            raise ValueError('smoke cannot be aggregated as formal training')
        for key in ('train_data','dev_data'):
            if file_hash(cfg[key]) != launch['resource_sha256'][cfg[key]]:
                raise ValueError('official Train/Dev data changed')
        if ds not in golds:
            examples,_ = load_labeled(cfg['dev_data'])
            golds[ds] = {ex.id:ex.aspects for ex in examples}
        provenance = json.loads((run/'provenance.json').read_text())
        run_sources = {'src/mact/'+n:v for n,v in provenance['source_sha256'].items()}
        expected_sources = {n:v for n,v in launch['code']['source_sha256'].items() if n.startswith('src/mact/')}
        if not expected_sources or run_sources != expected_sources:
            raise ValueError('training source differs from launch manifest')
        for source in ('online','ema'):
            directory = run if source=='online' else run/'ema'
            summary = json.loads((directory/'summary.json').read_text())
            check = json.loads((directory/'reload-verification.json').read_text())
            if (summary['is_smoke'] or summary['completed_epochs'] != 35 or summary['evaluated_epochs'] != 35
                    or summary['weight_source'] != source or summary['trajectory'] != trajectory
                    or not 1 <= summary['best_epoch'] <= 35
                    or not all(check.get(k) is True for k in ('full_dev','predictions_exact','metrics_exact'))
                    or file_hash(directory/'best.pt') != summary['checkpoint_sha256']):
                raise ValueError(f'invalid completed checkpoint: {directory}')
            if source=='ema' and summary['ema_updates'] != summary['optimizer_updates']:
                raise ValueError('EMA update count mismatch')
            pred = json.loads((directory/'best_dev_predictions.json').read_text())
            metrics = evaluate_predictions(golds[ds],pred)
            if metrics != summary['best_dev']:
                raise ValueError('saved metrics disagree with Dev predictions')
            arm = ARMS[trajectory,source]
            predictions[ds,seed,arm] = pred
            rows.append(dict(dataset=ds,seed=seed,arm=arm,trajectory=trajectory,weight_source=source,
                             joint_f1=metrics['joint']['f1'],metrics=metrics,best_epoch=summary['best_epoch'],
                             checkpoint=str(directory/'best.pt'),checkpoint_sha256=summary['checkpoint_sha256']))
    result = summarize(rows,seeds)
    result['jobs'] = states
    result['common_boundary_sentiment'] = [dict(dataset=ds,seed=s,arm=arm,
        **common_boundary_sentiment(golds[ds],predictions[ds,s,'A'],predictions[ds,s,arm]))
        for ds in DATASETS for s in seeds for arm in 'BCD'
        if (ds,s,'A') in predictions and (ds,s,arm) in predictions]
    write_json(root/'results.json',result)
    return result
