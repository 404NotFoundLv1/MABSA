"""Strict, paired Dev aggregation for the frozen-baseline OCR experiment."""
import json
import math
import statistics
from pathlib import Path

from .common.cache import file_hash
from .common.metrics import evaluate_predictions
from .data import load_labeled

TREATMENT_KEYS=('epochs','ocr_max_tokens','attention_heads','learning_rate','weight_decay',
                'warmup_ratio','gradient_clip','accumulation','batch_size','eval_batch_size','num_threads')


def same_treatment(a,b):
    if any(a[k]!=b[k] for k in TREATMENT_KEYS):
        return False
    x,y=a['baseline_run'],b['baseline_run']
    return (x['model']==y['model'] and x['max_length']==y['max_length']
            and a.get('ocr_manifest_sha256')==b.get('ocr_manifest_sha256')
            and a.get('ocr_cache')==b.get('ocr_cache'))


def paired_summary(rows,seeds):
    seeds=list(seeds)
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError('expected seeds must be unique and nonempty')
    expected={(d,s) for d in ('twitter2015','twitter2017') for s in seeds}
    keys=[(r['dataset'],r['seed']) for r in rows]
    if set(keys)!=expected or len(keys)!=len(expected):
        raise ValueError('missing, duplicate or unexpected dataset/seed results')
    def stats(values):
        return dict(mean=statistics.mean(values),sample_std=statistics.stdev(values) if len(values)>1 else None)
    result={}
    for dataset in ('twitter2015','twitter2017'):
        selected=[r for r in rows if r['dataset']==dataset]
        for row in selected:
            if any(not math.isfinite(row[k]) or not 0<=row[k]<=1 for k in ('baseline_f1','model_f1')):
                raise ValueError('invalid F1')
        baseline=[100*r['baseline_f1'] for r in selected]
        model=[100*r['model_f1'] for r in selected]
        delta=[100*(r['model_f1']-r['baseline_f1']) for r in selected]
        result[dataset]=dict(seeds=seeds,baseline_f1_percent=stats(baseline),model_f1_percent=stats(model),
            delta_points=stats(delta),wins=sum(v>0 for v in delta),ties=sum(v==0 for v in delta),
            losses=sum(v<0 for v in delta))
    return dict(state='completed',evaluate_test=False,datasets=result)


def verified_row(run,expected_config,root):
    run,root=Path(run),Path(root)
    read=lambda name:json.loads((run/name).read_text())
    summary,cfg,history,diagnostic,provenance=[read(n) for n in
        ('summary.json','config.json','history.json','diagnostic.json','provenance.json')]
    if (read('status.json')['state']!='completed' or summary['is_smoke'] or summary['evaluate_test']
            or summary['completed_epochs']!=35 or [r['epoch'] for r in history]!=list(range(1,36))):
        raise ValueError('only complete 35-epoch Dev runs can be aggregated')
    if not same_treatment(cfg,expected_config):
        raise ValueError('treatment settings differ across paired runs')
    for key in ('dataset','seed'):
        if summary[key]!=cfg[key] or cfg[key]!=expected_config[key]:
            raise ValueError('dataset/seed identity mismatch')
    selected=max(history,key=lambda r:r['dev']['joint']['f1'])
    if selected['epoch']!=summary['best_epoch'] or selected['dev']!=summary['best_dev']:
        raise ValueError('checkpoint was not selected on Dev')
    if file_hash(run/'best.pt')!=summary['checkpoint_sha256']:
        raise ValueError('selected residual checkpoint changed')
    for path,sha in provenance['protected_sha256'].items():
        if file_hash(path)!=sha:raise ValueError(f'protected input changed: {path}')
    reference=root/'reference/runs'/cfg['dataset']/f"seed{cfg['seed']}"
    baseline=read('baseline_dev_predictions.json')
    if baseline!=json.loads((reference/'best_dev_predictions.json').read_text()):
        raise ValueError('paired baseline differs from archived same-seed predictions')
    baseline_summary=json.loads((reference/'summary.json').read_text())
    if file_hash(cfg['baseline'])!=baseline_summary['checkpoint_sha256']:
        raise ValueError('wrong same-seed baseline checkpoint')
    gold={e.id:e.aspects for e in load_labeled(cfg['baseline_run']['dev_data'])[0]}
    for name,predictions,metrics in (
        ('baseline',baseline,summary['baseline_dev']),
        ('normal',read('normal-predictions.json'),summary['best_dev'])):
        if evaluate_predictions(gold,predictions)!=metrics or diagnostic['metrics'][name]['all']!=metrics:
            raise ValueError('stored Dev metrics do not recompute')
    if read('normal-predictions.json')!=read('best_dev_predictions.json') or read('removed-predictions.json')!=baseline:
        raise ValueError('reload/removal prediction mismatch')
    for key in ('baseline_frozen_exact','reload_exact','inputs_unchanged','source_unchanged'):
        if not summary[key]:raise ValueError(f'failed integrity proof: {key}')
    return dict(dataset=cfg['dataset'],seed=cfg['seed'],best_epoch=summary['best_epoch'],
        baseline_f1=summary['baseline_dev']['joint']['f1'],model_f1=summary['best_dev']['joint']['f1'],
        summary=summary,diagnostic=diagnostic,source=str(run))
