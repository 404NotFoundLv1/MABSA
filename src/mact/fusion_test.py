"""Explicitly authorized Test evaluation of a pre-locked raw 1:1 mixture.

No calibration, parameter search, checkpoint selection or training occurs here.
The separate Dev-only entry point and its access guard remain unchanged.
"""
import csv
import gc
import json
import statistics
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .calibrated_fusion import aspect_polarity_logits, fuse_labels
from .common.cache import file_hash, open_cache
from .common.metrics import evaluate_predictions
from .data import load_labeled, SequenceCollator
from .experiment import ROOT, code_identity
from .fusion_experiment import SEEDS, _hashes, _verify_hashes, _same
from .labels import tags_to_aspects
from .runtime import load_checkpoint, model_inputs, write_json
from .target_context import TargetDataset, TargetCollator, load_target, relabel
from .train import make_dataset

FIXED=dict(temperature_base=1.,temperature_target=1.,alpha=.5)


def _read(path):
    return json.loads(Path(path).read_text())


def load_decision(path):
    path=Path(path)
    if file_hash(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('decision checksum changed')
    d=_read(path)
    if (d.get('format') != 'mact_fixed_equal_test_v1' or d.get('parameters') != FIXED
            or d.get('selection_uses_test') is not False):
        raise ValueError('only fixed, Dev-selected parameters are permitted')
    seeds,datasets=d['seeds'],d['datasets']
    if (not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s not in SEEDS for s in seeds)
            or not datasets or len(set(datasets)) != len(datasets)
            or any(x not in ('twitter2015','twitter2017') for x in datasets)):
        raise ValueError('invalid declared cohort')
    actual=[(r['dataset'],r['seed']) for r in d['models']]
    if len(set(actual)) != len(actual) or set(actual) != {(ds,s) for ds in datasets for s in seeds}:
        raise ValueError('incomplete or duplicate locked model cohort')
    if file_hash(Path(__file__).with_name('calibrated_fusion.py')) != d['fusion_source_sha256']:
        raise ValueError('fusion rule source changed after decision')
    return d


def test_paths(dataset,seed):
    if dataset not in ('twitter2015','twitter2017') or seed not in SEEDS:
        raise ValueError('unknown dataset or seed')
    baseline=(ROOT/'outputs/online_test_s42_43_44_20261017'/dataset/f'seed{seed}'
              if seed in (42,43,44) else ROOT/'reference/runs'/dataset/f'seed{seed}'/'test')
    target=(ROOT/'outputs'/f'target_context_test_s{seed}_20261017'/dataset/f'seed{seed}'
            if seed in (42,43,44) else ROOT/'outputs/target_context_historical12_20261017/evaluations'/dataset/f'seed{seed}')
    return dict(test_data=str(ROOT/'assets/data'/dataset/'test.json'),
        visual_cache=str(ROOT/'assets/cache'/dataset/'test'),vision_model=str(ROOT/'assets/models/siglip2'),
        image_root=str(ROOT/'assets/images'/dataset),baseline_predictions=str(baseline/'predictions.json'),
        baseline_metrics=str(baseline/'metrics.json'),target_predictions=str(target/'predictions.json'),
        target_metrics=str(target/'metrics.json'))


def run_pair(decision,dataset,seed,device,output,paths):
    decision,output=Path(decision),Path(output)
    if output.exists(): raise FileExistsError('Test output must be fresh')
    lock=load_decision(decision)
    records=[r for r in lock['models'] if (r['dataset'],r['seed'])==(dataset,seed)]
    if len(records) != 1: raise ValueError('requested identity outside locked cohort')
    selected=records[0];run=Path(selected['run'])
    data=Path(paths['test_data']).resolve()
    if data.name != 'test.json' or data.parent.name != dataset:
        raise ValueError('Test data identity mismatch')
    cfg,summary=_read(run/'config.json'),_read(run/'summary.json')
    if cfg['seed'] != seed or file_hash(run/'config.json') != selected['config_sha256']:
        raise ValueError('target configuration changed')
    if (Path(cfg['baseline_checkpoint']).resolve() != Path(selected['baseline_checkpoint']).resolve()
            or (run/'best.pt').resolve() != Path(selected['target_checkpoint']).resolve()):
        raise ValueError('checkpoint paths differ from locked decision')
    if (file_hash(selected['baseline_checkpoint']) != selected['baseline_sha256']
            or file_hash(selected['target_checkpoint']) != selected['target_sha256']
            or summary['checkpoint_sha256'] != selected['target_sha256']):
        raise ValueError('checkpoint hash changed')
    base_summary=_read(cfg['baseline_summary'])
    if base_summary['checkpoint_sha256'] != selected['baseline_sha256']:
        raise ValueError('baseline summary identity mismatch')
    base_tokenizer=Path(cfg['baseline_summary']).parent/'tokenizer'
    input_paths=[decision,decision.with_suffix('.sha256'),data,run/'config.json',run/'summary.json',
                 Path(selected['baseline_checkpoint']),Path(selected['target_checkpoint']),Path(cfg['baseline_summary'])]
    input_paths += [Path(paths[k]) for k in ('baseline_predictions','baseline_metrics','target_predictions','target_metrics')]
    input_paths += [p for d in (base_tokenizer,run/'tokenizer') for p in d.rglob('*') if p.is_file()]
    hashes=_hashes(input_paths);identity=code_identity()
    torch.set_num_threads(cfg['num_threads'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    labeled,_=load_labeled(data)
    gold={e.id:e.aspects for e in labeled}
    examples=[replace(e,aspects=[],ignored_spans=[]) for e in labeled]
    model,meta=load_checkpoint(selected['baseline_checkpoint'],device)
    if meta['run_config']['seed'] != seed or model.config.label_scheme != 'five':
        raise ValueError('baseline checkpoint metadata mismatch')
    bc=dict(meta['run_config'])
    bc.update({k:paths[k] for k in ('visual_cache','vision_model','image_root') if k in paths})
    features=None
    if model.config.mode != 'text':
        hashes.update(_hashes([Path(bc['visual_cache'])/n for n in ('manifest.json','features.pt')]))
        features,_=open_cache(bc['visual_cache'],bc['vision_model'],bc['image_root'],examples)
    tok=AutoTokenizer.from_pretrained(base_tokenizer,local_files_only=True)
    loader=DataLoader(make_dataset(examples,tok,bc,features),batch_size=1,collate_fn=SequenceCollator(tok.pad_token_id))
    baseline,keys,base_logits={},[],[]
    with torch.no_grad():
        for batch in loader:
            inputs=model_inputs(batch,device)
            if 'tags' in inputs: raise ValueError('inference must not forward tags')
            z=model(**inputs)['emissions']
            decoded=model.crf.decode(z,inputs['word_mask'])
            for i,(e,tags) in enumerate(zip(batch['examples'],decoded)):
                spans=tags_to_aspects(tags);baseline[e.id]=spans
                scores=aspect_polarity_logits(model.crf,z[i,:len(e.words)],spans)
                if len(spans):
                    labels=torch.tensor([p for _,_,p in spans],device=scores.device)
                    if (scores.max(-1).values-scores[torch.arange(len(spans),device=scores.device),labels]>1e-4).any():
                        raise ValueError('conditional scores contradict Viterbi labels')
                keys.extend((e.id,s,end) for s,end,_ in spans)
                base_logits.extend(scores.cpu().tolist())
    if not _same(baseline,_read(paths['baseline_predictions'])):
        raise ValueError('baseline Test replay differs from archive')
    del model,loader,features
    gc.collect()
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    tok=AutoTokenizer.from_pretrained(run/'tokenizer',local_files_only=True)
    ds=TargetDataset(examples,baseline,tok,False,cfg['max_length'])
    if ds.keys != keys: raise ValueError('target boundary ordering mismatch')
    model,meta=load_target(run/'best.pt',device)
    if meta['config']['seed'] != seed or meta['epoch'] != summary['best_epoch']:
        raise ValueError('target checkpoint metadata mismatch')
    loader=DataLoader(ds,batch_size=cfg['eval_batch_size'],collate_fn=TargetCollator(tok.pad_token_id))
    target_logits=[]
    with torch.no_grad():
        for batch in loader:
            if 'labels' in batch: raise ValueError('inference must not forward labels')
            z=model(**{k:v.to(device) for k,v in batch.items()})['logits']
            if not torch.isfinite(z).all():raise ValueError('nonfinite target scores')
            target_logits.extend(z.cpu().tolist())
    target_scores=torch.tensor(target_logits).reshape(-1,3)
    target_labels=target_scores.argmax(-1)
    target=relabel(baseline,keys,target_labels.tolist())
    if not _same(target,_read(paths['target_predictions'])):
        raise ValueError('target Test replay differs from archive')
    old=torch.tensor([p for v in baseline.values() for _,_,p in v],dtype=torch.long)
    fused_labels=fuse_labels(torch.tensor(base_logits).reshape(-1,3),target_scores,old,target_labels,**FIXED)
    fusion=relabel(baseline,keys,fused_labels.tolist())
    predictions=dict(baseline=baseline,target=target,fusion=fusion)
    metrics={k:evaluate_predictions(gold,v) for k,v in predictions.items()}
    for role in ('baseline','target'):
        if metrics[role] != _read(paths[role+'_metrics']): raise ValueError(f'{role} Test metric replay mismatch')
    if any(m['mate'] != metrics['baseline']['mate'] for m in metrics.values()):
        raise ValueError('aspect boundaries changed')
    _verify_hashes(hashes)
    if code_identity()['source_sha256'] != identity['source_sha256']:
        raise ValueError('source changed during Test evaluation')
    result=dict(format='mact_fixed_equal_test_result_v1',dataset=dataset,seed=seed,parameters=FIXED,
        metrics=metrics,predictions=predictions,keys=keys,base_logits=base_logits,target_logits=target_logits,
        provenance=dict(input_sha256=hashes,code=identity,decision_sha256=file_hash(decision),
            gold_removed_from_inference=True,baseline_replay_exact=True,target_replay_exact=True,
            boundaries_unchanged=True,test_used_for_selection=False,full_test=True))
    output.mkdir(parents=True)
    write_json(output/'result.json',result)
    (output/'result.sha256').write_text(file_hash(output/'result.json')+'\n')
    return json.loads(json.dumps(result))


def summarize(decision,root):
    decision,root=Path(decision),Path(root)
    if (root/'comparison.json').exists():raise FileExistsError('Test summary must be fresh')
    lock=load_decision(decision);rows=[]
    for dataset in lock['datasets']:
        for seed in lock['seeds']:
            path=root/dataset/f'seed{seed}'/'result.json'
            if file_hash(path) != path.with_suffix('.sha256').read_text().strip():
                raise ValueError('result checksum mismatch')
            r=_read(path);p=r['provenance']
            if (r['format'] != 'mact_fixed_equal_test_result_v1' or r['dataset'] != dataset or r['seed'] != seed
                    or r['parameters'] != FIXED or p['decision_sha256'] != file_hash(decision)
                    or p['test_used_for_selection'] or not all(p[k] for k in ('full_test','baseline_replay_exact',
                        'target_replay_exact','gold_removed_from_inference','boundaries_unchanged'))):
                raise ValueError('result identity/verification mismatch')
            _verify_hashes(p['input_sha256'])
            if p['code']['source_sha256'] != code_identity()['source_sha256']:
                raise ValueError('Test result source changed')
            # The exact evaluated Test dataset is already bound in input hashes.
            candidates=[Path(x) for x in p['input_sha256'] if Path(x).name=='test.json']
            if len(candidates) != 1 or candidates[0].parent.name != dataset:
                raise ValueError('ambiguous Test data identity')
            examples,_=load_labeled(candidates[0]);gold={e.id:e.aspects for e in examples}
            for method in ('baseline','target','fusion'):
                if evaluate_predictions(gold,r['predictions'][method]) != r['metrics'][method]:
                    raise ValueError('reported metric differs from saved predictions')
                for key,spans in r['predictions']['baseline'].items():
                    if [v[:2] for v in spans] != [v[:2] for v in r['predictions'][method][key]]:
                        raise ValueError('saved aspect boundaries differ')
            rows.append(dict(dataset=dataset,seed=seed,**{k:100*v['joint']['f1'] for k,v in r['metrics'].items()}))
    datasets={};lines=['Fixed 1:1 fusion — Test results','All values are F1 percentage points; SD is sample SD.',
        'Same pre-locked checkpoints and parameters for every seed. No Test-based selection.',
        'Historical seeds and prior Test results were already known; not an untouched benchmark.','']
    for dataset in lock['datasets']:
        cohorts={}
        for name,seeds in [('all',lock['seeds']),('initial3',[42,43,44]),('historical12',SEEDS[3:])]:
            selected=[r for r in rows if r['dataset']==dataset and r['seed'] in seeds]
            if not selected:continue
            c=dict(seeds=[r['seed'] for r in selected])
            for method in ('baseline','target','fusion'):
                values=[r[method] for r in selected]
                c[method]=dict(mean=statistics.mean(values),sample_std=statistics.stdev(values) if len(values)>1 else None)
            for control in ('baseline','target'):
                delta=[r['fusion']-r[control] for r in selected]
                c['delta_vs_'+control]=dict(mean=statistics.mean(delta),wins=sum(x>1e-10 for x in delta),
                    ties=sum(abs(x)<=1e-10 for x in delta),losses=sum(x < -1e-10 for x in delta))
            cohorts[name]=c;lines.append(dataset+' / '+name+': '+json.dumps(c))
        datasets[dataset]=cohorts
    report=dict(parameters=FIXED,completed_pairs=len(rows),datasets=datasets,rows=rows,
                decision_sha256=file_hash(decision),test_used_for_selection=False)
    write_json(root/'comparison.json',report)
    with (root/'per_seed.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (root/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    return report
