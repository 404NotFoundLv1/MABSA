"""Verified Dev score extraction and sentence-grouped calibration experiments."""
import gc
import hashlib
import json
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .calibrated_fusion import aspect_polarity_logits, fit_temperature, fuse_labels
from .common.cache import file_hash, open_cache
from .common.metrics import evaluate_predictions
from .data import load_labeled, SequenceCollator
from .experiment import ROOT, code_identity
from .labels import tags_to_aspects, aspects_to_tags
from .runtime import load_checkpoint, model_inputs, write_json
from .target_context import TargetDataset, TargetCollator, load_target, relabel
from .train import make_dataset

SEEDS = (42,43,44,24,25,17,16,31,22,27,26,30,28,23,19)
ALPHAS = (0.,.25,.5,.75,1.)
LIMITATION = ('Exploratory Dev only: checkpoints were already selected on this Dev set; '
              'grouped cross-fitting does not undo checkpoint selection. Historical seeds '
              'were selected using prior results. No independent validation or Test gain claim.')


def _read(path):
    return json.loads(Path(path).read_text())


def _same(a,b):
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


def target_run(dataset,seed):
    if dataset not in ('twitter2015','twitter2017') or seed not in SEEDS:
        raise ValueError('unexpected dataset or seed')
    parent = ('target_context_s42_20261017' if seed == 42 else
              'target_context_s43_44_20261017' if seed in (43,44) else
              'target_context_historical12_20261017/runs')
    return ROOT/'outputs'/parent/dataset/f'seed{seed}'


def _hashes(paths):
    return {str(Path(p).resolve()):file_hash(p) for p in paths}


def _verify_hashes(hashes):
    for path,sha in hashes.items():
        if not Path(path).is_file() or file_hash(path) != sha:
            raise ValueError(f'input changed or stale cache: {path}')


def extract_run(run,dataset,seed,device,output,resources):
    """Read-only model inference. Resources relocate only visual cache/model/images."""
    run,output=Path(run),Path(output)
    if output.exists() or output.with_suffix('.sha256').exists():
        raise FileExistsError('score output must be fresh')
    if set(resources)-{'visual_cache','vision_model','image_root'}:
        raise ValueError('unsupported resource relocation')
    cfg,summary=_read(run/'config.json'),_read(run/'summary.json')
    if cfg['seed'] != seed or Path(cfg['dev_data']).name != 'dev.json':
        raise ValueError('seed mismatch or non-Dev data')
    base_summary=_read(cfg['baseline_summary'])
    tokenizer_path=Path(cfg['baseline_summary']).parent/'tokenizer'
    paths=[run/name for name in ('config.json','summary.json','best.pt','best_dev_predictions.json')]
    paths += [Path(cfg[k]) for k in ('dev_data','baseline_summary','baseline_predictions','baseline_checkpoint')]
    paths += [p for d in (tokenizer_path,run/'tokenizer') for p in d.rglob('*') if p.is_file()]
    if resources.get('visual_cache'):
        paths += [Path(resources['visual_cache'])/n for n in ('manifest.json','features.pt')]
    hashes=_hashes(paths)
    if (hashes[str((run/'best.pt').resolve())] != summary['checkpoint_sha256'] or
            hashes[str(Path(cfg['baseline_checkpoint']).resolve())] != base_summary['checkpoint_sha256']):
        raise ValueError('checkpoint identity mismatch')
    identity=code_identity()
    torch.set_num_threads(cfg['num_threads'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    examples,_=load_labeled(cfg['dev_data'])
    gold={e.id:e.aspects for e in examples}
    archived_base,archived_target=_read(cfg['baseline_predictions']),_read(run/'best_dev_predictions.json')
    model,meta=load_checkpoint(cfg['baseline_checkpoint'],device)
    if meta['run_config']['seed'] != seed or model.config.label_scheme != 'five':
        raise ValueError('wrong baseline seed or label scheme')
    base_cfg={**meta['run_config'],**resources}
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    features=None
    if model.config.mode != 'text':
        features,_=open_cache(base_cfg['visual_cache'],base_cfg['vision_model'],base_cfg['image_root'],examples)
    loader=DataLoader(make_dataset(examples,tokenizer,base_cfg,features),batch_size=1,
                      collate_fn=SequenceCollator(tokenizer.pad_token_id))
    baseline,keys,base_logits={ },[],[]
    with torch.no_grad():
        for batch in loader:
            inputs=model_inputs(batch,device)
            assert 'tags' not in inputs
            emissions=model(**inputs)['emissions']
            decoded=model.crf.decode(emissions,inputs['word_mask'])
            for i,(ex,tags) in enumerate(zip(batch['examples'],decoded)):
                aspects=tags_to_aspects(tags)
                baseline[ex.id]=aspects
                scores=aspect_polarity_logits(model.crf,emissions[i,:len(ex.words)],aspects)
                if len(aspects):
                    labels=torch.tensor([p for _,_,p in aspects],device=scores.device)
                    if (scores.max(-1).values-scores[torch.arange(len(scores),device=scores.device),labels] > 1e-4).any():
                        raise ValueError('conditional polarity contradicts Viterbi optimum')
                keys.extend((ex.id,s,e) for s,e,_ in aspects)
                base_logits.extend(scores.cpu().tolist())
    if (not _same(baseline,archived_base) or evaluate_predictions(gold,baseline) != base_summary['best_dev']
            or base_summary['best_dev'] != summary['baseline_dev']):
        raise ValueError('baseline full Dev replay mismatch')
    del model,loader,features
    gc.collect()
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    tokenizer=AutoTokenizer.from_pretrained(run/'tokenizer',local_files_only=True)
    ds=TargetDataset(examples,baseline,tokenizer,False,cfg['max_length'])
    if ds.keys != keys:
        raise ValueError('target boundary ordering mismatch')
    model,meta=load_target(run/'best.pt',device)
    if meta['config']['seed'] != seed or meta['epoch'] != summary['best_epoch']:
        raise ValueError('target checkpoint metadata mismatch')
    loader=DataLoader(ds,batch_size=cfg['eval_batch_size'],collate_fn=TargetCollator(tokenizer.pad_token_id))
    target_logits=[]
    with torch.no_grad():
        for batch in loader:
            assert 'labels' not in batch
            scores=model(**{k:v.to(device) for k,v in batch.items()})['logits']
            if not torch.isfinite(scores).all(): raise ValueError('nonfinite target scores')
            target_logits.extend(scores.cpu().tolist())
    labels=torch.tensor(target_logits).reshape(-1,3).argmax(-1).tolist()
    target=relabel(baseline,keys,labels)
    if not _same(target,archived_target) or evaluate_predictions(gold,target) != summary['best_dev']:
        raise ValueError('target full Dev replay mismatch')
    _verify_hashes(hashes)
    if code_identity()['source_sha256'] != identity['source_sha256']:
        raise ValueError('source changed during extraction')
    result=dict(format='mact_fusion_scores_v1',dataset=dataset,seed=seed,gold=gold,
        baseline=baseline,target=target,keys=keys,base_logits=base_logits,target_logits=target_logits,
        provenance=dict(input_sha256=hashes,code=identity,baseline_replay_exact=True,
                        target_replay_exact=True,test_used=False))
    validate_runs([result],[seed])
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,result)
    output.with_suffix('.sha256').write_text(file_hash(output)+'\n')
    return result


def load_verified(path):
    path=Path(path)
    if file_hash(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('corrupted score cache')
    run=_read(path)
    if run.get('format') != 'mact_fusion_scores_v1': raise ValueError('wrong cache format')
    p=run['provenance']
    if not p['baseline_replay_exact'] or not p['target_replay_exact'] or p['test_used']:
        raise ValueError('unverified score cache')
    _verify_hashes(p['input_sha256'])
    if p['code']['source_sha256'] != code_identity()['source_sha256']:
        raise ValueError('stale source identity')
    validate_runs([run],[run['seed']])
    return run


def sentence_fold(identifier):
    if not isinstance(identifier,str): raise ValueError('sentence ID must be a string')
    return int.from_bytes(hashlib.sha256(('fusion-dev-v1:'+identifier).encode()).digest()[:8],'big')%5


def validate_runs(runs,expected_seeds):
    expected=list(expected_seeds)
    if not expected or len(set(expected)) != len(expected): raise ValueError('invalid expected cohort')
    if len(runs) != len(expected) or sorted(r['seed'] for r in runs) != sorted(expected):
        raise ValueError('incomplete or duplicate seed cohort')
    reference=runs[0]
    for r in runs:
        if r['dataset'] not in ('twitter2015','twitter2017') or r['dataset'] != reference['dataset']:
            raise ValueError('mixed or invalid dataset')
        if not _same(r['gold'],reference['gold']): raise ValueError('gold differs across seeds')
        if set(r['gold']) != set(r['baseline']) or set(r['gold']) != set(r['target']):
            raise ValueError('sentence ID mismatch')
        expected_keys=[]
        for key,values in r['baseline'].items():
            sentence_fold(key)
            if [(s,e) for s,e,_ in values] != [(s,e) for s,e,_ in r['target'][key]]:
                raise ValueError('target boundaries changed')
            for triples in (values,r['target'][key],r['gold'][key]):
                aspects_to_tags(max([e for _,e,_ in triples]+[1]),triples)
            expected_keys.extend((key,s,e) for s,e,_ in values)
        if [tuple(k) for k in r['keys']] != expected_keys:
            raise ValueError('missing, duplicate or misordered score keys')
        # Validates finite scores, label range, and tensor dimensions.
        base=[p for v in r['baseline'].values() for _,_,p in v]
        target=[p for v in r['target'].values() for _,_,p in v]
        fuse_labels(torch.tensor(r['base_logits']).reshape(-1,3),torch.tensor(r['target_logits']).reshape(-1,3),
                    torch.tensor(base,dtype=torch.long),torch.tensor(target,dtype=torch.long),1.,1.,.5)


def _subset(run,ids):
    indices=[i for i,k in enumerate(run['keys']) if k[0] in ids]
    return dict(dataset=run['dataset'],seed=run['seed'],
        **{name:{k:v for k,v in run[name].items() if k in ids} for name in ('gold','baseline','target')},
        **{name:[run[name][i] for i in indices] for name in ('keys','base_logits','target_logits')})


def score_run(run,parameters):
    old=[p for v in run['baseline'].values() for _,_,p in v]
    new=[p for v in run['target'].values() for _,_,p in v]
    labels=fuse_labels(torch.tensor(run['base_logits']).reshape(-1,3),
        torch.tensor(run['target_logits']).reshape(-1,3),torch.tensor(old,dtype=torch.long),
        torch.tensor(new,dtype=torch.long),**parameters).tolist()
    predictions=relabel(run['baseline'],[tuple(k) for k in run['keys']],labels)
    return evaluate_predictions(run['gold'],predictions),predictions


def _fit(runs):
    b,t,y=[],[],[]
    for r in runs:
        matched={(k,s,e):p for k,v in r['gold'].items() for s,e,p in v}
        for key,bs,ts in zip(r['keys'],r['base_logits'],r['target_logits']):
            if tuple(key) in matched:
                b.append(bs);t.append(ts);y.append(matched[tuple(key)])
    parameters=dict(temperature_base=fit_temperature(b,y),temperature_target=fit_temperature(t,y))
    grid=[]
    for alpha in ALPHAS:
        value=statistics.mean(score_run(r,dict(**parameters,alpha=alpha))[0]['joint']['f1'] for r in runs)
        grid.append(dict(alpha=alpha,mean_joint_f1=value))
    # Iteration order makes exact ties prefer the smaller target weight.
    alpha=max(grid,key=lambda x:x['mean_joint_f1'])['alpha']
    return dict(**parameters,alpha=alpha,selection_grid=grid,matched_calibration_count=len(y))


def _decision_params(parameters):
    return {k:parameters[k] for k in ('temperature_base','temperature_target','alpha')}


def crossfit(runs,expected_seeds=SEEDS):
    validate_runs(runs,expected_seeds)
    ids=set(runs[0]['gold'])
    folds={k:sentence_fold(k) for k in ids}
    if set(folds.values()) != set(range(5)): raise ValueError('all five folds must contain sentences')
    predicted={r['seed']:{} for r in runs}
    fold_parameters=[]
    for fold in range(5):
        fit_ids={k for k in ids if folds[k] != fold}
        held_ids=ids-fit_ids
        parameters=_fit([_subset(r,fit_ids) for r in runs])
        fold_parameters.append(dict(fold=fold,fit_sentences=len(fit_ids),held_sentences=len(held_ids),**parameters))
        for r in runs:
            _,pred=score_run(_subset(r,held_ids),_decision_params(parameters))
            predicted[r['seed']].update(pred)
    rows=[]
    for r in runs:
        baseline=evaluate_predictions(r['gold'],r['baseline'])
        target=evaluate_predictions(r['gold'],r['target'])
        raw,_=score_run(r,dict(temperature_base=1.,temperature_target=1.,alpha=.5))
        calibrated=evaluate_predictions(r['gold'],predicted[r['seed']])
        if any(m['mate'] != baseline['mate'] for m in (target,raw,calibrated)):
            raise ValueError('frozen boundary invariant violated')
        rows.append(dict(seed=r['seed'],baseline=baseline,target=target,raw_half=raw,
                         calibrated=calibrated,crossfit_predictions=predicted[r['seed']]))
    cohorts={}
    for name,seeds in [('all',expected_seeds),('initial3',[42,43,44]),('historical12',SEEDS[3:])]:
        selected=[r for r in rows if r['seed'] in seeds]
        if not selected: continue
        values={}
        for method in ('baseline','target','raw_half','calibrated'):
            v=[r[method]['joint']['f1']*100 for r in selected]
            values[method]=dict(mean=statistics.mean(v),sample_std=statistics.stdev(v) if len(v)>1 else None)
        for control in ('baseline','target'):
            d=[100*(r['calibrated']['joint']['f1']-r[control]['joint']['f1']) for r in selected]
            values['delta_vs_'+control]=dict(mean=statistics.mean(d),wins=sum(x>1e-10 for x in d),
                ties=sum(abs(x)<=1e-10 for x in d),losses=sum(x < -1e-10 for x in d))
        cohorts[name]=dict(seeds=[r['seed'] for r in selected],**values)
    return dict(dataset=runs[0]['dataset'],rows=rows,fold_parameters=fold_parameters,
                full_dev_parameters=_fit(runs),fold_assignment=folds,cohorts=cohorts,
                limitation=LIMITATION,test_used=False)
