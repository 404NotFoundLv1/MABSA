"""Train only OCR residual parameters, then diagnose the selected Dev checkpoint."""
import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .common.cache import CachedVisionDataset, file_hash, open_cache
from .common.metrics import evaluate_predictions
from .data import SequenceDataset, SequenceCollator, load_labeled
from .experiment import ROOT, DATASETS, code_identity, resolve_experiment
from .logging import TrainingLogger
from .ocr import open_ocr
from .ocr_residual import (OCRDataset, OCRCollator, OCRResidualModel, residual_train_epoch,
                           save_residual, load_residual)
from .runtime import evaluate_model, load_checkpoint, model_inputs, write_json


def state_hash(model):
    digest = hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def read_config(path,dataset,device,seed=None):
    config = json.loads(Path(path).read_text())
    if seed is not None:
        config['seed']=seed
    defaults = json.loads((ROOT/'configs/ocr_residual.json').read_text())
    optional={'memory_source','standalone_weight'}
    if set(config)-optional!=set(defaults):
        raise ValueError('missing or unknown OCR residual configuration keys')
    if config.get('memory_source','ocr') not in ('ocr','text'):
        raise ValueError('memory_source must be ocr or text')
    weight=config.get('standalone_weight',0.)
    if type(weight) not in (int,float) or not math.isfinite(weight) or weight<0:
        raise ValueError('standalone_weight must be finite and nonnegative')
    for key in ('seed','epochs','ocr_max_tokens','attention_heads','accumulation','batch_size',
                'eval_batch_size','num_threads'):
        if type(config[key]) is not int or config[key]<1:
            raise ValueError(f'{key} must be a positive integer')
    if config['seed']>100:
        raise ValueError('seed must be in [1,100]')
    if not 0<=config['warmup_ratio']<1:
        raise ValueError('invalid warmup ratio')
    for key in ('learning_rate','weight_decay','gradient_clip'):
        if not math.isfinite(config[key]) or config[key]<0 or (key!='weight_decay' and config[key]==0):
            raise ValueError(f'invalid {key}')
    base,_ = resolve_experiment(ROOT/'configs/base.json',dataset,seed=config['seed'],device=device)
    for key in ('baseline','ocr_cache'):
        path = (ROOT/config[key].format(dataset=dataset,seed=config['seed'])).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f'{key} must be inside MACT')
        config[key] = str(path)
    return dict(config,dataset=dataset,device=device,baseline_run=base)


def shuffled_ocr(dataset,identities,seed):
    """Derange nonempty OCR, minimizing content-length mismatch; leave empty rows empty."""
    active = [i for i,x in enumerate(dataset.ocr) if x['ocr_content_mask'].any()]
    if len(active)<2:
        raise ValueError('at least two nonempty OCR examples required for shuffle diagnostic')
    tokens = [dataset.ocr[i]['ocr_input_ids'][dataset.ocr[i]['ocr_content_mask']].tolist() for i in active]
    sizes = np.array(list(map(len,tokens)))
    costs = np.abs(sizes[:,None]-sizes[None,:]).astype(float)
    costs += np.random.default_rng(seed).random(costs.shape)/(len(active)+1)
    forbidden = np.array([[identities[i]==identities[j] or tokens[a]==tokens[b]
                          for b,j in enumerate(active)] for a,i in enumerate(active)])
    costs[forbidden] = 1e9
    rows,cols = linear_sum_assignment(costs)
    if forbidden[rows,cols].any():
        raise ValueError('no different-image/different-text OCR derangement exists')
    donors = dict(zip(active,[active[j] for j in cols]))
    mismatch = [abs(len(tokens[i])-len(tokens[j])) for i,j in zip(rows,cols)]
    return donors,dict(seed=seed,donors=donors,exact_lengths=mismatch.count(0),
        mean_length_difference=float(np.mean(mismatch)),max_length_difference=max(mismatch))


class OCRIntervention:
    def __init__(self,dataset,donors=None,remove=False):
        self.dataset,self.donors,self.remove = dataset,donors,remove

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        item = self.dataset[index]
        if self.remove:
            item['ocr_content_mask'].zero_()
        elif index in self.donors:
            item.update({k:v.clone() for k,v in self.dataset.ocr[self.donors[index]].items()})
        return item


def normalized(value):
    return json.loads(json.dumps(value))


def group_metrics(examples,predictions,groups):
    gold = {e.id:e.aspects for e in examples}
    return {name:evaluate_predictions({i:gold[i] for i in ids},{i:predictions[i] for i in ids})
            for name,ids in groups.items()}


def diagnostic(model,dev_ds,examples,tokenizer,device,baseline_predictions,expected_predictions,
               identities,output,smoke=False):
    groups = dict(all=[e.id for e in examples],has_ocr=[],no_ocr=[])
    for i,e in enumerate(examples):
        groups['has_ocr' if dev_ds.ocr[i]['ocr_content_mask'].any() else 'no_ocr'].append(e.id)
    result = dict(groups={k:len(v) for k,v in groups.items()},metrics={
        'baseline':group_metrics(examples,baseline_predictions,groups)},shuffles={})
    modes = {'normal':dev_ds,'removed':OCRIntervention(dev_ds,remove=True)}
    if not smoke:
        for seed in (17,43,97):
            donors,details = shuffled_ocr(dev_ds,identities,seed)
            result['shuffles'][f'shuffle{seed}'] = details
            modes[f'shuffle{seed}'] = OCRIntervention(dev_ds,donors=donors)
    for mode,dataset in modes.items():
        loader = DataLoader(dataset,batch_size=1,collate_fn=OCRCollator(tokenizer.pad_token_id))
        metrics,predictions = evaluate_model(model,loader,device)
        predictions = normalized(predictions)
        if mode=='normal' and predictions!=normalized(expected_predictions):
            raise ValueError('residual checkpoint reload predictions differ')
        if mode=='removed' and predictions!=normalized(baseline_predictions):
            raise ValueError('removed OCR must reproduce baseline exactly')
        if any(predictions[i]!=baseline_predictions[i] for i in groups['no_ocr']):
            raise ValueError('no-OCR prediction changed')
        if (model.memory_source=='text' and mode.startswith('shuffle')
                and predictions!=normalized(expected_predictions)):
            raise ValueError('text control must ignore shuffled OCR content')
        result['metrics'][mode] = group_metrics(examples,predictions,groups)
        write_json(output/f'{mode}-predictions.json',predictions)
        print(f"DEV diagnostic {mode}: aesc_f:{100*metrics['joint']['f1']:.2f}",flush=True)
    result.update(reload_exact=True,no_ocr_predictions_exact=True,removed_predictions_exact=True,
                  memory_source=model.memory_source,evaluate_test=False,is_smoke=smoke)
    write_json(output/'diagnostic.json',result)
    return result


def run(config_path,dataset,output,device='cuda:0',smoke=False,seed=None):
    started = time.monotonic()
    cfg = read_config(config_path,dataset,device,seed=seed)
    if smoke:
        cfg['epochs']=1
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'output must be fresh: {output}')
    output.mkdir(parents=True)
    write_json(output/'status.json',dict(state='preparing',evaluate_test=False,is_smoke=smoke))
    try:
        return _run(cfg,output,smoke,started,config_path)
    except BaseException as error:
        write_json(output/'status.json',dict(state='failed',error=repr(error),evaluate_test=False,is_smoke=smoke))
        raise


def protected_paths(cfg,config_path,reference):
    """Exact files whose identities must survive the complete experiment."""
    base_cfg = cfg['baseline_run']
    protected = [Path(cfg['baseline']),Path(config_path),reference/'summary.json',
        reference/'best_dev_predictions.json',reference/'provenance.json',
        Path(cfg['ocr_cache'])/'manifest.json',Path(cfg['ocr_cache'])/'texts.json',
        Path(base_cfg['visual_cache'])/'manifest.json']
    protected += [Path(base_cfg[k+'_data']) for k in ('train','dev')]
    protected += sorted(p for p in (reference/'tokenizer').rglob('*') if p.is_file())
    protected.append(Path(base_cfg['visual_cache'])/'features.pt')
    return protected


def _run(cfg,output,smoke,started,config_path):
    dataset,device = cfg['dataset'],cfg['device']
    base_cfg = cfg['baseline_run']
    identity = code_identity()
    reference = ROOT/'reference/runs'/dataset/f"seed{cfg['seed']}"
    reference_summary = json.loads((reference/'summary.json').read_text())
    protected = protected_paths(cfg,config_path,reference)
    hashes = {str(p):file_hash(p) for p in protected}
    if hashes[cfg['baseline']]!=reference_summary['checkpoint_sha256']:
        raise ValueError('baseline checkpoint does not match archived seed42 run')
    if reference_summary['completed_epochs']!=35 or reference_summary['is_smoke']:
        raise ValueError('reference baseline must be a full 35-epoch run')
    baseline_provenance = json.loads((reference/'provenance.json').read_text())
    for split in ('train','dev'):
        if hashes[base_cfg[split+'_data']]!=baseline_provenance[split+'_sha256']:
            raise ValueError('official split differs from baseline')
    torch.set_num_threads(cfg['num_threads'])
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no silent CPU fallback')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    train,_=load_labeled(base_cfg['train_data'])
    dev,_=load_labeled(base_cfg['dev_data'])
    if smoke:
        train,dev=train[:24],dev[:12]
    features,visual_meta = open_cache(base_cfg['visual_cache'],base_cfg['vision_model'],base_cfg['image_root'],train+dev)
    texts,ocr_meta = open_ocr(cfg['ocr_cache'],base_cfg['image_root'],train+dev)
    cfg['ocr_manifest_sha256'] = ocr_meta['manifest_sha256']
    tokenizer=AutoTokenizer.from_pretrained(reference/'tokenizer',local_files_only=True,use_fast=True)
    tokenizer.save_pretrained(output/'tokenizer')
    originals=[CachedVisionDataset(SequenceDataset(examples,tokenizer,max_length=base_cfg['max_length']),features)
               for examples in (train,dev)]
    train_ds,dev_ds=[OCRDataset(ds,tokenizer,texts,cfg['ocr_max_tokens']) for ds in originals]
    collator=OCRCollator(tokenizer.pad_token_id)
    train_loader=DataLoader(train_ds,batch_size=cfg['batch_size'],shuffle=True,
        generator=torch.Generator().manual_seed(cfg['seed']),collate_fn=collator,num_workers=0)
    dev_loader=DataLoader(dev_ds,batch_size=cfg['eval_batch_size'],collate_fn=collator,num_workers=0)
    baseline,base_meta=load_checkpoint(cfg['baseline'],device)
    if base_meta['run_config'].get('ocr_cache') or base_meta['run_config'].get('rdrop_weight',0):
        raise ValueError('baseline has a previous treatment')
    if base_meta['run_config']['seed']!=cfg['seed'] or base_meta['epoch']!=reference_summary['best_epoch']:
        raise ValueError('baseline seed or selected epoch mismatch')
    baseline.requires_grad_(False).eval()
    frozen_hash=state_hash(baseline)
    old_loader=DataLoader(originals[1],batch_size=cfg['eval_batch_size'],
                          collate_fn=SequenceCollator(tokenizer.pad_token_id))
    baseline_metrics,baseline_predictions=evaluate_model(baseline,old_loader,device)
    baseline_predictions=normalized(baseline_predictions)
    archived=json.loads((reference/'best_dev_predictions.json').read_text())
    archived={e.id:archived[e.id] for e in dev}
    if baseline_predictions!=archived:
        raise ValueError('baseline does not reproduce archived Dev predictions exactly')
    if not smoke and baseline_metrics!=reference_summary['best_dev']:
        raise ValueError('baseline does not reproduce full Dev metrics')
    model=OCRResidualModel(baseline,heads=cfg['attention_heads'],
        memory_source=cfg.get('memory_source','ocr'),standalone_weight=cfg.get('standalone_weight',0.),
        memory_max_tokens=cfg['ocr_max_tokens']).to(device)
    # Check exact zero-init scores on every Dev row; comparison uses the frozen baseline.
    model.eval()
    with torch.no_grad():
        for batch in dev_loader:
            prediction=model(**model_inputs(batch,device))
            if not torch.equal(prediction['emissions'],prediction['base_emissions']):
                raise ValueError('zero initialized residual changed baseline emissions')
    initial_metrics,initial_predictions=evaluate_model(model,dev_loader,device)
    if initial_metrics!=baseline_metrics or normalized(initial_predictions)!=baseline_predictions:
        raise ValueError('initial OCR residual model does not match baseline')
    write_json(output/'baseline-verification.json',dict(archived_dev_predictions_exact=True,
        full_dev=not smoke,zero_initial_emissions_exact=True,baseline_state_sha256=frozen_hash,
        baseline_metrics=baseline_metrics))
    write_json(output/'baseline_dev_predictions.json',baseline_predictions)
    write_json(output/'config.json',cfg)
    trainable=sum(p.numel() for p in model.branch.parameters())
    write_json(output/'provenance.json',dict(code=identity,protected_sha256=hashes,ocr=ocr_meta,
        visual_cache_sha256=visual_meta['features_sha256'],baseline_state_sha256=frozen_hash,
        trainable_parameters=trainable,baseline_frozen=True,precision='fp32',evaluate_test=False,
        is_smoke=smoke,selection='best Dev joint F1 among trained epochs 1..35; epoch0 reported separately',
        objective='sentence_mean_fused_crf_plus_weighted_active_standalone_crf' if model.standalone_weight
            else 'sentence_mean_crf_nll_all_sentences_no_ocr_constant',
        memory_source=model.memory_source,standalone_weight=model.standalone_weight,
        scheduling='all accumulation windows; optimizer skips all-empty windows',
        frozen_OCR_encoder='same frozen baseline DeBERTa',
        memory_description='independent OCR sequence' if model.memory_source=='ocr'
            else 'original sentence encoded; first OCR-budget content subwords as memory; OCR presence gate only'))
    logger=TrainingLogger(output)
    logger.info(f'Frozen MACT + independent OCR residual | {dataset}',pad=True)
    logger.info(f"seed:{cfg['seed']} epochs:{cfg['epochs']} trainable:{trainable} LR:{cfg['learning_rate']:g}")
    logger.info(f'Memory: {model.memory_source}; standalone loss weight: {model.standalone_weight:g}; active rows: OCR present')
    logger.info('Selection: Dev JMASA F1, trained epochs only. Baseline and CRF frozen; no Test evaluation.')
    logger.joint_metrics('BASELINE DEV',baseline_metrics['joint'])
    groups=[dict(params=[p for p in model.branch.parameters() if (p.ndim>=2)==decay],
                 weight_decay=cfg['weight_decay'] if decay else 0.) for decay in (True,False)]
    optimizer=torch.optim.AdamW(groups,lr=cfg['learning_rate'])
    planned=math.ceil(len(train_loader)/cfg['accumulation'])*cfg['epochs']
    scheduler=get_linear_schedule_with_warmup(optimizer,int(planned*cfg['warmup_ratio']),planned)
    best=-1.;history=[]
    no_ocr_ids=[dev[i].id for i,x in enumerate(dev_ds.ocr) if not x['ocr_content_mask'].any()]
    write_json(output/'status.json',dict(state='training',evaluate_test=False,is_smoke=smoke))
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,cfg['epochs']+1):
        result=residual_train_epoch(model,train_loader,optimizer,device,cfg['accumulation'],cfg['gradient_clip'],scheduler)
        metrics,predictions=evaluate_model(model,dev_loader,device)
        predictions=normalized(predictions)
        if any(predictions[i]!=baseline_predictions[i] for i in no_ocr_ids):
            raise ValueError('no-OCR Dev predictions changed during training')
        if any(p.requires_grad or p.grad is not None for p in model.baseline.parameters()):
            raise ValueError('baseline must remain frozen')
        record=dict(epoch=epoch,train=result,dev=metrics,no_ocr_predictions_exact=True,
                    elapsed_seconds=time.monotonic()-started)
        history.append(record)
        write_json(output/'history.json',history)
        logger.event(dict(event='epoch',**record))
        logger.info(f"Epoch [{epoch}/{cfg['epochs']}], Mean Loss: {result['objective_loss']:.4f}",pad=True)
        if cfg.get('standalone_weight',0.):
            logger.info(f"Fused NLL: {result['sentence_nll']:.4f}; Standalone NLL: {result['standalone_nll']:.4f} (all-sentence means)")
        logger.joint_metrics('DEV',metrics['joint'])
        logger.info(f"DEV AE  ae_p:{100*metrics['mate']['precision']:.2f} ae_r:{100*metrics['mate']['recall']:.2f} ae_f:{100*metrics['mate']['f1']:.2f}")
        if metrics['joint']['f1']>best:
            best,best_epoch,best_metrics=metrics['joint']['f1'],epoch,metrics
            save_residual(output/'best.pt',model,cfg['baseline'],cfg,epoch,metrics)
            write_json(output/'best_dev_predictions.json',predictions)
            logger.info(f'Saved best Dev checkpoint at epoch {epoch}.')
    save_residual(output/'last.pt',model,cfg['baseline'],cfg,epoch,metrics)
    if state_hash(model.baseline)!=frozen_hash:
        raise ValueError('frozen baseline state changed')
    if not torch.isfinite(model.branch.output.weight).all() or not model.branch.output.weight.abs().sum():
        raise ValueError('residual output failed to learn finite nonzero weights')
    peak=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None
    del optimizer,scheduler,model,baseline
    gc.collect()
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    restored,metadata=load_residual(output/'best.pt',device)
    expected=json.loads((output/'best_dev_predictions.json').read_text())
    report=diagnostic(restored,dev_ds,dev,tokenizer,device,baseline_predictions,expected,
        [visual_meta['image_sha256'][e.image_id] for e in dev],output,smoke)
    if report['metrics']['normal']['all']!=best_metrics or metadata['epoch']!=best_epoch:
        raise ValueError('reloaded selected metrics/epoch mismatch')
    for path,sha in hashes.items():
        if file_hash(path)!=sha:raise ValueError(f'protected input changed: {path}')
    for path,sha in identity['source_sha256'].items():
        if file_hash(ROOT/path)!=sha:raise ValueError(f'source changed during run: {path}')
    summary=dict(dataset=dataset,seed=cfg['seed'],best_epoch=best_epoch,best_dev=best_metrics,
        baseline_dev=baseline_metrics,delta_f1_points=100*(best-baseline_metrics['joint']['f1']),
        completed_epochs=len(history),optimizer_updates=sum(r['train']['updates'] for r in history),
        windows=sum(r['train']['windows'] for r in history),trainable_parameters=trainable,
        baseline_frozen_exact=True,reload_exact=True,inputs_unchanged=True,source_unchanged=True,
        elapsed_seconds=time.monotonic()-started,peak_cuda_allocated_bytes=peak,
        checkpoint_sha256=file_hash(output/'best.pt'),is_smoke=smoke,evaluate_test=False)
    write_json(output/'summary.json',summary)
    write_json(output/'status.json',dict(state='completed',evaluate_test=False,is_smoke=smoke))
    logger.info(f'Best trained epoch: {best_epoch}; delta vs baseline: {summary["delta_f1_points"]:+.2f} points')
    logger.joint_metrics('BEST DEV',best_metrics['joint'])
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/ocr_residual.json'))
    parser.add_argument('--dataset',required=True,choices=DATASETS)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    run(args.config,args.dataset,args.output,args.device,args.smoke,seed=args.seed)


if __name__=='__main__':main()
