"""Real-backbone CUDA admission checks for the fixed paired experiment."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
from mact.experiment import resolve_experiment, code_identity
from mact.train import run_training
from mact.evaluate import run_evaluation
from mact.common.cache import file_hash
from mact.runtime import write_json
from run_llrd_ema import resource_hashes, verify_files, validate_smokes


def run_worker(output, dataset, trajectory, physical_gpu):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for real smoke; no CPU fallback')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(physical_gpu):
        raise ValueError('worker GPU visibility does not match assigned physical GPU')
    output=Path(output)
    identity=code_identity()
    config_path=ROOT/f'configs/ema_{trajectory}.json'
    cfg,_=resolve_experiment(config_path,dataset,seed=42,device='cuda:0')
    cfg.update(epochs=1,train_limit=16,dev_limit=8,max_updates=2,log_every=0)
    norms,observed_groups=[],[]
    original_clip=torch.nn.utils.clip_grad_norm_
    original_step=torch.optim.AdamW.step
    def clipped(*args,**kwargs):
        norm=original_clip(*args,**kwargs)
        norms.append(float(norm))
        return norm
    def step(optimizer,*args,**kwargs):
        observed_groups.append([dict(lr=g['lr'],initial_lr=g['initial_lr'],weight_decay=g['weight_decay'],
                                     numel=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups])
        return original_step(optimizer,*args,**kwargs)
    with patch.object(torch.nn.utils,'clip_grad_norm_',clipped),patch.object(torch.optim.AdamW,'step',step):
        summary=run_training(cfg,output)
    write_json(output/'resolved_smoke_config.json',cfg)
    records=json.loads((output/'provenance.json').read_text())['optimizer_groups']
    lr_match=len(observed_groups)==2
    for index,groups in enumerate(observed_groups):
        lr_match=lr_match and len(groups)==len(records)
        for actual,expected in zip(groups,records):
            lr_match=lr_match and (actual['initial_lr']==expected['lr'] and actual['weight_decay']==expected['weight_decay']
                and actual['numel']==expected['numel'] and math.isclose(actual['lr'],expected['lr']*(1-index/2),rel_tol=1e-12))
    dev=json.loads(Path(cfg['dev_data']).read_text())[:8]
    for i,row in enumerate(dev):row.setdefault('id',f"{Path(cfg['dev_data']).stem}:{i}:{row['image_id']}")
    data_path=output/'smoke_dev.json';write_json(data_path,dev)
    exact={}
    for source in ('online','ema'):
        gc.collect();torch.cuda.empty_cache()
        directory=output if source=='online' else output/'ema'
        metrics,predictions=run_evaluation(directory/'best.pt',data_path,directory/'reloaded_dev',device='cuda:0',
            visual_cache=cfg['visual_cache'],resource_paths={k:cfg[k] for k in ('vision_model','image_root')})
        saved=json.loads((directory/'summary.json').read_text())
        exact[source]=(metrics==saved['best_dev'] and json.loads(json.dumps(predictions))==
                       json.loads((directory/'best_dev_predictions.json').read_text()))
    ema_summary=json.loads((output/'ema/summary.json').read_text())
    check=dict(status='passed',dataset=dataset,trajectory=trajectory,device='cuda:0',physical_gpu=physical_gpu,
        peak_cuda_allocated_bytes=summary['peak_cuda_allocated_bytes'],optimizer_updates=summary['optimizer_updates'],
        ema_updates=ema_summary['ema_updates'],head_updated=summary['emission_head_delta_l2']>0,
        finite_gradients=len(norms)==2 and all(math.isfinite(n) and n>0 for n in norms),gradient_norms=norms,
        lr_groups_match=bool(lr_match),observed_optimizer_groups=observed_groups,
        online_reload_exact=exact['online'],ema_reload_exact=exact['ema'],
        source_sha256=identity['source_sha256'],config_sha256=file_hash(config_path))
    if (code_identity()['source_sha256']!=identity['source_sha256'] or check['optimizer_updates']!=2
            or check['ema_updates']!=2 or check['peak_cuda_allocated_bytes']<=0
            or not all(check[k] for k in ('head_updated','finite_gradients','lr_groups_match','online_reload_exact','ema_reload_exact'))):
        check['status']='failed'
    write_json(output/'integration_check.json',check)
    if check['status']!='passed':raise RuntimeError('real CUDA verification failed')
    print(json.dumps({k:check[k] for k in ('status','dataset','trajectory','optimizer_updates','ema_updates','peak_cuda_allocated_bytes')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--gpus',nargs=2,type=int,default=[0,1])
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--dataset',choices=['twitter2015','twitter2017'])
    parser.add_argument('--trajectory',choices=['uniform','llrd'])
    parser.add_argument('--physical-gpu',type=int)
    args=parser.parse_args()
    if args.worker:
        run_worker(Path(args.output),args.dataset,args.trajectory,args.physical_gpu)
        return
    if args.gpus != [0,1]:raise ValueError('approved GPU mapping is 0 and 1')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    identity=code_identity();resources=resource_hashes()
    write_json(output/'manifest.json',dict(source_sha256=identity['source_sha256'],resource_sha256=resources))
    for trajectory in ('uniform','llrd'):
        for gpu,dataset in zip(args.gpus,('twitter2015','twitter2017')):
            destination=output/trajectory/dataset
            with (output/f'{trajectory}-{dataset}.log').open('x') as log:
                subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--output',str(destination),
                    '--dataset',dataset,'--trajectory',trajectory,'--physical-gpu',str(gpu)],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
            print(f'CUDA smoke passed: {trajectory} {dataset}',flush=True)
    verify_files(resources);validate_smokes(output)
    print(f'Four final-source CUDA smokes passed: {output}',flush=True)


if __name__=='__main__':main()
