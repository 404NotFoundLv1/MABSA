#!/usr/bin/env python3
"""Detached two-GPU paired-seed runs, Dev diagnostics and complete-seed aggregation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mact.common.cache import file_hash
from mact.experiment import code_identity, DATASETS
from mact.runtime import write_json
from mact.ocr_residual_train import read_config
from mact.ocr_residual_results import paired_summary, same_treatment, verified_row

CONFIG=ROOT/'configs/ocr_residual.json'
OUT=ROOT/'outputs/ocr_residual_s42_20261015'
VERIFY=ROOT/'verification/ocr_residual'
SEEDS=(42,)
PRIOR=None


def make_jobs(seeds,output):
    if (not seeds or len(set(seeds))!=len(seeds)
            or any(type(s) is not int or not 1<=s<=100 for s in seeds)):
        raise ValueError('seeds must be unique integers in [1,100]')
    return [dict(dataset=dataset,seed=seed,gpu=gpu,output=Path(output)/dataset/f'seed{seed}')
            for gpu,dataset in enumerate(DATASETS) for seed in seeds]


def resolved(dataset,seed):
    cfg=read_config(CONFIG,dataset,'cuda:0',seed=seed)
    cfg['ocr_manifest_sha256']=file_hash(Path(cfg['ocr_cache'])/'manifest.json')
    return cfg


def smoke_path(dataset,seed):
    suffix='' if SEEDS==(42,) else f'-s{seed}'
    return VERIFY/f'final-smoke-{dataset}{suffix}'


def smoke_succeeded(summary,status):
    return (status['state']=='completed' and not summary['evaluate_test'] and summary['is_smoke']
            and summary['baseline_frozen_exact'] and summary['reload_exact']
            and summary['windows']==3 and summary['completed_epochs']==1
            and 0<summary['optimizer_updates']<=summary['windows']
            and (summary['peak_cuda_allocated_bytes'] or 0)>0)


def preflight():
    identity=code_identity()
    if identity['git_dirty']:
        raise ValueError('commit source before launching the pilot')
    config=json.loads(CONFIG.read_text())
    if config['epochs']!=35 or config['batch_size']!=1 or config['eval_batch_size']!=1:
        raise ValueError('pilot requires 35 epochs, batch size 1')
    proofs={}
    for job in make_jobs(SEEDS,OUT):
        dataset,seed=job['dataset'],job['seed']
        smoke=smoke_path(dataset,seed)
        summary=json.loads((smoke/'summary.json').read_text())
        provenance=json.loads((smoke/'provenance.json').read_text())
        status=json.loads((smoke/'status.json').read_text())
        if not smoke_succeeded(summary,status):
            raise ValueError(f'{dataset}: real GPU smoke did not pass')
        if provenance['code']['source_sha256']!=identity['source_sha256']:
            raise ValueError('smoke verification is stale after source edits')
        smoke_cfg=json.loads((smoke/'config.json').read_text())
        smoke_cfg['epochs']=35
        expected=resolved(dataset,seed)
        if (summary['dataset']!=dataset or summary['seed']!=seed
                or smoke_cfg['baseline']!=expected['baseline'] or not same_treatment(smoke_cfg,expected)):
            raise ValueError('smoke used a different seed, baseline or treatment')
        for path,sha in provenance['protected_sha256'].items():
            if file_hash(path)!=sha:raise ValueError(f'verified input changed: {path}')
        proofs[f'{dataset}-s{seed}']=file_hash(smoke/'summary.json')
    prior_proofs={}
    if PRIOR is not None:
        if 42 in SEEDS:raise ValueError('seed42 already supplied by prior results')
        for dataset in DATASETS:
            run=PRIOR/dataset/'seed42'
            verified_row(run,resolved(dataset,42),ROOT)
            prior_proofs[dataset]=file_hash(run/'summary.json')
            # Encoder/attention/CRF computation must be exactly the seed42 code.
            provenance=json.loads((run/'provenance.json').read_text())
            for path in ('src/mact/model.py','src/mact/ocr_residual.py','src/mact/crf.py','src/mact/runtime.py'):
                if file_hash(ROOT/path)!=provenance['code']['source_sha256'][path]:
                    raise ValueError(f'core computation differs from seed42: {path}')
    return dict(code=identity,config_sha256=file_hash(CONFIG),smoke_proofs=proofs,prior_proofs=prior_proofs)


def collect():
    rows=[]
    for job in make_jobs(SEEDS,OUT):
        rows.append(verified_row(job['output'],resolved(job['dataset'],job['seed']),ROOT))
    seeds=list(SEEDS)
    if PRIOR is not None:
        seeds=[42]+seeds
        for dataset in DATASETS:
            rows.append(verified_row(PRIOR/dataset/'seed42',resolved(dataset,42),ROOT))
    rows.sort(key=lambda r:(r['dataset'],r['seed']))
    aggregate=paired_summary(rows,seeds)
    result=dict(state='completed',seeds=seeds,evaluate_test=False,rows=rows,aggregate=aggregate)
    write_json(OUT/'results.json',result)
    lines=['Dev only; best among trained epochs 1..35. Baseline epoch0 reported separately.',
           'dataset seed best_epoch baseline_F1 residual_P residual_R residual_F1 delta_F1']
    for row in rows:
        s=row['summary'];m=s['best_dev']['joint']
        lines.append(f"{row['dataset']} {row['seed']} {s['best_epoch']} {100*s['baseline_dev']['joint']['f1']:.2f} "
            f"{100*m['precision']:.2f} {100*m['recall']:.2f} {100*m['f1']:.2f} {s['delta_f1_points']:+.2f}")
        for mode,groups in row['diagnostic']['metrics'].items():
            lines.append(f"  {mode}: "+' '.join(f"{g}={100*v['joint']['f1']:.2f}" for g,v in groups.items()))
    lines.append('Paired aggregate (percent / percentage points; sample SD):')
    lines.append(json.dumps(aggregate['datasets'],indent=2))
    (OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    return result


def worker():
    if OUT.exists():raise FileExistsError(OUT)
    proof=preflight()
    OUT.mkdir(parents=True)
    started=time.time()
    write_json(OUT/'launch.json',dict(**proof,seeds=list(SEEDS),epochs=35,evaluate_test=False,
        prior_results=str(PRIOR) if PRIOR else None,
        assignment=dict(zip(DATASETS,[0,1])),started_at=started,automatic_dev_diagnostics=True))
    children=[]
    handles=[]
    try:
        for job in make_jobs(SEEDS,OUT):
            gpu,dataset,seed=job['gpu'],job['dataset'],job['seed']
            run=job['output']
            run.parent.mkdir(exist_ok=True)
            log_name='train.log' if SEEDS==(42,) else f'seed{seed}.log'
            log=(run.parent/log_name).open('x');handles.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
            cmd=[sys.executable,'-u',str(ROOT/'scripts/train_ocr_residual.py'),'--config',str(CONFIG),
                 '--dataset',dataset,'--seed',str(seed),'--output',str(run),'--device','cuda:0']
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            children.append((f'{dataset}-s{seed}',process))
            print(f'{dataset} seed{seed} GPU{gpu} PID {process.pid}',flush=True)
        write_json(OUT/'status.json',dict(state='running',started_at=started,
            pids={name:p.pid for name,p in children},evaluate_test=False))
        codes={name:p.wait() for name,p in children}
        if any(codes.values()):raise RuntimeError(f'training failed: {codes}')
        collect()
        write_json(OUT/'status.json',dict(state='completed',started_at=started,ended_at=time.time(),evaluate_test=False))
    except BaseException as error:
        for _,p in children:
            if p.poll() is None:p.terminate()
        for _,p in children:
            if p.poll() is None:
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        write_json(OUT/'status.json',dict(state='failed',error=repr(error),started_at=started,
            ended_at=time.time(),evaluate_test=False))
        raise
    finally:
        for log in handles:log.close()


def main():
    global CONFIG,OUT,VERIFY,SEEDS,PRIOR
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--collect',action='store_true')
    parser.add_argument('--config',default=str(CONFIG))
    parser.add_argument('--output',default=str(OUT))
    parser.add_argument('--verification',default=str(VERIFY))
    parser.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS))
    parser.add_argument('--prior-output')
    args=parser.parse_args()
    CONFIG,OUT,VERIFY=(Path(x).resolve() for x in (args.config,args.output,args.verification))
    SEEDS=tuple(args.seeds)
    PRIOR=Path(args.prior_output).resolve() if args.prior_output else None
    make_jobs(SEEDS,OUT)
    for path in (CONFIG,OUT,VERIFY)+((PRIOR,) if PRIOR else ()):
        if not path.is_relative_to(ROOT):raise ValueError('all pilot paths must be inside MACT')
    if args.collect:
        collect();return
    if args.worker:
        worker();return
    proof=preflight()
    if args.check_only:
        print(json.dumps(proof,ensure_ascii=False));return
    if OUT.exists():raise FileExistsError(OUT)
    with OUT.with_name(OUT.name+'.log').open('x') as log:
        worker_args=['--config',str(CONFIG),'--output',str(OUT),'--verification',str(VERIFY),
                     '--seeds',*[str(s) for s in SEEDS]]
        if PRIOR is not None:worker_args+=['--prior-output',str(PRIOR)]
        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',*worker_args],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(OUT.with_name(OUT.name+'.pid.json'),dict(pid=process.pid))
    print(f'Pilot supervisor PID: {process.pid}',flush=True)


if __name__=='__main__':main()
