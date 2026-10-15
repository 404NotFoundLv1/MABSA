#!/usr/bin/env python3
"""Detached two-GPU seed42 pilot; train both datasets then collect Dev diagnostics."""
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

CONFIG=ROOT/'configs/ocr_residual.json'
OUT=ROOT/'outputs/ocr_residual_s42_20261015'
VERIFY=ROOT/'verification/ocr_residual'


def preflight():
    identity=code_identity()
    if identity['git_dirty']:
        raise ValueError('commit source before launching the pilot')
    config=json.loads(CONFIG.read_text())
    if config['seed']!=42 or config['epochs']!=35 or config['batch_size']!=1 or config['eval_batch_size']!=1:
        raise ValueError('pilot requires seed42, 35 epochs, batch size 1')
    proofs={}
    for dataset in DATASETS:
        smoke=VERIFY/f'final-smoke-{dataset}'
        summary=json.loads((smoke/'summary.json').read_text())
        provenance=json.loads((smoke/'provenance.json').read_text())
        status=json.loads((smoke/'status.json').read_text())
        if (status['state']!='completed' or summary['evaluate_test'] or not summary['is_smoke']
                or not summary['baseline_frozen_exact'] or not summary['reload_exact']
                or summary['optimizer_updates']!=3):
            raise ValueError(f'{dataset}: real GPU smoke did not pass')
        if provenance['code']['source_sha256']!=identity['source_sha256']:
            raise ValueError('smoke verification is stale after source edits')
        for path,sha in provenance['protected_sha256'].items():
            if file_hash(path)!=sha:raise ValueError(f'verified input changed: {path}')
        proofs[dataset]=file_hash(smoke/'summary.json')
    return dict(code=identity,config_sha256=file_hash(CONFIG),smoke_proofs=proofs)


def collect():
    rows=[]
    for dataset in DATASETS:
        run=OUT/dataset/'seed42'
        status=json.loads((run/'status.json').read_text())
        if status['state']!='completed':
            raise ValueError(f'{dataset} did not complete: {status}')
        summary=json.loads((run/'summary.json').read_text())
        diagnostic=json.loads((run/'diagnostic.json').read_text())
        if summary['completed_epochs']!=35 or summary['evaluate_test'] or summary['is_smoke']:
            raise ValueError('incomplete or invalid full run')
        rows.append(dict(dataset=dataset,summary=summary,diagnostic=diagnostic))
    result=dict(state='completed',seed=42,evaluate_test=False,rows=rows)
    write_json(OUT/'results.json',result)
    lines=['Dev only; best among trained epochs 1..35. Baseline epoch0 reported separately.',
           'dataset best_epoch baseline_F1 residual_P residual_R residual_F1 delta_F1']
    for row in rows:
        s=row['summary'];m=s['best_dev']['joint']
        lines.append(f"{row['dataset']} {s['best_epoch']} {100*s['baseline_dev']['joint']['f1']:.2f} "
            f"{100*m['precision']:.2f} {100*m['recall']:.2f} {100*m['f1']:.2f} {s['delta_f1_points']:+.2f}")
        for mode,groups in row['diagnostic']['metrics'].items():
            lines.append(f"  {mode}: "+' '.join(f"{g}={100*v['joint']['f1']:.2f}" for g,v in groups.items()))
    (OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    return result


def worker():
    if OUT.exists():raise FileExistsError(OUT)
    proof=preflight()
    OUT.mkdir(parents=True)
    started=time.time()
    write_json(OUT/'launch.json',dict(**proof,seed=42,epochs=35,evaluate_test=False,
        assignment=dict(zip(DATASETS,[0,1])),started_at=started,automatic_dev_diagnostics=True))
    children=[]
    handles=[]
    try:
        for gpu,dataset in enumerate(DATASETS):
            run=OUT/dataset/'seed42'
            run.parent.mkdir()
            log=(run.parent/'train.log').open('x');handles.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
            cmd=[sys.executable,'-u',str(ROOT/'scripts/train_ocr_residual.py'),'--config',str(CONFIG),
                 '--dataset',dataset,'--output',str(run),'--device','cuda:0']
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            children.append((dataset,process))
            print(f'{dataset} seed42 GPU{gpu} PID {process.pid}',flush=True)
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--collect',action='store_true')
    args=parser.parse_args()
    if args.collect:
        collect();return
    if args.worker:
        worker();return
    proof=preflight()
    if args.check_only:
        print(json.dumps(proof,ensure_ascii=False));return
    if OUT.exists():raise FileExistsError(OUT)
    with OUT.with_name(OUT.name+'.log').open('x') as log:
        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(OUT.with_name(OUT.name+'.pid.json'),dict(pid=process.pid))
    print(f'Pilot supervisor PID: {process.pid}',flush=True)


if __name__=='__main__':main()
