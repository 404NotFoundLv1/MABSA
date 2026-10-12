"""Run both datasets in sequential seed groups, optionally detached."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mact.experiment import code_identity,resolve_experiment
from mact.common.cache import file_hash
from mact.runtime import write_json
from mact.schedule import build_groups,run_schedule
from collect_results import collect

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',nargs='+',type=int,required=True)
    p.add_argument('--group-size',type=int,default=4)
    p.add_argument('--gpus',nargs='+',type=int,default=[0,1])
    p.add_argument('--config',default='configs/base.json')
    p.add_argument('--output',required=True)
    p.add_argument('--detach',action='store_true')
    a=p.parse_args();out=Path(a.output).resolve()
    groups=build_groups(a.seeds,a.group_size,a.gpus,out)
    config=Path(a.config);config=(config if config.is_absolute() else ROOT/config).resolve()
    for dataset in ('twitter2015','twitter2017'):resolve_experiment(config,dataset)
    if out.exists():raise FileExistsError(out)
    if a.detach:
        out.parent.mkdir(parents=True,exist_ok=True)
        argv=[x for x in sys.argv[1:] if x!='--detach']
        with out.with_name(out.name+'.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),*argv],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(out.with_name(out.name+'.pid.json'),dict(pid=child.pid))
        print(f'Batch PID: {child.pid}; status: {out / "batch-status.json"}')
        return
    identity=code_identity()
    config_hash=file_hash(config)
    def verify_code():
        for name,sha in identity['source_sha256'].items():
            if file_hash(ROOT/name)!=sha:raise ValueError(f'code/config changed during batch: {name}')
        if file_hash(config)!=config_hash:raise ValueError('batch configuration changed')
        if not (out/'launch.json').exists():
            write_json(out/'launch.json',dict(groups=groups,seeds=a.seeds,code=identity,
                config=str(config),config_sha256=config_hash,executable=sys.executable))
    def command(job):
        return [sys.executable,'-u',str(ROOT/'scripts/train.py'),'--config',str(config),
                '--dataset',job['dataset'],'--seed',str(job['seed']),'--device','cuda:0','--output',job['output']]
    result=run_schedule(groups,out,command,before_group=verify_code,
                        after_group=lambda:collect(out,a.seeds))
    if result['state']!='completed':raise SystemExit(1)

if __name__=='__main__':main()
