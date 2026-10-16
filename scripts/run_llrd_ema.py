"""Run the fixed twelve-job Dev-only LLRD/EMA cohort after CUDA admission checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mact.common.cache import file_hash, model_hashes
from mact.experiment import resolve_experiment, code_identity
from mact.llrd_ema_results import collect
from mact.runtime import write_json
from mact.schedule import build_groups, run_schedule

SEEDS = [42,43,44]
DATASETS = ('twitter2015','twitter2017')
TRAJECTORIES = ('uniform','llrd')


def cohort_groups(output):
    result = {}
    for arm in TRAJECTORIES:
        result[arm] = build_groups(SEEDS,1,[0,1],Path(output)/arm)
        for group in result[arm]:
            for job in group:
                job['trajectory'] = arm
    return result


def training_command(job):
    return [sys.executable,'-u',str(ROOT/'scripts/train.py'),'--config',
            str(ROOT/f"configs/ema_{job['trajectory']}.json"),'--dataset',job['dataset'],
            '--seed',str(job['seed']),'--device','cuda:0','--output',job['output'],'--train-only']


def verify_files(hashes):
    for path,sha in hashes.items():
        if file_hash(path) != sha:
            raise ValueError(f'frozen file changed: {path}')


def resource_hashes():
    paths = set()
    for ds in DATASETS:
        cfg,_ = resolve_experiment('configs/ema_uniform.json',ds)
        paths.update(cfg[k] for k in ('train_data','dev_data'))
        paths.update(str(Path(cfg['visual_cache'])/n) for n in ('manifest.json','features.pt'))
        for key in ('text_model','vision_model'):
            paths.update(str(Path(cfg[key])/n) for n in model_hashes(cfg[key]))
        paths.update(str(p) for p in Path(cfg['text_model']).iterdir() if p.is_file())
    return {p:file_hash(p) for p in sorted(paths)}


def validate_smokes(smoke_root):
    manifest = json.loads((Path(smoke_root)/'manifest.json').read_text())
    identity = code_identity()
    if manifest.get('source_sha256') != identity['source_sha256']:
        raise ValueError('smoke source identity differs from current source')
    verify_files(manifest['resource_sha256'])
    for arm in TRAJECTORIES:
        for gpu,ds in enumerate(DATASETS):
            path = Path(smoke_root)/arm/ds/'integration_check.json'
            check = json.loads(path.read_text())
            if (check.get('status') != 'passed' or check.get('device') != 'cuda:0'
                    or check.get('physical_gpu') != gpu or check.get('peak_cuda_allocated_bytes',0) <= 0
                    or check.get('optimizer_updates') != 2 or check.get('ema_updates') != 2
                    or check.get('head_updated') is not True or check.get('finite_gradients') is not True
                    or check.get('source_sha256') != identity['source_sha256']
                    or check.get('config_sha256') != file_hash(ROOT/f'configs/ema_{arm}.json')
                    or not all(check.get(k) is True for k in ('online_reload_exact','ema_reload_exact'))):
                raise ValueError(f'CUDA smoke rejected: {path}')
    return manifest


def run_cohort(output, smoke_root):
    output = Path(output).resolve()
    # Admission precedes any new output directory or child training process.
    smoke = validate_smokes(smoke_root)
    resources = resource_hashes()
    if smoke and resources != smoke['resource_sha256']:
        raise ValueError('smoke resource set differs from current resources')
    identity = code_identity()
    groups = cohort_groups(output)
    jobs = []
    for arm in TRAJECTORIES:
        for group in groups[arm]:
            for job in group:
                cfg,_ = resolve_experiment(f'configs/ema_{arm}.json',job['dataset'],job['seed'],'cuda:0')
                jobs.append(dict(job,run_config=cfg))
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'launch.json',dict(seeds=SEEDS,jobs=jobs,code=identity,resource_sha256=resources,
        smoke_root=str(Path(smoke_root).resolve()),executable=sys.executable,evaluate_test=False))
    state = dict(state='running',pid=os.getpid(),started_at=time.time())
    write_json(output/'status.json',state)
    def verify():
        if code_identity()['source_sha256'] != identity['source_sha256']:
            raise ValueError('source or configuration changed during cohort')
        verify_files(resources)
    try:
        states = []
        for arm in TRAJECTORIES:
            states.append(run_schedule(groups[arm],output/arm,training_command,before_group=verify,
                                       after_group=lambda:collect(output)))
        result = collect(output)
        state.update(state='completed' if result['state']=='completed' and all(s['state']=='completed' for s in states)
                     else 'failed',ended_at=time.time())
        write_json(output/'status.json',state)
        if state['state'] != 'completed':
            raise RuntimeError('cohort incomplete; inspect per-job status')
        return result
    except BaseException as error:
        state.update(state='failed',ended_at=time.time(),error=repr(error))
        write_json(output/'status.json',state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--smoke-root',required=True)
    parser.add_argument('--detach',action='store_true')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.detach:
        if output.exists():
            raise FileExistsError(output)
        validate_smokes(Path(args.smoke_root))
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.with_name(output.name+'.log').open('x') as log:
            child = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(output),
                '--smoke-root',str(Path(args.smoke_root).resolve())],stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
        write_json(output.with_name(output.name+'.pid.json'),dict(pid=child.pid))
        print(json.dumps(dict(pid=child.pid,status=str(output/'status.json')),ensure_ascii=False))
    else:
        run_cohort(output,Path(args.smoke_root))


if __name__=='__main__':
    main()
