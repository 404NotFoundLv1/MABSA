"""One source tree, concurrent seeds within a group, sequential groups."""
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from .runtime import write_json


def build_groups(seeds, group_size, gpus, output):
    seeds=list(seeds);gpus=list(gpus)
    if (not seeds or len(set(seeds))!=len(seeds) or any(type(s)!=int or s<0 for s in seeds)
            or type(group_size)!=int or group_size<1 or not gpus or any(type(g)!=int or g<0 for g in gpus)):
        raise ValueError('unique nonnegative seeds, positive group size and GPU indices required')
    return [[dict(dataset=ds,seed=seed,gpu=gpus[i%len(gpus)],output=str(Path(output)/ds/f'seed{seed}'))
             for i,ds in enumerate(('twitter2015','twitter2017')) for seed in seeds[start:start+group_size]]
            for start in range(0,len(seeds),group_size)]


def run_schedule(groups, output, command, before_group=None, after_group=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'logs').mkdir()
    state=dict(state='running',started_at=time.time(),active_group=None,groups=len(groups),jobs=[])
    write_json(output/'batch-status.json',state)
    children=[]
    handlers={}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'scheduler received signal {signum}')
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT,signal.SIGTERM):
            handlers[sig]=signal.signal(sig,interrupted)
    try:
        for index,jobs in enumerate(groups,1):
            if before_group:before_group()
            state['active_group']=index
            write_json(output/'batch-status.json',state)
            print(f'Starting group {index}/{len(groups)}: seeds {sorted({j["seed"] for j in jobs})}',flush=True)
            children=[]
            for job in jobs:
                entry=dict(job,group=index,state='starting',started_at=time.time())
                state['jobs'].append(entry)
                try:
                    with (output/'logs'/f"{job['dataset']}-seed{job['seed']}.log").open('x') as log:
                        child=subprocess.Popen(command(job),env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['gpu'])),
                            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    children.append((child,entry));entry.update(pid=child.pid,state='running')
                except Exception as error:
                    entry.update(state='failed',error=repr(error),ended_at=time.time())
                write_json(output/'batch-status.json',state)
            pending=list(children)
            while pending:
                for child,entry in list(pending):
                    code=child.poll()
                    if code is None:continue
                    pending.remove((child,entry))
                    status=Path(entry['output']).with_name(f"seed{entry['seed']}.status.json")
                    try:
                        finished=json.loads(status.read_text()) if status.exists() else {}
                    except (ValueError,OSError) as error:
                        finished=dict(state='failed',error=repr(error))
                    entry.update(returncode=code,ended_at=time.time(),
                                 state='completed' if code==0 and finished.get('state')=='completed' else 'failed')
                    write_json(output/'batch-status.json',state)
                if pending:time.sleep(.1)
            # Each directly owned process has exited; no intermediate supervisor can orphan it.
            print(f'Finished group {index}/{len(groups)}',flush=True)
            if after_group:after_group()
        state.update(state='completed' if all(j['state']=='completed' for j in state['jobs']) else 'failed',
                     active_group=None,ended_at=time.time())
        write_json(output/'batch-status.json',state)
        return state
    except BaseException as error:
        for child,_ in children:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for child,_ in children:
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        for _,entry in children:
            if entry['state'] in ('starting','running'):
                entry.update(state='failed',error='scheduler interrupted',ended_at=time.time())
        state.update(state='failed',ended_at=time.time(),error=repr(error))
        write_json(output/'batch-status.json',state)
        raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
