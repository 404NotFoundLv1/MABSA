import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from mact.schedule import build_groups, run_schedule


class ScheduleTests(unittest.TestCase):
    def test_interrupt_and_termination_cleanup_all_workers(self):
        source=Path(__file__).resolve().parents[1]/'src'
        for sig in (signal.SIGINT,signal.SIGTERM):
            with self.subTest(signal=sig),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'batch'
                program='''import sys
from pathlib import Path
from mact.schedule import build_groups,run_schedule
out=Path(sys.argv[1])
run_schedule(build_groups([20],1,[0,1],out),out,lambda j:[sys.executable,'-c','import time; time.sleep(30)'])
'''
                scheduler=subprocess.Popen([sys.executable,'-c',program,str(out)],
                    env=dict(os.environ,PYTHONPATH=str(source)),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                workers=[]
                try:
                    deadline=time.monotonic()+15
                    while time.monotonic()<deadline:
                        path=out/'batch-status.json'
                        state=json.loads(path.read_text()) if path.exists() else {}
                        workers=[j['pid'] for j in state.get('jobs',[]) if 'pid' in j]
                        if len(workers)==2:break
                        time.sleep(.05)
                    self.assertEqual(len(workers),2)
                    time.sleep(.2)
                    scheduler.send_signal(sig)
                    scheduler.wait(timeout=5)
                    state=json.loads((out/'batch-status.json').read_text())
                    self.assertEqual(state['state'],'failed')
                    for pid in workers:
                        path=Path(f'/proc/{pid}/stat')
                        self.assertTrue(not path.exists() or path.read_text().rsplit(')',1)[1].split()[0]=='Z')
                finally:
                    if scheduler.poll() is None:scheduler.kill();scheduler.wait()
                    for pid in workers:
                        try:os.killpg(pid,signal.SIGKILL)
                        except ProcessLookupError:pass

    def test_seeds_are_grouped_and_both_datasets_receive_each_seed(self):
        groups=build_groups(range(20,32),4,[0,1],Path('/tmp/unused'))
        self.assertEqual([len(g) for g in groups],[8,8,8])
        self.assertEqual({j['seed'] for j in groups[1]},{24,25,26,27})
        self.assertTrue(all(j['gpu']==(0 if j['dataset']=='twitter2015' else 1) for g in groups for j in g))
        with self.assertRaises(ValueError):build_groups([1,1],4,[0,1],Path('/tmp/unused'))

    def test_group_barrier_and_failed_worker_do_not_skip_remaining_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'batch';script=root/'worker.py'
            script.write_text('''import json,sys,time,os
from pathlib import Path
p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);p.mkdir()
assert os.environ['CUDA_VISIBLE_DEVICES']==sys.argv[2]
start=time.monotonic_ns();time.sleep(.05)
state='failed' if 'seed20' in str(p) else 'completed'
(p/'times.json').write_text(json.dumps([start,time.monotonic_ns()]))
p.with_name(p.name+'.status.json').write_text(json.dumps({'state':state}))
raise SystemExit(1 if state=='failed' else 0)
''')
            groups=build_groups(range(20,23),1,[0,1],out)
            result=run_schedule(groups,out,lambda j:[sys.executable,str(script),j['output'],str(j['gpu'])])
            self.assertEqual(result['state'],'failed')
            self.assertEqual(len(result['jobs']),6)
            for i in (0,1):
                ended=max(json.loads((Path(j['output'])/'times.json').read_text())[1] for j in groups[i])
                started=min(json.loads((Path(j['output'])/'times.json').read_text())[0] for j in groups[i+1])
                self.assertLessEqual(ended,started)
            self.assertEqual(sum(j['state']=='completed' for j in result['jobs']),4)
