"""Prepare/run the sequential seed42 three-fold Train/Dev correction experiment."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from mact.oof_correction import prepare_pipeline, run_pipeline, verify_launch
from mact.runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, default=ROOT/'outputs/oof_correction_s42_20261018')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.detach: parser.error('detach applies only to run')
        launch = prepare_pipeline(args.root, args.device)
        print('Prepared', len(launch['jobs']), 'fold pairs at', args.root)
    elif args.detach:
        verify_launch(args.root)
        log = args.root/'worker.log'
        with log.open('x') as handle:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'run', '--root', str(args.root.resolve())],
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
        write_json(args.root/'worker.pid.json', dict(pid=child.pid))
        print('Worker PID:', child.pid, 'status:', args.root/'status.json', flush=True)
    else:
        run_pipeline(args.root)


if __name__ == '__main__': main()
