"""Train one seed; select on Dev and evaluate full Test once after training."""
import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from mact.experiment import run_experiment

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/base.json')
    p.add_argument('--dataset',choices=('twitter2015','twitter2017'),required=True)
    p.add_argument('--seed',type=int)
    p.add_argument('--device')
    p.add_argument('--output')
    p.add_argument('--train-only',action='store_true')
    a=vars(p.parse_args());a['evaluate_test']=not a.pop('train_only')
    run_experiment(**a)
