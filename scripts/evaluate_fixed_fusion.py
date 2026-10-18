"""Evaluate the user-approved, locked raw half-mixture on Test; no tuning flags."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from mact.fusion_test import load_decision,run_pair,test_paths,summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','summarize'])
    parser.add_argument('--root',required=True)
    parser.add_argument('--dataset',choices=['twitter2015','twitter2017'])
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    root=Path(args.root);decision=root/'decision.json'
    lock=load_decision(decision)
    if args.action=='run':
        if args.dataset not in lock['datasets']:parser.error('run requires a locked dataset')
        for seed in lock['seeds']:
            result=run_pair(decision,args.dataset,seed,args.device,root/args.dataset/f'seed{seed}',test_paths(args.dataset,seed))
            print(json.dumps(dict(dataset=args.dataset,seed=seed,state='verified',
                joint_f1={k:100*v['joint']['f1'] for k,v in result['metrics'].items()})),flush=True)
    else:
        result=summarize(decision,root)
        print(json.dumps(result['datasets'],indent=2),flush=True)


if __name__=='__main__':main()
