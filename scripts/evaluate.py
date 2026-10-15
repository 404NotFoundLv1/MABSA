"""Evaluate an archived or new checkpoint with project-local resources."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from mact.experiment import evaluate

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--dataset',choices=('twitter2015','twitter2017'),required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--config',default='configs/base.json')
    p.add_argument('--split',choices=('dev','test'),default='test')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--predict-only',action='store_true')
    p.add_argument('--ocr-cache',help='New split OCR asset, verified against checkpoint training pipeline')
    metrics,_=evaluate(**vars(p.parse_args()))
    print(json.dumps(metrics,indent=2))
