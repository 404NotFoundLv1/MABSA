"""Recompute the complete paired Dev results, without opening Test resources."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mact.llrd_ema_results import collect

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=collect(Path(args.output))
    print(json.dumps({k:result[k] for k in ('state','aggregate','selected_candidate')},indent=2))
