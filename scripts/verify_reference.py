"""Re-evaluate all four archived seed24/25 checkpoints with local resources."""
import argparse,gc,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
from mact.experiment import evaluate
from mact.runtime import write_json

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:0');p.add_argument('--output',default=str(ROOT/'verification/reference-evaluation'));a=p.parse_args()
 report=[]
 for dataset in ('twitter2015','twitter2017'):
  for seed in (24,25):
   run=ROOT/'reference/runs'/dataset/f'seed{seed}'
   output=Path(a.output)/f'{dataset}-seed{seed}'
   metrics,predictions=evaluate(run/'best.pt',dataset,output,device=a.device)
   expected_metrics=json.loads((run/'test/metrics.json').read_text());expected_predictions=json.loads((run/'test/predictions.json').read_text())
   assert metrics==expected_metrics,(dataset,seed,metrics,expected_metrics)
   assert json.loads(json.dumps(predictions))==expected_predictions,(dataset,seed,'prediction mismatch')
   item=dict(dataset=dataset,seed=seed,metrics_exact=True,predictions_exact=True,examples=len(predictions),metrics=metrics)
   report.append(item);print(json.dumps(item),flush=True)
   gc.collect()
   if a.device.startswith('cuda'):torch.cuda.empty_cache()
 Path(a.output).mkdir(parents=True,exist_ok=True)
 write_json(Path(a.output)/'verification.json',dict(status='passed',runs=report))
if __name__=='__main__':main()
