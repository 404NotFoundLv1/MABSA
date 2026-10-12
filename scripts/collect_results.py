"""Collect complete runs and report mean/sample SD only for the requested seed set."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mact.experiment import summarize_rows
from mact.runtime import write_json

def collect(output,seeds):
    output=Path(output);rows=[]
    for dataset in ('twitter2015','twitter2017'):
        for seed in seeds:
            run=output/dataset/f'seed{seed}';status=run.with_name(run.name+'.status.json')
            if not status.exists() or json.loads(status.read_text()).get('state')!='completed':continue
            summary=json.loads((run/'summary.json').read_text())
            cfg=json.loads((run/'config.json').read_text())
            selection=json.loads((run/'test/selection.json').read_text())
            metrics=json.loads((run/'test/metrics.json').read_text())
            if (summary['is_smoke'] or summary['completed_epochs']!=cfg['epochs'] or cfg['seed']!=seed
                    or selection['selected_on']!='dev_joint_f1' or selection['test_used_for_selection']
                    or selection['selected_epoch']!=summary['best_epoch']
                    or selection['checkpoint_sha256']!=summary['checkpoint_sha256']):
                raise ValueError(f'inconsistent run: {run}')
            rows.append(dict(dataset=dataset,seed=seed,epochs=cfg['epochs'],best_epoch=summary['best_epoch'],
                test_joint_f1=metrics['joint']['f1'],test_metrics=metrics))
    summary=summarize_rows(rows,expected_seeds=seeds)
    result=dict(state='completed' if all(s['state']=='completed' for s in summary.values()) else 'incomplete',
                expected_seeds=list(seeds),rows=rows,summary=summary)
    write_json(output/'results.json',result)
    lines=['Full official Test; checkpoint selected by Dev Joint F1; metrics in percent.']
    for row in rows:
        j=row['test_metrics']['joint']
        lines.append(f"{row['dataset']} seed {row['seed']}: P={j['precision']*100:.2f} R={j['recall']*100:.2f} F1={j['f1']*100:.2f}")
    for dataset,s in summary.items():
        lines.append(f"{dataset}: {s['test_joint_f1_mean']*100:.2f} +/- {s['test_joint_f1_sample_std']*100:.2f}" if s['state']=='completed' and len(seeds)>1 else f"{dataset}: {s}")
    (output/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--seeds',nargs='+',type=int,required=True)
    a=p.parse_args();print(json.dumps(collect(a.output,a.seeds),indent=2))
