"""Extract all paired Dev scores or summarize the locked cross-fitting protocol."""
import argparse
import json
import os
import re
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from mact.experiment import ROOT
from mact.fusion_experiment import SEEDS, target_run, extract_run, load_verified, crossfit, LIMITATION
from mact.runtime import write_json
import torch


def deny_test(event,args,output_root=None):
    if event == 'open' and isinstance(args[0],(str,bytes)):
        p=Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(ROOT/'assets/data') and p.name != 'dev.json':
            raise RuntimeError('only Dev data allowed')
        if p.is_relative_to(ROOT/'outputs') or p.is_relative_to(ROOT/'reference'):
            if any(re.search(r'(^|[_\-.])test([_\-.]|$)',x.lower()) or x == 'evaluations' for x in p.parts):
                raise RuntimeError('Test artifacts forbidden')
            if output_root is not None and p.is_relative_to(Path(output_root).resolve()):
                return
            allowed={'config.json','summary.json','best.pt','best_dev_predictions.json'}
            tokenizer_files={'tokenizer.json','tokenizer_config.json','special_tokens_map.json',
                'added_tokens.json','spm.model','sentencepiece.bpe.model','tokenizer.model',
                'vocab.txt','vocab.json','merges.txt','chat_template.jinja'}
            if p.name not in allowed and not ('tokenizer' in p.parts and p.name in tokenizer_files):
                raise RuntimeError('experiment input is not an approved Dev artifact')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['extract','summarize'])
    parser.add_argument('--root',required=True)
    parser.add_argument('--dataset',choices=['twitter2015','twitter2017'])
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    output=Path(args.root)
    sys.addaudithook(lambda event,values:deny_test(event,values,output))
    for path in [ROOT/'assets/data/twitter2015/test.json',
                 ROOT/'outputs/rdrop_s42_43_44_20261013/test-comparison.json',
                 ROOT/'outputs/run/seed42-test.log']:
        try: path.read_text()
        except RuntimeError: pass
        else: raise AssertionError('Test read guard failed')
    output.mkdir(parents=True,exist_ok=True)
    if args.action == 'extract':
        if not args.dataset: parser.error('extract requires --dataset')
        resources=dict(visual_cache=str(ROOT/'assets/cache'/args.dataset/'train_dev'),
                       vision_model=str(ROOT/'assets/models/siglip2'),image_root=str(ROOT/'assets/images'/args.dataset))
        seeds=[args.seed] if args.seed is not None else SEEDS
        for seed in seeds:
            path=output/args.dataset/f'seed{seed}.json'
            run=target_run(args.dataset,seed)
            cfg=json.loads((run/'config.json').read_text())
            if Path(cfg['dev_data']).resolve() != (ROOT/'assets/data'/args.dataset/'dev.json').resolve():
                raise ValueError('extraction requires the canonical Dev dataset')
            result=extract_run(run,args.dataset,seed,args.device,path,resources)
            print(json.dumps(dict(dataset=args.dataset,seed=seed,state='verified',aspects=len(result['keys']))),flush=True)
    else:
        if (output/'comparison.json').exists(): raise FileExistsError('summary must be fresh')
        torch.set_num_threads(4)
        results={}
        for dataset in ('twitter2015','twitter2017'):
            runs=[load_verified(output/dataset/f'seed{s}.json',dataset,s) for s in SEEDS]
            results[dataset]=crossfit(runs)
        eligible=all(r['cohorts']['all']['delta_vs_'+c]['mean']>0 for r in results.values() for c in ('baseline','target'))
        report=dict(results=results,test_used=False,test_read_guard_verified=True,
                    eligible_for_test=eligible,limitation=LIMITATION)
        write_json(output/'comparison.json',report)
        lines=[LIMITATION,'','Cross-fitted Dev results (F1 points; sample SD):']
        for dataset,r in results.items():
            for cohort,c in r['cohorts'].items():
                lines.append(f'{dataset} {cohort}: '+json.dumps({k:v for k,v in c.items() if k != 'seeds'}))
            lines.append('Fold parameters: '+json.dumps(r['fold_parameters']))
            lines.append('Full-Dev deployment parameters (not cross-fitted score): '+json.dumps(r['full_dev_parameters']))
        lines.append(f'Eligible for Test consideration: {eligible}; no Test evaluation performed.')
        (output/'REPORT.txt').write_text('\n'.join(lines)+'\n')
        print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
