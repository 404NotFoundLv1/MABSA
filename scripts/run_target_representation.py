"""Sequential A/B/C seed42 pilot; Train/Dev only, immutable MACT and 1:1 fusion."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from mact.common.cache import file_hash
from mact.data import load_labeled
from mact.experiment import ROOT, code_identity
from mact.fusion_experiment import target_run, extract_run, load_verified, _verify_hashes
from mact.representation_experiment import pilot_config, run_training, verify_completed
from mact.representation_diagnostic import diagnose, dataset_statistics, error_transitions
from mact.runtime import write_json


def schedule():
    return [(d, v) for d in ('twitter2015', 'twitter2017') for v in ('marker', 'span', 'context')]


def deny_test(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = Path(os.fsdecode(args[0])).resolve()
        if any(path.is_relative_to(ROOT / p) for p in ('assets', 'outputs', 'reference')):
            if any(re.search(r'(^|[_\-.])test([_\-.]|$)', p.lower()) for p in path.parts):
                raise RuntimeError('Test data, cache and results are forbidden in this pilot')


def lock_decision(output, decision):
    output = Path(output)
    path = output / 'decision.json'
    sha_path = output / 'decision.sha256'
    if path.exists():
        if not sha_path.is_file() or file_hash(path) != sha_path.read_text().strip():
            raise ValueError('decision hash mismatch')
        if json.loads(path.read_text()) != decision:
            raise ValueError('locked decision differs from current config/source/inputs')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('output must be fresh; partial directory found')
        output.mkdir(parents=True, exist_ok=True)
        write_json(path, decision)
        sha_path.write_text(file_hash(path) + '\n')


def make_decision(output, device, smoke):
    configs, paths = {}, set()
    for dataset, variant in schedule():
        cfg = pilot_config(dataset, variant, output / 'scores' / (dataset + '.json'), smoke)
        cfg['device'] = device
        configs[dataset + '/' + variant] = cfg
        paths.update(Path(cfg[k]).resolve() for k in ('train_data', 'dev_data', 'baseline_predictions',
            'baseline_summary', 'baseline_checkpoint'))
        paths.update(p for p in Path(cfg['text_model']).rglob('*') if p.is_file())
        historical = target_run(dataset, 42)
        paths.update(historical / name for name in ('best.pt', 'summary.json', 'config.json', 'best_dev_predictions.json'))
        for directory in (historical / 'tokenizer', Path(cfg['baseline_checkpoint']).parent / 'tokenizer'):
            paths.update(p for p in directory.rglob('*') if p.is_file())
        paths.update(ROOT / 'assets/cache' / dataset / 'train_dev' / name for name in ('manifest.json', 'features.pt'))
    return dict(format='mact_representation_decision_v1', code=code_identity()['source_sha256'],
        configs=configs, input_sha256={str(p): file_hash(p) for p in sorted(paths)},
        alpha=.5, temperatures=[1., 1.], test_used=False, smoke=smoke, seed=42)


def paired_rows(rows):
    if set(rows) != set(schedule()):
        raise ValueError('incomplete A/B/C cohort')
    result = {}
    for dataset, variant in schedule():
        row = dict(rows[dataset, variant])
        current = row['best_fusion']['joint']['f1']
        row['delta_vs_A_pp'] = 100 * (current - rows[dataset, 'marker']['best_fusion']['joint']['f1'])
        row['delta_vs_historical_pp'] = 100 * (current - row['historical_reference']['fusion']['joint']['f1'])
        result.setdefault(dataset, {})[variant] = row
    return result


def summarize(output, decision):
    rows = {}
    for dataset, variant in schedule():
        cfg = decision['configs'][dataset + '/' + variant]
        rows[dataset, variant] = verify_completed(output / dataset / variant, cfg)
    paired = paired_rows(rows)
    for dataset in paired:
        cache = load_verified(output / 'scores' / (dataset + '.json'), dataset, 42)
        a = json.loads((output / dataset / 'marker' / 'best_fusion_predictions.json').read_text())
        baseline = {(k, s, e): p for k, vals in cache['baseline'].items() for s, e, p in vals}
        target = {(k, s, e): p for k, vals in cache['target'].items() for s, e, p in vals}
        for variant in paired[dataset]:
            p = json.loads((output / dataset / variant / 'best_fusion_predictions.json').read_text())
            lookup = {(k, s, e): y for k, vals in p.items() for s, e, y in vals}
            paired[dataset][variant]['vs_A'] = error_transitions(cache['gold'], a, p)
            paired[dataset][variant]['historical_shared_error_rescues'] = sum(
                baseline[key] != y and target[key] != y and lookup[key] == y
                for k, vals in cache['gold'].items() for s, e, y in vals
                if (key := (k, s, e)) in baseline)
    report = dict(results=paired, seed=42, test_used=False, is_smoke=decision['smoke'],
        limitation='Single-seed exploratory Dev selection, no Test/generalization or novelty claim.')
    write_json(output / 'comparison.json', report)
    lines = ['A/B/C target representation pilot — ' + ('SMOKE ONLY' if decision['smoke'] else 'Dev seed42'),
        'A=marker; B=marker+aspect; C=B+target-conditioned context.',
        'All checkpoints selected by earliest best fixed-fusion predicted-boundary Dev Joint F1.',
        'variant | dataset | epoch | target F1 | fusion F1 | delta vs A | delta vs historical']
    for dataset, variant in schedule():
        r = paired[dataset][variant]
        lines.append(f"{variant} | {dataset} | {r['best_epoch']} | {100*r['best_target']['joint']['f1']:.4f} | "
            f"{100*r['best_fusion']['joint']['f1']:.4f} | {r['delta_vs_A_pp']:+.4f} | {r['delta_vs_historical_pp']:+.4f}")
    lines.append(report['limitation'])
    (output / 'SUMMARY.txt').write_text('\n'.join(lines) + '\n')
    return report


def run_child(args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), *args],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('batch', 'extract', 'train', 'summarize'), default='batch')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dataset', choices=('twitter2015', 'twitter2017'))
    parser.add_argument('--variant', choices=('marker', 'span', 'context'))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    sys.addaudithook(deny_test)
    if args.action != 'batch':
        decision_path = output / 'decision.json'
        if file_hash(decision_path) != (output / 'decision.sha256').read_text().strip():
            raise ValueError('decision integrity failure')
        decision = json.loads(decision_path.read_text())
        if decision['code'] != code_identity()['source_sha256']:
            raise ValueError('source changed after decision')
        if args.action == 'extract':
            if not args.dataset: parser.error('extract needs dataset')
            resources = dict(visual_cache=str(ROOT / 'assets/cache' / args.dataset / 'train_dev'),
                vision_model=str(ROOT / 'assets/models/siglip2'), image_root=str(ROOT / 'assets/images' / args.dataset))
            extract_run(target_run(args.dataset, 42), args.dataset, 42, args.device,
                        output / 'scores' / (args.dataset + '.json'), resources)
        elif args.action == 'train':
            if not args.dataset or not args.variant: parser.error('train needs dataset and variant')
            cfg = decision['configs'][args.dataset + '/' + args.variant]
            run_training(cfg, output / args.dataset / args.variant)
        else:
            summarize(output, decision)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    # A parent-directory lock keeps partial root directories distinguishable.
    with output.with_name(output.name + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        decision = make_decision(output, args.device, args.smoke)
        lock_decision(output, decision)
        started = time.time()
        status = dict(state='running', pid=os.getpid(), started_at=started, completed_jobs=[],
                      seed=42, test_used=False, is_smoke=args.smoke)
        write_json(output / 'status.json', status)
        try:
            for dataset in ('twitter2015', 'twitter2017'):
                status['current_job'] = dataset + '/extract-and-diagnose'
                write_json(output / 'status.json', status)
                score_path = output / 'scores' / (dataset + '.json')
                if not score_path.exists():
                    run_child(['--action', 'extract', '--output', str(output), '--dataset', dataset,
                               '--device', args.device], output / 'logs' / (dataset + '-extract.log'))
                cache = load_verified(score_path, dataset, 42)
                cfg = decision['configs'][dataset + '/marker']
                train, _ = load_labeled(cfg['train_data']); dev, _ = load_labeled(cfg['dev_data'])
                report = dict(train=dataset_statistics(train), dev=diagnose(dev, cache))
                write_json(output / (dataset + '-diagnostic.json'), report)
            for dataset, variant in schedule():
                key = dataset + '/' + variant
                cfg = decision['configs'][key]
                job = output / dataset / variant
                status.update(current_job=key)
                write_json(output / 'status.json', status)
                if job.exists() or job.with_name(job.name + '.status.json').exists():
                    verify_completed(job, cfg)
                else:
                    print(f'Starting {key}', flush=True)
                    run_child(['--action', 'train', '--output', str(output), '--dataset', dataset,
                               '--variant', variant], output / 'logs' / (dataset + '-' + variant + '.log'))
                    verify_completed(job, cfg)
                status['completed_jobs'].append(key)
                write_json(output / 'status.json', status)
            _verify_hashes(decision['input_sha256'])
            summarize(output, decision)
            status.update(state='completed', ended_at=time.time(), current_job=None)
            write_json(output / 'status.json', status)
        except BaseException as error:
            status.update(state='failed', ended_at=time.time(), error=repr(error))
            write_json(output / 'status.json', status)
            raise


if __name__ == '__main__':
    main()
