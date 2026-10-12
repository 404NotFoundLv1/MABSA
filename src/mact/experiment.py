"""Project configuration, run provenance and complete-seed result aggregation."""
import gc
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

from .common.cache import file_hash
from .train import resolve_config, run_training
from .evaluate import run_evaluation
from .runtime import write_json
from .logging import TrainingLogger

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ('twitter2015', 'twitter2017')
PATH_KEYS = ('text_model','vision_model','visual_cache','train_data','dev_data','image_root',
             'test_data','test_visual_cache')


def resolve_experiment(config, dataset, seed=None, device=None):
    if dataset not in DATASETS:
        raise ValueError(f'unknown dataset: {dataset}')
    config = Path(config)
    if not config.is_absolute():
        config = ROOT / config
    cfg = json.loads(config.read_text())
    cfg.update(json.loads((ROOT / 'configs/datasets' / f'{dataset}.json').read_text()))
    for key in PATH_KEYS:
        path = Path(cfg[key])
        path = (path if path.is_absolute() else ROOT / path).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f'{key} must be inside the independent MACT project: {path}')
        cfg[key] = str(path)
    if seed is not None:
        cfg['seed'] = seed
    if device is not None:
        cfg['device'] = device
    test = {key: cfg.pop(key) for key in ('test_data', 'test_visual_cache')}
    return resolve_config(cfg), test


def code_identity():
    files = sorted((ROOT/'src').rglob('*.py')) + sorted((ROOT/'scripts').glob('*.py')) + sorted((ROOT/'configs').rglob('*.json'))
    result = dict(source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in files})
    try:
        result['git_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,
                                                       stderr=subprocess.DEVNULL).strip()
        result['git_dirty'] = bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        result.update(git_commit=None, git_dirty=None)
    return result


def evaluate(checkpoint, dataset, output, config='configs/base.json', split='test', device='cpu',
             predict_only=False):
    cfg, test = resolve_experiment(config, dataset, device=device)
    if split not in ('dev','test'):
        raise ValueError('evaluation split must be dev or test')
    return run_evaluation(checkpoint, test['test_data'] if split=='test' else cfg['dev_data'], output,
        device=device, batch_size=cfg['eval_batch_size'], predict_only=predict_only,
        visual_cache=test['test_visual_cache'] if split=='test' else cfg['visual_cache'],
        resource_paths={key:cfg[key] for key in ('vision_model','image_root')})


def run_experiment(config, dataset, seed=None, device=None, output=None, evaluate_test=True):
    import torch
    cfg, test = resolve_experiment(config, dataset, seed=seed, device=device)
    output = Path(output).resolve() if output else ROOT/'outputs'/dataset/f"seed{cfg['seed']}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'output must be fresh: {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    status_path = output.parent / (output.name+'.status.json')
    if status_path.exists():
        raise FileExistsError(f'run status already exists: {status_path}')
    identity = code_identity()
    data_hashes = {p:file_hash(p) for p in (cfg['train_data'],cfg['dev_data'],test['test_data'])}
    started = time.time()
    write_json(status_path,dict(state='running',dataset=dataset,seed=cfg['seed'],started_at=started))
    try:
        summary = run_training(cfg,output)
        if summary['is_smoke'] or summary['completed_epochs'] != cfg['epochs']:
            raise ValueError('full experiment did not complete its epoch budget')
        gc.collect()
        if cfg['device'].startswith('cuda'):
            torch.cuda.empty_cache()
        dev, predictions = evaluate(output/'best.pt',dataset,output/'reloaded_dev',config,split='dev',device=cfg['device'])
        if dev != summary['best_dev'] or json.loads(json.dumps(predictions)) != json.loads((output/'best_dev_predictions.json').read_text()):
            raise ValueError('checkpoint reload did not reproduce Dev predictions exactly')
        write_json(output/'reload-verification.json',dict(full_dev=True,predictions_exact=True,metrics_exact=True))
        for name, sha in identity['source_sha256'].items():
            if file_hash(ROOT/name) != sha:
                raise ValueError(f'source changed during run: {name}')
        for path, sha in data_hashes.items():
            if file_hash(path) != sha:
                raise ValueError(f'official data changed: {path}')
        metrics = None
        if evaluate_test:
            metrics,_ = evaluate(output/'best.pt',dataset,output/'test',config,device=cfg['device'])
            write_json(output/'test/selection.json',dict(seed=cfg['seed'],mode=cfg['model']['mode'],
                selected_on='dev_joint_f1',selected_epoch=summary['best_epoch'],
                checkpoint_sha256=summary['checkpoint_sha256'],test_used_for_selection=False))
            logger = TrainingLogger(output)
            logger.info('BEST DEV TEST:-----')
            logger.joint_metrics('BEST DEV--TEST',metrics['joint'])
            logger.info(f"TEST AE  ae_p:{metrics['mate']['precision']*100:.2f} ae_r:{metrics['mate']['recall']*100:.2f} ae_f:{metrics['mate']['f1']*100:.2f}")
        for path, sha in data_hashes.items():
            if file_hash(path) != sha:
                raise ValueError(f'official data changed: {path}')
        write_json(output/'experiment.json',dict(dataset=dataset,seed=cfg['seed'],code=identity,
            evaluation=test,data_sha256=data_hashes,started_at=started,ended_at=time.time()))
        write_json(status_path,dict(state='completed',dataset=dataset,seed=cfg['seed'],
            started_at=started,ended_at=time.time(),best_epoch=summary['best_epoch'],test=metrics))
        return summary
    except BaseException as error:
        write_json(status_path,dict(state='failed',dataset=dataset,seed=cfg['seed'],
            started_at=started,ended_at=time.time(),error=repr(error)))
        raise


def summarize_rows(rows, expected_seeds):
    expected_seeds = tuple(expected_seeds)
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError('expected seeds must be nonempty and unique')
    seen = set()
    for row in rows:
        key = (row['dataset'],row['seed'])
        if key in seen or row['dataset'] not in DATASETS or row['seed'] not in expected_seeds:
            raise ValueError('duplicate or unexpected dataset/seed')
        if not math.isfinite(row['test_joint_f1']) or not 0 <= row['test_joint_f1'] <= 1:
            raise ValueError('invalid Test F1')
        seen.add(key)
    result = {}
    for dataset in DATASETS:
        selected = [r for r in rows if r['dataset']==dataset]
        missing = sorted(set(expected_seeds)-{r['seed'] for r in selected})
        values = [r['test_joint_f1'] for r in selected]
        result[dataset] = dict(state='completed' if not missing else 'incomplete',missing_seeds=missing,
            completed_seeds=sorted(r['seed'] for r in selected),
            test_joint_f1_mean=statistics.mean(values) if not missing else None,
            test_joint_f1_sample_std=statistics.stdev(values) if not missing and len(values)>1 else None)
    return result
