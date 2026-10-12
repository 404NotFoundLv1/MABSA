"""Actual-backbone train/save/reload check on 16 Train and 8 Dev examples."""

import argparse
import gc
import json
from pathlib import Path

import torch

from .evaluate import run_evaluation
from .runtime import write_json
from .train import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--mode', choices=['text', 'fusion', 'evidence'])
    args = parser.parse_args()
    output = Path(args.output)
    cfg = json.loads(Path(args.config).read_text())
    if args.mode:
        cfg['model']['mode'] = args.mode
    cfg.update(epochs=1, train_limit=16, dev_limit=8, max_updates=2, device=args.device, log_every=1)
    summary = run_training(cfg, output)
    if summary['emission_head_delta_l2'] <= 0:
        raise AssertionError('emission head did not train')
    gc.collect()
    torch.cuda.empty_cache()
    # Preserve original implicit IDs when writing the explicitly limited Dev file.
    dev_records = json.loads(Path(cfg['dev_data']).read_text())[:8]
    for i, row in enumerate(dev_records):
        row.setdefault('id', f"{Path(cfg['dev_data']).stem}:{i}:{row['image_id']}")
    write_json(output / 'smoke_dev.json', dev_records)
    metrics, predictions = run_evaluation(output / 'best.pt', output / 'smoke_dev.json',
                                           output / 'reloaded', device=args.device)
    expected = json.loads((output / 'best_dev_predictions.json').read_text())
    if json.loads(json.dumps(predictions)) != expected or metrics != summary['best_dev']:
        raise AssertionError('checkpoint reload changed predictions or metrics')
    write_json(output / 'integration_check.json', {'status': 'passed', 'head_updated': True,
               'checkpoint_prediction_equality': True, 'checkpoint_metric_equality': True,
               'train_examples': 16, 'dev_examples': 8, 'optimizer_updates': 2})
    print(json.dumps({'event': 'integration_passed', 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
