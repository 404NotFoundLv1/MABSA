#!/usr/bin/env python3
"""Run the prespecified seed42 frozen correction pilot; deliberately no Test path."""
import argparse
import json
from pathlib import Path

from mact.correction_experiment import extract_pilot, train_pair, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('extract', 'train', 'summarize'))
    parser.add_argument('--dataset', choices=('twitter2015', 'twitter2017'))
    parser.add_argument('--root', type=Path, default=Path('outputs/sentiment_correction_s42_20261018'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.action != 'summarize' and args.dataset is None: parser.error('--dataset required')
    if args.action == 'extract':
        extract_pilot(args.dataset, 42, args.device, args.root)
    elif args.action == 'train':
        path = args.root/args.dataset
        train_pair(path/'features/train.pt', path/'features/dev.pt', path/'fit')
    else:
        print(json.dumps(summarize(args.root), indent=2))


if __name__ == '__main__': main()
